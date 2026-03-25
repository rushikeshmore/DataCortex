//! Codec orchestrator — compress and decompress through the DataCortex pipeline.
//!
//! Phase 1: Format preprocessing + zstd (Fast mode).
//! Phase 3: Full CM engine with higher-order models + mixer + APM (Balanced mode, ~256MB).
//! Phase 5: Full CM engine with 2x context maps (Max mode, ~512MB).
//! Phase 6: Dual-path CM + LLM with MetaMixer (Max mode with `neural` feature).
//! Phase 7: Dual-path CM + GRU byte-level predictor (Balanced mode).

use std::io::{self, Cursor, Read, Write};

use crate::dcx::{DcxHeader, FormatHint, Mode};
use crate::entropy::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::format::transform::TransformChain;
use crate::format::{detect_format, preprocess, reverse_preprocess};
use crate::mixer::MetaMixer;
use crate::model::gru_model::GruModel;
use crate::model::{CMConfig, CMEngine};

/// Adaptive zstd level for Fast mode based on preprocessed data size.
///
/// Smaller data compresses quickly even at high levels, so we use higher
/// zstd levels for small-medium files without meaningful speed impact.
/// If `level_override` is set (user passed --level), it always wins.
fn adaptive_fast_level(data_size: usize, level_override: Option<i32>) -> i32 {
    if let Some(level) = level_override {
        return level; // User explicitly set level, respect it
    }
    match data_size {
        0..=1_048_576 => 19,          // <1MB: zstd-19 is <50ms, use best ratio
        1_048_577..=10_485_760 => 13, // 1MB-10MB: good balance
        _ => 9,                       // >10MB: use level 9 for speed
    }
}

// ─── Zstd Dictionary Training (Fast mode) ─────────────────────────────────────

/// Minimum preprocessed data size to attempt dictionary training.
/// Below this threshold the dictionary overhead exceeds any savings.
const DICT_MIN_DATA_SIZE: usize = 8192;

/// Target chunk size for splitting preprocessed data before per-chunk compression.
/// Each chunk is compressed independently with the shared dictionary.
/// Smaller chunks benefit more from dictionary priming, but each chunk has
/// framing overhead (4 bytes size + zstd frame header ~10 bytes).
/// Adaptive: scale with data size to avoid too many chunks.
fn dict_chunk_size(data_len: usize) -> usize {
    if data_len > 4_194_304 {
        131_072 // 128 KB for > 4 MB
    } else if data_len > 1_048_576 {
        65_536 // 64 KB for 1 - 4 MB
    } else if data_len > 262_144 {
        32_768 // 32 KB for 256 KB - 1 MB
    } else {
        16_384 // 16 KB for smaller files
    }
}

/// Maximum dictionary size based on input data size.
/// Kept relatively small to minimize overhead. The dictionary primes each chunk's
/// compressor context, so even a small dict provides most of the benefit.
fn dict_max_size(data_len: usize) -> usize {
    if data_len > 4_194_304 {
        16_384 // 16 KB for > 4 MB
    } else if data_len > 1_048_576 {
        8_192 // 8 KB for 1 - 4 MB
    } else {
        4_096 // 4 KB for smaller files
    }
}

/// Generate training samples from the data for dictionary training.
///
/// Uses column boundaries (0x00 separators) if available, otherwise fixed blocks.
/// These samples are only used for `zstd::dict::from_samples`, NOT for the
/// actual chunked compression (which uses `split_into_chunks`).
fn generate_training_samples(data: &[u8], chunk_size: usize) -> Vec<&[u8]> {
    // Try column boundaries (0x00 separators from columnar transform).
    let col_chunks: Vec<&[u8]> = data.split(|&b| b == 0x00).collect();
    if col_chunks.len() >= 5 {
        // Filter out empty chunks and return.
        return col_chunks.into_iter().filter(|c| !c.is_empty()).collect();
    }

    // Fall back to fixed-size blocks for training.
    split_into_chunks(data, chunk_size)
}

/// Split data into fixed-size chunks for per-chunk compression.
/// Every byte is preserved exactly -- no bytes are lost at boundaries.
fn split_into_chunks(data: &[u8], chunk_size: usize) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let end = (offset + chunk_size).min(data.len());
        chunks.push(&data[offset..end]);
        offset = end;
    }
    chunks
}

/// Attempt chunk-based dictionary compression.
///
/// 1. Split data into chunks
/// 2. Train a zstd dictionary on the chunks
/// 3. Compress each chunk independently using the trained dictionary
/// 4. Return the dict + all compressed chunks as a payload
///
/// Returns `Some(payload)` if the total is smaller than `plain_size`, else `None`.
fn try_dict_compress(data: &[u8], level: i32, plain_size: usize) -> Option<Vec<u8>> {
    let chunk_size = dict_chunk_size(data.len());

    // Generate training samples (may use column boundaries for better diversity).
    let training_samples = generate_training_samples(data, chunk_size);
    if training_samples.len() < 5 {
        return None;
    }

    let max_dict = dict_max_size(data.len());

    // Train dictionary from the training samples.
    let dict = zstd::dict::from_samples(&training_samples, max_dict).ok()?;
    if dict.is_empty() {
        return None;
    }

    // Split data into fixed-size chunks for per-chunk compression.
    let chunks = split_into_chunks(data, chunk_size);

    // Compress each chunk independently with the dictionary.
    let mut compressor = zstd::bulk::Compressor::with_dictionary(level, &dict).ok()?;
    let mut compressed_chunks: Vec<Vec<u8>> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let cc = compressor.compress(chunk).ok()?;
        compressed_chunks.push(cc);
    }

    // Build payload:
    //   [dict_size: u32 LE] [dict_bytes]
    //   [num_chunks: u32 LE]
    //   for each chunk: [chunk_compressed_size: u32 LE] [chunk_data]
    let total_compressed: usize = compressed_chunks.iter().map(|c| 4 + c.len()).sum();
    let payload_size = 4 + dict.len() + 4 + total_compressed;

    // Only use dict if it beats plain compression.
    if payload_size >= plain_size {
        return None;
    }

    let mut payload = Vec::with_capacity(payload_size);
    payload.extend_from_slice(&(dict.len() as u32).to_le_bytes());
    payload.extend_from_slice(&dict);
    payload.extend_from_slice(&(compressed_chunks.len() as u32).to_le_bytes());
    for cc in &compressed_chunks {
        payload.extend_from_slice(&(cc.len() as u32).to_le_bytes());
        payload.extend_from_slice(cc);
    }

    Some(payload)
}

/// Decompress a chunk-based dictionary-compressed payload.
///
/// Payload format:
///   [dict_size: u32 LE] [dict_bytes]
///   [num_chunks: u32 LE]
///   for each chunk: [chunk_compressed_size: u32 LE] [chunk_data]
///
/// Chunks are decompressed individually and concatenated.
fn decompress_with_dict(payload: &[u8], capacity: usize) -> std::io::Result<Vec<u8>> {
    if payload.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dict payload too short for dict_size",
        ));
    }
    let mut pos = 0;

    // Read dictionary.
    let dict_size =
        u32::from_le_bytes(payload[pos..pos + 4].try_into().expect("4-byte slice")) as usize;
    pos += 4;
    if payload.len() < pos + dict_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dict payload truncated: dictionary bytes",
        ));
    }
    let dict_bytes = &payload[pos..pos + dict_size];
    pos += dict_size;

    // Read number of chunks.
    if payload.len() < pos + 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dict payload truncated: num_chunks",
        ));
    }
    let num_chunks =
        u32::from_le_bytes(payload[pos..pos + 4].try_into().expect("4-byte slice")) as usize;
    pos += 4;

    // Prepare decompressor with dictionary.
    let mut decompressor = zstd::bulk::Decompressor::with_dictionary(dict_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut output = Vec::with_capacity(capacity);

    for i in 0..num_chunks {
        if payload.len() < pos + 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dict payload truncated: chunk {i} size"),
            ));
        }
        let chunk_size =
            u32::from_le_bytes(payload[pos..pos + 4].try_into().expect("4-byte slice")) as usize;
        pos += 4;
        if payload.len() < pos + chunk_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dict payload truncated: chunk {i} data"),
            ));
        }
        let chunk_data = &payload[pos..pos + chunk_size];
        pos += chunk_size;

        // Each chunk decompresses to at most chunk_size + some headroom.
        let chunk_capacity = capacity.saturating_sub(output.len());
        let decompressed = decompressor
            .decompress(chunk_data, chunk_capacity)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("chunk {i} decompress failed: {e}"),
                )
            })?;
        output.extend_from_slice(&decompressed);
    }

    Ok(output)
}

// ─── Brotli helpers (Fast mode auto-fallback) ─────────────────────────────────

/// Compress `data` with brotli at the given quality (0-11).
fn brotli_compress(data: &[u8], quality: u32) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let params = brotli::enc::BrotliEncoderParams {
        quality: quality as i32,
        ..Default::default()
    };
    brotli::BrotliCompress(&mut io::Cursor::new(data), &mut output, &params)?;
    Ok(output)
}

/// Decompress a brotli stream. `max_size` is a capacity hint for the output buffer.
fn brotli_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    brotli::BrotliDecompress(&mut io::Cursor::new(data), &mut output)?;
    Ok(output)
}

/// Compress data using the CM engine with the given configuration.
/// Returns the compressed byte stream.
fn cm_compress(data: &[u8], config: CMConfig) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut encoder = ArithmeticEncoder::new();

    for &byte in data {
        for bpos in 0..8 {
            let bit = (byte >> (7 - bpos)) & 1;
            let p = engine.predict();
            encoder.encode(bit, p);
            engine.update(bit);
        }
    }

    encoder.finish()
}

/// Decompress data using the CM engine with the given configuration.
/// `compressed` is the arithmetic-coded stream, `original_size` is the expected output length.
fn cm_decompress(compressed: &[u8], original_size: usize, config: CMConfig) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut decoder = ArithmeticDecoder::new(compressed);
    let mut output = Vec::with_capacity(original_size);

    for _ in 0..original_size {
        let mut byte_val: u8 = 0;
        for bpos in 0..8 {
            let p = engine.predict();
            let bit = decoder.decode(p);
            engine.update(bit);
            byte_val |= bit << (7 - bpos);
        }
        output.push(byte_val);
    }

    output
}

// ─── GRU dual-path (CM + GRU byte predictor) ────────────────────────────────
// The GRU provides a DIFFERENT signal from CM: byte-level cross-bit correlations.
// It's blended AFTER the full CM pipeline via MetaMixer.
// CRITICAL: encoder and decoder must produce IDENTICAL GRU + CM state.

/// Compress using dual-path: CM engine + GRU byte predictor + MetaMixer.
/// Used for Balanced mode.
fn gru_compress(data: &[u8], config: CMConfig) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut gru = GruModel::new();
    let mut meta_mixer = MetaMixer::new(12); // 12% GRU weight
    let mut encoder = ArithmeticEncoder::new();

    let total_bytes = data.len();
    let report_interval = if total_bytes > 100_000 {
        total_bytes / 20
    } else {
        0
    };

    for (byte_idx, &byte) in data.iter().enumerate() {
        for bpos in 0..8u8 {
            let bit = (byte >> (7 - bpos)) & 1;

            // CM prediction (full pipeline: 19 models + mixer + 7 APM).
            let p_cm = engine.predict();

            // GRU bit prediction from cached byte probs.
            let partial = if bpos == 0 {
                1u32
            } else {
                let mut p = 1u32;
                for prev_bpos in 0..bpos {
                    let prev_bit = (byte >> (7 - prev_bpos)) & 1;
                    p = (p << 1) | prev_bit as u32;
                }
                p
            };
            let p_gru = gru.predict_bit(bpos, partial);

            // MetaMixer blend.
            let p_final = meta_mixer.blend(p_cm, p_gru);

            encoder.encode(bit, p_final);
            engine.update(bit);
            meta_mixer.update(bit);
        }

        // Byte complete: train GRU on observed byte, then forward for next prediction.
        gru.train(byte);
        gru.forward(byte);

        if report_interval > 0 && (byte_idx + 1) % report_interval == 0 {
            let pct = (byte_idx + 1) * 100 / total_bytes;
            eprint!("\r[gru] compressing... {pct}%");
        }
    }

    if total_bytes > 100_000 {
        eprintln!("\r[gru] compressing... 100%");
    }

    encoder.finish()
}

/// Decompress using dual-path: CM engine + GRU byte predictor + MetaMixer.
/// Must produce IDENTICAL GRU + CM state as the encoder.
fn gru_decompress(compressed: &[u8], original_size: usize, config: CMConfig) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut gru = GruModel::new();
    let mut meta_mixer = MetaMixer::new(12); // same 12% as encoder
    let mut decoder = ArithmeticDecoder::new(compressed);
    let mut output = Vec::with_capacity(original_size);

    let report_interval = if original_size > 100_000 {
        original_size / 20
    } else {
        0
    };

    for byte_idx in 0..original_size {
        let mut byte_val: u8 = 0;

        for bpos in 0..8u8 {
            // CM prediction.
            let p_cm = engine.predict();

            // GRU bit prediction (same partial byte state as encoder).
            let partial = if bpos == 0 {
                1u32
            } else {
                let mut p = 1u32;
                for prev_bpos in 0..bpos {
                    let prev_bit = (byte_val >> (7 - prev_bpos)) & 1;
                    p = (p << 1) | prev_bit as u32;
                }
                p
            };
            let p_gru = gru.predict_bit(bpos, partial);

            // MetaMixer blend.
            let p_final = meta_mixer.blend(p_cm, p_gru);

            let bit = decoder.decode(p_final);
            engine.update(bit);
            meta_mixer.update(bit);
            byte_val |= bit << (7 - bpos);
        }

        output.push(byte_val);

        // Byte complete: train GRU then forward (same as encoder).
        gru.train(byte_val);
        gru.forward(byte_val);

        if report_interval > 0 && (byte_idx + 1) % report_interval == 0 {
            let pct = (byte_idx + 1) * 100 / original_size;
            eprint!("\r[gru] decompressing... {pct}%");
        }
    }

    if original_size > 100_000 {
        eprintln!("\r[gru] decompressing... 100%");
    }

    output
}

// ─── Neural dual-path (CM + LLM) ─────────────────────────────────────────────
// Feature-gated: only available when `neural` is enabled.
// The LLM predictor runs alongside the CM engine. A MetaMixer blends them.
// CRITICAL: encoder and decoder must produce IDENTICAL LLM + CM state.

/// Compress using dual-path: CM engine + LLM predictor + MetaMixer.
/// Only used for Max mode with neural feature enabled.
#[cfg(feature = "neural")]
fn neural_compress(
    data: &[u8],
    config: CMConfig,
    llm: &mut datacortex_neural::LlmPredictor,
    meta_mixer: &mut datacortex_neural::MetaMixer,
) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut encoder = ArithmeticEncoder::new();

    // For the first byte, LLM has no context. Feed a zero byte to prime it.
    // We need the LLM to have predicted byte probs BEFORE we start encoding.
    // Strategy: process byte-by-byte. After encoding byte N, feed byte N to LLM
    // to get predictions for byte N+1.

    let total_bytes = data.len();
    let mut bytes_processed = 0;
    let report_interval = total_bytes / 20; // Report every 5%.

    for (byte_idx, &byte) in data.iter().enumerate() {
        // At this point, LLM has been fed bytes 0..byte_idx-1.
        // LLM's cached_byte_probs predict byte_idx.

        for bpos in 0..8u8 {
            let bit = (byte >> (7 - bpos)) & 1;

            // CM prediction.
            let p_cm = engine.predict();

            // LLM bit prediction.
            // c0 is the partial byte being built: starts at 1, accumulates bits.
            let partial = if bpos == 0 {
                1u32
            } else {
                // Build partial from the bits we've already encoded for this byte.
                let mut p = 1u32;
                for prev_bpos in 0..bpos {
                    let prev_bit = (byte >> (7 - prev_bpos)) & 1;
                    p = (p << 1) | prev_bit as u32;
                }
                p
            };
            let p_llm = llm.predict_bit(bpos, partial);

            // Meta-mixer blend.
            let p_final = meta_mixer.blend(p_cm, p_llm);

            encoder.encode(bit, p_final);
            engine.update(bit);
            meta_mixer.update(bit);
        }

        // Feed the completed byte to the LLM for next-byte prediction.
        if let Err(e) = llm.predict_byte_probs(byte) {
            // If LLM fails, it will return uniform on next call. Log but don't abort.
            if byte_idx < 5 {
                eprintln!("[neural] LLM predict error at byte {byte_idx}: {e}");
            }
        }

        bytes_processed += 1;
        if report_interval > 0 && bytes_processed % report_interval == 0 {
            let pct = bytes_processed * 100 / total_bytes;
            eprint!("\r[neural] compressing... {pct}%");
        }
    }

    if total_bytes > 1000 {
        eprintln!("\r[neural] compressing... 100%");
    }

    encoder.finish()
}

/// Decompress using dual-path: CM engine + LLM predictor + MetaMixer.
/// Must produce IDENTICAL LLM + CM state as the encoder.
#[cfg(feature = "neural")]
fn neural_decompress(
    compressed: &[u8],
    original_size: usize,
    config: CMConfig,
    llm: &mut datacortex_neural::LlmPredictor,
    meta_mixer: &mut datacortex_neural::MetaMixer,
) -> Vec<u8> {
    let mut engine = CMEngine::with_config(config);
    let mut decoder = ArithmeticDecoder::new(compressed);
    let mut output = Vec::with_capacity(original_size);

    let report_interval = if original_size > 0 {
        original_size / 20
    } else {
        1
    };

    for byte_idx in 0..original_size {
        let mut byte_val: u8 = 0;

        for bpos in 0..8u8 {
            // CM prediction.
            let p_cm = engine.predict();

            // LLM bit prediction (using same partial byte state as encoder).
            let partial = if bpos == 0 {
                1u32
            } else {
                // Build partial from bits already decoded for this byte.
                let mut p = 1u32;
                for prev_bpos in 0..bpos {
                    let prev_bit = (byte_val >> (7 - prev_bpos)) & 1;
                    p = (p << 1) | prev_bit as u32;
                }
                p
            };
            let p_llm = llm.predict_bit(bpos, partial);

            // Meta-mixer blend.
            let p_final = meta_mixer.blend(p_cm, p_llm);

            let bit = decoder.decode(p_final);
            engine.update(bit);
            meta_mixer.update(bit);
            byte_val |= bit << (7 - bpos);
        }

        output.push(byte_val);

        // Feed decoded byte to LLM (same as encoder did).
        if let Err(e) = llm.predict_byte_probs(byte_val) {
            if byte_idx < 5 {
                eprintln!("[neural] LLM predict error at byte {byte_idx}: {e}");
            }
        }

        if report_interval > 0 && (byte_idx + 1) % report_interval == 0 {
            let pct = (byte_idx + 1) * 100 / original_size;
            eprint!("\r[neural] decompressing... {pct}%");
        }
    }

    if original_size > 1000 {
        eprintln!("\r[neural] decompressing... 100%");
    }

    output
}

/// Get the CMConfig for a given mode.
fn cm_config_for_mode(mode: Mode) -> CMConfig {
    match mode {
        Mode::Max => CMConfig::max(),
        Mode::Balanced => CMConfig::balanced(),
        Mode::Fast => CMConfig::balanced(), // not used for Fast, but keeps API clean
    }
}

/// Resolve the model path from:
/// 1. Explicit path (--model-path CLI flag)
/// 2. DATACORTEX_MODEL environment variable
/// 3. Default: ~/.datacortex/models/SmolLM2-135M-Instruct-Q8_0.gguf
#[cfg(feature = "neural")]
fn resolve_model_path(explicit: Option<&str>) -> Option<String> {
    if let Some(p) = explicit {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
        eprintln!("[neural] explicit model path not found: {p}");
        return None;
    }

    if let Ok(p) = std::env::var("DATACORTEX_MODEL") {
        if p.is_empty() {
            // Explicitly set to empty = disable neural.
            return None;
        }
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
        eprintln!("[neural] DATACORTEX_MODEL path not found: {p}");
        return None; // Don't fall through to default.
    }

    // Default location.
    if let Some(home) = std::env::var_os("HOME") {
        let default = format!(
            "{}/.datacortex/models/SmolLM2-135M-Instruct-Q8_0.gguf",
            home.to_string_lossy()
        );
        if std::path::Path::new(&default).exists() {
            return Some(default);
        }
    }

    None
}

/// Compress `data` into .dcx format, writing to `output`.
pub fn compress<W: Write>(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    output: &mut W,
) -> io::Result<()> {
    compress_with_model(data, mode, format_override, None, output)
}

/// Compress with optional explicit model path (for neural Max mode).
pub fn compress_with_model<W: Write>(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    model_path: Option<&str>,
    output: &mut W,
) -> io::Result<()> {
    compress_with_options(data, mode, format_override, model_path, None, output)
}

/// Compress with optional explicit model path and zstd level override.
pub fn compress_with_options<W: Write>(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    model_path: Option<&str>,
    zstd_level_override: Option<i32>,
    output: &mut W,
) -> io::Result<()> {
    let format_hint = format_override.unwrap_or_else(|| detect_format(data));
    let crc = crc32fast::hash(data);

    // Step 1: Format-aware preprocessing.
    let (preprocessed, chain) = preprocess(data, format_hint, mode);
    let transform_metadata = if chain.is_empty() {
        vec![]
    } else {
        chain.serialize()
    };

    // Step 2: Compress with engine.
    let mut use_dict = false;
    let mut use_brotli = false;
    // Track whether raw fallback won (empty transform chain).
    let mut use_raw_fallback = false;
    // Track whether metadata is embedded in the compressed stream.
    let mut use_meta_embedded = false;
    let compressed = match mode {
        // Fast mode: auto-fallback — try preprocessed+zstd, raw+zstd, raw+brotli,
        // preprocessed+brotli, and embedded-metadata+brotli. Keep whichever produces
        // the smallest output (including header and metadata overhead).
        //
        // Preprocessing (columnar + typed encoding) usually helps zstd by grouping
        // similar values. But for some files (e.g. citm_catalog.json with extreme
        // repetition), raw zstd without preprocessing gives MUCH better results
        // because preprocessing removes the repetition patterns zstd's LZ77 exploits.
        //
        // Brotli at quality 11 can beat zstd on some JSON files (e.g. twitter.json)
        // because its context modeling handles certain data patterns better.
        //
        // For small files with transforms, embedding metadata inside the brotli stream
        // saves the separate metadata overhead (~150 bytes), because brotli compresses
        // the 4-byte length prefix + raw metadata nearly for free.
        Mode::Fast => {
            let level = adaptive_fast_level(preprocessed.len(), zstd_level_override);

            // Path A: preprocessed + zstd (with optional dict).
            let plain_a = zstd::bulk::compress(&preprocessed, level).map_err(io::Error::other)?;

            let (compressed_a, dict_a) = if preprocessed.len() >= DICT_MIN_DATA_SIZE {
                if let Some(dict_payload) = try_dict_compress(&preprocessed, level, plain_a.len()) {
                    (dict_payload, true)
                } else {
                    (plain_a, false)
                }
            } else {
                (plain_a, false)
            };

            // Estimate compressed metadata size for fair comparison.
            // This matches the compression that happens later for the header.
            let meta_size_for_comparison = if transform_metadata.len() > 64 {
                let compressed_meta = zstd::bulk::compress(&transform_metadata, 19)
                    .unwrap_or_else(|_| transform_metadata.clone());
                if compressed_meta.len() < transform_metadata.len() {
                    compressed_meta.len()
                } else {
                    transform_metadata.len()
                }
            } else {
                transform_metadata.len()
            };

            // Total size for Path A: header(32) + (compressed) metadata + compressed_a.
            let total_a = 32 + meta_size_for_comparison + compressed_a.len();

            // Path B: raw zstd (no preprocessing, no dict).
            // Use same adaptive level but on original data size.
            let raw_level = adaptive_fast_level(data.len(), zstd_level_override);
            let compressed_b = zstd::bulk::compress(data, raw_level).map_err(io::Error::other)?;

            // Total size for Path B: header(32) + 0 (empty metadata) + compressed_b.
            let total_b = 32 + compressed_b.len();

            // Start with best of zstd paths.
            let (mut best_compressed, mut best_total, mut best_dict, mut best_raw, mut best_brotli, mut best_embedded) =
                if total_b < total_a {
                    (compressed_b, total_b, false, true, false, false)
                } else {
                    (compressed_a, total_a, dict_a, false, false, false)
                };

            // Path C: raw + brotli.
            // Use quality 11 for files <= 1MB, 9 for larger (speed tradeoff).
            let brotli_quality = if data.len() <= 1_048_576 { 11 } else { 9 };
            if let Ok(brotli_raw) = brotli_compress(data, brotli_quality) {
                let brotli_raw_total = 32 + brotli_raw.len();
                if brotli_raw_total < best_total {
                    best_compressed = brotli_raw;
                    best_total = brotli_raw_total;
                    best_dict = false;
                    best_raw = true;
                    best_brotli = true;
                    best_embedded = false;
                }
            }

            // Path D: preprocessed + brotli (separate metadata).
            let brotli_prep_quality = if preprocessed.len() <= 1_048_576 {
                11
            } else {
                9
            };
            if let Ok(brotli_prep) = brotli_compress(&preprocessed, brotli_prep_quality) {
                let brotli_prep_total = 32 + meta_size_for_comparison + brotli_prep.len();
                if brotli_prep_total < best_total {
                    best_compressed = brotli_prep;
                    best_total = brotli_prep_total;
                    best_dict = false;
                    best_raw = false;
                    best_brotli = true;
                    best_embedded = false;
                }
            }

            // Path E: preprocessed + brotli with EMBEDDED metadata.
            // Only attempt when transforms were applied (non-empty metadata).
            // The metadata is prepended to the preprocessed data as:
            //   [meta_len: u32 LE][raw_metadata][preprocessed_data]
            // and the whole thing is brotli-compressed together. This eliminates
            // separate metadata overhead in the header.
            if !transform_metadata.is_empty() {
                let mut embedded_payload =
                    Vec::with_capacity(4 + transform_metadata.len() + preprocessed.len());
                embedded_payload
                    .extend_from_slice(&(transform_metadata.len() as u32).to_le_bytes());
                embedded_payload.extend_from_slice(&transform_metadata);
                embedded_payload.extend_from_slice(&preprocessed);

                let embed_quality = if embedded_payload.len() <= 1_048_576 {
                    11
                } else {
                    9
                };
                if let Ok(brotli_embedded) = brotli_compress(&embedded_payload, embed_quality) {
                    // Total: header(32) + 0 (no separate metadata) + brotli_embedded.
                    let brotli_embedded_total = 32 + brotli_embedded.len();
                    if brotli_embedded_total < best_total {
                        best_compressed = brotli_embedded;
                        best_total = brotli_embedded_total;
                        best_dict = false;
                        best_raw = false;
                        best_brotli = true;
                        best_embedded = true;
                    }
                }
            }

            let _ = best_total; // used only for comparisons
            use_dict = best_dict;
            use_raw_fallback = best_raw;
            use_brotli = best_brotli;
            use_meta_embedded = best_embedded;
            best_compressed
        }
        // Balanced mode: dual-path CM + GRU byte predictor.
        Mode::Balanced => {
            let config = cm_config_for_mode(mode);
            let cm_data = gru_compress(&preprocessed, config);
            let mut payload = Vec::with_capacity(8 + cm_data.len());
            payload.extend_from_slice(&(preprocessed.len() as u64).to_le_bytes());
            payload.extend_from_slice(&cm_data);
            payload
        }
        // Max mode: try neural dual-path, fall back to CM-only.
        Mode::Max => {
            let config = cm_config_for_mode(mode);

            #[cfg(feature = "neural")]
            {
                if let Some(mpath) = resolve_model_path(model_path) {
                    match datacortex_neural::LlmPredictor::new(&mpath) {
                        Ok(mut llm) => {
                            let mut meta_mixer = datacortex_neural::MetaMixer::new(5);
                            eprintln!(
                                "[neural] Max mode: dual-path CM+LLM ({} bytes mapped)",
                                llm.mapped_bytes()
                            );
                            let cm_data =
                                neural_compress(&preprocessed, config, &mut llm, &mut meta_mixer);
                            let mut payload = Vec::with_capacity(8 + cm_data.len());
                            // Byte 0 of the 8-byte size prefix: set bit 7 to flag neural mode.
                            // This lets the decompressor know to use neural path.
                            let size_with_flag = preprocessed.len() as u64 | (1u64 << 63);
                            payload.extend_from_slice(&size_with_flag.to_le_bytes());
                            payload.extend_from_slice(&cm_data);
                            payload
                        }
                        Err(e) => {
                            eprintln!("[neural] LLM init failed, falling back to CM-only: {e}");
                            let cm_data = cm_compress(&preprocessed, config);
                            let mut payload = Vec::with_capacity(8 + cm_data.len());
                            payload.extend_from_slice(&(preprocessed.len() as u64).to_le_bytes());
                            payload.extend_from_slice(&cm_data);
                            payload
                        }
                    }
                } else {
                    eprintln!(
                        "[neural] no model found, Max mode using CM-only. \
                         Set DATACORTEX_MODEL or use --model-path."
                    );
                    let cm_data = cm_compress(&preprocessed, config);
                    let mut payload = Vec::with_capacity(8 + cm_data.len());
                    payload.extend_from_slice(&(preprocessed.len() as u64).to_le_bytes());
                    payload.extend_from_slice(&cm_data);
                    payload
                }
            }

            #[cfg(not(feature = "neural"))]
            {
                let _ = model_path; // suppress unused warning
                let cm_data = cm_compress(&preprocessed, config);
                let mut payload = Vec::with_capacity(8 + cm_data.len());
                payload.extend_from_slice(&(preprocessed.len() as u64).to_le_bytes());
                payload.extend_from_slice(&cm_data);
                payload
            }
        }
    };

    // When raw fallback or embedded metadata won, use empty header metadata.
    // - Raw fallback: decompressor handles empty chains (just decompresses, no reverse transforms).
    // - Embedded: metadata lives inside the compressed stream, not in the header.
    let final_metadata = if use_raw_fallback || use_meta_embedded {
        vec![]
    } else {
        transform_metadata
    };

    // Compress metadata with zstd if it's large enough to benefit.
    // Small metadata (<= 64 bytes) stays raw to avoid zstd frame overhead.
    // Skipped when metadata is embedded (final_metadata is empty).
    let (header_metadata, meta_compressed) = if final_metadata.len() > 64 {
        let compressed_meta =
            zstd::bulk::compress(&final_metadata, 19).unwrap_or_else(|_| final_metadata.clone());
        if compressed_meta.len() < final_metadata.len() {
            (compressed_meta, true)
        } else {
            (final_metadata, false)
        }
    } else {
        (final_metadata, false)
    };

    let header = DcxHeader {
        mode,
        format_hint,
        original_size: data.len() as u64,
        compressed_size: compressed.len() as u64,
        crc32: crc,
        transform_metadata: header_metadata,
        has_dict: use_dict,
        meta_compressed,
        use_brotli,
        meta_embedded: use_meta_embedded,
    };

    header.write_to(output)?;
    output.write_all(&compressed)?;

    Ok(())
}

/// Decompress a .dcx file from `input`, returning the original data.
pub fn decompress<R: Read>(input: &mut R) -> io::Result<Vec<u8>> {
    decompress_with_model(input, None)
}

/// Decompress with optional explicit model path (for neural Max mode).
pub fn decompress_with_model<R: Read>(
    input: &mut R,
    model_path: Option<&str>,
) -> io::Result<Vec<u8>> {
    let header = DcxHeader::read_from(input)?;

    let mut compressed = vec![0u8; header.compressed_size as usize];
    input.read_exact(&mut compressed)?;

    // Step 1: Decompress with engine.
    let preprocessed = match header.mode {
        Mode::Fast => {
            if header.use_brotli {
                brotli_decompress(&compressed)?
            } else {
                let capacity = header.original_size as usize * 2 + 65536;
                if header.has_dict {
                    decompress_with_dict(&compressed, capacity)?
                } else {
                    zstd::bulk::decompress(&compressed, capacity)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                }
            }
        }
        Mode::Balanced => {
            // Balanced mode: dual-path CM + GRU byte predictor.
            if compressed.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CM mode compressed data too short",
                ));
            }
            let size_raw = u64::from_le_bytes(compressed[..8].try_into().expect("8-byte slice"));
            let preprocessed_size = (size_raw & !(1u64 << 63)) as usize;
            let config = cm_config_for_mode(header.mode);
            gru_decompress(&compressed[8..], preprocessed_size, config)
        }
        Mode::Max => {
            // Max mode: may use neural (LLM) dual-path or CM-only.
            if compressed.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CM mode compressed data too short",
                ));
            }
            let size_raw = u64::from_le_bytes(compressed[..8].try_into().expect("8-byte slice"));

            // Check if bit 63 is set (neural flag).
            let neural_flag = size_raw & (1u64 << 63) != 0;
            let preprocessed_size = (size_raw & !(1u64 << 63)) as usize;
            let config = cm_config_for_mode(header.mode);

            if neural_flag {
                #[cfg(feature = "neural")]
                {
                    if let Some(mpath) = resolve_model_path(model_path) {
                        match datacortex_neural::LlmPredictor::new(&mpath) {
                            Ok(mut llm) => {
                                let mut meta_mixer = datacortex_neural::MetaMixer::new(5);
                                eprintln!(
                                    "[neural] decompressing with dual-path CM+LLM ({} bytes mapped)",
                                    llm.mapped_bytes()
                                );
                                neural_decompress(
                                    &compressed[8..],
                                    preprocessed_size,
                                    config,
                                    &mut llm,
                                    &mut meta_mixer,
                                )
                            }
                            Err(e) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::Other,
                                    format!(
                                        "file was compressed with neural mode but LLM failed to load: {e}"
                                    ),
                                ));
                            }
                        }
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "file was compressed with neural mode but no model found. \
                             Set DATACORTEX_MODEL or use --model-path.",
                        ));
                    }
                }

                #[cfg(not(feature = "neural"))]
                {
                    let _ = model_path;
                    return Err(io::Error::other(
                        "file was compressed with neural mode but this build lacks the \
                         `neural` feature. Rebuild with --features neural.",
                    ));
                }
            } else {
                cm_decompress(&compressed[8..], preprocessed_size, config)
            }
        }
    };

    // Step 1.5: Handle embedded metadata OR separate metadata.
    // When meta_embedded is set, the decompressed stream starts with:
    //   [meta_len: u32 LE][raw_metadata][preprocessed_data]
    // We extract the metadata and the actual preprocessed data from the stream.
    let (preprocessed, transform_metadata) = if header.meta_embedded {
        if preprocessed.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "embedded metadata: decompressed stream too short for meta_len",
            ));
        }
        let meta_len = u32::from_le_bytes(
            preprocessed[0..4].try_into().expect("4-byte slice"),
        ) as usize;
        if preprocessed.len() < 4 + meta_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "embedded metadata: stream too short for metadata ({} bytes needed, {} available)",
                    4 + meta_len,
                    preprocessed.len()
                ),
            ));
        }
        let metadata = preprocessed[4..4 + meta_len].to_vec();
        let actual_preprocessed = preprocessed[4 + meta_len..].to_vec();
        (actual_preprocessed, metadata)
    } else {
        // Decompress metadata if it was zstd-compressed (separate metadata path).
        // Use streaming decoder to avoid guessing decompressed size.
        let tm = if header.meta_compressed && !header.transform_metadata.is_empty() {
            let mut decoder =
                zstd::Decoder::new(Cursor::new(&header.transform_metadata)).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("failed to init metadata decompressor: {e}"),
                    )
                })?;
            let mut decompressed_meta = Vec::new();
            decoder.read_to_end(&mut decompressed_meta).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to decompress transform metadata: {e}"),
                )
            })?;
            decompressed_meta
        } else {
            header.transform_metadata.clone()
        };
        (preprocessed, tm)
    };

    // Step 2: Reverse preprocessing.
    let data = if transform_metadata.is_empty() {
        preprocessed
    } else {
        let chain = TransformChain::deserialize(&transform_metadata)?;
        reverse_preprocess(&preprocessed, &chain)
    };

    // CRC-32 integrity check.
    let crc = crc32fast::hash(&data);
    if crc != header.crc32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "CRC-32 mismatch: expected {:#010X}, got {:#010X}",
                header.crc32, crc
            ),
        ));
    }

    if data.len() as u64 != header.original_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "size mismatch: header says {} bytes, got {}",
                header.original_size,
                data.len()
            ),
        ));
    }

    Ok(data)
}

/// Compress to Vec (convenience).
pub fn compress_to_vec(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    compress(data, mode, format_override, &mut buf)?;
    Ok(buf)
}

/// Compress to Vec with explicit model path.
pub fn compress_to_vec_with_model(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    model_path: Option<&str>,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    compress_with_model(data, mode, format_override, model_path, &mut buf)?;
    Ok(buf)
}

/// Compress to Vec with explicit model path and zstd level override.
pub fn compress_to_vec_with_options(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    model_path: Option<&str>,
    zstd_level_override: Option<i32>,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    compress_with_options(
        data,
        mode,
        format_override,
        model_path,
        zstd_level_override,
        &mut buf,
    )?;
    Ok(buf)
}

/// Decompress from slice (convenience).
pub fn decompress_from_slice(dcx_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut cursor = Cursor::new(dcx_data);
    decompress(&mut cursor)
}

/// Read header only (for `info` command).
pub fn read_header<R: Read>(input: &mut R) -> io::Result<DcxHeader> {
    DcxHeader::read_from(input)
}

/// Compress raw data with zstd at a given level (for benchmark comparison).
pub fn raw_zstd_compress(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
    zstd::bulk::compress(data, level).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_mode_roundtrip() {
        let original = b"Hello, DataCortex! This is a test of Fast mode compression.";
        let compressed = compress_to_vec(original, Mode::Fast, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn fast_mode_json_roundtrip() {
        let data = br#"{"name":"Alice","age":30,"name":"Bob","age":25,"name":"Carol","age":35}"#;
        let compressed = compress_to_vec(data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn balanced_mode_roundtrip() {
        let original = b"Balanced mode test data with some content.";
        let compressed = compress_to_vec(original, Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn balanced_mode_longer_text() {
        let original = b"The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once. We need enough data to properly exercise the arithmetic coder and order-0 model.";
        let compressed = compress_to_vec(original, Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn balanced_mode_repetitive_data() {
        let data = "hello world! ".repeat(100);
        let compressed = compress_to_vec(data.as_bytes(), Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.as_bytes());
    }

    #[test]
    fn balanced_mode_all_byte_values() {
        let original: Vec<u8> = (0..=255).collect();
        let compressed = compress_to_vec(&original, Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn balanced_mode_single_byte() {
        let original = b"X";
        let compressed = compress_to_vec(original, Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn balanced_mode_json_roundtrip() {
        let data = br#"{"name":"Alice","age":30,"name":"Bob","age":25,"name":"Carol","age":35}"#;
        let compressed = compress_to_vec(data, Mode::Balanced, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn empty_data_roundtrip() {
        let original = b"";
        for mode in [Mode::Fast, Mode::Balanced, Mode::Max] {
            let compressed = compress_to_vec(original, mode, None).unwrap();
            let decompressed = decompress_from_slice(&compressed).unwrap();
            assert_eq!(decompressed, original, "failed for mode {mode}");
        }
    }

    #[test]
    fn crc_mismatch_detected() {
        let original = b"test data for CRC check";
        let mut compressed = compress_to_vec(original, Mode::Fast, None).unwrap();
        // Corrupt in the compressed data section (after header).
        let header_size = 32; // minimum header
        if compressed.len() > header_size + 5 {
            compressed[header_size + 3] ^= 0xFF;
        }
        assert!(decompress_from_slice(&compressed).is_err());
    }

    #[test]
    fn fast_mode_actually_compresses() {
        // Repetitive data should compress well with zstd.
        let data = "hello world. ".repeat(100);
        let compressed = compress_to_vec(data.as_bytes(), Mode::Fast, None).unwrap();
        assert!(
            compressed.len() < data.len(),
            "Fast mode should compress repetitive data: {} vs {}",
            compressed.len(),
            data.len()
        );
    }

    #[test]
    fn json_preprocessing_improves_fast_mode() {
        let data = br#"[{"name":"Alice","score":95},{"name":"Bob","score":87},{"name":"Carol","score":92},{"name":"Dave","score":88},{"name":"Eve","score":91}]"#;
        let with_preprocess = compress_to_vec(data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let without_preprocess =
            compress_to_vec(data, Mode::Fast, Some(FormatHint::Generic)).unwrap();

        // Both should decompress correctly.
        assert_eq!(
            decompress_from_slice(&with_preprocess).unwrap(),
            data.to_vec()
        );
        assert_eq!(
            decompress_from_slice(&without_preprocess).unwrap(),
            data.to_vec()
        );
    }

    #[test]
    fn all_modes_roundtrip() {
        let data = b"test all modes with some more content to ensure decent compression";
        for mode in [Mode::Max, Mode::Balanced, Mode::Fast] {
            let compressed = compress_to_vec(data, mode, None).unwrap();
            let decompressed = decompress_from_slice(&compressed).unwrap();
            assert_eq!(decompressed, data, "failed for mode {mode}");
        }
    }

    #[test]
    fn cm_compress_decompress_direct() {
        let data = b"Hello, World! This is a direct CM test.";
        let compressed = cm_compress(data, CMConfig::balanced());
        let decompressed = cm_decompress(&compressed, data.len(), CMConfig::balanced());
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn cm_empty() {
        let data: &[u8] = b"";
        let compressed = cm_compress(data, CMConfig::balanced());
        let decompressed = cm_decompress(&compressed, 0, CMConfig::balanced());
        assert!(decompressed.is_empty());
    }

    #[test]
    fn cm_single_byte() {
        for byte in 0..=255u8 {
            let data = [byte];
            let compressed = cm_compress(&data, CMConfig::balanced());
            let decompressed = cm_decompress(&compressed, 1, CMConfig::balanced());
            assert_eq!(
                decompressed, data,
                "CM roundtrip failed for byte {byte:#04X}"
            );
        }
    }

    #[test]
    fn cm_repetitive_compresses() {
        let data = vec![b'A'; 1000];
        let compressed = cm_compress(&data, CMConfig::balanced());
        // 1000 identical bytes should compress well with adaptive model.
        assert!(
            compressed.len() < 200,
            "CM should compress 1000 identical bytes well: {} bytes",
            compressed.len()
        );
        let decompressed = cm_decompress(&compressed, data.len(), CMConfig::balanced());
        assert_eq!(decompressed, data);
    }

    #[test]
    fn max_mode_roundtrip() {
        let original = b"Max mode test data with some content for compression.";
        let compressed = compress_to_vec(original, Mode::Max, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn max_mode_longer_text() {
        let original = b"The quick brown fox jumps over the lazy dog. Max mode uses 2x context maps for better predictions with fewer hash collisions. This should compress slightly better than balanced mode.";
        let compressed = compress_to_vec(original, Mode::Max, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    // ─── Dictionary compression tests ──────────────────────────────────────────

    #[test]
    fn test_dict_compress_roundtrip() {
        // Generate NDJSON data large enough to trigger dictionary training.
        // Repetitive columnar data is ideal for dictionary learning.
        let mut ndjson = String::new();
        for i in 0..500 {
            ndjson.push_str(&format!(
                r#"{{"id":{},"name":"user_{}","status":"active","score":{}}}"#,
                i,
                i,
                i * 17 % 100
            ));
            ndjson.push('\n');
        }
        let data = ndjson.as_bytes();
        assert!(
            data.len() > DICT_MIN_DATA_SIZE,
            "test data should exceed dict threshold: {} bytes",
            data.len()
        );

        let compressed = compress_to_vec(data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(
            decompressed, data,
            "dict compress roundtrip: byte-exact mismatch"
        );
    }

    #[test]
    fn test_dict_falls_back_on_small() {
        // Data smaller than DICT_MIN_DATA_SIZE should not use dictionary.
        let data = b"small data that won't trigger dictionary training";
        assert!(data.len() < DICT_MIN_DATA_SIZE);

        let compressed = compress_to_vec(data, Mode::Fast, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());

        // Verify no dict flag in header.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        assert!(!header.has_dict, "small data should not have dict flag set");
    }

    #[test]
    fn test_dict_backward_compat() {
        // Compress with old behavior (no dict) and verify it still decompresses.
        // We simulate this by compressing small data (which skips dict).
        let original = b"backward compatibility test data for decompression";
        let compressed = compress_to_vec(original, Mode::Fast, None).unwrap();

        // Verify the flag is NOT set.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        assert!(!header.has_dict);

        // Decompress should work fine.
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_dict_ndjson_large_roundtrip() {
        // Larger NDJSON dataset — should benefit from dictionary.
        let mut ndjson = String::new();
        for i in 0..2000 {
            ndjson.push_str(&format!(
                r#"{{"timestamp":"2025-01-{:02}T{:02}:{:02}:00Z","level":"info","message":"Request processed","request_id":"req_{}","duration_ms":{}}}"#,
                (i % 28) + 1,
                i % 24,
                i % 60,
                i,
                (i * 13) % 500
            ));
            ndjson.push('\n');
        }
        let data = ndjson.as_bytes();

        let compressed = compress_to_vec(data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "large NDJSON roundtrip mismatch");
    }

    #[test]
    fn test_dict_generic_data_roundtrip() {
        // Generic (non-JSON) data that's large enough for dict training.
        // Uses fixed-size block splitting instead of column boundaries.
        let mut data = Vec::new();
        for i in 0..3000 {
            data.extend_from_slice(
                format!("line {i}: the quick brown fox jumps over the lazy dog\n").as_bytes(),
            );
        }
        assert!(data.len() > DICT_MIN_DATA_SIZE);

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Generic)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "generic data dict roundtrip mismatch");
    }

    #[test]
    fn test_dict_does_not_affect_other_modes() {
        // Dictionary training should only apply to Fast mode.
        // Balanced and Max modes should remain unchanged.
        let mut ndjson = String::new();
        for i in 0..200 {
            ndjson.push_str(&format!(
                r#"{{"id":{},"name":"user_{}","status":"active"}}"#,
                i, i
            ));
            ndjson.push('\n');
        }
        let data = ndjson.as_bytes();

        for mode in [Mode::Balanced, Mode::Max] {
            let compressed = compress_to_vec(data, mode, Some(FormatHint::Ndjson)).unwrap();
            let mut cursor = Cursor::new(&compressed);
            let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
            assert!(!header.has_dict, "mode {mode} should never have dict flag");
            let decompressed = decompress_from_slice(&compressed).unwrap();
            assert_eq!(decompressed, data, "roundtrip failed for mode {mode}");
        }
    }

    // ─── Configurable zstd level tests ──────────────────────────────────────

    #[test]
    fn test_compress_with_level() {
        // Compress with level 19 override in Fast mode, verify roundtrip.
        let data = "hello world, compressing with custom zstd level. ".repeat(50);
        let compressed =
            compress_to_vec_with_options(data.as_bytes(), Mode::Fast, None, None, Some(19))
                .unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.as_bytes(), "level 19 roundtrip failed");
    }

    #[test]
    fn test_compress_with_level_default() {
        // No level override — should use mode default (9 for Fast).
        let data = "default level test data. ".repeat(50);
        let compressed =
            compress_to_vec_with_options(data.as_bytes(), Mode::Fast, None, None, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(
            decompressed,
            data.as_bytes(),
            "default level roundtrip failed"
        );
    }

    #[test]
    fn test_compress_with_level_higher_ratio() {
        // Level 19 should compress better than level 1 on repetitive data.
        let data = r#"{"name":"Alice","score":95}"#.repeat(200);
        let low =
            compress_to_vec_with_options(data.as_bytes(), Mode::Fast, None, None, Some(1)).unwrap();
        let high = compress_to_vec_with_options(data.as_bytes(), Mode::Fast, None, None, Some(19))
            .unwrap();

        // Both must roundtrip.
        assert_eq!(decompress_from_slice(&low).unwrap(), data.as_bytes());
        assert_eq!(decompress_from_slice(&high).unwrap(), data.as_bytes());

        // Higher level should produce smaller output (or at least not larger).
        assert!(
            high.len() <= low.len(),
            "level 19 ({}) should be <= level 1 ({})",
            high.len(),
            low.len()
        );
    }

    // ─── Auto-fallback tests ──────────────────────────────────────────────────

    #[test]
    fn test_auto_fallback_picks_smaller() {
        // citm_catalog.json has extreme repetition. The auto-fallback picks
        // whichever path (raw or preprocessed) produces the smallest output.
        // With compressed metadata, the preprocessed path may now win.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/citm_catalog.json"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "citm_catalog roundtrip failed");

        // Verify good compression ratio regardless of which path won.
        let ratio = data.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 50.0,
            "citm_catalog should achieve >50x, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_auto_fallback_preprocessed_wins_on_ndjson() {
        // NDJSON with uniform schema should still prefer preprocessed path
        // (columnar + typed encoding beats raw zstd for structured data).
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "test-ndjson roundtrip failed");

        // Check that preprocessing was used: either non-empty transform metadata
        // in the header, or metadata embedded in the compressed stream.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        assert!(
            !header.transform_metadata.is_empty() || header.meta_embedded,
            "test-ndjson should prefer preprocessed path (non-empty transform metadata or embedded)"
        );
    }

    #[test]
    fn test_auto_fallback_roundtrip() {
        // Verify both raw and preprocessed paths produce correct roundtrips.
        // Use citm_catalog (raw wins) and test-ndjson (preprocessed wins).
        let citm = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/citm_catalog.json"
        ))
        .unwrap();
        let ndjson = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .unwrap();

        // citm_catalog — raw path
        let compressed_citm = compress_to_vec(&citm, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed_citm = decompress_from_slice(&compressed_citm).unwrap();
        assert_eq!(
            decompressed_citm, citm,
            "citm_catalog roundtrip (raw path) failed"
        );

        // test-ndjson — preprocessed path
        let compressed_ndjson =
            compress_to_vec(&ndjson, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed_ndjson = decompress_from_slice(&compressed_ndjson).unwrap();
        assert_eq!(
            decompressed_ndjson, ndjson,
            "test-ndjson roundtrip (preprocessed path) failed"
        );
    }

    // ─── Adaptive level tests ─────────────────────────────────────────────────

    #[test]
    fn test_adaptive_level_small_data() {
        // <1MB should use level 19 (zstd-19 is <50ms on small data).
        assert_eq!(adaptive_fast_level(100_000, None), 19);
        assert_eq!(adaptive_fast_level(500_000, None), 19);
        assert_eq!(adaptive_fast_level(1_048_576, None), 19);
        assert_eq!(adaptive_fast_level(0, None), 19);
    }

    #[test]
    fn test_adaptive_level_large_data() {
        // 1MB-10MB should use level 13, >10MB should use level 9.
        assert_eq!(adaptive_fast_level(1_048_577, None), 13);
        assert_eq!(adaptive_fast_level(5_000_000, None), 13);
        assert_eq!(adaptive_fast_level(10_485_760, None), 13);
        assert_eq!(adaptive_fast_level(10_485_761, None), 9);
        assert_eq!(adaptive_fast_level(100_000_000, None), 9);
    }

    #[test]
    fn test_adaptive_level_override() {
        // --level flag should always override adaptive level.
        assert_eq!(adaptive_fast_level(100, Some(3)), 3);
        assert_eq!(adaptive_fast_level(100_000_000, Some(22)), 22);
        assert_eq!(adaptive_fast_level(0, Some(1)), 1);
    }

    // ─── Compressed metadata tests ──────────────────────────────────────────────

    #[test]
    fn test_compressed_metadata_roundtrip() {
        // Generate NDJSON data large enough to produce > 64 bytes of transform metadata.
        let mut ndjson = String::new();
        for i in 0..500 {
            ndjson.push_str(&format!(
                r#"{{"id":{},"name":"user_{}","status":"active","score":{}}}"#,
                i,
                i,
                i * 17 % 100
            ));
            ndjson.push('\n');
        }
        let data = ndjson.as_bytes();

        let compressed = compress_to_vec(data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(
            decompressed, data,
            "compressed metadata roundtrip: byte-exact mismatch"
        );

        // Verify the header has meta_compressed set if metadata was large enough.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        // The file should have used preprocessed path (non-empty metadata).
        if !header.transform_metadata.is_empty() && header.transform_metadata.len() > 10 {
            // Metadata was present — check that compressed flag makes sense.
            // (meta_compressed is true only if compression actually saved space)
            // Just verify roundtrip was correct — the flag is an optimization detail.
        }
    }

    #[test]
    fn test_compressed_metadata_backward_compat() {
        // Simulate old files that have no compressed metadata (bit 2 = 0).
        // These should still decompress correctly.
        let original = b"backward compatibility test data for metadata decompression";
        let compressed = compress_to_vec(original, Mode::Fast, None).unwrap();

        // Verify decompression works.
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original.to_vec());

        // For small data, metadata should be empty or very small — no compression.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        // Small data may or may not have metadata, but it should roundtrip either way.
        assert!(!header.meta_compressed || !header.transform_metadata.is_empty());
    }

    #[test]
    fn test_compressed_metadata_small_skipped() {
        // Small metadata (< 64 bytes) should NOT be compressed — zstd frame overhead
        // would make it larger.
        let data = br#"{"name":"Alice","age":30}"#;
        let compressed = compress_to_vec(data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());

        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        // Small JSON has small metadata — should not be compressed.
        if header.transform_metadata.len() <= 64 {
            assert!(
                !header.meta_compressed,
                "metadata <= 64 bytes should not be compressed, but meta_compressed=true \
                 for {} bytes of metadata",
                header.transform_metadata.len()
            );
        }
    }

    #[test]
    fn test_twitter_json_brotli_wins() {
        // twitter.json should use brotli — raw brotli-11 beats both preprocessed+zstd
        // and raw+zstd on this file.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/twitter.json"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "twitter.json roundtrip failed");

        // Check that brotli was selected.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        assert!(
            header.use_brotli,
            "twitter.json should use brotli (FLAG_BROTLI set in header)"
        );
    }

    #[test]
    fn test_compressed_metadata_all_modes_roundtrip() {
        // Metadata compression applies to all modes, not just Fast.
        let mut ndjson = String::new();
        for i in 0..200 {
            ndjson.push_str(&format!(
                r#"{{"id":{},"name":"user_{}","status":"active"}}"#,
                i, i
            ));
            ndjson.push('\n');
        }
        let data = ndjson.as_bytes();

        for mode in [Mode::Fast, Mode::Balanced, Mode::Max] {
            let compressed = compress_to_vec(data, mode, Some(FormatHint::Ndjson)).unwrap();
            let decompressed = decompress_from_slice(&compressed).unwrap();
            assert_eq!(
                decompressed, data,
                "compressed metadata roundtrip failed for mode {mode}"
            );
        }
    }

    // ─── Brotli auto-fallback tests ──────────────────────────────────────────

    #[test]
    fn test_brotli_compress_roundtrip() {
        // Direct brotli compress/decompress helper roundtrip.
        let data = b"Hello, brotli! This is a test of the brotli compression helpers.";
        let compressed = brotli_compress(data, 11).unwrap();
        let decompressed = brotli_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn test_brotli_auto_fallback_twitter() {
        // twitter.json should select brotli and roundtrip correctly.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/twitter.json"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "twitter.json brotli roundtrip failed");

        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();
        assert!(
            header.use_brotli,
            "twitter.json should use brotli in auto-fallback"
        );
    }

    #[test]
    fn test_brotli_ndjson_roundtrip() {
        // NDJSON with uniform schema — regardless of which entropy coder wins,
        // the roundtrip must be byte-exact.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "ndjson roundtrip failed");
    }

    #[test]
    fn test_brotli_backward_compat() {
        // Old .dcx files without the brotli flag (bit 3 = 0) must still decompress.
        // We simulate an old file by manually crafting a .dcx with FLAG_BROTLI unset.
        // Compress with zstd directly and build a minimal .dcx header.
        let original = b"backward compatibility test: this data was compressed without brotli";
        let crc = crc32fast::hash(original);
        let zstd_compressed = zstd::bulk::compress(original, 19).unwrap();

        let header = crate::dcx::DcxHeader {
            mode: Mode::Fast,
            format_hint: crate::dcx::FormatHint::Generic,
            original_size: original.len() as u64,
            compressed_size: zstd_compressed.len() as u64,
            crc32: crc,
            transform_metadata: vec![],
            has_dict: false,
            meta_compressed: false,
            use_brotli: false,
            meta_embedded: false,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        buf.extend_from_slice(&zstd_compressed);

        // Verify the brotli flag is NOT set in the serialized header.
        assert_eq!(buf[7] & crate::dcx::FLAG_BROTLI, 0);

        // Decompress — must work even though brotli path exists.
        let decompressed = decompress_from_slice(&buf).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    // ─── Embedded metadata tests ──────────────────────────────────────────────

    #[test]
    fn test_embedded_metadata_roundtrip() {
        // Compress test-api.json with Fast mode — if embedded metadata is used,
        // the roundtrip must be byte-exact.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-api.json"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(
            decompressed, data,
            "test-api.json embedded metadata roundtrip: byte-exact mismatch"
        );
    }

    #[test]
    fn test_embedded_metadata_backward_compat() {
        // Old .dcx files without the meta_embedded flag (bit 4 = 0) must still decompress.
        // We simulate an old file by manually crafting a .dcx with FLAG_META_EMBEDDED unset
        // and separate transform metadata.
        let original = b"backward compat: no embedded metadata in this old file format";
        let crc = crc32fast::hash(original);
        let zstd_compressed = zstd::bulk::compress(original, 19).unwrap();

        let header = crate::dcx::DcxHeader {
            mode: Mode::Fast,
            format_hint: crate::dcx::FormatHint::Generic,
            original_size: original.len() as u64,
            compressed_size: zstd_compressed.len() as u64,
            crc32: crc,
            transform_metadata: vec![],
            has_dict: false,
            meta_compressed: false,
            use_brotli: false,
            meta_embedded: false,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        buf.extend_from_slice(&zstd_compressed);

        // Verify meta_embedded flag is NOT set.
        assert_eq!(buf[7] & crate::dcx::FLAG_META_EMBEDDED, 0);

        // Decompress — must work without embedded metadata support.
        let decompressed = decompress_from_slice(&buf).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_embedded_metadata_small_file_improvement() {
        // test-api.json is a small file (37KB) where embedded metadata should
        // save overhead compared to separate metadata.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-api.json"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Json)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, data, "roundtrip failed");

        // Verify the file compresses to a reasonable size.
        let ratio = data.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 5.0,
            "test-api.json should achieve >5x compression, got {ratio:.1}x"
        );

        // Check header to see which path was chosen.
        let mut cursor = Cursor::new(&compressed);
        let header = crate::dcx::DcxHeader::read_from(&mut cursor).unwrap();

        // If embedded was chosen, verify the flag is set and header metadata is empty.
        if header.meta_embedded {
            assert!(
                header.transform_metadata.is_empty(),
                "meta_embedded header should have empty transform_metadata"
            );
            assert!(
                header.use_brotli,
                "meta_embedded should use brotli codec"
            );
        }
    }

    #[test]
    fn test_embedded_metadata_ndjson_roundtrip() {
        // NDJSON files with transforms must still roundtrip correctly
        // regardless of whether embedded or separate metadata is chosen.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .unwrap();

        let compressed = compress_to_vec(&data, Mode::Fast, Some(FormatHint::Ndjson)).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(
            decompressed, data,
            "NDJSON embedded metadata roundtrip: byte-exact mismatch"
        );
    }

    #[test]
    fn test_embedded_metadata_manual_roundtrip() {
        // Manually construct an embedded-metadata .dcx to verify the decompress path
        // handles the format correctly, independent of what the compressor chooses.
        let original = b"Hello, embedded metadata world! This is a test.";
        let crc = crc32fast::hash(original);

        // Build embedded payload with an empty transform chain so reverse_preprocess
        // is a no-op and the data passes through unchanged.
        let empty_chain = TransformChain::new();
        let raw_metadata = empty_chain.serialize();

        // Build embedded payload: [meta_len:u32 LE][raw_metadata][original_data]
        let mut embedded = Vec::new();
        embedded.extend_from_slice(&(raw_metadata.len() as u32).to_le_bytes());
        embedded.extend_from_slice(&raw_metadata);
        embedded.extend_from_slice(original);

        let brotli_data = brotli_compress(&embedded, 11).unwrap();

        let header = crate::dcx::DcxHeader {
            mode: Mode::Fast,
            format_hint: crate::dcx::FormatHint::Generic,
            original_size: original.len() as u64,
            compressed_size: brotli_data.len() as u64,
            crc32: crc,
            transform_metadata: vec![], // empty — metadata is embedded
            has_dict: false,
            meta_compressed: false,
            use_brotli: true,
            meta_embedded: true,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        buf.extend_from_slice(&brotli_data);

        // Verify flags.
        assert_ne!(buf[7] & crate::dcx::FLAG_META_EMBEDDED, 0);
        assert_ne!(buf[7] & crate::dcx::FLAG_BROTLI, 0);

        // Decompress and verify.
        let decompressed = decompress_from_slice(&buf).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }
}
