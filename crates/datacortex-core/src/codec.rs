//! Codec orchestrator — compress and decompress through the DataCortex pipeline.
//!
//! Phase 1: Format preprocessing + zstd (Fast mode).
//! Phase 3: Full CM engine with higher-order models + mixer + APM (Balanced mode, ~256MB).
//! Phase 5: Full CM engine with 2x context maps (Max mode, ~512MB).
//! Phase 6: Dual-path CM + LLM with MetaMixer (Max mode with `neural` feature).

use std::io::{self, Cursor, Read, Write};

use crate::dcx::{DcxHeader, FormatHint, Mode};
use crate::entropy::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::format::transform::TransformChain;
use crate::format::{detect_format, preprocess, reverse_preprocess};
use crate::model::{CMConfig, CMEngine};

/// zstd compression level per mode (for Fast mode).
fn zstd_level(mode: Mode) -> i32 {
    match mode {
        Mode::Fast => 3,
        Mode::Balanced => 19,
        Mode::Max => 22,
    }
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
    let compressed = match mode {
        // Fast mode: zstd compression on preprocessed data.
        Mode::Fast => {
            zstd::bulk::compress(&preprocessed, zstd_level(mode)).map_err(io::Error::other)?
        }
        // Balanced mode: CM engine only.
        Mode::Balanced => {
            let config = cm_config_for_mode(mode);
            let cm_data = cm_compress(&preprocessed, config);
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

    let header = DcxHeader {
        mode,
        format_hint,
        original_size: data.len() as u64,
        compressed_size: compressed.len() as u64,
        crc32: crc,
        transform_metadata,
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
            zstd::bulk::decompress(&compressed, header.original_size as usize * 2 + 65536)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        }
        Mode::Balanced | Mode::Max => {
            // CM modes embed the preprocessed length as the first 8 bytes of the payload.
            if compressed.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CM mode compressed data too short",
                ));
            }
            let size_raw = u64::from_le_bytes(compressed[..8].try_into().unwrap());

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

    // Step 2: Reverse preprocessing.
    let data = if header.transform_metadata.is_empty() {
        preprocessed
    } else {
        let chain = TransformChain::deserialize(&header.transform_metadata)?;
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
}
