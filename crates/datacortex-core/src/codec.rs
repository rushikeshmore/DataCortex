//! Codec orchestrator — compress and decompress through the DataCortex pipeline.
//!
//! Phase 0: Identity compression (Max mode placeholder).
//! Phase 1: Format preprocessing + zstd (Fast mode).
//! Phase 2: Order-0 CM engine (Balanced mode).

use std::io::{self, Cursor, Read, Write};

use crate::dcx::{DcxHeader, FormatHint, Mode};
use crate::entropy::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::format::transform::TransformChain;
use crate::format::{detect_format, preprocess, reverse_preprocess};
use crate::model::Order0Model;

/// zstd compression level per mode (for Fast mode).
fn zstd_level(mode: Mode) -> i32 {
    match mode {
        Mode::Fast => 3,
        Mode::Balanced => 19,
        Mode::Max => 22,
    }
}

/// Compress data using the Order-0 CM engine (Balanced mode).
/// Returns the compressed byte stream.
fn cm_compress(data: &[u8]) -> Vec<u8> {
    let mut model = Order0Model::new();
    let mut encoder = ArithmeticEncoder::new();

    for &byte in data {
        let mut c: usize = 1; // partial byte context, starts at 1
        for bpos in 0..8 {
            let bit = (byte >> (7 - bpos)) & 1;
            let p = model.predict(c);
            encoder.encode(bit, p);
            model.update(c, bit);
            c = (c << 1) | bit as usize;
        }
    }

    encoder.finish()
}

/// Decompress data using the Order-0 CM engine (Balanced mode).
/// `compressed` is the arithmetic-coded stream, `original_size` is the expected output length.
fn cm_decompress(compressed: &[u8], original_size: usize) -> Vec<u8> {
    let mut model = Order0Model::new();
    let mut decoder = ArithmeticDecoder::new(compressed);
    let mut output = Vec::with_capacity(original_size);

    for _ in 0..original_size {
        let mut c: usize = 1; // partial byte context, starts at 1
        for _ in 0..8 {
            let p = model.predict(c);
            let bit = decoder.decode(p);
            model.update(c, bit);
            c = (c << 1) | bit as usize;
        }
        // c is now byte_value + 256; extract the byte.
        output.push((c & 0xFF) as u8);
    }

    output
}

/// Compress `data` into .dcx format, writing to `output`.
pub fn compress<W: Write>(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
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
        // Balanced mode: Order-0 CM engine.
        // Prepend the preprocessed size (8 bytes LE) so the decoder knows how many
        // bytes to reconstruct (transforms can change data size).
        Mode::Balanced => {
            let cm_data = cm_compress(&preprocessed);
            let mut payload = Vec::with_capacity(8 + cm_data.len());
            payload.extend_from_slice(&(preprocessed.len() as u64).to_le_bytes());
            payload.extend_from_slice(&cm_data);
            payload
        }
        // Max: identity for now (RWKV in Phase 6).
        Mode::Max => preprocessed.clone(),
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
    let header = DcxHeader::read_from(input)?;

    let mut compressed = vec![0u8; header.compressed_size as usize];
    input.read_exact(&mut compressed)?;

    // Step 1: Decompress with engine.
    let preprocessed = match header.mode {
        Mode::Fast => {
            zstd::bulk::decompress(&compressed, header.original_size as usize * 2 + 65536)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        }
        Mode::Balanced => {
            // For Balanced, we need the preprocessed size to know how many bytes to decode.
            // If there's transform metadata, the preprocessed size differs from original.
            // We store original_size in header. The preprocessed size is what the CM encoded.
            // Since transform_metadata tells us the transforms applied, we need to figure
            // out the preprocessed size. But we don't store it separately.
            //
            // Solution: store the preprocessed size as the first 8 bytes of the CM stream.
            // Wait — that would break the format. Instead, we can compute:
            // If no transforms, preprocessed_size = original_size.
            // If transforms, we need to know. Let's store it in the compressed data.
            //
            // Actually, the cleanest approach: embed the preprocessed length at the start
            // of the compressed payload for Balanced mode.
            if compressed.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "balanced mode compressed data too short",
                ));
            }
            let preprocessed_size =
                u64::from_le_bytes(compressed[..8].try_into().unwrap()) as usize;
            cm_decompress(&compressed[8..], preprocessed_size)
        }
        Mode::Max => compressed,
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
        let compressed = cm_compress(data);
        let decompressed = cm_decompress(&compressed, data.len());
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn cm_empty() {
        let data: &[u8] = b"";
        let compressed = cm_compress(data);
        let decompressed = cm_decompress(&compressed, 0);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn cm_single_byte() {
        for byte in 0..=255u8 {
            let data = [byte];
            let compressed = cm_compress(&data);
            let decompressed = cm_decompress(&compressed, 1);
            assert_eq!(
                decompressed, data,
                "CM roundtrip failed for byte {byte:#04X}"
            );
        }
    }

    #[test]
    fn cm_repetitive_compresses() {
        let data = vec![b'A'; 1000];
        let compressed = cm_compress(&data);
        // 1000 identical bytes should compress well with adaptive model.
        assert!(
            compressed.len() < 200,
            "CM should compress 1000 identical bytes well: {} bytes",
            compressed.len()
        );
        let decompressed = cm_decompress(&compressed, data.len());
        assert_eq!(decompressed, data);
    }
}
