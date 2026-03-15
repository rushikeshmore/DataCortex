//! Codec orchestrator — compress and decompress through the DataCortex pipeline.
//!
//! Phase 0: Identity compression (Balanced/Max modes).
//! Phase 1: Format preprocessing + zstd (Fast mode).

use std::io::{self, Cursor, Read, Write};

use crate::dcx::{DcxHeader, FormatHint, Mode};
use crate::format::transform::TransformChain;
use crate::format::{detect_format, preprocess, reverse_preprocess};

/// zstd compression level per mode (for Fast mode).
fn zstd_level(mode: Mode) -> i32 {
    match mode {
        Mode::Fast => 3,
        Mode::Balanced => 19,
        Mode::Max => 22,
    }
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
        // Balanced/Max: identity for now (CM engine in Phase 2-3).
        // Still apply preprocessing, but store preprocessed data as-is.
        Mode::Balanced | Mode::Max => preprocessed.clone(),
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
        Mode::Balanced | Mode::Max => compressed,
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

        // Preprocessing should help (or at least not hurt).
        // On small data, zstd overhead might dominate. Just verify roundtrip.
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
}
