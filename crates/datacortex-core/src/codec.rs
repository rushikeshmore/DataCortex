//! Codec orchestrator — compress and decompress through the DataCortex pipeline.
//!
//! Phase 0: Identity compression (store raw bytes in .dcx container).
//! Future phases will add: format preprocessing → model prediction → entropy coding.

use std::io::{self, Cursor, Read, Write};

use crate::dcx::{DcxHeader, FormatHint, Mode};
use crate::format::detect_format;

/// Compress `data` into .dcx format, writing to `output`.
///
/// Auto-detects format unless `format_override` is provided.
pub fn compress<W: Write>(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
    output: &mut W,
) -> io::Result<()> {
    let format_hint = format_override.unwrap_or_else(|| detect_format(data));
    let crc = crc32fast::hash(data);

    // Phase 0: identity — compressed data IS the original data.
    // Future: format preprocess → model → entropy encode.
    let compressed = data;

    let header = DcxHeader {
        mode,
        format_hint,
        original_size: data.len() as u64,
        compressed_size: compressed.len() as u64,
        crc32: crc,
        transform_metadata: vec![],
    };

    header.write_to(output)?;
    output.write_all(compressed)?;

    Ok(())
}

/// Decompress a .dcx file from `input`, returning the original data.
pub fn decompress<R: Read>(input: &mut R) -> io::Result<Vec<u8>> {
    let header = DcxHeader::read_from(input)?;

    let mut compressed = vec![0u8; header.compressed_size as usize];
    input.read_exact(&mut compressed)?;

    // Phase 0: identity — compressed data IS the original data.
    // Future: entropy decode → model → format reverse-preprocess.
    let data = compressed;

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

/// Compress and return as a Vec<u8> (convenience wrapper).
pub fn compress_to_vec(
    data: &[u8],
    mode: Mode,
    format_override: Option<FormatHint>,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    compress(data, mode, format_override, &mut buf)?;
    Ok(buf)
}

/// Decompress from a byte slice (convenience wrapper).
pub fn decompress_from_slice(dcx_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut cursor = Cursor::new(dcx_data);
    decompress(&mut cursor)
}

/// Read .dcx header without decompressing (for `info` command).
pub fn read_header<R: Read>(input: &mut R) -> io::Result<DcxHeader> {
    DcxHeader::read_from(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip() {
        let original = b"Hello, DataCortex!";
        let compressed = compress_to_vec(original, Mode::Balanced, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn empty_data_roundtrip() {
        let original = b"";
        let compressed = compress_to_vec(original, Mode::Fast, None).unwrap();
        let decompressed = decompress_from_slice(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn crc_mismatch_detected() {
        let original = b"test data";
        let mut compressed = compress_to_vec(original, Mode::Max, None).unwrap();
        // Corrupt a byte in the data section (after 32-byte header).
        if compressed.len() > 32 {
            compressed[32] ^= 0xFF;
        }
        assert!(decompress_from_slice(&compressed).is_err());
    }

    #[test]
    fn format_override_used() {
        let data = b"not actually json";
        let compressed = compress_to_vec(data, Mode::Balanced, Some(FormatHint::Json)).unwrap();
        let mut cursor = Cursor::new(&compressed);
        let header = DcxHeader::read_from(&mut cursor).unwrap();
        assert_eq!(header.format_hint, FormatHint::Json);
    }

    #[test]
    fn all_modes_roundtrip() {
        let data = b"test all modes";
        for mode in [Mode::Max, Mode::Balanced, Mode::Fast] {
            let compressed = compress_to_vec(data, mode, None).unwrap();
            let decompressed = decompress_from_slice(&compressed).unwrap();
            assert_eq!(decompressed, data, "failed for mode {mode}");
        }
    }
}
