//! .dcx file format — v3 header with CRC-32 integrity.
//!
//! Layout (32-byte fixed header + variable metadata + compressed data):
//!   Bytes  0-3:  Magic "DCX\x03"
//!   Byte   4:    Version (3)
//!   Byte   5:    Mode (0=Max, 1=Balanced, 2=Fast)
//!   Byte   6:    Format hint (0-10)
//!   Byte   7:    Flags (bit 0: has_transform_metadata, bit 1: has_zstd_dictionary)
//!   Bytes  8-15: Original size (u64 LE)
//!   Bytes 16-23: Compressed data size (u64 LE)
//!   Bytes 24-27: CRC-32 of original data (u32 LE)
//!   Bytes 28-31: Transform metadata length (u32 LE)
//!   [Transform metadata: variable]
//!   [Compressed data]

use std::io::{self, Read, Write};

const MAGIC: [u8; 4] = [b'D', b'C', b'X', 0x03];
const VERSION: u8 = 3;
const HEADER_SIZE: usize = 32;

/// Compression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    Max = 0,
    Balanced = 1,
    Fast = 2,
}

impl Mode {
    pub fn from_u8(v: u8) -> io::Result<Self> {
        match v {
            0 => Ok(Self::Max),
            1 => Ok(Self::Balanced),
            2 => Ok(Self::Fast),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown mode: {v}"),
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Balanced => "balanced",
            Self::Fast => "fast",
        }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Detected or declared file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FormatHint {
    Generic = 0,
    Json = 1,
    // Legacy variants kept for backward compat decoding
    Markdown = 2,
    Ndjson = 3,
    // Legacy variants kept for backward compat decoding
    Csv = 4,
    Code = 5,
    Log = 6,
    Logfmt = 7,
    Prometheus = 8,
    Yaml = 9,
    Xml = 10,
}

impl FormatHint {
    pub fn from_u8(v: u8) -> io::Result<Self> {
        match v {
            0 => Ok(Self::Generic),
            1 => Ok(Self::Json),
            2 => Ok(Self::Markdown),
            3 => Ok(Self::Ndjson),
            4 => Ok(Self::Csv),
            5 => Ok(Self::Code),
            6 => Ok(Self::Log),
            7 => Ok(Self::Logfmt),
            8 => Ok(Self::Prometheus),
            9 => Ok(Self::Yaml),
            10 => Ok(Self::Xml),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown format: {v}"),
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Json => "json",
            Self::Markdown => "markdown",
            Self::Ndjson => "ndjson",
            Self::Csv => "csv",
            Self::Code => "code",
            Self::Log => "log",
            Self::Logfmt => "logfmt",
            Self::Prometheus => "prometheus",
            Self::Yaml => "yaml",
            Self::Xml => "xml",
        }
    }
}

impl std::fmt::Display for FormatHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Flags byte layout:
///   bit 0: has_transform_metadata
///   bit 1: has_zstd_dictionary (Fast mode only)
pub const FLAG_HAS_TRANSFORM: u8 = 1 << 0;
pub const FLAG_HAS_DICT: u8 = 1 << 1;

/// .dcx file header.
#[derive(Debug, Clone)]
pub struct DcxHeader {
    pub mode: Mode,
    pub format_hint: FormatHint,
    pub original_size: u64,
    pub compressed_size: u64,
    pub crc32: u32,
    pub transform_metadata: Vec<u8>,
    /// True if the compressed payload embeds a zstd dictionary.
    pub has_dict: bool,
}

impl DcxHeader {
    /// Serialize header to writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&MAGIC)?;
        w.write_all(&[VERSION])?;
        w.write_all(&[self.mode as u8])?;
        w.write_all(&[self.format_hint as u8])?;
        let mut flags: u8 = 0;
        if !self.transform_metadata.is_empty() {
            flags |= FLAG_HAS_TRANSFORM;
        }
        if self.has_dict {
            flags |= FLAG_HAS_DICT;
        }
        w.write_all(&[flags])?;
        w.write_all(&self.original_size.to_le_bytes())?;
        w.write_all(&self.compressed_size.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        w.write_all(&(self.transform_metadata.len() as u32).to_le_bytes())?;
        if !self.transform_metadata.is_empty() {
            w.write_all(&self.transform_metadata)?;
        }
        Ok(())
    }

    /// Deserialize header from reader.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;

        if buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a .dcx file",
            ));
        }
        if buf[4] != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .dcx version: {} (expected {VERSION})", buf[4]),
            ));
        }

        let mode = Mode::from_u8(buf[5])?;
        let format_hint = FormatHint::from_u8(buf[6])?;
        let flags = buf[7];
        let has_dict = flags & FLAG_HAS_DICT != 0;
        let original_size = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let crc32 = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let transform_metadata_len = u32::from_le_bytes(buf[28..32].try_into().unwrap()) as usize;

        let transform_metadata = if flags & FLAG_HAS_TRANSFORM != 0 && transform_metadata_len > 0 {
            let mut meta = vec![0u8; transform_metadata_len];
            r.read_exact(&mut meta)?;
            meta
        } else {
            Vec::new()
        };

        Ok(DcxHeader {
            mode,
            format_hint,
            original_size,
            compressed_size,
            crc32,
            transform_metadata,
            has_dict,
        })
    }

    /// Total header size including transform metadata.
    pub fn total_size(&self) -> usize {
        HEADER_SIZE + self.transform_metadata.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = DcxHeader {
            mode: Mode::Balanced,
            format_hint: FormatHint::Json,
            original_size: 12345,
            compressed_size: 6789,
            crc32: 0xDEADBEEF,
            transform_metadata: vec![],
            has_dict: false,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = io::Cursor::new(&buf);
        let decoded = DcxHeader::read_from(&mut cursor).unwrap();

        assert_eq!(decoded.mode, Mode::Balanced);
        assert_eq!(decoded.format_hint, FormatHint::Json);
        assert_eq!(decoded.original_size, 12345);
        assert_eq!(decoded.compressed_size, 6789);
        assert_eq!(decoded.crc32, 0xDEADBEEF);
        assert!(decoded.transform_metadata.is_empty());
    }

    #[test]
    fn header_with_metadata() {
        let meta = vec![1, 2, 3, 4, 5];
        let header = DcxHeader {
            mode: Mode::Max,
            format_hint: FormatHint::Ndjson,
            original_size: 999,
            compressed_size: 500,
            crc32: 0x12345678,
            transform_metadata: meta.clone(),
            has_dict: false,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE + 5);

        let mut cursor = io::Cursor::new(&buf);
        let decoded = DcxHeader::read_from(&mut cursor).unwrap();
        assert_eq!(decoded.transform_metadata, meta);
        assert_eq!(decoded.total_size(), HEADER_SIZE + 5);
    }

    #[test]
    fn header_dict_flag_roundtrip() {
        let header = DcxHeader {
            mode: Mode::Fast,
            format_hint: FormatHint::Ndjson,
            original_size: 5000,
            compressed_size: 2000,
            crc32: 0xCAFEBABE,
            transform_metadata: vec![10, 20],
            has_dict: true,
        };

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        // Flags byte at offset 7 should have both bit 0 and bit 1 set.
        assert_eq!(buf[7], FLAG_HAS_TRANSFORM | FLAG_HAS_DICT);

        let mut cursor = io::Cursor::new(&buf);
        let decoded = DcxHeader::read_from(&mut cursor).unwrap();
        assert!(decoded.has_dict);
        assert_eq!(decoded.transform_metadata, vec![10, 20]);
    }

    #[test]
    fn bad_magic_rejected() {
        let buf = [0u8; HEADER_SIZE];
        let mut cursor = io::Cursor::new(&buf);
        assert!(DcxHeader::read_from(&mut cursor).is_err());
    }

    #[test]
    fn bad_version_rejected() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4] = 99; // wrong version
        let mut cursor = io::Cursor::new(&buf);
        assert!(DcxHeader::read_from(&mut cursor).is_err());
    }
}
