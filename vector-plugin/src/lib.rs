//! `DataCortex` compression codec for Vector.dev log pipelines.
//!
//! Provides [`DataCortexEncoder`] and [`DataCortexDecoder`] that match
//! Vector's `io::Write`-based compression pattern (same as `SnappyEncoder`).
//!
//! # How it works
//!
//! Vector encodes events to NDJSON bytes, then passes those bytes through a
//! compressor via `io::Write`. `DataCortex` needs the full payload for columnar
//! reorg, so we buffer all bytes and compress on `finish()` -- identical to
//! how Vector's Snappy codec works.
//!
//! # Example
//!
//! ```
//! use std::io::Write;
//! use datacortex_vector::{DataCortexEncoder, DataCortexDecoder};
//!
//! let ndjson = b"{\"level\":\"info\",\"msg\":\"hello\"}\n{\"level\":\"warn\",\"msg\":\"world\"}\n";
//!
//! // Compress
//! let mut compressed = Vec::new();
//! let mut encoder = DataCortexEncoder::new(&mut compressed);
//! encoder.write_all(ndjson).unwrap();
//! encoder.finish().unwrap();
//!
//! // Decompress
//! let original = DataCortexDecoder::decompress(&compressed).unwrap();
//! assert_eq!(original, ndjson);
//! ```

mod config;

#[doc(inline)]
pub use config::DataCortexConfig;
#[doc(inline)]
pub use datacortex_core::dcx::{FormatHint, Mode};

use std::io;

use datacortex_core::codec;

/// NDJSON compressed output is typically 2-25% of input size.
/// We pre-allocate at 25% to avoid reallocation in most cases.
const COMPRESSED_OUTPUT_RATIO: usize = 4;

/// Compression encoder matching Vector's `io::Write` pattern.
///
/// Buffers all written bytes, then compresses the complete payload on
/// [`finish()`](Self::finish) using the `DataCortex` format-aware pipeline
/// (columnar reorg + typed encoding + auto-fallback across 6+ paths).
///
/// This follows the same buffer-then-compress pattern Vector uses for Snappy.
pub struct DataCortexEncoder<W: io::Write> {
    writer: W,
    buffer: Vec<u8>,
    config: DataCortexConfig,
}

impl<W: io::Write> DataCortexEncoder<W> {
    /// Create a new encoder with default config (Fast mode, NDJSON format).
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buffer: Vec::new(),
            config: DataCortexConfig::default(),
        }
    }

    /// Create a new encoder with a capacity hint for the internal buffer.
    ///
    /// Use this when you know the approximate batch size (e.g., 1 MB for
    /// typical Vector batches) to avoid reallocation during writes.
    pub fn with_capacity(writer: W, capacity: usize) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(capacity),
            config: DataCortexConfig::default(),
        }
    }

    /// Create a new encoder with custom config.
    pub fn with_config(writer: W, config: DataCortexConfig) -> Self {
        Self {
            writer,
            buffer: Vec::new(),
            config,
        }
    }

    /// Compress buffered data and write to the underlying writer.
    ///
    /// Consumes the encoder. Returns the underlying writer for reuse.
    ///
    /// # Errors
    ///
    /// Returns `io::Error` if compression or writing fails.
    pub fn finish(mut self) -> io::Result<W> {
        if !self.buffer.is_empty() {
            let mode = self.config.mode();
            let format = self.config.format_hint();

            let mut compressed = Vec::with_capacity(self.buffer.len() / COMPRESSED_OUTPUT_RATIO);
            if self.config.is_turbo() {
                codec::compress_turbo(&self.buffer, Some(format), &mut compressed)?;
            } else {
                codec::compress(&self.buffer, mode, Some(format), &mut compressed)?;
            }

            self.writer.write_all(&compressed)?;
        }

        Ok(self.writer)
    }

    /// Reference to the underlying writer.
    pub const fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Returns `true` if no data has been written to the encoder.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Number of uncompressed bytes buffered so far.
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }
}

impl<W: io::Write> io::Write for DataCortexEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<W: io::Write + std::fmt::Debug> std::fmt::Debug for DataCortexEncoder<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataCortexEncoder")
            .field("writer", &self.get_ref())
            .field("buffered_bytes", &self.buffer.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Decoder for `DataCortex` compressed data.
#[derive(Debug, Clone, Copy)]
pub struct DataCortexDecoder;

impl DataCortexDecoder {
    /// Decompress a `.dcx` byte slice back to the original data.
    ///
    /// # Errors
    ///
    /// Returns `io::Error` if the data is not valid `.dcx` format.
    pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
        codec::decompress_from_slice(data)
    }

    /// Decompress `.dcx` data from a reader.
    ///
    /// # Errors
    ///
    /// Returns `io::Error` if reading or decompression fails.
    pub fn decompress_from_reader<R: io::Read>(reader: &mut R) -> io::Result<Vec<u8>> {
        codec::decompress(reader)
    }
}

/// File extension for `DataCortex` compressed files (Vector file/S3 sinks).
pub const EXTENSION: &str = "log.dcx";

/// Content-Encoding header value (Vector HTTP sinks).
pub const CONTENT_ENCODING: &str = "datacortex";

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::io::Write;

    fn sample_ndjson() -> Vec<u8> {
        let mut data = String::new();
        for i in 0..100 {
            writeln!(
                data,
                "{{\"timestamp\":\"2026-04-02T12:{:02}:00Z\",\"level\":\"info\",\
                 \"service\":\"api\",\"msg\":\"request processed\",\
                 \"duration_ms\":{},\"status\":200}}",
                i % 60,
                42 + (i % 50)
            )
            .unwrap();
        }
        data.into_bytes()
    }

    #[test]
    fn roundtrip_preserves_ndjson_bytes() {
        let ndjson = sample_ndjson();

        let mut compressed = Vec::new();
        let mut encoder = DataCortexEncoder::new(&mut compressed);
        encoder.write_all(&ndjson).unwrap();
        encoder.finish().unwrap();

        assert!(!compressed.is_empty());
        assert!(compressed.len() < ndjson.len(), "should actually compress");

        let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
        assert_eq!(decompressed, ndjson);
    }

    #[test]
    fn finish_on_empty_buffer_writes_nothing() {
        let mut compressed = Vec::new();
        let encoder = DataCortexEncoder::new(&mut compressed);
        encoder.finish().unwrap();

        assert!(compressed.is_empty());
    }

    #[test]
    fn roundtrip_preserves_single_line() {
        let ndjson = b"{\"level\":\"info\",\"msg\":\"hello world\"}\n";

        let mut compressed = Vec::new();
        let mut encoder = DataCortexEncoder::new(&mut compressed);
        encoder.write_all(ndjson).unwrap();
        encoder.finish().unwrap();

        let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
        assert_eq!(decompressed, ndjson);
    }

    #[test]
    fn roundtrip_with_incremental_writes() {
        let ndjson = sample_ndjson();

        // Write in small chunks (simulates Vector's batched encoding)
        let mut compressed = Vec::new();
        let mut encoder = DataCortexEncoder::new(&mut compressed);
        for chunk in ndjson.chunks(64) {
            encoder.write_all(chunk).unwrap();
        }
        encoder.finish().unwrap();

        let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
        assert_eq!(decompressed, ndjson);
    }

    #[test]
    fn is_empty_tracks_buffer_state() {
        let mut output = Vec::new();
        let mut encoder = DataCortexEncoder::new(&mut output);

        assert!(encoder.is_empty());
        assert_eq!(encoder.buffered_len(), 0);

        encoder.write_all(b"{}").unwrap();

        assert!(!encoder.is_empty());
        assert_eq!(encoder.buffered_len(), 2);
    }

    #[test]
    fn with_config_uses_specified_mode_and_format() {
        let ndjson = sample_ndjson();

        let config = DataCortexConfig::new(Mode::Fast, FormatHint::Ndjson);
        let mut compressed = Vec::new();
        let mut encoder = DataCortexEncoder::with_config(&mut compressed, config);
        encoder.write_all(&ndjson).unwrap();
        encoder.finish().unwrap();

        let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
        assert_eq!(decompressed, ndjson);
    }

    #[test]
    fn turbo_mode_roundtrip() {
        let ndjson = sample_ndjson();

        let config = DataCortexConfig::turbo();
        let mut compressed = Vec::new();
        let mut encoder = DataCortexEncoder::with_config(&mut compressed, config);
        encoder.write_all(&ndjson).unwrap();
        encoder.finish().unwrap();

        assert!(!compressed.is_empty());
        assert!(compressed.len() < ndjson.len(), "turbo should compress");

        let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
        assert_eq!(decompressed, ndjson);
    }

    #[test]
    fn turbo_config_builder() {
        let config = DataCortexConfig::new(Mode::Fast, FormatHint::Ndjson).with_turbo(true);
        assert!(config.is_turbo());
        assert_eq!(config.mode(), Mode::Fast);

        let default = DataCortexConfig::default();
        assert!(!default.is_turbo());
    }

    #[test]
    fn constants_match_expected_values() {
        assert_eq!(EXTENSION, "log.dcx");
        assert_eq!(CONTENT_ENCODING, "datacortex");
    }
}
