//! DataCortex — lossless text compression engine.
//!
//! Format-aware preprocessing + bit-level context mixing + entropy coding.

pub mod codec;
pub mod dcx;
pub mod entropy;
pub mod format;
pub mod mixer;
pub mod model;
pub mod ramanujan;
pub mod state;

pub use codec::{compress, decompress, raw_zstd_compress, read_header};
pub use dcx::{DcxHeader, FormatHint, Mode};
pub use format::detect_format;
pub use model::CMEngine;
