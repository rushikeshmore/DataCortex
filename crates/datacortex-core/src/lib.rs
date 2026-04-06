//! DataCortex -- lossless JSON/NDJSON compression engine.
//!
//! Combines format-aware preprocessing (schema inference, columnar reorg,
//! typed encoding) with bit-level context mixing and entropy coding to
//! achieve compression ratios that beat zstd-19 and brotli-11 on every
//! JSON file tested.
//!
//! # Quick Start
//!
//! ```
//! use datacortex_core::codec::{compress_to_vec, decompress_from_slice};
//! use datacortex_core::dcx::Mode;
//!
//! let data = br#"{"id":1,"name":"test"}"#;
//! let compressed = compress_to_vec(data, Mode::Fast, None).unwrap();
//! let original = decompress_from_slice(&compressed).unwrap();
//! assert_eq!(data.as_slice(), original.as_slice());
//! ```

// Lint policy (see STYLE.md for rationale).
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::struct_excessive_bools,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::if_not_else,
    clippy::redundant_else,
    clippy::match_same_arms,
    clippy::inline_always,
    clippy::wildcard_imports,
    clippy::unnecessary_wraps,
    clippy::range_plus_one,
    clippy::single_match_else,
    clippy::uninlined_format_args,
    clippy::unused_self,
    clippy::struct_field_names,
    clippy::default_trait_access,
    clippy::large_types_passed_by_value,
    clippy::fn_params_excessive_bools,
    clippy::trivially_copy_pass_by_ref,
    clippy::verbose_bit_mask,
    clippy::format_push_string
)]

pub mod codec;
pub mod dcx;
pub mod entropy;
pub mod format;
pub mod mixer;
pub mod model;
pub mod state;

pub use codec::{
    compress, compress_turbo, compress_with_model, compress_with_options, decompress,
    decompress_with_model, raw_zstd_compress, read_header,
};
pub use dcx::{DcxHeader, FormatHint, Mode};
pub use format::detect_format;
pub use model::{CMConfig, CMEngine};
