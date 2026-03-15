//! Entropy coding — binary arithmetic coder (12-bit precision, carry-free).
//!
//! Phase 2: PAQ8-style binary arithmetic encoder/decoder.

pub mod arithmetic;

pub use arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
