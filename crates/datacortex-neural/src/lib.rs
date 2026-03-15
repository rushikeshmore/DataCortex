//! DataCortex neural module — LLM byte-level prediction for Max mode.
//!
//! Feature-gated behind `neural`. When enabled, provides an LLM predictor
//! that generates byte-level probability distributions using a pre-trained
//! GGUF model via llama.cpp.
//!
//! Architecture (dual-path compression):
//!   Input bytes → [CM Engine] → P_cm (12-bit)
//!                → [LLM]      → P_llm (12-bit)
//!                → [MetaMixer] → P_final → [Arithmetic Coder]
//!
//! The LLM predicts TOKEN probabilities, which we convert to BYTE probabilities
//! by mapping each byte (0-255) to its corresponding token(s) in the vocabulary.

#[cfg(feature = "neural")]
pub mod llm;

#[cfg(feature = "neural")]
pub use llm::{LlmError, LlmPredictor};

/// Meta-mixer that blends CM and LLM predictions.
/// This is always available (not feature-gated) since it's pure logic.
pub mod meta_mixer;
pub use meta_mixer::MetaMixer;
