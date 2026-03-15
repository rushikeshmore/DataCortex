//! LLM byte-level predictor using llama.cpp via llama-cpp-2 crate.
//!
//! Converts LLM token-level predictions into byte-level probabilities.
//!
//! Strategy:
//! 1. Build a byte→token mapping from the vocabulary at init time.
//! 2. Feed byte history as tokens to the LLM.
//! 3. After each forward pass, extract logits for tokens that correspond to
//!    single bytes, then normalize to get P(next_byte).
//! 4. Convert byte probabilities to bit probabilities for the arithmetic coder.
//!
//! The tricky part: LLMs predict TOKENS, not bytes. Most tokenizers have
//! single-byte tokens for all 256 byte values (especially byte-level BPE models
//! like SmolLM). We extract those and ignore multi-byte tokens.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

use std::num::NonZeroU32;
use std::path::Path;

/// Errors from the LLM predictor.
#[derive(Debug)]
pub enum LlmError {
    BackendInit(String),
    ModelLoad(String),
    ContextCreate(String),
    Decode(String),
    NoByteMappingFound,
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::BackendInit(e) => write!(f, "LLM backend init failed: {e}"),
            LlmError::ModelLoad(e) => write!(f, "LLM model load failed: {e}"),
            LlmError::ContextCreate(e) => write!(f, "LLM context creation failed: {e}"),
            LlmError::Decode(e) => write!(f, "LLM decode failed: {e}"),
            LlmError::NoByteMappingFound => {
                write!(f, "no single-byte tokens found in vocabulary")
            }
        }
    }
}

impl std::error::Error for LlmError {}

/// Maximum context window for the LLM. We keep this small for compression
/// (each byte generates 1+ tokens). 512 is enough for local context.
const DEFAULT_CTX_SIZE: u32 = 512;

/// LLM predictor that produces byte-level probability distributions.
///
/// Uses a two-level ownership to avoid self-referential struct issues:
/// - `model` is `Box<LlamaModel>` so it has a stable heap address.
/// - `context` borrows from the boxed model via unsafe lifetime extension.
/// - Drop order: context is dropped before model because Rust drops fields
///   in declaration order (context before model).
pub struct LlmPredictor {
    /// The llama.cpp backend (must stay alive).
    _backend: LlamaBackend,
    /// Inference context. MUST be declared before `_model` for correct drop order.
    /// (Rust drops fields in declaration order.)
    context: LlamaContext<'static>,
    /// The loaded model on heap (stable address). Dropped AFTER context.
    _model: Box<LlamaModel>,
    /// Mapping: byte_value (0-255) → LlamaToken. None if no single-byte token.
    byte_to_token: [Option<LlamaToken>; 256],
    /// Number of bytes successfully mapped to tokens.
    mapped_count: usize,
    /// Vocabulary size.
    n_vocab: usize,
    /// Token history for feeding to the model.
    token_history: Vec<LlamaToken>,
    /// Current position in the KV cache.
    kv_pos: usize,
    /// Maximum context size.
    max_ctx: usize,
    /// Cached byte probabilities from last LLM forward pass.
    /// 256 entries, each is a probability (f32).
    cached_byte_probs: [f32; 256],
    /// Whether cached_byte_probs is valid.
    cache_valid: bool,
    /// Number of bytes processed since last LLM forward pass.
    bytes_since_predict: usize,
}

impl LlmPredictor {
    /// Create a new LLM predictor from a GGUF model path.
    pub fn new(model_path: &str) -> Result<Box<Self>, LlmError> {
        Self::with_ctx_size(model_path, DEFAULT_CTX_SIZE)
    }

    /// Create with a custom context size.
    pub fn with_ctx_size(model_path: &str, ctx_size: u32) -> Result<Box<Self>, LlmError> {
        if !Path::new(model_path).exists() {
            return Err(LlmError::ModelLoad(format!(
                "model file not found: {model_path}"
            )));
        }

        // Initialize backend.
        let backend = LlamaBackend::init().map_err(|e| LlmError::BackendInit(format!("{e:?}")))?;

        // Load model into a Box for stable heap address.
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| LlmError::ModelLoad(format!("{e:?}")))?;
        let model = Box::new(model);

        let n_vocab = model.n_vocab() as usize;

        // Build byte->token mapping.
        let byte_to_token = build_byte_token_map(&model);
        let mapped_count = byte_to_token.iter().filter(|t| t.is_some()).count();

        if mapped_count == 0 {
            return Err(LlmError::NoByteMappingFound);
        }

        eprintln!(
            "[neural] loaded model: {} vocab, {}/{} bytes mapped to tokens",
            n_vocab, mapped_count, 256
        );

        // Create context from the boxed model.
        // SAFETY: The Box<LlamaModel> has a stable heap address. The context
        // borrows from it. We declare context BEFORE _model in the struct so
        // Rust drops context first, then model. The 'static lifetime is safe
        // because model outlives context.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_n_threads(1)
            .with_n_threads_batch(1);

        let model_ref: &'static LlamaModel = unsafe { &*(model.as_ref() as *const LlamaModel) };

        let context = model_ref
            .new_context(&backend, ctx_params)
            .map_err(|e| LlmError::ContextCreate(format!("{e:?}")))?;

        Ok(Box::new(LlmPredictor {
            _backend: backend,
            context,
            _model: model,
            byte_to_token,
            mapped_count,
            n_vocab,
            token_history: Vec::with_capacity(ctx_size as usize),
            kv_pos: 0,
            max_ctx: ctx_size as usize,
            cached_byte_probs: [0.0f32; 256],
            cache_valid: false,
            bytes_since_predict: 0,
        }))
    }

    /// Feed a byte to the LLM and get updated byte probabilities.
    /// Returns 256-element array of probabilities (NOT log-probs), normalized to sum to 1.0.
    ///
    /// This runs the LLM forward pass, which is slow (~10-50ms per byte).
    pub fn predict_byte_probs(&mut self, byte: u8) -> Result<&[f32; 256], LlmError> {
        // Convert byte to token.
        let token = match self.byte_to_token[byte as usize] {
            Some(t) => t,
            None => {
                // Unmapped byte: use a fallback (the model's unknown token or just skip).
                // For now, return uniform distribution.
                self.cached_byte_probs = [1.0 / 256.0; 256];
                self.cache_valid = true;
                return Ok(&self.cached_byte_probs);
            }
        };

        // Add to history.
        self.token_history.push(token);
        self.bytes_since_predict += 1;

        // Handle context window overflow: shift window.
        if self.token_history.len() > self.max_ctx {
            // Keep the last max_ctx/2 tokens.
            let keep = self.max_ctx / 2;
            let drain_count = self.token_history.len() - keep;
            self.token_history.drain(..drain_count);
            // Clear KV cache and re-encode from scratch.
            self.context.clear_kv_cache();
            self.kv_pos = 0;
        }

        // Run LLM forward pass on new tokens.
        self.run_forward()?;
        self.bytes_since_predict = 0;

        Ok(&self.cached_byte_probs)
    }

    /// Run the LLM forward pass on tokens from kv_pos to end of history.
    /// Extracts byte probabilities from the output logits.
    fn run_forward(&mut self) -> Result<(), LlmError> {
        let start = self.kv_pos;
        let end = self.token_history.len();

        if start >= end {
            // Nothing new to process.
            return Ok(());
        }

        let n_new = end - start;

        // Create batch with new tokens.
        let mut batch = LlamaBatch::new(n_new, 1);
        for (i, &token) in self.token_history[start..end].iter().enumerate() {
            let pos = (start + i) as i32;
            let logits = i == n_new - 1; // Only need logits for the last token.
            batch
                .add(token, pos, &[0], logits)
                .map_err(|e| LlmError::Decode(format!("batch add failed: {e:?}")))?;
        }

        // Decode (forward pass).
        self.context
            .decode(&mut batch)
            .map_err(|e| LlmError::Decode(format!("decode failed: {e:?}")))?;

        // Extract logits for the last token.
        // Copy logits to avoid borrow conflict (self.context borrows self immutably,
        // but extract_byte_probs borrows self mutably).
        let logits_slice = self.context.get_logits_ith((n_new - 1) as i32);
        let mut logits_copy = vec![0.0f32; logits_slice.len()];
        logits_copy.copy_from_slice(logits_slice);

        // Convert token logits to byte probabilities.
        self.extract_byte_probs(&logits_copy);

        self.kv_pos = end;
        self.cache_valid = true;

        Ok(())
    }

    /// Convert raw token logits into normalized byte probabilities.
    /// For each byte 0-255, find the corresponding token's logit.
    /// Apply softmax over the 256 byte-tokens only.
    fn extract_byte_probs(&mut self, logits: &[f32]) {
        // Step 1: Collect logits for mapped byte tokens.
        let mut byte_logits = [-100.0f32; 256]; // Very negative default for unmapped.
        let mut max_logit = f32::NEG_INFINITY;

        for byte_val in 0..256usize {
            if let Some(token) = self.byte_to_token[byte_val] {
                let token_id = token.0 as usize;
                if token_id < logits.len() {
                    byte_logits[byte_val] = logits[token_id];
                    if logits[token_id] > max_logit {
                        max_logit = logits[token_id];
                    }
                }
            }
        }

        // Step 2: Softmax over byte logits (numerically stable).
        if max_logit == f32::NEG_INFINITY {
            // No mapped bytes at all -- uniform.
            self.cached_byte_probs = [1.0 / 256.0; 256];
            return;
        }

        let mut sum = 0.0f64; // Use f64 for sum to avoid precision loss.
        for byte_val in 0..256 {
            let exp = ((byte_logits[byte_val] - max_logit) as f64).exp();
            self.cached_byte_probs[byte_val] = exp as f32;
            sum += exp;
        }

        // Normalize.
        if sum > 0.0 {
            let inv_sum = (1.0 / sum) as f32;
            for p in self.cached_byte_probs.iter_mut() {
                *p *= inv_sum;
            }
        }
    }

    /// Convert byte probabilities to a bit-level prediction.
    ///
    /// Given the current byte probabilities and the partial byte state,
    /// compute P(next_bit = 1).
    ///
    /// `bpos`: bit position (0-7, MSB first).
    /// `partial_byte`: the bits decoded so far (c0 in CM engine terms).
    ///   At bpos=0, partial_byte=1 (just the leading sentinel bit).
    ///   At bpos=3, partial_byte has 1 + 3 decoded bits.
    ///
    /// Returns: 12-bit probability [1, 4095] of the next bit being 1.
    pub fn predict_bit(&self, bpos: u8, partial_byte: u32) -> u32 {
        if !self.cache_valid {
            return 2048; // No prediction available yet -- return 50/50.
        }

        // The partial_byte encodes which bits have been decoded.
        // At bpos=0: partial=0b1 (sentinel only, no bits decoded yet)
        // At bpos=1: partial=0b1X (sentinel + 1 decoded bit)
        // At bpos=k: partial has k decoded bits after the sentinel
        //
        // We need to sum byte probabilities over all bytes consistent with
        // the already-decoded bits AND next bit = 0, vs next bit = 1.

        let mut prob_bit1 = 0.0f64;
        let mut prob_total = 0.0f64;

        for byte_val in 0..256u32 {
            // Check if this byte is consistent with the bits decoded so far.
            // The partial_byte has bpos bits already decided.
            // We need to check the top `bpos` bits of byte_val match.
            let consistent = if bpos == 0 {
                true // No bits decided yet -- all bytes possible.
            } else {
                // Extract top `bpos` bits from byte_val.
                let shift = 8 - bpos;
                let top_bits = byte_val >> shift;
                // Extract the decided bits from partial_byte.
                // partial_byte = 1 followed by bpos bits.
                // The decided bits are: partial_byte with the leading 1 removed.
                let decided = partial_byte & ((1 << bpos as u32) - 1);
                top_bits == decided
            };

            if consistent {
                let p = self.cached_byte_probs[byte_val as usize] as f64;
                prob_total += p;

                // Check if the next bit (at position bpos) is 1.
                let next_bit = (byte_val >> (7 - bpos as u32)) & 1;
                if next_bit == 1 {
                    prob_bit1 += p;
                }
            }
        }

        if prob_total <= 0.0 {
            return 2048; // Shouldn't happen, but safe fallback.
        }

        let p = prob_bit1 / prob_total;
        // Convert to 12-bit: [1, 4095].
        let p12 = (p * 4096.0).round() as u32;
        p12.clamp(1, 4095)
    }

    /// Get the number of mapped bytes (out of 256).
    pub fn mapped_bytes(&self) -> usize {
        self.mapped_count
    }

    /// Get vocab size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }
}

/// Build the byte→token mapping from the model's vocabulary.
///
/// Strategy: iterate over all tokens, find those whose string representation
/// is exactly one byte. Map that byte value to the token ID.
///
/// Most modern BPE tokenizers (GPT-2 style, used by SmolLM2) have single-byte
/// tokens for all 256 byte values, though they may be encoded as special
/// characters (e.g., byte 0x00 might be token "Ā" in the GPT-2 byte-encoding).
fn build_byte_token_map(model: &LlamaModel) -> [Option<LlamaToken>; 256] {
    let mut map: [Option<LlamaToken>; 256] = [None; 256];
    let n_vocab = model.n_vocab();

    // Strategy 1: Check each token's string representation.
    // For GPT-2 style BPE, single-byte tokens use a byte-to-unicode mapping:
    // bytes 33-126 and 161-172 and 174-255 map to themselves as Unicode chars,
    // bytes 0-32 and 127-160 and 173 map to Unicode chars starting at U+0100.
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    for token_id in 0..n_vocab {
        let token = LlamaToken::new(token_id);
        if let Ok(s) = model.token_to_piece(token, &mut decoder, true, None) {
            if s.len() == 1 {
                // Direct single-ASCII-char token.
                let byte_val = s.as_bytes()[0];
                if map[byte_val as usize].is_none() {
                    map[byte_val as usize] = Some(token);
                }
            } else if s.chars().count() == 1 {
                // Single Unicode char -- might be a GPT-2 byte encoding.
                let ch = s.chars().next().unwrap();
                if let Some(byte_val) = gpt2_char_to_byte(ch) {
                    if map[byte_val as usize].is_none() {
                        map[byte_val as usize] = Some(token);
                    }
                }
            }
        }
    }

    // Strategy 2: For any unmapped bytes, try tokenizing the byte directly.
    for byte_val in 0..256u16 {
        if map[byte_val as usize].is_none() {
            // Try to find a token for this exact byte.
            let byte_str = String::from(byte_val as u8 as char);
            if let Ok(tokens) = model.str_to_token(&byte_str, llama_cpp_2::model::AddBos::Never) {
                if tokens.len() == 1 {
                    map[byte_val as usize] = Some(tokens[0]);
                }
            }
        }
    }

    map
}

/// GPT-2 byte-to-Unicode mapping (inverse).
/// In GPT-2's tokenizer, bytes that aren't printable ASCII are mapped to
/// Unicode characters starting at U+0100 (Ā, ā, Ă, etc.).
/// This function reverses that mapping.
fn gpt2_char_to_byte(ch: char) -> Option<u8> {
    let c = ch as u32;

    // Direct mapping: printable ASCII and some high bytes map to themselves.
    // GPT-2 maps these ranges to themselves:
    //   '!' (33) to '~' (126)
    //   '¡' (161) to '¬' (172)
    //   '®' (174) to 'ÿ' (255)
    if (33..=126).contains(&c) || (161..=172).contains(&c) || (174..=255).contains(&c) {
        return Some(c as u8);
    }

    // Remapped bytes: 0-32, 127-160, 173 are mapped to U+0100..U+0143
    // The mapping is: bytes not in the direct set get mapped in order to
    // consecutive Unicode codepoints starting at 256 (U+0100).
    // There are 68 such bytes: 0-32 (33), 127-160 (34), 173 (1) = 68 total.
    if (256..=323).contains(&c) {
        // Map back: the 68 excluded bytes in order.
        static EXCLUDED_BYTES: [u8; 68] = {
            let mut arr = [0u8; 68];
            let mut idx = 0;
            let mut b = 0u16;
            while b <= 255 {
                let in_direct =
                    (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
                if !in_direct {
                    arr[idx] = b as u8;
                    idx += 1;
                }
                b += 1;
            }
            arr
        };

        let offset = (c - 256) as usize;
        if offset < EXCLUDED_BYTES.len() {
            return Some(EXCLUDED_BYTES[offset]);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_mapping_printable_ascii() {
        // Printable ASCII should map to themselves.
        for b in 33..=126u8 {
            let ch = b as char;
            assert_eq!(gpt2_char_to_byte(ch), Some(b), "byte {b} ('{ch}')");
        }
    }

    #[test]
    fn gpt2_mapping_remapped_bytes() {
        // Byte 0 should map to U+0100 (Ā).
        assert_eq!(gpt2_char_to_byte('\u{0100}'), Some(0));
        // Byte 1 should map to U+0101 (ā).
        assert_eq!(gpt2_char_to_byte('\u{0101}'), Some(1));
        // Space (32) is excluded from direct mapping, should map from U+0100 range.
        // It's the 33rd excluded byte (0-32 = 33 bytes, space=32 is the last).
        assert_eq!(gpt2_char_to_byte('\u{0120}'), Some(32));
    }

    #[test]
    fn gpt2_mapping_coverage() {
        // The GPT-2 mapping should cover all 256 bytes.
        let mut covered = [false; 256];

        // Direct mappings.
        for b in 0..=255u8 {
            let ch = b as char;
            if let Some(mapped) = gpt2_char_to_byte(ch) {
                covered[mapped as usize] = true;
            }
        }

        // Remapped range.
        for c in 256..=323u32 {
            if let Some(ch) = char::from_u32(c) {
                if let Some(mapped) = gpt2_char_to_byte(ch) {
                    covered[mapped as usize] = true;
                }
            }
        }

        let count = covered.iter().filter(|&&c| c).count();
        // Should cover most/all 256 bytes.
        assert!(
            count >= 250,
            "GPT-2 mapping should cover most bytes, only covers {count}/256"
        );
    }

    #[test]
    fn predict_bit_uniform_gives_half() {
        // Create a fake predictor state with uniform byte probs.
        let probs = [1.0 / 256.0; 256];
        // At bpos=0, partial=1, all bytes are consistent.
        // Half have bit 7 = 1, half have bit 7 = 0.
        let mut prob_bit1 = 0.0f64;
        for byte_val in 0..256u32 {
            if (byte_val >> 7) & 1 == 1 {
                prob_bit1 += probs[byte_val as usize] as f64;
            }
        }
        let p = (prob_bit1 * 4096.0).round() as u32;
        assert!(
            (2000..=2100).contains(&p),
            "uniform probs at bpos 0 should give ~2048, got {p}"
        );
    }

    #[test]
    fn predict_bit_with_constraint() {
        // If byte probs are uniform and we've already decoded bit 7 = 1,
        // then at bpos=1, only bytes 128-255 are consistent.
        // Among those, half have bit 6 = 1 → P(bit6=1) ≈ 0.5.
        let probs = [1.0 / 256.0; 256];
        let bpos = 1u8;
        let partial = 0b11; // sentinel 1 + decoded bit = 1

        let mut prob_bit1 = 0.0f64;
        let mut prob_total = 0.0f64;
        for byte_val in 0..256u32 {
            let top_bits = byte_val >> (8 - bpos);
            let decided = partial & ((1 << bpos) - 1);
            if top_bits == decided {
                prob_total += probs[byte_val as usize] as f64;
                if (byte_val >> (7 - bpos as u32)) & 1 == 1 {
                    prob_bit1 += probs[byte_val as usize] as f64;
                }
            }
        }
        let p = (prob_bit1 / prob_total * 4096.0).round() as u32;
        assert!(
            (2000..=2100).contains(&p),
            "constrained uniform should be ~2048, got {p}"
        );
    }
}
