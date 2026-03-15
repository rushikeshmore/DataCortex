//! CMEngine -- orchestrates all context models + mixer + APM.
//!
//! Phase 5: Scaled context mixing engine with configurable sizes (CMConfig).
//!
//! Balanced preset (~256MB):
//! - Order-0 through Order-7 context models (16-32MB each)
//! - Match model: ring buffer (16MB) + hash table (8M entries)
//! - Word, Sparse, Run, JSON models
//!
//! Max preset (~512MB):
//! - 2x all ContextMap sizes vs Balanced
//! - Match model: ring buffer (32MB) + hash table (16M entries)
//! - Fewer hash collisions = better predictions
//!
//! Both presets use:
//! - Triple logistic mixer (fine 64K + med 16K + coarse 4K)
//! - 3-stage APM cascade
//!
//! Probability always in [1, 4095] -- clamped after every operation.
//! CRITICAL: Encoder/decoder must use IDENTICAL prediction sequence.

use crate::mixer::apm::APMStage;
use crate::mixer::dual_mixer::{DualMixer, byte_class};
use crate::model::cm_model::{AssociativeContextModel, ChecksumContextModel, ContextModel};
use crate::model::json_model::JsonModel;
use crate::model::match_model::MatchModel;
use crate::model::order0::Order0Model;
use crate::model::run_model::RunModel;
use crate::model::sparse_model::SparseModel;
use crate::model::word_model::WordModel;

/// Configuration for the CM engine -- controls memory/quality trade-off.
///
/// Each field specifies the ContextMap size in bytes for the corresponding model.
/// Larger sizes reduce hash collisions and improve prediction accuracy at the
/// cost of more memory and slightly slower cache performance.
#[derive(Debug, Clone)]
pub struct CMConfig {
    /// Order-1 ContextMap size (default: 32MB).
    pub order1_size: usize,
    /// Order-2 ContextMap size (default: 16MB).
    pub order2_size: usize,
    /// Order-3 ChecksumContextMap size (default: 32MB).
    pub order3_size: usize,
    /// Order-4 ChecksumContextMap size (default: 32MB).
    pub order4_size: usize,
    /// Order-5 AssociativeContextMap size (default: 32MB).
    pub order5_size: usize,
    /// Order-6 AssociativeContextMap size (default: 16MB).
    pub order6_size: usize,
    /// Order-7 AssociativeContextMap size (default: 32MB).
    pub order7_size: usize,
    /// Match model ring buffer size in bytes (default: 16MB). Must be power of 2.
    pub match_ring_size: usize,
    /// Match model hash table entry count (default: 8M). Must be power of 2.
    pub match_hash_size: usize,
    /// Word model ContextMap size (default: 16MB).
    pub word_size: usize,
    /// Sparse model ContextMap size per gap context (default: 8MB, 16MB total).
    pub sparse_size: usize,
    /// Run model ContextMap size (default: 4MB).
    pub run_size: usize,
    /// JSON model ContextMap size (default: 8MB).
    pub json_size: usize,
}

impl CMConfig {
    /// Balanced preset: current production sizes (~256MB total).
    pub fn balanced() -> Self {
        CMConfig {
            order1_size: 1 << 25,      // 32MB
            order2_size: 1 << 24,      // 16MB
            order3_size: 1 << 25,      // 32MB
            order4_size: 1 << 25,      // 32MB
            order5_size: 1 << 25,      // 32MB
            order6_size: 1 << 24,      // 16MB
            order7_size: 1 << 25,      // 32MB
            match_ring_size: 16 << 20, // 16MB
            match_hash_size: 8 << 20,  // 8M entries
            word_size: 1 << 24,        // 16MB
            sparse_size: 1 << 23,      // 8MB per gap (16MB total)
            run_size: 1 << 22,         // 4MB
            json_size: 1 << 23,        // 8MB
        }
    }

    /// Max preset: 2x everything for better compression (~512MB total).
    /// Doubles all ContextMap sizes, match ring, and hash table.
    /// More context slots = fewer collisions = better predictions.
    pub fn max() -> Self {
        CMConfig {
            order1_size: 1 << 26,      // 64MB
            order2_size: 1 << 25,      // 32MB
            order3_size: 1 << 26,      // 64MB
            order4_size: 1 << 26,      // 64MB
            order5_size: 1 << 26,      // 64MB
            order6_size: 1 << 25,      // 32MB
            order7_size: 1 << 26,      // 64MB
            match_ring_size: 32 << 20, // 32MB
            match_hash_size: 16 << 20, // 16M entries
            word_size: 1 << 25,        // 32MB
            sparse_size: 1 << 24,      // 16MB per gap (32MB total)
            run_size: 1 << 23,         // 8MB
            json_size: 1 << 24,        // 16MB
        }
    }
}

/// Context mixing engine -- orchestrates all models, mixer, and APM.
pub struct CMEngine {
    // --- Models ---
    /// Order-0: 256-context partial byte predictor.
    order0: Order0Model,
    /// Order-1: previous byte + partial byte context. ContextMap 32MB.
    order1: ContextModel,
    /// Order-2: previous 2 bytes + partial byte context. ContextMap 16MB.
    order2: ContextModel,
    /// Order-3: previous 3 bytes + partial byte context. ChecksumContextMap 32MB.
    order3: ChecksumContextModel,
    /// Order-4: previous 4 bytes + partial byte context. ChecksumContextMap 32MB.
    order4: ChecksumContextModel,
    /// Order-5: previous 5 bytes + partial byte context. AssociativeContextMap 32MB.
    order5: AssociativeContextModel,
    /// Order-6: previous 6 bytes + partial byte context. AssociativeContextMap 16MB.
    order6: AssociativeContextModel,
    /// Order-7: previous 7 bytes + partial byte context. AssociativeContextMap 32MB.
    order7: AssociativeContextModel,
    /// Match model: ring buffer (16MB) + hash table (8M entries).
    match_model: MatchModel,
    /// Word model: word boundary context. ContextMap 16MB.
    word_model: WordModel,
    /// Sparse model: skip-byte context for periodic patterns. 8MB total.
    sparse_model: SparseModel,
    /// Run model: run-length context. 2MB.
    run_model: RunModel,
    /// JSON model: structure-aware context. 4MB.
    json_model: JsonModel,

    // --- Mixer + APM ---
    /// Triple logistic mixer (fine 64K + medium 16K + coarse 4K).
    mixer: DualMixer,
    /// APM Stage 1: 2K contexts (c0 * bpos), 50% blend.
    apm1: APMStage,
    /// APM Stage 2: 16K contexts (c1 * bpos * byte_class), 25% blend.
    apm2: APMStage,
    /// APM Stage 3: 4K contexts (match_q * c1_top4 * bpos), 20% blend.
    apm3: APMStage,

    // --- Context state ---
    /// Partial byte being decoded (1-255). Starts at 1.
    c0: u32,
    /// Last completed byte.
    c1: u8,
    /// Second-to-last completed byte.
    c2: u8,
    /// Third-to-last completed byte.
    c3: u8,
    /// Fourth-to-last completed byte.
    c4: u8,
    /// Fifth-to-last completed byte.
    c5: u8,
    /// Sixth-to-last completed byte.
    c6: u8,
    /// Seventh-to-last completed byte.
    c7: u8,
    /// Bit position within current byte (0-7).
    bpos: u8,
    /// Byte-level run length (consecutive identical bytes).
    run_len: u8,
    /// Distance since last newline (quantized).
    line_pos: u16,
}

impl CMEngine {
    /// Create a new CM engine with balanced (default) configuration.
    pub fn new() -> Self {
        Self::with_config(CMConfig::balanced())
    }

    /// Create a CM engine with a specific configuration.
    pub fn with_config(config: CMConfig) -> Self {
        CMEngine {
            order0: Order0Model::new(),
            order1: ContextModel::new(config.order1_size),
            order2: ContextModel::new(config.order2_size),
            order3: ChecksumContextModel::new(config.order3_size),
            order4: ChecksumContextModel::new(config.order4_size),
            order5: AssociativeContextModel::new(config.order5_size),
            order6: AssociativeContextModel::new(config.order6_size),
            order7: AssociativeContextModel::new(config.order7_size),
            match_model: MatchModel::with_sizes(config.match_ring_size, config.match_hash_size),
            word_model: WordModel::with_size(config.word_size),
            sparse_model: SparseModel::with_size(config.sparse_size),
            run_model: RunModel::with_size(config.run_size),
            json_model: JsonModel::with_size(config.json_size),
            mixer: DualMixer::new(),
            apm1: APMStage::new(2048, 55),  // c0(256) * bpos(8) = 2048
            apm2: APMStage::new(16384, 30), // c1*bpos*byte_class = 256*8*8 = 16K
            apm3: APMStage::new(4096, 25), // match_q(4) * c2_top2(4) * c1_top4(16) * bpos(8) = 2048, use 4K
            c0: 1,
            c1: 0,
            c2: 0,
            c3: 0,
            c4: 0,
            c5: 0,
            c6: 0,
            c7: 0,
            bpos: 0,
            run_len: 0,
            line_pos: 0,
        }
    }

    /// Predict probability of the next bit being 1.
    /// Returns 12-bit probability in [1, 4095].
    ///
    /// CRITICAL: encoder and decoder must call this with identical state.
    #[inline(always)]
    pub fn predict(&mut self) -> u32 {
        // --- Gather predictions from each model ---
        let c0 = self.c0;
        let c1 = self.c1;
        let c2 = self.c2;
        let c3 = self.c3;
        let c4 = self.c4;
        let c5 = self.c5;
        let c6 = self.c6;
        let c7 = self.c7;
        let bpos = self.bpos;

        // Order-0: context is the partial byte.
        let p0 = self.order0.predict(c0 as usize);

        // Order-1: context hash = c1 combined with partial byte.
        let h1 = order1_hash(c1, c0);
        let p1 = self.order1.predict(h1);

        // Order-2: context hash = c2, c1, partial byte.
        let h2 = order2_hash(c2, c1, c0);
        let p2 = self.order2.predict(h2);

        // Order-3: context hash = c3, c2, c1, partial byte.
        let h3 = order3_hash(c3, c2, c1, c0);
        let p3 = self.order3.predict(h3);

        // Order-4: context hash = c4, c3, c2, c1, partial byte.
        let h4 = order4_hash(c4, c3, c2, c1, c0);
        let p4 = self.order4.predict(h4);

        // Order-5: context hash = c5, c4, c3, c2, c1, partial byte.
        let h5 = order5_hash(c5, c4, c3, c2, c1, c0);
        let p5 = self.order5.predict(h5);

        // Order-6: context hash = c6, c5, c4, c3, c2, c1, partial byte.
        let h6 = order6_hash(c6, c5, c4, c3, c2, c1, c0);
        let p6 = self.order6.predict(h6);

        // Order-7: context hash = c7..c1, partial byte.
        let h7 = order7_hash(c7, c6, c5, c4, c3, c2, c1, c0);
        let p7 = self.order7.predict(h7);

        // Match model.
        let p_match = self.match_model.predict(c0, bpos, c1, c2, c3);

        // Word model.
        let p_word = self.word_model.predict(c0, bpos, c1);

        // Sparse model (skip-byte patterns).
        let p_sparse = self.sparse_model.predict(c0, c1, c2, c3);

        // Run model (run-length patterns).
        let p_run = self.run_model.predict(c0, bpos, c1);

        // JSON model (structure-aware).
        let p_json = self.json_model.predict(c0, bpos, c1);

        // --- Mix ---
        let predictions = [
            p0, p1, p2, p3, p4, p5, p6, p7, p_match, p_word, p_sparse, p_run, p_json,
        ];
        let bclass = byte_class(c1);
        let match_q = self.match_model.match_length_quantized();
        let run_q = quantize_run_for_mixer(self.run_len);

        let mixed = self
            .mixer
            .predict(&predictions, c0, c1, bpos, bclass, match_q, run_q);

        // --- APM cascade ---
        // Stage 1: context = c0_partial(8b) + bpos(3b) = 11 bits -> 2048 contexts.
        let apm1_ctx = ((c0 as usize & 0xFF) << 3) | bpos as usize;
        let after_apm1 = self.apm1.predict(mixed, apm1_ctx);

        // Stage 2: context = c1(8b) * bpos(3b) + byte_class(3b).
        let apm2_ctx = ((c1 as usize) << 3 | bpos as usize) * 8 + bclass as usize;
        let after_apm2 = self.apm2.predict(after_apm1, apm2_ctx);

        // Stage 3: context = match_q(2b) * c1_top4(4b) * bpos(3b) * c2_top2(2b).
        let apm3_ctx = (match_q as usize * 512)
            + ((c2 as usize >> 6) << 7)
            + ((c1 as usize >> 4) << 3)
            + bpos as usize;
        let final_p = self.apm3.predict(after_apm2, apm3_ctx);

        final_p.clamp(1, 4095)
    }

    /// Update all models after observing `bit`.
    ///
    /// CRITICAL: encoder and decoder must call this with identical state and bit.
    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        // Update APM (reverse order: stage 3 first, then 2, then 1).
        self.apm3.update(bit);
        self.apm2.update(bit);
        self.apm1.update(bit);

        // Update mixer.
        self.mixer.update(bit);

        // Update all models.
        self.order0.update(self.c0 as usize, bit);
        self.order1.update(bit);
        self.order2.update(bit);
        self.order3.update(bit);
        self.order4.update(bit);
        self.order5.update(bit);
        self.order6.update(bit);
        self.order7.update(bit);
        self.match_model
            .update(bit, self.bpos, self.c0, self.c1, self.c2);
        self.word_model.update(bit);
        self.sparse_model.update(bit);
        self.run_model.update(bit);
        self.json_model.update(bit);

        // Advance context state.
        self.c0 = (self.c0 << 1) | bit as u32;
        self.bpos += 1;

        if self.bpos >= 8 {
            // Byte complete. Extract byte value and reset.
            let byte = (self.c0 & 0xFF) as u8;
            // Track run length.
            if byte == self.c1 {
                self.run_len = self.run_len.saturating_add(1);
            } else {
                self.run_len = 1;
            }
            // Track line position.
            if byte == b'\n' {
                self.line_pos = 0;
            } else {
                self.line_pos = self.line_pos.saturating_add(1);
            }

            self.c7 = self.c6;
            self.c6 = self.c5;
            self.c5 = self.c4;
            self.c4 = self.c3;
            self.c3 = self.c2;
            self.c2 = self.c1;
            self.c1 = byte;
            self.c0 = 1; // reset partial byte
            self.bpos = 0;
        }
    }
}

impl Default for CMEngine {
    fn default() -> Self {
        Self::new()
    }
}

// --- Context hash functions ---
// Use FNV-1a style hashing with different seeds per order for speed
// and reasonable distribution. The seed ensures different orders produce
// different hashes even with overlapping byte sequences.

/// FNV-1a offset basis.
const FNV_OFFSET: u32 = 0x811C9DC5;
/// FNV-1a prime.
const FNV_PRIME: u32 = 0x01000193;

/// Order-1 context hash: combines last byte with partial byte.
#[inline]
fn order1_hash(c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-2 context hash: combines last 2 bytes with partial byte.
#[inline]
fn order2_hash(c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-3 context hash: combines last 3 bytes with partial byte.
#[inline]
fn order3_hash(c3: u8, c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-4 context hash: combines last 4 bytes with partial byte.
#[inline]
fn order4_hash(c4: u8, c3: u8, c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c4 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-5 context hash: combines last 5 bytes with partial byte.
#[inline]
fn order5_hash(c5: u8, c4: u8, c3: u8, c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c5 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c4 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-6 context hash: combines last 6 bytes with partial byte.
#[inline]
fn order6_hash(c6: u8, c5: u8, c4: u8, c3: u8, c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c6 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c5 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c4 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Order-7 context hash: combines last 7 bytes with partial byte.
#[inline]
#[allow(clippy::too_many_arguments)]
fn order7_hash(c7: u8, c6: u8, c5: u8, c4: u8, c3: u8, c2: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c7 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c6 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c5 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c4 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Quantize run length to 0-3 for mixer context.
#[inline]
fn quantize_run_for_mixer(run_len: u8) -> u8 {
    match run_len {
        0..=1 => 0,
        2..=3 => 1,
        4..=8 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_balanced() {
        let mut engine = CMEngine::new();
        let p = engine.predict();
        // Should be near 2048 (all models start balanced).
        assert!(
            (1800..=2200).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_always_in_range() {
        let mut engine = CMEngine::new();
        let data = b"Hello, World! This is a test of the CM engine.";
        for &byte in data {
            for bpos in 0..8 {
                let p = engine.predict();
                assert!(
                    (1..=4095).contains(&p),
                    "prediction out of range at bpos {bpos}: {p}"
                );
                let bit = (byte >> (7 - bpos)) & 1;
                engine.update(bit);
            }
        }
    }

    #[test]
    fn context_state_tracks_correctly() {
        let mut engine = CMEngine::new();
        // Feed byte 0x42 (01000010)
        let byte: u8 = 0x42;
        for bpos in 0..8 {
            let _p = engine.predict();
            let bit = (byte >> (7 - bpos)) & 1;
            engine.update(bit);
        }
        // After byte complete, c1 should be 0x42, c0 should be 1.
        assert_eq!(engine.c1, 0x42);
        assert_eq!(engine.c0, 1);
        assert_eq!(engine.bpos, 0);
    }

    #[test]
    fn repeated_byte_adapts() {
        let mut engine = CMEngine::new();
        let byte: u8 = b'A';
        let mut total_bits: f64 = 0.0;
        let mut first_byte_bits: f64 = 0.0;

        for iteration in 0..50 {
            let mut byte_bits: f64 = 0.0;
            for bpos in 0..8 {
                let p = engine.predict();
                let bit = (byte >> (7 - bpos)) & 1;
                let prob_of_bit = if bit == 1 {
                    p as f64 / 4096.0
                } else {
                    1.0 - p as f64 / 4096.0
                };
                byte_bits += -prob_of_bit.max(0.001).log2();
                engine.update(bit);
            }
            if iteration == 0 {
                first_byte_bits = byte_bits;
            }
            total_bits += byte_bits;
        }

        let avg = total_bits / 50.0;
        assert!(
            avg < first_byte_bits,
            "engine should improve: first={first_byte_bits:.2}, avg={avg:.2}"
        );
    }

    #[test]
    fn hash_functions_differ() {
        let h1 = order1_hash(65, 1);
        let h2 = order2_hash(0, 65, 1);
        let h3 = order3_hash(0, 0, 65, 1);
        // All should produce different hashes.
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
    }

    #[test]
    fn engine_deterministic() {
        // Two engines with same input must produce same predictions.
        let data = b"determinism test";
        let mut e1 = CMEngine::new();
        let mut e2 = CMEngine::new();

        for &byte in data.iter() {
            for bpos in 0..8 {
                let p1 = e1.predict();
                let p2 = e2.predict();
                assert_eq!(p1, p2, "engines diverged at bpos {bpos}");
                let bit = (byte >> (7 - bpos)) & 1;
                e1.update(bit);
                e2.update(bit);
            }
        }
    }
}
