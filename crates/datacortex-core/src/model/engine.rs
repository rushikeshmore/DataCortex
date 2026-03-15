//! CMEngine — orchestrates all context models + mixer + APM.
//!
//! Phase 3: Full context mixing engine with:
//! - Order-0: 256-entry direct model
//! - Order-1: ContextMap(4MB) + StateMap (context = prev byte + partial)
//! - Order-2: ContextMap(512K) + StateMap (context = prev 2 bytes + partial)
//! - Order-3: ContextMap(2MB) + StateMap (context = prev 3 bytes + partial)
//! - Match model: ring buffer + hash table
//! - Word model: word boundary context
//! - Dual logistic mixer (fine 64K + coarse 4K)
//! - Two-stage APM cascade
//!
//! Probability always in [1, 4095] — clamped after every operation.
//! CRITICAL: Encoder/decoder must use IDENTICAL prediction sequence.

use crate::mixer::apm::APMStage;
use crate::mixer::dual_mixer::{DualMixer, byte_class};
use crate::model::cm_model::ContextModel;
use crate::model::match_model::MatchModel;
use crate::model::order0::Order0Model;
use crate::model::word_model::WordModel;

/// Context mixing engine — orchestrates all models, mixer, and APM.
pub struct CMEngine {
    // --- Models ---
    /// Order-0: 256-context partial byte predictor.
    order0: Order0Model,
    /// Order-1: previous byte + partial byte context. ContextMap 4MB (2^22).
    order1: ContextModel,
    /// Order-2: previous 2 bytes + partial byte context. ContextMap 512KB (2^19).
    order2: ContextModel,
    /// Order-3: previous 3 bytes + partial byte context. ContextMap 2MB (2^21).
    order3: ContextModel,
    /// Order-4: previous 4 bytes + partial byte context. ContextMap 4MB (2^22).
    order4: ContextModel,
    /// Match model: ring buffer + hash table.
    match_model: MatchModel,
    /// Word model: word boundary context. ContextMap 2MB (2^21).
    word_model: WordModel,

    // --- Mixer + APM ---
    /// Dual logistic mixer (fine 64K + coarse 4K).
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
    /// Bit position within current byte (0-7).
    bpos: u8,
}

impl CMEngine {
    /// Create a new CM engine with all models, mixer, and APM initialized.
    pub fn new() -> Self {
        CMEngine {
            order0: Order0Model::new(),
            order1: ContextModel::new(1 << 23), // 8MB
            order2: ContextModel::new(1 << 21), // 2MB
            order3: ContextModel::new(1 << 22), // 4MB
            order4: ContextModel::new(1 << 22), // 4MB
            match_model: MatchModel::new(),
            word_model: WordModel::new(),
            mixer: DualMixer::new(),
            apm1: APMStage::new(2048, 55),  // c0(256) * bpos(8) = 2048
            apm2: APMStage::new(16384, 30), // c1*bpos*byte_class = 256*8*8 = 16K
            apm3: APMStage::new(4096, 25),  // match_q(4) * c1_top4(16) * bpos(8) = 512, use 4K
            c0: 1,
            c1: 0,
            c2: 0,
            c3: 0,
            c4: 0,
            bpos: 0,
        }
    }

    /// Predict probability of the next bit being 1.
    /// Returns 12-bit probability in [1, 4095].
    ///
    /// CRITICAL: encoder and decoder must call this with identical state.
    #[inline]
    pub fn predict(&mut self) -> u32 {
        // --- Gather predictions from each model ---
        let c0 = self.c0;
        let c1 = self.c1;
        let c2 = self.c2;
        let c3 = self.c3;
        let c4 = self.c4;
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

        // Match model.
        let p_match = self.match_model.predict(c0, bpos, c1, c2, c3);

        // Word model.
        let p_word = self.word_model.predict(c0, bpos, c1);

        // --- Mix ---
        let predictions = [p0, p1, p2, p3, p4, p_match, p_word];
        let bclass = byte_class(c1);
        let match_q = self.match_model.match_length_quantized();

        let mixed = self
            .mixer
            .predict(&predictions, c0, c1, bpos, bclass, match_q);

        // --- APM cascade ---
        // Stage 1: context = c0_partial(8b) + bpos(3b) = 11 bits → 2048 contexts.
        let apm1_ctx = ((c0 as usize & 0xFF) << 3) | bpos as usize;
        let after_apm1 = self.apm1.predict(mixed, apm1_ctx);

        // Stage 2: context = c1(8b) * bpos(3b) + byte_class(3b).
        let apm2_ctx = ((c1 as usize) << 3 | bpos as usize) * 8 + bclass as usize;
        let after_apm2 = self.apm2.predict(after_apm1, apm2_ctx);

        // Stage 3: context = match_q(2b) * c1_top4(4b) * bpos(3b).
        let apm3_ctx = (match_q as usize * 128) + ((c1 as usize >> 4) << 3) + bpos as usize;
        let final_p = self.apm3.predict(after_apm2, apm3_ctx);

        final_p.clamp(1, 4095)
    }

    /// Update all models after observing `bit`.
    ///
    /// CRITICAL: encoder and decoder must call this with identical state and bit.
    #[inline]
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
        self.match_model
            .update(bit, self.bpos, self.c0, self.c1, self.c2);
        self.word_model.update(bit);

        // Advance context state.
        self.c0 = (self.c0 << 1) | bit as u32;
        self.bpos += 1;

        if self.bpos >= 8 {
            // Byte complete. Extract byte value and reset.
            let byte = (self.c0 & 0xFF) as u8;
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
