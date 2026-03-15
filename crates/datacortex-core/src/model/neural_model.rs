//! NeuralModel -- bit-level cross-context model.
//!
//! Uses context hashes that combine bit-level information from the current
//! byte being decoded with byte-level context. This captures patterns that
//! traditional byte-context CM models miss because they don't condition on
//! the bits already decoded in the current byte.
//!
//! Key contexts:
//!   - c0_full (all decoded bits so far, not just low 8) x c1 nibbles
//!   - Byte boundary contexts (position in line, word boundary)
//!   - Bit-pattern contexts (repeated bit runs, alternating patterns)
//!
//! Uses the same ContextModel (ContextMap + StateMap) machinery as other models.
//!
//! CRITICAL: Encoder and decoder must produce IDENTICAL neural model state.

use crate::model::cm_model::ContextModel;

/// Number of internal context models.
const N_MODELS: usize = 4;

/// Neural model using bit-level cross-contexts.
pub struct NeuralModel {
    /// Internal context models.
    models: Vec<ContextModel>,
}

impl NeuralModel {
    /// Create a new neural model with default sizes.
    pub fn new() -> Self {
        Self::with_size(1 << 21) // 2MB per model, 8MB total
    }

    /// Create with custom size per internal model.
    pub fn with_size(size: usize) -> Self {
        let mut models = Vec::with_capacity(N_MODELS);
        for _ in 0..N_MODELS {
            models.push(ContextModel::new(size));
        }
        NeuralModel { models }
    }

    /// Predict using bit-level cross-contexts. Returns single 12-bit probability.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &mut self,
        c0: u32,
        bpos: u8,
        c1: u8,
        c2: u8,
        c3: u8,
        run_len: u8,
        match_q: u8,
    ) -> u32 {
        // c0 contains the FULL partial byte (1-255 range, MSB first).
        // At bpos=0, c0=1. At bpos=3, c0 has 1 + 3 decoded bits.
        // This is a unique signal that no other model uses directly.
        let c0_full = c0; // up to 9 bits of context

        // Context 0: c0_full (partial byte with all bits) x c1_hi_nibble
        // This is UNIQUE because other models use c0 & 0xFF (losing the leading 1-bit).
        // c0_full encodes exactly which bits have been decoded so far.
        let c1_hi = (c1 >> 4) as u32;
        let h0 = fhash3(c0_full, c1_hi, 0xA1B2_C3D4, 0xDEAD_1001);
        let p0 = self.models[0].predict(h0);

        // Context 1: c0_full x byte_class_pair(c1, c2)
        // Combines partial byte with a transition context
        let class_pair = byte_class_pair(c1, c2) as u32;
        let h1 = fhash3(c0_full, class_pair, 0xE5F6_0718, 0xBEEF_2002);
        let p1 = self.models[1].predict(h1);

        // Context 2: c0_full x c1 x run_q x bpos -- conditioning on whether we're in a run
        let rq = quantize_run(run_len) as u32;
        let h2 = fhash4(c0_full, c1 as u32, rq, bpos as u32, 0xCAFE_3003);
        let p2 = self.models[2].predict(h2);

        // Context 3: c0_full x c2_lo x match_q -- skip-1 with match awareness
        let c2_lo = (c2 & 0x0F) as u32;
        let h3 = fhash4(c0_full, c2_lo, match_q as u32, c3 as u32, 0xFACE_4004);
        let p3 = self.models[3].predict(h3);

        // Average predictions
        let sum = p0 + p1 + p2 + p3;
        let avg = sum / N_MODELS as u32;
        avg.clamp(1, 4095)
    }

    /// Update all internal models after observing bit.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        for model in &mut self.models {
            model.update(bit);
        }
    }
}

impl Default for NeuralModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode byte class pair as a single value (6x6 = 36 values).
#[inline]
fn byte_class_pair(c1: u8, c2: u8) -> u8 {
    byte_class_6(c1) * 6 + byte_class_6(c2)
}

/// 6-class byte classification.
#[inline]
fn byte_class_6(b: u8) -> u8 {
    match b {
        b'a'..=b'z' => 0,
        b'A'..=b'Z' => 1,
        b'0'..=b'9' => 2,
        b' ' | b'\t' => 3,
        b'\n' | b'\r' => 4,
        _ => 5,
    }
}

/// Quantize run length to 0-3.
#[inline]
fn quantize_run(run_len: u8) -> u8 {
    match run_len {
        0..=1 => 0,
        2..=3 => 1,
        4..=8 => 2,
        _ => 3,
    }
}

/// Fast hash of 3 values with seed. FNV-1a style.
#[inline]
fn fhash3(a: u32, b: u32, c: u32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= a;
    h = h.wrapping_mul(0x0100_0193);
    h ^= b;
    h = h.wrapping_mul(0x0100_0193);
    h ^= c;
    h = h.wrapping_mul(0x0100_0193);
    h
}

/// Fast hash of 4 values with seed.
#[inline]
fn fhash4(a: u32, b: u32, c: u32, d: u32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= a;
    h = h.wrapping_mul(0x0100_0193);
    h ^= b;
    h = h.wrapping_mul(0x0100_0193);
    h ^= c;
    h = h.wrapping_mul(0x0100_0193);
    h ^= d;
    h = h.wrapping_mul(0x0100_0193);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_near_half() {
        let mut model = NeuralModel::new();
        let p = model.predict(1, 0, 0, 0, 0, 0, 0);
        assert!(
            (1800..=2200).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_always_in_range() {
        let mut model = NeuralModel::new();
        for c1 in [0u8, 65, 128, 255] {
            for bpos in 0..8u8 {
                let p = model.predict(1, bpos, c1, 0, 0, 0, 0);
                assert!((1..=4095).contains(&p), "prediction out of range: {p}");
                model.update(1);
            }
        }
    }

    #[test]
    fn deterministic() {
        let mut m1 = NeuralModel::new();
        let mut m2 = NeuralModel::new();

        let data: &[u8] = b"Hello World";
        for &byte in data {
            for bpos in 0..8u8 {
                let p1 = m1.predict(1, bpos, byte, 0, 0, 0, 0);
                let p2 = m2.predict(1, bpos, byte, 0, 0, 0, 0);
                assert_eq!(p1, p2, "neural models diverged");
                let bit = (byte >> (7 - bpos)) & 1;
                m1.update(bit);
                m2.update(bit);
            }
        }
    }

    #[test]
    fn adapts_to_data() {
        let mut model = NeuralModel::new();
        let mut first_p = 0;
        for i in 0..200 {
            let p = model.predict(1, 0, b'A', b'B', b'C', 1, 0);
            if i == 0 {
                first_p = p;
            }
            model.update(1);
        }
        let final_p = model.predict(1, 0, b'A', b'B', b'C', 1, 0);
        assert!(
            final_p > first_p,
            "model should adapt: first={first_p}, final={final_p}"
        );
    }

    #[test]
    fn byte_class_categories() {
        assert_eq!(byte_class_6(b'a'), 0);
        assert_eq!(byte_class_6(b'Z'), 1);
        assert_eq!(byte_class_6(b'5'), 2);
        assert_eq!(byte_class_6(b' '), 3);
        assert_eq!(byte_class_6(b'\n'), 4);
        assert_eq!(byte_class_6(b'.'), 5);
    }
}
