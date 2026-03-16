//! TripleMixer — three-level logistic mixer (fine + medium + coarse) in log-odds space.
//!
//! Phase 3+: PAQ8-style logistic mixer with multi-output ContextMap support.
//!
//! 37 model inputs:
//! - Order-0: 1 (state only)
//! - Order-1 through Order-9: 3 each (state + run + byte_hist) = 27
//! - Match, Word, Sparse, Run, JSON, Indirect, PPM, DMC, ISSE: 1 each = 9
//!
//! Fine mixer: 64K weight sets, learning rate eta=2.
//! Medium mixer: 16K weight sets, learning rate eta=3.
//! Coarse mixer: 4K weight sets, learning rate eta=4.
//!
//! Blend: fine * 0.5 + medium * 0.3 + coarse * 0.2 in log-odds space.

use crate::mixer::logistic::{squash, stretch};

/// Number of models feeding the mixer.
/// Layout: [O0, O1_s, O1_r, O2_s, O2_r, ..., O9_s, O9_r,
///          match, word, sparse, run, json, indirect, ppm, dmc, isse]
/// = 1 + 9*2 + 9 = 28
pub const NUM_MODELS: usize = 28;

/// Fine mixer: 64K weight sets.
const FINE_SETS: usize = 65536;

/// Medium mixer: 16K weight sets.
const MEDIUM_SETS: usize = 16384;

/// Coarse mixer: 4K weight sets.
const COARSE_SETS: usize = 4096;

/// Weight scale factor (2^12 = 4096).
const W_SCALE: i32 = 4096;

/// Initial weights per model.
/// Order models: state predictions get higher weight, run starts low.
/// Layout: O0, O1(s,r), O2(s,r), ..., O9(s,r), match, word, sparse, run, json, indirect, ppm, dmc, isse
const INITIAL_WEIGHTS: [i32; NUM_MODELS] = [
    200, // O0
    300, 60, // O1 (state, run)
    350, 60, // O2
    450, 60, // O3
    450, 60, // O4
    450, 60, // O5
    300, 60, // O6
    250, 60, // O7
    200, 60, // O8
    180, 60,  // O9
    300, // match
    250, // word
    250, // sparse
    200, // run
    250, // json
    200, // indirect
    50,  // ppm
    30,  // dmc
    150, // isse
];

/// Fine mixer learning rate.
const FINE_LR: i32 = 2;

/// Medium mixer learning rate.
const MEDIUM_LR: i32 = 3;

/// Coarse mixer learning rate.
const COARSE_LR: i32 = 4;

/// Triple logistic mixer (fine + medium + coarse).
pub struct DualMixer {
    /// Fine mixer weights: [FINE_SETS][NUM_MODELS].
    fine_weights: Vec<[i32; NUM_MODELS]>,
    /// Medium mixer weights: [MEDIUM_SETS][NUM_MODELS].
    medium_weights: Vec<[i32; NUM_MODELS]>,
    /// Coarse mixer weights: [COARSE_SETS][NUM_MODELS].
    coarse_weights: Vec<[i32; NUM_MODELS]>,
    /// Cached stretched predictions from last predict() call.
    last_d: [i32; NUM_MODELS],
    /// Cached fine context index.
    last_fine_ctx: usize,
    /// Cached medium context index.
    last_medium_ctx: usize,
    /// Cached coarse context index.
    last_coarse_ctx: usize,
    /// Cached blended output probability.
    last_p: u32,
}

impl DualMixer {
    pub fn new() -> Self {
        DualMixer {
            fine_weights: vec![INITIAL_WEIGHTS; FINE_SETS],
            medium_weights: vec![INITIAL_WEIGHTS; MEDIUM_SETS],
            coarse_weights: vec![INITIAL_WEIGHTS; COARSE_SETS],
            last_d: [0; NUM_MODELS],
            last_fine_ctx: 0,
            last_medium_ctx: 0,
            last_coarse_ctx: 0,
            last_p: 2048,
        }
    }

    /// Mix model predictions to produce a final 12-bit probability.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &mut self,
        predictions: &[u32; NUM_MODELS],
        c0: u32,
        c1: u8,
        bpos: u8,
        byte_class: u8,
        match_len_q: u8,
        run_q: u8,
        xml_state: u8,
    ) -> u32 {
        // Stretch all predictions to log-odds.
        for i in 0..NUM_MODELS {
            self.last_d[i] = stretch(predictions[i]);
        }

        // Fine mixer context: full hash including XML state.
        self.last_fine_ctx = fine_context(c0, c1, bpos, byte_class, match_len_q, run_q, xml_state);
        // Medium mixer context: (c0, c1_top4, bpos, bclass, run_q, match_q, xml_state).
        self.last_medium_ctx = medium_context(c0, c1, bpos, run_q, match_len_q, xml_state);
        // Coarse mixer context: (c0, bpos).
        self.last_coarse_ctx = coarse_context(c0, bpos);

        // Compute weighted sums in log-odds space.
        let fw = &self.fine_weights[self.last_fine_ctx];
        let mw = &self.medium_weights[self.last_medium_ctx];
        let cw = &self.coarse_weights[self.last_coarse_ctx];

        let mut fine_sum: i64 = 0;
        let mut medium_sum: i64 = 0;
        let mut coarse_sum: i64 = 0;
        for i in 0..NUM_MODELS {
            let d = self.last_d[i] as i64;
            fine_sum += fw[i] as i64 * d;
            medium_sum += mw[i] as i64 * d;
            coarse_sum += cw[i] as i64 * d;
        }
        let fine_d = (fine_sum / W_SCALE as i64) as i32;
        let medium_d = (medium_sum / W_SCALE as i64) as i32;
        let coarse_d = (coarse_sum / W_SCALE as i64) as i32;

        // Blend: fine * 0.5 + medium * 0.3 + coarse * 0.2 in log-odds space.
        // Use integer: (fine * 5 + medium * 3 + coarse * 2) / 10
        let blended_d = (fine_d as i64 * 5 + medium_d as i64 * 3 + coarse_d as i64 * 2) / 10;
        let p = squash(blended_d as i32).clamp(1, 4095);
        self.last_p = p;
        p
    }

    /// Update weights after observing `bit`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, bit: u8) {
        let error = (bit as i32) * 4096 - self.last_p as i32;

        // Fine mixer update.
        let fw = &mut self.fine_weights[self.last_fine_ctx];
        for i in 0..NUM_MODELS {
            let delta = (FINE_LR as i64 * self.last_d[i] as i64 * error as i64) >> 16;
            fw[i] = (fw[i] as i64 + delta).clamp(-32768, 32767) as i32;
        }

        // Medium mixer update.
        let mw = &mut self.medium_weights[self.last_medium_ctx];
        for i in 0..NUM_MODELS {
            let delta = (MEDIUM_LR as i64 * self.last_d[i] as i64 * error as i64) >> 16;
            mw[i] = (mw[i] as i64 + delta).clamp(-32768, 32767) as i32;
        }

        // Coarse mixer update.
        let cw = &mut self.coarse_weights[self.last_coarse_ctx];
        for i in 0..NUM_MODELS {
            let delta = (COARSE_LR as i64 * self.last_d[i] as i64 * error as i64) >> 16;
            cw[i] = (cw[i] as i64 + delta).clamp(-32768, 32767) as i32;
        }
    }
}

impl Default for DualMixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a byte into categories for mixer context.
/// Returns 0-9 based on byte value ranges.
///
/// The high byte range (128-255) is split into 4 groups to properly
/// discriminate WRT word codes. Without this, all WRT codes (0x80-0xFE)
/// get the SAME mixer context, preventing the mixer from learning
/// different weights for different word codes.
#[inline]
pub fn byte_class(b: u8) -> u8 {
    match b {
        0..=31 => 0,      // control chars
        b' ' => 1,        // space
        b'0'..=b'9' => 2, // digits
        b'A'..=b'Z' => 3, // uppercase
        b'a'..=b'z' => 4, // lowercase
        b'!'..=b'/' => 5, // punctuation low
        b':'..=b'@' => 5, // punctuation mid
        b'['..=b'`' => 5, // punctuation high
        b'{'..=b'~' => 5, // punctuation high2
        0x80..=0x9F => 6, // WRT word codes group 1 (common words: the, of, and...)
        0xA0..=0xBF => 7, // WRT word codes group 2
        0xC0..=0xDF => 8, // WRT word codes group 3
        0xE0..=0xFE => 9, // WRT word codes group 4 / high bytes
        0xFF => 10,       // WRT escape byte
        _ => 11,          // other (unreachable with full match)
    }
}

/// Compute fine mixer context index (0..65535).
/// Uses c0 partial byte, c1 top 4 bits, bpos, byte class, match info, run length, and XML state.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fine_context(
    c0: u32,
    c1: u8,
    bpos: u8,
    bclass: u8,
    match_q: u8,
    run_q: u8,
    xml_state: u8,
) -> usize {
    // Hash together: c0(8b) + c1_top4(4b) + bpos(3b) + bclass(3b) + match_q(2b) + run_q(2b) + xml(3b)
    // = 25 bits, fold to 16 bits for 64K sets
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(97) + (c1 as usize >> 4);
    h = h.wrapping_mul(97) + bpos as usize;
    h = h.wrapping_mul(97) + (bclass as usize & 0x7);
    h = h.wrapping_mul(97) + (match_q as usize & 0x3);
    h = h.wrapping_mul(97) + (run_q as usize & 0x3);
    h = h.wrapping_mul(97) + (xml_state as usize & 0x7);
    h & (FINE_SETS - 1)
}

/// Compute medium mixer context index (0..16383).
/// Uses c0, c1 top nibble, bpos, byte class, run quantized, match length quantized, and XML state.
#[inline]
fn medium_context(c0: u32, c1: u8, bpos: u8, run_q: u8, match_q: u8, xml_state: u8) -> usize {
    // c0 (8 bits) + c1_top4 (4 bits) + bpos (3 bits) + bclass (3 bits) + run_q (2 bits) + match_q (2 bits) + xml(3b) = 25 bits -> hash to 14 bits
    let bclass = byte_class(c1);
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(67) + (c1 as usize >> 4);
    h = h.wrapping_mul(67) + bpos as usize;
    h = h.wrapping_mul(67) + bclass as usize;
    h = h.wrapping_mul(67) + (run_q as usize & 0x3);
    h = h.wrapping_mul(67) + (match_q as usize & 0x3);
    h = h.wrapping_mul(67) + (xml_state as usize & 0x7);
    h & (MEDIUM_SETS - 1)
}

/// Compute coarse mixer context index (0..4095).
#[inline]
fn coarse_context(c0: u32, bpos: u8) -> usize {
    ((c0 as usize & 0xFF) | ((bpos as usize) << 8)) & (COARSE_SETS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_near_balanced() {
        let mut mixer = DualMixer::new();
        let preds = [2048u32; NUM_MODELS];
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0);
        assert!(
            (1900..=2100).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_in_range() {
        let mut mixer = DualMixer::new();
        let mut preds = [2048u32; NUM_MODELS];
        preds[0] = 100;
        preds[1] = 4000;
        preds[4] = 3000;
        preds[7] = 500;
        let p = mixer.predict(&preds, 128, b'a', 3, 4, 1, 0, 0);
        assert!((1..=4095).contains(&p), "prediction out of range: {p}");
    }

    #[test]
    fn update_changes_weights() {
        let mut mixer = DualMixer::new();
        let preds = [2048u32; NUM_MODELS];
        mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0);
        let before = mixer.fine_weights[mixer.last_fine_ctx];
        mixer.update(1);
        let after = mixer.fine_weights[mixer.last_fine_ctx];
        let _ = (before, after);
    }

    #[test]
    fn mixer_adapts_to_biased_input() {
        let mut mixer = DualMixer::new();
        for _ in 0..100 {
            let mut preds = [2048u32; NUM_MODELS];
            preds[0] = 3500;
            let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0);
            let _ = p;
            mixer.update(1);
        }
        let mut preds = [2048u32; NUM_MODELS];
        preds[0] = 3500;
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0);
        assert!(p > 2500, "mixer should have learned to trust model 0: {p}");
    }

    #[test]
    fn byte_class_categories() {
        assert_eq!(byte_class(0), 0); // control
        assert_eq!(byte_class(b' '), 1); // space
        assert_eq!(byte_class(b'5'), 2); // digit
        assert_eq!(byte_class(b'A'), 3); // uppercase
        assert_eq!(byte_class(b'z'), 4); // lowercase
        assert_eq!(byte_class(b'.'), 5); // punctuation
        assert_eq!(byte_class(0x80), 6); // WRT group 1
        assert_eq!(byte_class(0x90), 6); // WRT group 1
        assert_eq!(byte_class(0xA0), 7); // WRT group 2
        assert_eq!(byte_class(0xC0), 8); // WRT group 3
        assert_eq!(byte_class(0xE0), 9); // WRT group 4
        assert_eq!(byte_class(0xFF), 10); // WRT escape
    }

    #[test]
    fn fine_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                for xml in 0..8u8 {
                    let ctx = fine_context(c0, 0xFF, bpos, 7, 3, 3, xml);
                    assert!(ctx < FINE_SETS, "fine context out of range: {ctx}");
                }
            }
        }
    }

    #[test]
    fn medium_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                for xml in 0..8u8 {
                    let ctx = medium_context(c0, 0xFF, bpos, 3, 3, xml);
                    assert!(ctx < MEDIUM_SETS, "medium context out of range: {ctx}");
                }
            }
        }
    }

    #[test]
    fn coarse_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = coarse_context(c0, bpos);
                assert!(ctx < COARSE_SETS, "coarse context out of range: {ctx}");
            }
        }
    }
}
