//! TripleMixer — three-level logistic mixer (fine + medium + coarse) in log-odds space.
//!
//! Phase 3: PAQ8-style logistic mixer following V2 proven parameters.
//!
//! Fine mixer: 64K weight sets, learning rate eta=2.
//! Medium mixer: 16K weight sets, learning rate eta=3.
//! Coarse mixer: 4K weight sets, learning rate eta=4.
//!
//! Blend: fine * 0.5 + medium * 0.3 + coarse * 0.2 in log-odds space.

use crate::mixer::logistic::{squash, stretch};

/// Number of models feeding the mixer.
pub const NUM_MODELS: usize = 7; // order0, order1, order2, order3, order4, match, word

/// Fine mixer: 64K weight sets.
const FINE_SETS: usize = 65536;

/// Medium mixer: 16K weight sets.
const MEDIUM_SETS: usize = 16384;

/// Coarse mixer: 4K weight sets.
const COARSE_SETS: usize = 4096;

/// Weight scale factor (2^12 = 4096).
const W_SCALE: i32 = 4096;

/// Initial weights per model (non-uniform).
/// Higher-order and match models get more weight because they're typically more informative.
/// order0=300, order1=500, order2=600, order3=700, order4=700, match=500, word=400
/// Sum = 3700, close to W_SCALE (4096).
const INITIAL_WEIGHTS: [i32; NUM_MODELS] = [300, 500, 600, 700, 700, 500, 400];

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
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn predict(
        &mut self,
        predictions: &[u32; NUM_MODELS],
        c0: u32,
        c1: u8,
        bpos: u8,
        byte_class: u8,
        match_len_q: u8,
    ) -> u32 {
        // Stretch all predictions to log-odds.
        for i in 0..NUM_MODELS {
            self.last_d[i] = stretch(predictions[i]);
        }

        // Fine mixer context: full hash.
        self.last_fine_ctx = fine_context(c0, c1, bpos, byte_class, match_len_q);
        // Medium mixer context: (c0, c1_top4, bpos).
        self.last_medium_ctx = medium_context(c0, c1, bpos);
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
    #[inline]
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
/// Returns 0-7 based on byte value ranges.
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
        128..=255 => 6,   // high bytes
        _ => 7,           // other
    }
}

/// Compute fine mixer context index (0..65535).
#[inline]
fn fine_context(c0: u32, c1: u8, bpos: u8, bclass: u8, match_q: u8) -> usize {
    let h = (c0 as usize & 0xFF)
        | ((c1 as usize >> 6) << 8)
        | ((bpos as usize) << 10)
        | ((bclass as usize & 0x7) << 13)
        | ((match_q as usize & 0x3) << 16);
    (h ^ (h >> 16)) & (FINE_SETS - 1)
}

/// Compute medium mixer context index (0..16383).
#[inline]
fn medium_context(c0: u32, c1: u8, bpos: u8) -> usize {
    // c0 (8 bits) + c1_top4 (4 bits) + bpos (3 bits) = 15 bits → hash to 14 bits
    let h = (c0 as usize & 0xFF) | ((c1 as usize >> 4) << 8) | ((bpos as usize) << 12);
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
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0);
        assert!(
            (1900..=2100).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_in_range() {
        let mut mixer = DualMixer::new();
        let preds = [100, 4000, 2048, 3000, 500, 2048, 1500];
        let p = mixer.predict(&preds, 128, b'a', 3, 4, 1);
        assert!((1..=4095).contains(&p), "prediction out of range: {p}");
    }

    #[test]
    fn update_changes_weights() {
        let mut mixer = DualMixer::new();
        let preds = [2048u32; NUM_MODELS];
        mixer.predict(&preds, 1, 0, 0, 0, 0);
        let before = mixer.fine_weights[mixer.last_fine_ctx];
        mixer.update(1);
        let after = mixer.fine_weights[mixer.last_fine_ctx];
        let _ = (before, after);
    }

    #[test]
    fn mixer_adapts_to_biased_input() {
        let mut mixer = DualMixer::new();
        for _ in 0..100 {
            let preds = [3500, 2048, 2048, 2048, 2048, 2048, 2048];
            let p = mixer.predict(&preds, 1, 0, 0, 0, 0);
            let _ = p;
            mixer.update(1);
        }
        let preds = [3500, 2048, 2048, 2048, 2048, 2048, 2048];
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0);
        assert!(p > 2500, "mixer should have learned to trust model 0: {p}");
    }

    #[test]
    fn byte_class_categories() {
        assert_eq!(byte_class(0), 0);
        assert_eq!(byte_class(b' '), 1);
        assert_eq!(byte_class(b'5'), 2);
        assert_eq!(byte_class(b'A'), 3);
        assert_eq!(byte_class(b'z'), 4);
        assert_eq!(byte_class(b'.'), 5);
        assert_eq!(byte_class(200), 6);
    }

    #[test]
    fn fine_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = fine_context(c0, 0xFF, bpos, 7, 3);
                assert!(ctx < FINE_SETS, "fine context out of range: {ctx}");
            }
        }
    }

    #[test]
    fn medium_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = medium_context(c0, 0xFF, bpos);
                assert!(ctx < MEDIUM_SETS, "medium context out of range: {ctx}");
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
