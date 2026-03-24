//! MultiSetMixer — PAQ8L-style multi-context-set logistic mixer.
//!
//! Instead of one mixer with one context hash, multiple independent logistic mixers
//! each use a different context hash to select weights. This allows the mixer to
//! learn different weight combinations for different situations:
//!
//! - Set 1: (c0, bpos) — partial byte context (what bits we've seen so far)
//! - Set 2: (c1, bpos) — last byte + bit position
//! - Set 3: (c1, c2_top4, bpos) — bigram history
//! - Set 4: (match_q, bpos, c1_class) — match-dependent weighting
//! - Set 5: (bpos, c1_class) — structure-dependent (xml_state removed)
//! - Set 6: (byte_class(c1), run_q, bpos) — character type + run length
//! - Set 7: (c1, c0, word_boundary_q) — word-level context
//!
//! Second layer: combines 7 set outputs with a simple logistic blend.
//! The second layer has its own small weight table (~512 entries).

use crate::mixer::dual_mixer::NUM_MODELS;
use crate::mixer::logistic::{squash, stretch};

/// Number of context sets (first layer).
const NUM_SETS: usize = 7;

/// Set sizes (number of weight tables per set).
/// Larger = more specialized weighting, but slower to learn.
const SET1_SIZE: usize = 2048; // c0(256) * bpos(8) = 2K
const SET2_SIZE: usize = 2048; // c1(256) * bpos(8) = 2K
const SET3_SIZE: usize = 8192; // c1(256) * c2_top4(16) -- folded to 8K via bpos
const SET4_SIZE: usize = 1024; // match_q(4) * bpos(8) * c1_class(12) -- folded to 1K
const SET5_SIZE: usize = 512; // bpos(8) * c1_class(8) -- folded to 512
const SET6_SIZE: usize = 1024; // bclass(12) * run_q(4) * bpos(8) -- folded to 1K
const SET7_SIZE: usize = 4096; // c1(256) * c0_top4(16) -- folded to 4K

/// Second layer mixer size.
const LAYER2_SIZE: usize = 512; // bpos(8) * c1_class(8) * run_q(4) -> 256, use 512

/// Learning rate for first-layer sets (shared).
const SET_LR: i32 = 3;

/// Learning rate for second-layer blend.
const LAYER2_LR: i32 = 4;

/// Weight scale factor.
const W_SCALE: i32 = 4096;

/// A single logistic mixer set with its own context hash and weight table.
struct MixerSet {
    /// Weight tables: [num_contexts][NUM_MODELS].
    weights: Vec<[i32; NUM_MODELS]>,
    /// Number of contexts (size of weight table).
    num_contexts: usize,
    /// Last context index used (for update).
    last_ctx: usize,
    /// Last output in log-odds (for layer 2 input).
    last_d: i32,
}

impl MixerSet {
    fn new(num_contexts: usize, initial_weights: &[i32; NUM_MODELS]) -> Self {
        MixerSet {
            weights: vec![*initial_weights; num_contexts],
            num_contexts,
            last_ctx: 0,
            last_d: 0,
        }
    }

    /// Compute weighted sum in log-odds space for a given context.
    /// Returns the log-odds output (not squashed).
    #[inline(always)]
    fn predict(&mut self, stretched: &[i32; NUM_MODELS], ctx: usize) -> i32 {
        self.last_ctx = ctx & (self.num_contexts - 1);
        let w = &self.weights[self.last_ctx];
        let mut sum: i64 = 0;
        for i in 0..NUM_MODELS {
            sum += w[i] as i64 * stretched[i] as i64;
        }
        let d = (sum / W_SCALE as i64) as i32;
        self.last_d = d;
        d
    }

    /// Update weights using the given error signal.
    #[inline(always)]
    fn update(&mut self, stretched: &[i32; NUM_MODELS], error: i32) {
        let w = &mut self.weights[self.last_ctx];
        for i in 0..NUM_MODELS {
            let delta = (SET_LR as i64 * stretched[i] as i64 * error as i64) >> 16;
            w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
        }
    }
}

/// Second-layer mixer: blends NUM_SETS outputs into one prediction.
#[allow(dead_code)]
struct Layer2Mixer {
    /// Weight tables: [LAYER2_SIZE][NUM_SETS].
    weights: Vec<[i32; NUM_SETS]>,
    /// Last context index.
    last_ctx: usize,
    /// Last stretched set outputs.
    last_d: [i32; NUM_SETS],
    /// Last output probability.
    last_p: u32,
}

#[allow(dead_code)]
impl Layer2Mixer {
    fn new() -> Self {
        // Initialize with equal weights for all sets.
        let initial = [300i32; NUM_SETS];
        Layer2Mixer {
            weights: vec![initial; LAYER2_SIZE],
            last_ctx: 0,
            last_d: [0; NUM_SETS],
            last_p: 2048,
        }
    }

    /// Blend set outputs into a final probability.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn predict(&mut self, set_outputs_d: &[i32; NUM_SETS], ctx: usize) -> u32 {
        self.last_ctx = ctx & (LAYER2_SIZE - 1);
        self.last_d.copy_from_slice(set_outputs_d);
        let w = &self.weights[self.last_ctx];
        let mut sum: i64 = 0;
        for i in 0..NUM_SETS {
            sum += w[i] as i64 * self.last_d[i] as i64;
        }
        let blended_d = (sum / W_SCALE as i64) as i32;
        let p = squash(blended_d).clamp(1, 4095);
        self.last_p = p;
        p
    }

    /// Update second-layer weights.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn update(&mut self, bit: u8) {
        let error = (bit as i32) * 4096 - self.last_p as i32;
        let w = &mut self.weights[self.last_ctx];
        for i in 0..NUM_SETS {
            let delta = (LAYER2_LR as i64 * self.last_d[i] as i64 * error as i64) >> 16;
            w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
        }
    }
}

/// Multi-set mixer: 7 first-layer sets + 1 second-layer blend.
pub struct MultiSetMixer {
    /// First-layer mixer sets, each with different context.
    sets: [MixerSet; NUM_SETS],
    /// Second-layer blender (reserved for future use).
    #[allow(dead_code)]
    layer2: Layer2Mixer,
    /// Cached stretched predictions (shared across all sets).
    last_stretched: [i32; NUM_MODELS],
    /// Cached output probability (for error computation).
    last_p: u32,
}

/// Initial weights for first-layer sets (28 inputs).
const INITIAL_SET_WEIGHTS: [i32; NUM_MODELS] = [
    200, // O0
    300, 60, // O1 (state, run)
    350, 60, // O2
    450, 60, // O3
    450, 60, // O4
    450, 60, // O5
    300, 60, // O6
    250, 60, // O7
    200, 60, // O8
    180, 60, // O9
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

impl MultiSetMixer {
    pub fn new() -> Self {
        MultiSetMixer {
            sets: [
                MixerSet::new(SET1_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET2_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET3_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET4_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET5_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET6_SIZE, &INITIAL_SET_WEIGHTS),
                MixerSet::new(SET7_SIZE, &INITIAL_SET_WEIGHTS),
            ],
            layer2: Layer2Mixer::new(),
            last_stretched: [0; NUM_MODELS],
            last_p: 2048,
        }
    }

    /// Mix model predictions using multi-set architecture.
    /// Returns 12-bit probability in [1, 4095].
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &mut self,
        predictions: &[u32; NUM_MODELS],
        c0: u32,
        c1: u8,
        c2: u8,
        bpos: u8,
        byte_class: u8,
        match_len_q: u8,
        run_q: u8,
        _xml_state: u8,
    ) -> u32 {
        // Stretch all predictions to log-odds (shared across all sets).
        for (i, &p) in predictions.iter().enumerate() {
            self.last_stretched[i] = stretch(p);
        }

        // Compute context for each set.
        // Set 1: (c0, bpos) — what partial byte have we seen
        let ctx1 = ((c0 as usize & 0xFF) << 3 | bpos as usize) & (SET1_SIZE - 1);

        // Set 2: (c1, bpos) — last full byte + position
        let ctx2 = ((c1 as usize) << 3 | bpos as usize) & (SET2_SIZE - 1);

        // Set 3: (c1, c2_top4, bpos) — bigram context
        let ctx3 = ((c1 as usize).wrapping_mul(67) + (c2 as usize >> 4))
            .wrapping_mul(67)
            .wrapping_add(bpos as usize)
            & (SET3_SIZE - 1);

        // Set 4: (match_q, bpos, byte_class) — match-dependent
        let ctx4 = (match_len_q as usize * 128 + byte_class as usize * 8 + bpos as usize)
            & (SET4_SIZE - 1);

        // Set 5: (bpos, byte_class) — structure-dependent (xml_state removed)
        let ctx5 =
            (byte_class as usize * 8 + bpos as usize) & (SET5_SIZE - 1);

        // Set 6: (byte_class, run_q, bpos) — character type + repetition
        let ctx6 =
            (byte_class as usize * 32 + run_q as usize * 8 + bpos as usize) & (SET6_SIZE - 1);

        // Set 7: (c1, c0_top4) — word/byte context
        let ctx7 = ((c1 as usize).wrapping_mul(67) + ((c0 as usize >> 4) & 0xF))
            .wrapping_mul(67)
            .wrapping_add(bpos as usize)
            & (SET7_SIZE - 1);

        // Run each set.
        let d1 = self.sets[0].predict(&self.last_stretched, ctx1);
        let d2 = self.sets[1].predict(&self.last_stretched, ctx2);
        let d3 = self.sets[2].predict(&self.last_stretched, ctx3);
        let d4 = self.sets[3].predict(&self.last_stretched, ctx4);
        let d5 = self.sets[4].predict(&self.last_stretched, ctx5);
        let d6 = self.sets[5].predict(&self.last_stretched, ctx6);
        let d7 = self.sets[6].predict(&self.last_stretched, ctx7);

        // Fixed-weight average in log-odds space (no learned second layer).
        // Sets 1-3 (byte context) get more weight than sets 4-7 (sparse context).
        // Weights: 4,4,3,2,1,1,1 (total=16)
        let blended_d = (d1 as i64 * 4 + d2 as i64 * 4 + d3 as i64 * 3
            + d4 as i64 * 2 + d5 as i64 + d6 as i64 + d7 as i64) / 16;
        let p = squash(blended_d as i32).clamp(1, 4095);
        self.last_p = p;
        p
    }

    /// Update all mixer sets after observing `bit`.
    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        // Update all first-layer sets with individual error signals.
        // Each set computes its own error based on its own output.
        for set in &mut self.sets {
            let set_p = squash(set.last_d);
            let error = (bit as i32) * 4096 - set_p as i32;
            set.update(&self.last_stretched, error);
        }
    }
}

impl Default for MultiSetMixer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_near_balanced() {
        let mut mixer = MultiSetMixer::new();
        let preds = [2048u32; NUM_MODELS];
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0, 0);
        assert!(
            (1800..=2200).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_in_range() {
        let mut mixer = MultiSetMixer::new();
        let mut preds = [2048u32; NUM_MODELS];
        preds[0] = 100;
        preds[1] = 4000;
        preds[4] = 3000;
        let p = mixer.predict(&preds, 128, b'a', b'b', 3, 4, 1, 0, 0);
        assert!((1..=4095).contains(&p), "prediction out of range: {p}");
    }

    #[test]
    fn mixer_adapts() {
        let mut mixer = MultiSetMixer::new();
        for _ in 0..100 {
            let mut preds = [2048u32; NUM_MODELS];
            preds[0] = 3500;
            mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0, 0);
            mixer.update(1);
        }
        let mut preds = [2048u32; NUM_MODELS];
        preds[0] = 3500;
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0, 0, 0);
        assert!(p > 2500, "mixer should adapt to biased model 0: {p}");
    }

    #[test]
    fn deterministic() {
        let mut m1 = MultiSetMixer::new();
        let mut m2 = MultiSetMixer::new();
        let preds = [2048u32; NUM_MODELS];
        for _ in 0..10 {
            let p1 = m1.predict(&preds, 1, 65, 66, 3, 4, 1, 0, 0);
            let p2 = m2.predict(&preds, 1, 65, 66, 3, 4, 1, 0, 0);
            assert_eq!(p1, p2, "mixers should be deterministic");
            m1.update(1);
            m2.update(1);
        }
    }
}
