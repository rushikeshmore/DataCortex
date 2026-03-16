//! HierarchicalMixer — two-stage grouped logistic mixer.
//!
//! Replaces the flat 13-input triple mixer with a two-stage hierarchy:
//!
//! Stage 1 — Group Mixers (3 small mixers, each specialized):
//!   - Order mixer: O0-O7 predictions (8 inputs) → mixed_order
//!   - Match mixer: match model prediction (1 input) → passthrough with adaptation
//!   - Context mixer: word + sparse + run + JSON (4 inputs) → mixed_context
//!
//! Stage 2 — Meta Mixer (1 mixer):
//!   Takes (mixed_order, mixed_match, mixed_context) = 3 inputs → final prediction
//!
//! Each group mixer uses its own specialized context hash.
//! This prevents weight dilution: high-order models don't compete with
//! word/sparse models for the same weight slots.

use crate::mixer::logistic::{squash, stretch};

/// Number of order models (O0-O9).
const NUM_ORDER: usize = 10;
/// Number of context models (word, sparse, run, json).
const NUM_CONTEXT: usize = 4;
/// Total models feeding the hierarchical mixer.
/// NOTE: Not used by default engine -- engine uses dual_mixer::NUM_MODELS.
pub const NUM_MODELS: usize = 15; // 10 order + 1 match + 4 context

// --- Order Mixer: 8 inputs ---
const ORDER_SETS: usize = 32768; // 32K weight sets (8 models need rich context)
const ORDER_LR: i32 = 2;
const ORDER_W_SCALE: i32 = 4096;

// --- Match Mixer: 1 input (adaptive passthrough) ---
const MATCH_SETS: usize = 2048; // 2K weight sets
const MATCH_LR: i32 = 3;
const MATCH_W_SCALE: i32 = 4096;

// --- Context Mixer: 4 inputs ---
const CONTEXT_SETS: usize = 8192; // 8K weight sets
const CONTEXT_LR: i32 = 3;
const CONTEXT_W_SCALE: i32 = 4096;

// --- Meta Mixer: 3 inputs ---
const META_SETS: usize = 65536; // 64K weight sets (rich context for meta-level decisions)
const META_LR: i32 = 2;
const META_W_SCALE: i32 = 4096;

/// Initial weights for order mixer (favor higher-order models).
const ORDER_INIT: [i32; NUM_ORDER] = [200, 300, 400, 500, 550, 550, 450, 400, 350, 300];

/// Initial weight for match mixer (single model).
const MATCH_INIT: [i32; 1] = [4096];

/// Initial weights for context mixer.
const CONTEXT_INIT: [i32; NUM_CONTEXT] = [1200, 1000, 800, 1000]; // word, sparse, run, json

/// Initial weights for meta mixer (order, match, context).
const META_INIT: [i32; 3] = [1800, 1000, 1200];

/// Hierarchical two-stage mixer.
pub struct HierarchicalMixer {
    // Stage 1: Group mixers.
    order_weights: Vec<[i32; NUM_ORDER]>,
    match_weights: Vec<[i32; 1]>,
    context_weights: Vec<[i32; NUM_CONTEXT]>,

    // Stage 2: Meta mixer.
    meta_weights: Vec<[i32; 3]>,

    // Cached stretched inputs for each group (for update).
    last_order_d: [i32; NUM_ORDER],
    last_match_d: [i32; 1],
    last_context_d: [i32; NUM_CONTEXT],
    last_meta_d: [i32; 3],

    // Raw log-odds output from each group mixer (avoids squash/stretch at boundary).
    last_order_d_raw: i32,
    last_match_d_raw: i32,
    last_context_d_raw: i32,

    last_order_ctx: usize,
    last_match_ctx: usize,
    last_context_ctx: usize,
    last_meta_ctx: usize,

    // Squashed probabilities for each group (for group-level update).
    last_order_p: u32,
    last_match_p: u32,
    last_context_p: u32,
    last_p: u32,
}

impl HierarchicalMixer {
    pub fn new() -> Self {
        HierarchicalMixer {
            order_weights: vec![ORDER_INIT; ORDER_SETS],
            match_weights: vec![MATCH_INIT; MATCH_SETS],
            context_weights: vec![CONTEXT_INIT; CONTEXT_SETS],
            meta_weights: vec![META_INIT; META_SETS],

            last_order_d: [0; NUM_ORDER],
            last_match_d: [0; 1],
            last_context_d: [0; NUM_CONTEXT],
            last_meta_d: [0; 3],

            last_order_d_raw: 0,
            last_match_d_raw: 0,
            last_context_d_raw: 0,

            last_order_ctx: 0,
            last_match_ctx: 0,
            last_context_ctx: 0,
            last_meta_ctx: 0,

            last_order_p: 2048,
            last_match_p: 2048,
            last_context_p: 2048,
            last_p: 2048,
        }
    }

    /// Mix model predictions to produce a final 12-bit probability.
    ///
    /// `predictions` layout: [o0, o1, o2, o3, o4, o5, o6, o7, match, word, sparse, run, json]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    pub fn predict(
        &mut self,
        predictions: &[u32; NUM_MODELS],
        c0: u32,
        c1: u8,
        bpos: u8,
        byte_class: u8,
        match_len_q: u8,
        run_q: u8,
    ) -> u32 {
        // --- Stage 1: Order Mixer (8 order model predictions) ---
        for i in 0..NUM_ORDER {
            self.last_order_d[i] = stretch(predictions[i]);
        }
        self.last_order_ctx = order_context(c0, c1, bpos, byte_class);
        let (order_p, order_d) = mix_group_raw(
            &self.order_weights[self.last_order_ctx],
            &self.last_order_d,
            ORDER_W_SCALE,
        );
        self.last_order_p = order_p;
        self.last_order_d_raw = order_d;

        // --- Stage 1: Match Mixer (1 match prediction) ---
        self.last_match_d[0] = stretch(predictions[8]);
        self.last_match_ctx = match_context(c0, bpos, match_len_q);
        let (match_p, match_d) = mix_group_raw(
            &self.match_weights[self.last_match_ctx],
            &self.last_match_d,
            MATCH_W_SCALE,
        );
        self.last_match_p = match_p;
        self.last_match_d_raw = match_d;

        // --- Stage 1: Context Mixer (word, sparse, run, json) ---
        for i in 0..NUM_CONTEXT {
            self.last_context_d[i] = stretch(predictions[9 + i]);
        }
        self.last_context_ctx = context_mixer_context(c0, bpos, byte_class, run_q);
        let (ctx_p, ctx_d) = mix_group_raw(
            &self.context_weights[self.last_context_ctx],
            &self.last_context_d,
            CONTEXT_W_SCALE,
        );
        self.last_context_p = ctx_p;
        self.last_context_d_raw = ctx_d;

        // --- Stage 2: Meta Mixer ---
        // IMPORTANT: Use raw log-odds from group mixers to avoid squash/stretch
        // quantization error at the boundary. Group mixers output in log-odds space.
        self.last_meta_d[0] = self.last_order_d_raw;
        self.last_meta_d[1] = self.last_match_d_raw;
        self.last_meta_d[2] = self.last_context_d_raw;

        self.last_meta_ctx = meta_context(c0, c1, bpos, byte_class, match_len_q, run_q);

        let meta_w = &self.meta_weights[self.last_meta_ctx];
        let mut sum: i64 = 0;
        for i in 0..3 {
            sum += meta_w[i] as i64 * self.last_meta_d[i] as i64;
        }
        let d = (sum / META_W_SCALE as i64) as i32;
        let p = squash(d).clamp(1, 4095);
        self.last_p = p;
        p
    }

    /// Update all mixer weights after observing `bit`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, bit: u8) {
        // Update meta mixer (stage 2).
        let meta_error = (bit as i32) * 4096 - self.last_p as i32;
        {
            let w = &mut self.meta_weights[self.last_meta_ctx];
            for i in 0..3 {
                let delta = (META_LR as i64 * self.last_meta_d[i] as i64 * meta_error as i64) >> 16;
                w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
            }
        }

        // Update order mixer (stage 1).
        let order_error = (bit as i32) * 4096 - self.last_order_p as i32;
        {
            let w = &mut self.order_weights[self.last_order_ctx];
            for i in 0..NUM_ORDER {
                let delta =
                    (ORDER_LR as i64 * self.last_order_d[i] as i64 * order_error as i64) >> 16;
                w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
            }
        }

        // Update match mixer (stage 1).
        let match_error = (bit as i32) * 4096 - self.last_match_p as i32;
        {
            let w = &mut self.match_weights[self.last_match_ctx];
            for i in 0..1 {
                let delta =
                    (MATCH_LR as i64 * self.last_match_d[i] as i64 * match_error as i64) >> 16;
                w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
            }
        }

        // Update context mixer (stage 1).
        let ctx_error = (bit as i32) * 4096 - self.last_context_p as i32;
        {
            let w = &mut self.context_weights[self.last_context_ctx];
            for i in 0..NUM_CONTEXT {
                let delta =
                    (CONTEXT_LR as i64 * self.last_context_d[i] as i64 * ctx_error as i64) >> 16;
                w[i] = (w[i] as i64 + delta).clamp(-32768, 32767) as i32;
            }
        }
    }
}

impl Default for HierarchicalMixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic logistic mixer for N inputs.
/// Returns (squashed_probability, raw_log_odds).
/// The raw log-odds avoids squash/stretch quantization at stage boundaries.
#[inline(always)]
fn mix_group_raw<const N: usize>(weights: &[i32; N], d: &[i32; N], w_scale: i32) -> (u32, i32) {
    let mut sum: i64 = 0;
    for i in 0..N {
        sum += weights[i] as i64 * d[i] as i64;
    }
    let out_d = (sum / w_scale as i64) as i32;
    (squash(out_d).clamp(1, 4095), out_d)
}

/// Order mixer context: (c0, c1_top4, bpos, byte_class) — byte-level context for order selection.
/// Rich context lets the order mixer learn which order is best per character type.
#[inline]
fn order_context(c0: u32, c1: u8, bpos: u8, byte_class: u8) -> usize {
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(97) + (c1 as usize >> 4); // c1 top nibble
    h = h.wrapping_mul(97) + bpos as usize;
    h = h.wrapping_mul(97) + (byte_class as usize & 0x7);
    h & (ORDER_SETS - 1)
}

/// Match mixer context: (c0, bpos, match_len_q) — match-aware.
#[inline]
fn match_context(c0: u32, bpos: u8, match_len_q: u8) -> usize {
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(67) + bpos as usize;
    h = h.wrapping_mul(67) + (match_len_q as usize & 0x3);
    h & (MATCH_SETS - 1)
}

/// Context mixer context: (c0, bpos, byte_class, run_q) — character-class + run-aware.
#[inline]
fn context_mixer_context(c0: u32, bpos: u8, byte_class: u8, run_q: u8) -> usize {
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(67) + bpos as usize;
    h = h.wrapping_mul(67) + (byte_class as usize & 0x7);
    h = h.wrapping_mul(67) + (run_q as usize & 0x3);
    h & (CONTEXT_SETS - 1)
}

/// Meta mixer context: rich hash combining all key signals.
/// (c0, c1_top4, bpos, byte_class, match_len_q, run_q)
#[inline]
fn meta_context(c0: u32, c1: u8, bpos: u8, byte_class: u8, match_len_q: u8, run_q: u8) -> usize {
    let mut h: usize = c0 as usize & 0xFF;
    h = h.wrapping_mul(97) + (c1 as usize >> 4);
    h = h.wrapping_mul(97) + bpos as usize;
    h = h.wrapping_mul(97) + (byte_class as usize & 0x7);
    h = h.wrapping_mul(97) + (match_len_q as usize & 0x3);
    h = h.wrapping_mul(97) + (run_q as usize & 0x3);
    h & (META_SETS - 1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_near_balanced() {
        let mut mixer = HierarchicalMixer::new();
        let preds = [2048u32; NUM_MODELS];
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0);
        assert!(
            (1800..=2200).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_in_range() {
        let mut mixer = HierarchicalMixer::new();
        let preds = [
            100, 4000, 2048, 3000, 500, 2048, 1500, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
        ];
        let p = mixer.predict(&preds, 128, b'a', 3, 4, 1, 0);
        assert!((1..=4095).contains(&p), "prediction out of range: {p}");
    }

    #[test]
    fn mixer_adapts_to_biased_input() {
        let mut mixer = HierarchicalMixer::new();
        for _ in 0..100 {
            let preds = [
                3500, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
                2048,
            ];
            let _p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0);
            mixer.update(1);
        }
        let preds = [
            3500, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
            2048,
        ];
        let p = mixer.predict(&preds, 1, 0, 0, 0, 0, 0);
        assert!(p > 2500, "mixer should have learned to trust model 0: {p}");
    }

    #[test]
    fn order_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = order_context(c0, 0xFF, bpos, 7);
                assert!(ctx < ORDER_SETS, "order context out of range: {ctx}");
            }
        }
    }

    #[test]
    fn meta_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = meta_context(c0, 0xFF, bpos, 7, 3, 3);
                assert!(ctx < META_SETS, "meta context out of range: {ctx}");
            }
        }
    }

    #[test]
    fn match_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = match_context(c0, bpos, 3);
                assert!(ctx < MATCH_SETS, "match context out of range: {ctx}");
            }
        }
    }

    #[test]
    fn context_mixer_context_in_range() {
        for c0 in [1u32, 128, 255] {
            for bpos in 0..8u8 {
                let ctx = context_mixer_context(c0, bpos, 7, 3);
                assert!(
                    ctx < CONTEXT_SETS,
                    "context mixer context out of range: {ctx}"
                );
            }
        }
    }
}
