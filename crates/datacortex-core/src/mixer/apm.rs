//! APM — Adaptive Probability Map for post-mixer refinement.
//!
//! Phase 3: Two-stage APM cascade following V2 proven parameters.
//!
//! APM maps (context, input_probability) → refined_probability using
//! an interpolation table. The table adapts over time.
//!
//! Stage 1: 512 contexts (bpos * byte_class), 50% blend with input.
//! Stage 2: 16K contexts (c1 * c0_top4), 25% blend with input.

/// Number of probability bins in the interpolation table.
/// 65 bins provides good resolution while keeping memory footprint small.
const NUM_BINS: usize = 65;

/// An adaptive probability map stage.
pub struct APMStage {
    /// Table: [num_contexts][NUM_BINS] of 12-bit probabilities.
    /// Each entry is the mapped output probability for (context, bin).
    table: Vec<[u32; NUM_BINS]>,
    /// Number of contexts.
    num_contexts: usize,
    /// Blend factor: how much of APM output vs input to use (0-256).
    /// blend=128 means 50% input + 50% APM output.
    blend: u32,
    /// Last context used (for update).
    last_ctx: usize,
    /// Last bin index (for update).
    last_bin: usize,
    /// Last weight for interpolation (for update).
    last_weight: u32,
}

impl APMStage {
    /// Create a new APM stage with the given number of contexts and blend factor.
    ///
    /// `num_contexts`: number of different contexts.
    /// `blend_pct`: blend percentage (0-100). 50 means 50% APM output + 50% input.
    pub fn new(num_contexts: usize, blend_pct: u32) -> Self {
        // Initialize table with linear mapping: bin i → i * 4096 / (NUM_BINS - 1).
        let mut table = vec![[0u32; NUM_BINS]; num_contexts];
        for ctx_row in table.iter_mut() {
            for (i, entry) in ctx_row.iter_mut().enumerate() {
                *entry = ((i as u64 * 4095 + (NUM_BINS as u64 - 1) / 2) / (NUM_BINS as u64 - 1))
                    .clamp(1, 4095) as u32;
            }
        }

        APMStage {
            table,
            num_contexts,
            blend: (blend_pct * 256 / 100).min(256),
            last_ctx: 0,
            last_bin: 0,
            last_weight: 0,
        }
    }

    /// Map an input probability through the APM.
    ///
    /// `prob`: input 12-bit probability [1, 4095].
    /// `context`: context index (0..num_contexts-1).
    ///
    /// Returns: refined 12-bit probability [1, 4095].
    #[inline]
    pub fn predict(&mut self, prob: u32, context: usize) -> u32 {
        let ctx = context % self.num_contexts;
        self.last_ctx = ctx;

        // Map probability to bin index with interpolation weight.
        // prob in [0, 4096] → bin in [0, NUM_BINS-1]
        let scaled = prob.min(4095) as u64 * (NUM_BINS as u64 - 1);
        let bin = (scaled / 4095) as usize;
        let bin = bin.min(NUM_BINS - 2); // clamp so bin+1 is valid
        let weight = (scaled % 4095) as u32; // interpolation weight (0-4094)

        self.last_bin = bin;
        self.last_weight = weight;

        // Linear interpolation between table[ctx][bin] and table[ctx][bin+1].
        let t = &self.table[ctx];
        let interp = t[bin] as i64 + (t[bin + 1] as i64 - t[bin] as i64) * weight as i64 / 4095;
        let apm_p = interp.clamp(1, 4095) as u32;

        // Blend APM output with input probability.
        let blended =
            (apm_p as u64 * self.blend as u64 + prob as u64 * (256 - self.blend) as u64) / 256;
        (blended as u32).clamp(1, 4095)
    }

    /// Update the APM after observing `bit`.
    /// Must be called after predict().
    #[inline]
    pub fn update(&mut self, bit: u8) {
        let target = if bit != 0 { 4095u32 } else { 1u32 };
        let t = &mut self.table[self.last_ctx];

        // Update both bins involved in interpolation.
        // Learning rate: move 1/16th toward target.
        let rate = 4; // shift amount: 1/16

        // Primary bin.
        let old = t[self.last_bin];
        let delta = (target as i32 - old as i32) >> rate;
        t[self.last_bin] = (old as i32 + delta).clamp(1, 4095) as u32;

        // Adjacent bin (with reduced learning).
        if self.last_bin + 1 < NUM_BINS {
            let old2 = t[self.last_bin + 1];
            let delta2 = (target as i32 - old2 as i32) >> (rate + 1);
            t[self.last_bin + 1] = (old2 as i32 + delta2).clamp(1, 4095) as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_passthrough() {
        let mut apm = APMStage::new(1, 0); // 0% blend = pure input
        let p = apm.predict(2048, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn initial_50_blend_near_identity() {
        let mut apm = APMStage::new(1, 50);
        // With identity initialization, 50% blend should give ~same as input.
        let p = apm.predict(2048, 0);
        assert!(
            (2000..=2096).contains(&p),
            "50% blend of identity should be near input: {p}"
        );
    }

    #[test]
    fn prediction_in_range() {
        let mut apm = APMStage::new(512, 50);
        for prob in [1u32, 100, 1000, 2048, 3000, 4000, 4095] {
            for ctx in [0usize, 100, 511] {
                let p = apm.predict(prob, ctx);
                assert!(
                    (1..=4095).contains(&p),
                    "out of range: prob={prob}, ctx={ctx}, got {p}"
                );
            }
        }
    }

    #[test]
    fn update_adapts() {
        let mut apm = APMStage::new(1, 100); // 100% APM
        // Predict at 2048, then update with bit=1 many times.
        for _ in 0..100 {
            apm.predict(2048, 0);
            apm.update(1);
        }
        // After many 1s, prediction at 2048 should shift higher.
        let p = apm.predict(2048, 0);
        assert!(p > 2048, "after many 1s, APM should predict higher: {p}");
    }

    #[test]
    fn different_contexts_independent() {
        let mut apm = APMStage::new(2, 100);
        // Train context 0 with all 1s.
        for _ in 0..50 {
            apm.predict(2048, 0);
            apm.update(1);
        }
        // Context 1 should still be near 2048.
        let p = apm.predict(2048, 1);
        assert!(
            (2000..=2096).contains(&p),
            "untrained context should be near 2048: {p}"
        );
    }

    #[test]
    fn extreme_inputs() {
        let mut apm = APMStage::new(1, 50);
        let p_low = apm.predict(1, 0);
        assert!((1..=100).contains(&p_low), "low input: {p_low}");

        let p_high = apm.predict(4095, 0);
        assert!((3995..=4095).contains(&p_high), "high input: {p_high}");
    }
}
