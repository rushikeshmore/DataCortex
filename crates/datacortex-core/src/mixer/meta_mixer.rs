//! MetaMixer — adaptive blending of CM and GRU bit-level predictions.
//!
//! Blends P_cm and P_gru into P_final using an adaptive interpolation table.
//! The table is indexed by (quantized P_cm, quantized P_gru) and learns the
//! optimal blend from the data stream.
//!
//! This is a 2D APM (Adaptive Probability Map), same design as the neural
//! crate's MetaMixer but lives in core so it works without the neural feature.

/// Number of quantization bins for each input probability.
/// 64 bins gives reasonable resolution: 4096/64 = 64 values per bin.
const N_BINS: usize = 64;

/// Total table size: N_BINS * N_BINS = 4096 entries.
const TABLE_SIZE: usize = N_BINS * N_BINS;

/// MetaMixer blends two 12-bit probability sources.
pub struct MetaMixer {
    /// 2D interpolation table: [cm_bin][gru_bin] -> blended 12-bit probability.
    /// Initialized to weighted average, then adapts.
    table: Vec<u32>,
    /// Fixed blend weight for the GRU signal (0-256).
    /// 128 = 50/50, 13 = ~5% GRU (conservative start).
    gru_weight: u32,
    /// Last table index used (for update).
    last_idx: usize,
    /// Last output probability (for update error calculation).
    last_p: u32,
    /// Learning rate shift (higher = slower learning). 5 = 1/32nd toward target.
    lr_shift: u32,
}

impl MetaMixer {
    /// Create a new MetaMixer with a given GRU weight percentage (0-100).
    /// Start conservative at 5% since GRU needs warmup time.
    pub fn new(gru_weight_pct: u32) -> Self {
        let mut table = vec![0u32; TABLE_SIZE];
        for cm_bin in 0..N_BINS {
            for gru_bin in 0..N_BINS {
                let cm_center =
                    (cm_bin as u32 * 4095 + (N_BINS as u32 - 1) / 2) / (N_BINS as u32 - 1);
                let gru_center =
                    (gru_bin as u32 * 4095 + (N_BINS as u32 - 1) / 2) / (N_BINS as u32 - 1);
                // Weighted average to start, biased toward CM.
                let w = (gru_weight_pct * 256 / 100).min(256);
                let avg =
                    (cm_center as u64 * (256 - w) as u64 + gru_center as u64 * w as u64) / 256;
                table[cm_bin * N_BINS + gru_bin] = (avg as u32).clamp(1, 4095);
            }
        }

        MetaMixer {
            table,
            gru_weight: (gru_weight_pct * 256 / 100).min(256),
            last_idx: 0,
            last_p: 2048,
            lr_shift: 5,
        }
    }

    /// Blend CM and GRU predictions.
    ///
    /// `p_cm`: CM engine 12-bit probability [1, 4095].
    /// `p_gru`: GRU 12-bit probability [1, 4095].
    ///
    /// Returns: blended 12-bit probability [1, 4095].
    #[inline(always)]
    pub fn blend(&mut self, p_cm: u32, p_gru: u32) -> u32 {
        let cm_bin = ((p_cm.min(4095) as u64 * (N_BINS as u64 - 1)) / 4095) as usize;
        let gru_bin = ((p_gru.min(4095) as u64 * (N_BINS as u64 - 1)) / 4095) as usize;
        let idx = cm_bin.min(N_BINS - 1) * N_BINS + gru_bin.min(N_BINS - 1);
        self.last_idx = idx;

        let table_p = self.table[idx];

        // Direct weighted average as fallback/blend.
        let direct = (p_cm as u64 * (256 - self.gru_weight) as u64
            + p_gru as u64 * self.gru_weight as u64)
            / 256;

        // Blend table output with direct average (50/50).
        let blended = (table_p as u64 + direct) / 2;
        self.last_p = (blended as u32).clamp(1, 4095);
        self.last_p
    }

    /// Update after observing a bit. Must be called after blend().
    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        let target = if bit != 0 { 4095u32 } else { 1u32 };
        let old = self.table[self.last_idx];
        let delta = (target as i32 - old as i32) >> self.lr_shift;
        self.table[self.last_idx] = (old as i32 + delta).clamp(1, 4095) as u32;
    }

    /// Get the last output probability (for diagnostics).
    pub fn last_prediction(&self) -> u32 {
        self.last_p
    }
}

impl Default for MetaMixer {
    fn default() -> Self {
        Self::new(5) // Conservative: 5% GRU weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_blend_biased_to_cm() {
        let mut mixer = MetaMixer::new(5);
        // When both inputs are 2048, output should be near 2048.
        let p = mixer.blend(2048, 2048);
        assert!(
            (1900..=2200).contains(&p),
            "equal inputs should give ~2048, got {p}"
        );
    }

    #[test]
    fn blend_always_in_range() {
        let mut mixer = MetaMixer::new(5);
        for cm in [1u32, 100, 1000, 2048, 3000, 4000, 4095] {
            for gru in [1u32, 100, 1000, 2048, 3000, 4000, 4095] {
                let p = mixer.blend(cm, gru);
                assert!(
                    (1..=4095).contains(&p),
                    "out of range: cm={cm}, gru={gru}, got {p}"
                );
            }
        }
    }

    #[test]
    fn cm_dominates_at_low_weight() {
        let mut mixer = MetaMixer::new(5); // 5% GRU
        // CM says high, GRU says low.
        let p = mixer.blend(3500, 500);
        // Should be much closer to CM.
        assert!(p > 2500, "5% GRU should let CM dominate: got {p}");
    }

    #[test]
    fn update_adapts() {
        let mut mixer = MetaMixer::new(5);
        for _ in 0..200 {
            mixer.blend(2048, 2048);
            mixer.update(1);
        }
        let p = mixer.blend(2048, 2048);
        assert!(p > 2048, "after many 1s, should predict higher: {p}");
    }
}
