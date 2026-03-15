//! MetaMixer — adaptive blending of CM and LLM bit-level predictions.
//!
//! Blends P_cm and P_llm into P_final using an adaptive interpolation table.
//! The table is indexed by (quantized P_cm, quantized P_llm) and learns the
//! optimal blend from the data stream.
//!
//! This is essentially a 2D APM (Adaptive Probability Map).

/// Number of quantization bins for each input probability.
/// 64 bins gives reasonable resolution: 4096/64 = 64 values per bin.
const N_BINS: usize = 64;

/// Total table size: N_BINS * N_BINS = 4096 entries.
const TABLE_SIZE: usize = N_BINS * N_BINS;

/// MetaMixer blends two 12-bit probability sources.
pub struct MetaMixer {
    /// 2D interpolation table: [cm_bin][llm_bin] → blended 12-bit probability.
    /// Initialized to simple average, then adapts.
    table: Vec<u32>,
    /// Fixed blend weight for the LLM signal (0-256).
    /// 128 = 50/50, 64 = 25% LLM, 192 = 75% LLM.
    llm_weight: u32,
    /// Last table index used (for update).
    last_idx: usize,
    /// Last output probability (for update error calculation).
    last_p: u32,
    /// Learning rate shift (higher = slower learning). 4 = 1/16th toward target.
    lr_shift: u32,
}

impl MetaMixer {
    /// Create a new MetaMixer with a given LLM weight percentage (0-100).
    /// 50 = equal weight. Start conservative (25%) since LLM signal may be noisy.
    pub fn new(llm_weight_pct: u32) -> Self {
        // Initialize table: for each (cm_bin, llm_bin), store the weighted average
        // of the bin centers.
        let mut table = vec![0u32; TABLE_SIZE];
        for cm_bin in 0..N_BINS {
            for llm_bin in 0..N_BINS {
                let cm_center =
                    (cm_bin as u32 * 4095 + (N_BINS as u32 - 1) / 2) / (N_BINS as u32 - 1);
                let llm_center =
                    (llm_bin as u32 * 4095 + (N_BINS as u32 - 1) / 2) / (N_BINS as u32 - 1);
                // Simple average to start
                let avg = (cm_center + llm_center) / 2;
                table[cm_bin * N_BINS + llm_bin] = avg.clamp(1, 4095);
            }
        }

        MetaMixer {
            table,
            llm_weight: (llm_weight_pct * 256 / 100).min(256),
            last_idx: 0,
            last_p: 2048,
            lr_shift: 4,
        }
    }

    /// Blend CM and LLM predictions.
    ///
    /// `p_cm`: CM engine 12-bit probability [1, 4095].
    /// `p_llm`: LLM 12-bit probability [1, 4095].
    ///
    /// Returns: blended 12-bit probability [1, 4095].
    #[inline(always)]
    pub fn blend(&mut self, p_cm: u32, p_llm: u32) -> u32 {
        let cm_bin = ((p_cm.min(4095) as u64 * (N_BINS as u64 - 1)) / 4095) as usize;
        let llm_bin = ((p_llm.min(4095) as u64 * (N_BINS as u64 - 1)) / 4095) as usize;
        let idx = cm_bin.min(N_BINS - 1) * N_BINS + llm_bin.min(N_BINS - 1);
        self.last_idx = idx;

        let table_p = self.table[idx];

        // Also compute a direct weighted average as a fallback/blend.
        let direct = (p_cm as u64 * (256 - self.llm_weight) as u64
            + p_llm as u64 * self.llm_weight as u64)
            / 256;

        // Blend table output with direct average (50/50 to start, table earns trust).
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
        Self::new(25) // Conservative: 25% LLM weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_blend_near_average() {
        let mut mixer = MetaMixer::new(50);
        let p = mixer.blend(2048, 2048);
        assert!(
            (1900..=2200).contains(&p),
            "equal inputs should give ~2048, got {p}"
        );
    }

    #[test]
    fn blend_always_in_range() {
        let mut mixer = MetaMixer::new(50);
        for cm in [1u32, 100, 1000, 2048, 3000, 4000, 4095] {
            for llm in [1u32, 100, 1000, 2048, 3000, 4000, 4095] {
                let p = mixer.blend(cm, llm);
                assert!(
                    (1..=4095).contains(&p),
                    "out of range: cm={cm}, llm={llm}, got {p}"
                );
            }
        }
    }

    #[test]
    fn update_adapts() {
        let mut mixer = MetaMixer::new(50);
        // Repeatedly blend at (2048, 2048) and update with bit=1.
        for _ in 0..200 {
            mixer.blend(2048, 2048);
            mixer.update(1);
        }
        let p = mixer.blend(2048, 2048);
        assert!(p > 2048, "after many 1s, should predict higher: {p}");
    }

    #[test]
    fn asymmetric_weight() {
        let mut mixer_low = MetaMixer::new(10); // 10% LLM
        let mut mixer_high = MetaMixer::new(90); // 90% LLM
        // CM says high (3500), LLM says low (500).
        let p_low = mixer_low.blend(3500, 500);
        let p_high = mixer_high.blend(3500, 500);
        // Low LLM weight should give higher overall probability.
        assert!(
            p_low > p_high,
            "low llm weight should trust CM more: low={p_low}, high={p_high}"
        );
    }
}
