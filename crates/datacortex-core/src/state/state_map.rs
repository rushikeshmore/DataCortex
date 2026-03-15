//! StateMap -- adaptive state-to-probability mapper.
//!
//! Maps each of 256 states to a 12-bit probability (1-4095) that adapts
//! based on observed bits. Uses variable learning rate:
//! - Fast learning for low-count states (early observations matter more)
//! - Slow learning for high-count states (well-established statistics)
//!
//! Initialization uses the StateTable's static probability as the prior.

use super::state_table::StateTable;

/// Maximum learning count. Higher = better convergence on stable statistics.
/// 192 is a good balance: fast enough to adapt, slow enough to converge well.
const MAX_COUNT: u16 = 192;

/// Minimum learning denominator. Prevents divide-by-zero and initial overshoot.
const MIN_DENOM: i32 = 4;

/// A single entry in the state map.
#[derive(Clone, Copy)]
struct Entry {
    /// 12-bit probability of bit=1, range [1, 4095].
    prob: u16,
    /// Number of updates (capped at MAX_COUNT).
    count: u16,
}

/// Maps 256 states to adaptive 12-bit probabilities.
pub struct StateMap {
    entries: [Entry; 256],
}

impl StateMap {
    /// Create a new StateMap initialized from the StateTable's static probabilities.
    pub fn new() -> Self {
        let mut entries = [Entry {
            prob: 2048,
            count: 0,
        }; 256];

        for (i, entry) in entries.iter_mut().enumerate() {
            let base_prob = StateTable::prob(i as u8);
            entry.prob = base_prob;
            // Give initial confidence proportional to how extreme the state is.
            // States near 50/50 get low initial count (easy to update).
            // Extreme states get higher initial count (harder to shift).
            let dist_from_center = base_prob.abs_diff(2048);
            // Scale: 0 at center, up to ~4 at extremes
            entry.count = ((dist_from_center as u32) / 600).min(4) as u16;
        }

        StateMap { entries }
    }

    /// Get the predicted probability of bit=1 for the given state.
    /// Returns a 12-bit value in [1, 4095].
    #[inline(always)]
    pub fn predict(&self, state: u8) -> u32 {
        self.entries[state as usize].prob as u32
    }

    /// Update the probability for the given state after observing `bit`.
    /// Uses adaptive learning: p += (target - p) / (count + MIN_DENOM).
    /// Learning rate decreases as count increases (1/n convergence).
    #[inline]
    pub fn update(&mut self, state: u8, bit: u8) {
        let e = &mut self.entries[state as usize];
        let target = if bit != 0 { 4095i32 } else { 0i32 };
        let p = e.prob as i32;
        let count = e.count as i32 + MIN_DENOM;
        let delta = (target - p) / count;
        let new_p = (p + delta).clamp(1, 4095);
        e.prob = new_p as u16;
        if e.count < MAX_COUNT {
            e.count += 1;
        }
    }
}

impl Default for StateMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_0_is_balanced() {
        let sm = StateMap::new();
        assert_eq!(sm.predict(0), 2048);
    }

    #[test]
    fn predictions_in_range() {
        let sm = StateMap::new();
        for s in 0..=255u8 {
            let p = sm.predict(s);
            assert!((1..=4095).contains(&p), "state {s}: pred {p} out of range");
        }
    }

    #[test]
    fn update_toward_one() {
        let mut sm = StateMap::new();
        let before = sm.predict(0);
        sm.update(0, 1);
        let after = sm.predict(0);
        assert!(
            after >= before,
            "after seeing 1, prob should increase: {before} -> {after}"
        );
    }

    #[test]
    fn update_toward_zero() {
        let mut sm = StateMap::new();
        let before = sm.predict(0);
        sm.update(0, 0);
        let after = sm.predict(0);
        assert!(
            after <= before,
            "after seeing 0, prob should decrease: {before} -> {after}"
        );
    }

    #[test]
    fn many_ones_converge_high() {
        let mut sm = StateMap::new();
        for _ in 0..200 {
            sm.update(0, 1);
        }
        let p = sm.predict(0);
        assert!(p > 3500, "many 1s should push probability high: {p}");
    }

    #[test]
    fn many_zeros_converge_low() {
        let mut sm = StateMap::new();
        for _ in 0..200 {
            sm.update(0, 0);
        }
        let p = sm.predict(0);
        assert!(p < 500, "many 0s should push probability low: {p}");
    }

    #[test]
    fn probability_stays_in_bounds() {
        let mut sm = StateMap::new();
        for _ in 0..1000 {
            sm.update(128, 1);
        }
        assert!(sm.predict(128) >= 1 && sm.predict(128) <= 4095);

        for _ in 0..1000 {
            sm.update(128, 0);
        }
        assert!(sm.predict(128) >= 1 && sm.predict(128) <= 4095);
    }

    #[test]
    fn different_states_independent() {
        let mut sm = StateMap::new();
        let before_10 = sm.predict(10);
        sm.update(20, 1);
        sm.update(20, 1);
        sm.update(20, 1);
        let after_10 = sm.predict(10);
        assert_eq!(
            before_10, after_10,
            "updating state 20 should not affect state 10"
        );
    }
}
