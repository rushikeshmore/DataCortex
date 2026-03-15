//! StateTable — PAQ8-style 256-state bit history machine.
//!
//! Each state encodes a compact (n0, n1) bit count pair with recency bias.
//! State 0 is the initial state (no history observed).
//!
//! The table maps (state, bit) -> next_state and provides an approximate
//! probability of bit=1 for each state based on the encoded counts.
//!
//! Design based on PAQ8 state table:
//! - States encode (n0, n1) count pairs compactly
//! - Low counts have fine granularity, high counts are coarsened
//! - Transitions favor recent observations (recency bias)
//! - State 0 = initial (equal probability)

/// Number of states in the table.
pub const NUM_STATES: usize = 256;

// State-to-(n0,n1) mapping. Generated to cover the useful range of
// (n0, n1) combinations with finer granularity near (0,0) and coarser
// at higher counts.

// The core state table: 256 entries with (n0, n1, next0, next1).
// Each state represents a compact (n0, n1) bit count history.
// Transitions:
//   On bit=0: move to state with (n0+1, n1) or scaled equivalent
//   On bit=1: move to state with (n0, n1+1) or scaled equivalent
// For high counts, we scale down the less-recent count to maintain
// recency bias (the table "forgets" old observations gradually).
//
// We generate the table programmatically for optimal coverage.
// Strategy: enumerate (n0, n1) pairs in a priority order that covers
// the most useful probability ranges, then create transition links.

// === COMPILE-TIME TABLE GENERATION ===

const fn build_table() -> ([u8; 512], [u16; 256]) {
    // NEXT[s*2 + bit] = next_state
    // PROB[s] = 12-bit probability of bit=1
    let mut next = [0u8; 512];
    let mut prob = [2048u16; 256];

    // (n0, n1) pairs for each state.
    // We assign states systematically.
    let mut n0 = [0u16; 256];
    let mut n1 = [0u16; 256];

    // State 0: initial (0, 0)
    n0[0] = 0;
    n1[0] = 0;

    // Fill states with (n0, n1) pairs in a spiral pattern.
    // Priority: small total counts first, balanced within each total.
    let mut state_idx: usize = 1;

    // Fine granularity: all (a, b) with a+b = 1..8 (excluding (0,0) already used)
    let mut total: u16 = 1;
    while total <= 8 {
        let mut b: u16 = 0;
        while b <= total {
            let a = total - b;
            if state_idx < 256 {
                n0[state_idx] = a;
                n1[state_idx] = b;
                state_idx += 1;
            }
            b += 1;
        }
        total += 1;
    }

    // Medium: selected pairs with total 9..20, stride 1
    total = 9;
    while total <= 20 {
        let mut b: u16 = 0;
        while b <= total {
            let a = total - b;
            // Skip some to save states - only include if balanced-ish or extreme
            let ratio_ok = {
                let min_val = if a < b { a } else { b };
                let max_val = if a > b { a } else { b };
                min_val == 0 || max_val <= min_val * 4
            };
            if ratio_ok && state_idx < 256 {
                n0[state_idx] = a;
                n1[state_idx] = b;
                state_idx += 1;
            }
            b += 1;
        }
        total += 1;
    }

    // Coarse: high-count runs and extreme ratios
    // Long run states for consecutive 0s
    let mut run: u16 = 21;
    while run <= 60 && state_idx < 256 {
        n0[state_idx] = run;
        n1[state_idx] = 0;
        state_idx += 1;
        n0[state_idx] = 0;
        n1[state_idx] = run;
        state_idx += 1;
        run += 3;
    }

    // Fill remaining with high-ratio states
    while state_idx < 256 {
        n0[state_idx] = 1;
        n1[state_idx] = 1;
        state_idx += 1;
    }

    // === Compute probabilities ===
    // P(1) = (n1 + 1) / (n0 + n1 + 2) mapped to [1, 4095]
    let mut s: usize = 0;
    while s < 256 {
        let t = n0[s] as u32 + n1[s] as u32 + 2;
        let p = ((n1[s] as u32 + 1) * 4095 + t / 2) / t;
        let p = if p < 1 {
            1
        } else if p > 4095 {
            4095
        } else {
            p
        };
        prob[s] = p as u16;
        s += 1;
    }

    // === Build lookup table for finding states by (n0, n1) ===
    // For transitions, we need to find the state with the closest (n0, n1).
    // Build transitions: for each state s, find best state for (n0+1, n1) and (n0, n1+1).

    s = 0;
    while s < 256 {
        let s_n0 = n0[s];
        let s_n1 = n1[s];

        // Transition on bit=0: want (s_n0 + 1, s_n1), possibly scaled
        let (target_n0_0, target_n1_0) = scale_counts(s_n0 + 1, s_n1);
        next[s * 2] = find_closest_state(&n0, &n1, target_n0_0, target_n1_0, s);

        // Transition on bit=1: want (s_n0, s_n1 + 1), possibly scaled
        let (target_n0_1, target_n1_1) = scale_counts(s_n0, s_n1 + 1);
        next[s * 2 + 1] = find_closest_state(&n0, &n1, target_n0_1, target_n1_1, s);

        s += 1;
    }

    (next, prob)
}

/// Scale down counts to maintain bounded state space with recency bias.
/// When total gets too large, scale the smaller count down.
const fn scale_counts(a: u16, b: u16) -> (u16, u16) {
    let total = a + b;
    if total <= 20 {
        (a, b)
    } else if total <= 40 {
        // Light scaling: reduce minority count by ~25%
        if a >= b {
            let new_b = b * 3 / 4;
            (
                a,
                if new_b > 0 {
                    new_b
                } else if b > 0 {
                    1
                } else {
                    0
                },
            )
        } else {
            let new_a = a * 3 / 4;
            (
                if new_a > 0 {
                    new_a
                } else if a > 0 {
                    1
                } else {
                    0
                },
                b,
            )
        }
    } else {
        // Heavy scaling: halve the minority count
        if a >= b {
            let new_b = b / 2;
            let new_a = if a > 60 { 60 } else { a };
            (new_a, new_b)
        } else {
            let new_a = a / 2;
            let new_b = if b > 60 { 60 } else { b };
            (new_a, new_b)
        }
    }
}

/// Find the state with (n0, n1) closest to (target_n0, target_n1).
/// Uses L1 distance weighted by total count difference.
const fn find_closest_state(
    n0: &[u16; 256],
    n1: &[u16; 256],
    target_n0: u16,
    target_n1: u16,
    current: usize,
) -> u8 {
    let mut best: usize = 0;
    let mut best_dist: u32 = u32::MAX;

    let mut i: usize = 0;
    while i < 256 {
        // Don't map back to the same state (except for state 0 which can self-loop)
        if i != current || current == 0 {
            let d0 = (n0[i] as u32).abs_diff(target_n0 as u32);
            let d1 = (n1[i] as u32).abs_diff(target_n1 as u32);

            // Weight ratio preservation: penalize flipping the majority
            let target_total = target_n0 as u32 + target_n1 as u32;
            let state_total = n0[i] as u32 + n1[i] as u32;
            let total_diff = state_total.abs_diff(target_total);

            let dist = d0 * 3 + d1 * 3 + total_diff;

            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
        i += 1;
    }

    best as u8
}

static TABLE: ([u8; 512], [u16; 256]) = build_table();

/// The state table -- provides transitions and initial probabilities.
pub struct StateTable;

impl StateTable {
    /// Get the next state after observing `bit` in state `s`.
    #[inline(always)]
    pub fn next(s: u8, bit: u8) -> u8 {
        TABLE.0[s as usize * 2 + (bit as usize & 1)]
    }

    /// Get the initial (static) probability of bit=1 for state `s`.
    /// Returns a 12-bit value in [1, 4095].
    #[inline(always)]
    pub fn prob(s: u8) -> u16 {
        TABLE.1[s as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_balanced() {
        assert_eq!(StateTable::prob(0), 2048);
    }

    #[test]
    fn state_0_transitions_are_distinct() {
        let next0 = StateTable::next(0, 0);
        let next1 = StateTable::next(0, 1);
        assert_ne!(next0, next1, "state 0 transitions should differ");
        // After seeing 0, probability should decrease
        assert!(
            StateTable::prob(next0) < 2048,
            "bit=0 should go to low-probability state, got {}",
            StateTable::prob(next0)
        );
        // After seeing 1, probability should increase
        assert!(
            StateTable::prob(next1) > 2048,
            "bit=1 should go to high-probability state, got {}",
            StateTable::prob(next1)
        );
    }

    #[test]
    fn probabilities_are_in_range() {
        for s in 0..NUM_STATES {
            let p = StateTable::prob(s as u8);
            assert!((1..=4095).contains(&p), "state {s}: prob {p} out of range");
        }
    }

    #[test]
    fn transitions_stay_in_range() {
        for s in 0..NUM_STATES {
            for bit in 0..2u8 {
                let next = StateTable::next(s as u8, bit);
                assert!(
                    (next as usize) < NUM_STATES,
                    "state {s}, bit {bit}: next={next} out of range"
                );
            }
        }
    }

    #[test]
    fn repeated_zeros_decrease_probability() {
        let mut s = 0u8;
        s = StateTable::next(s, 0); // first 0
        let p1 = StateTable::prob(s);
        s = StateTable::next(s, 0); // second 0
        let p2 = StateTable::prob(s);
        assert!(p2 <= p1, "more 0s should decrease P(1): p1={p1}, p2={p2}");
    }

    #[test]
    fn repeated_ones_increase_probability() {
        let mut s = 0u8;
        s = StateTable::next(s, 1); // first 1
        let p1 = StateTable::prob(s);
        s = StateTable::next(s, 1); // second 1
        let p2 = StateTable::prob(s);
        assert!(p2 >= p1, "more 1s should increase P(1): p1={p1}, p2={p2}");
    }

    #[test]
    fn no_state_maps_to_zero_from_nonzero() {
        // After first observation, state should not return to 0
        for s in 1..NUM_STATES {
            for bit in 0..2u8 {
                let next = StateTable::next(s as u8, bit);
                // State 0 is okay only if current state is also near-initial
                if s > 2 {
                    assert!(
                        next != 0,
                        "state {s}, bit {bit}: should not transition back to state 0"
                    );
                }
            }
        }
    }

    #[test]
    fn convergence_all_zeros() {
        let mut s = 0u8;
        for _ in 0..50 {
            s = StateTable::next(s, 0);
        }
        let p = StateTable::prob(s);
        assert!(p < 200, "50 zeros should give very low P(1): {p}");
    }

    #[test]
    fn convergence_all_ones() {
        let mut s = 0u8;
        for _ in 0..50 {
            s = StateTable::next(s, 1);
        }
        let p = StateTable::prob(s);
        assert!(p > 3900, "50 ones should give very high P(1): {p}");
    }

    #[test]
    fn mixed_sequence_stays_moderate() {
        let mut s = 0u8;
        // Alternating bits should keep probability moderate
        for i in 0..100 {
            s = StateTable::next(s, (i & 1) as u8);
        }
        let p = StateTable::prob(s);
        assert!(
            (500..=3500).contains(&p),
            "alternating bits should give moderate P(1): {p}"
        );
    }
}
