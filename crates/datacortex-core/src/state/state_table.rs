//! StateTable — 256-state bit history machine.
//!
//! Each state encodes a compact bit history. State 0 is the initial state
//! (no history observed). Transitions move toward the observed bit.
//!
//! The table maps (state, bit) → next_state and provides an approximate
//! probability of bit=1 for each state.

/// Number of states in the table.
pub const NUM_STATES: usize = 256;

/// Transition table: NEXT[state][bit] = next_state.
/// Built to track a compact bit history with asymmetric transitions.
///
/// Design: States 0-127 are "lean toward 0" (P(1) low),
/// states 128-255 are "lean toward 1" (P(1) high).
/// State 0 = initial (50/50). Transitions move toward the observed bit
/// with step sizes that decrease as we approach certainty (adaptive).
static NEXT: [[u8; 2]; NUM_STATES] = {
    let mut table = [[0u8; 2]; NUM_STATES];
    let mut s: usize = 0;
    while s < NUM_STATES {
        // On bit=0: move state toward 0 (lower values)
        // On bit=1: move state toward 255 (higher values)
        // Step size decreases near extremes (diminishing returns on certainty).

        // Distance from center (128)
        let center = 128i32;
        let pos = s as i32;

        // Step size: larger near center, smaller near extremes.
        let dist = if pos >= center {
            pos - center
        } else {
            center - pos
        };
        let raw = 256 - dist * 2;
        let step_base = if raw > 16 { raw } else { 16 } / 16;
        let step = if step_base > 1 { step_base } else { 1 };

        // bit=0: move toward 0
        let n0 = pos - step;
        let next0 = if n0 > 1 { n0 as u8 } else { 1u8 };
        // bit=1: move toward 255
        let n1 = pos + step;
        let next1 = if n1 < 255 { n1 as u8 } else { 255u8 };

        // Special case: state 0 (initial — no history)
        if s == 0 {
            // First bit: go to state skewed slightly toward that bit
            table[s][0] = 108; // lean toward 0
            table[s][1] = 148; // lean toward 1
        } else {
            table[s][0] = next0;
            table[s][1] = next1;
        }

        s += 1;
    }
    table
};

/// Approximate probability of bit=1 for each state, in 12-bit (1-4095).
/// State 0 = 2048 (50/50), state 1 = lowest, state 255 = highest.
static STATE_PROB: [u16; NUM_STATES] = {
    let mut probs = [0u16; NUM_STATES];
    let mut s: usize = 0;
    while s < NUM_STATES {
        if s == 0 {
            probs[s] = 2048; // 50/50
        } else {
            // Linear mapping: state 1 → ~32, state 255 → ~4063
            // Keeps values in [1, 4095]
            let p = 32 + ((s as u32 - 1) * (4063 - 32)) / 254;
            probs[s] = p as u16;
        }
        s += 1;
    }
    probs
};

/// The state table — provides transitions and initial probabilities.
pub struct StateTable;

impl StateTable {
    /// Get the next state after observing `bit` in state `s`.
    #[inline(always)]
    pub fn next(s: u8, bit: u8) -> u8 {
        NEXT[s as usize][bit as usize & 1]
    }

    /// Get the initial (static) probability of bit=1 for state `s`.
    /// Returns a 12-bit value in [1, 4095].
    #[inline(always)]
    pub fn prob(s: u8) -> u16 {
        STATE_PROB[s as usize]
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
        assert!(next0 < 128, "bit=0 should go to low state");
        assert!(next1 > 128, "bit=1 should go to high state");
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
    fn no_state_maps_to_zero() {
        // State 0 is the entry state; no transition should produce state 0
        // (except we allow it for state 0 itself as initial).
        for s in 1..NUM_STATES {
            for bit in 0..2u8 {
                let next = StateTable::next(s as u8, bit);
                assert!(next >= 1, "state {s}, bit {bit}: transitioned to state 0");
            }
        }
    }
}
