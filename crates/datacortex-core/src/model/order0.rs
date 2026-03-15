//! Order-0 Context Model — predicts next bit from partial byte context.
//!
//! Context = the partial byte being decoded. `c` starts at 1, and after each
//! bit, c = (c << 1) | bit. After 8 bits, c holds the full byte value + 256
//! and gets reset to 1 for the next byte.
//!
//! Each context (c in range 1..=255) maps to a state in the StateTable,
//! which is then mapped to a probability via StateMap.

use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// Order-0 bit prediction model.
///
/// Uses 256 contexts (indexed by partial byte value c, range 1-255).
/// Index 0 is unused (c starts at 1).
pub struct Order0Model {
    /// State per context (256 entries, index 0 unused).
    states: [u8; 256],
    /// Adaptive state → probability map.
    state_map: StateMap,
}

impl Order0Model {
    /// Create a new Order-0 model with all states initialized to 0.
    pub fn new() -> Self {
        Order0Model {
            states: [0u8; 256],
            state_map: StateMap::new(),
        }
    }

    /// Predict the probability of bit=1 given the current partial byte context.
    ///
    /// `context`: partial byte value (c, range 1-255).
    /// Returns: 12-bit probability in [1, 4095].
    #[inline]
    pub fn predict(&self, context: usize) -> u32 {
        let state = self.states[context & 0xFF];
        self.state_map.predict(state)
    }

    /// Update the model after observing `bit` in the given context.
    ///
    /// `context`: partial byte value (c, range 1-255).
    /// `bit`: the observed bit (0 or 1).
    #[inline]
    pub fn update(&mut self, context: usize, bit: u8) {
        let ctx = context & 0xFF;
        let state = self.states[ctx];

        // Update the state map (adaptive probability).
        self.state_map.update(state, bit);

        // Transition to next state.
        self.states[ctx] = StateTable::next(state, bit);
    }
}

impl Default for Order0Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_balanced() {
        let model = Order0Model::new();
        // Context 1 (start of byte), state 0 → ~2048
        let p = model.predict(1);
        assert_eq!(p, 2048, "initial prediction should be 2048");
    }

    #[test]
    fn prediction_in_range() {
        let model = Order0Model::new();
        for c in 1..=255 {
            let p = model.predict(c);
            assert!(
                (1..=4095).contains(&p),
                "context {c}: pred {p} out of range"
            );
        }
    }

    #[test]
    fn update_adapts_prediction() {
        let mut model = Order0Model::new();
        let before = model.predict(1);
        model.update(1, 1);
        let after = model.predict(1);
        // After seeing bit=1, the state transitions, and the state map
        // probability for the OLD state gets updated. The prediction changes
        // because the state itself changes.
        assert_ne!(before, after, "prediction should change after update");
    }

    #[test]
    fn different_contexts_have_separate_states() {
        let mut model = Order0Model::new();
        // Update context 10 with many 1s.
        for _ in 0..50 {
            model.update(10, 1);
        }
        // Update context 5 with many 0s.
        for _ in 0..50 {
            model.update(5, 0);
        }
        let p5 = model.predict(5);
        let p10 = model.predict(10);
        // Context 5 (trained on 0s) should predict lower than context 10 (trained on 1s).
        assert!(
            p10 > p5,
            "context 10 (all 1s) should predict higher than context 5 (all 0s): p10={p10}, p5={p5}"
        );
    }

    #[test]
    fn simulate_byte_encoding() {
        let mut model = Order0Model::new();
        let byte: u8 = 0x42; // 01000010

        let mut c: usize = 1;
        for bpos in 0..8 {
            let bit = (byte >> (7 - bpos)) & 1;
            let _p = model.predict(c);
            model.update(c, bit);
            c = (c << 1) | bit as usize;
        }
        // After 8 bits, c should be byte + 256.
        assert_eq!(c, 0x42 + 256);
    }

    #[test]
    fn repeated_pattern_adapts() {
        let mut model = Order0Model::new();
        // Encode 'A' (0x41) many times — model should adapt.
        let byte: u8 = 0x41;
        let mut total_surprise: f64 = 0.0;
        let mut first_byte_surprise: f64 = 0.0;

        for iteration in 0..20 {
            let mut c: usize = 1;
            let mut byte_surprise: f64 = 0.0;
            for bpos in 0..8 {
                let bit = (byte >> (7 - bpos)) & 1;
                let p = model.predict(c);
                // Surprise = -log2(P(bit))
                let prob_of_bit = if bit == 1 {
                    p as f64 / 4096.0
                } else {
                    1.0 - p as f64 / 4096.0
                };
                byte_surprise += -prob_of_bit.log2();
                model.update(c, bit);
                c = (c << 1) | bit as usize;
            }
            if iteration == 0 {
                first_byte_surprise = byte_surprise;
            }
            total_surprise += byte_surprise;
        }

        let last_avg = total_surprise / 20.0;
        assert!(
            last_avg < first_byte_surprise,
            "model should improve: first byte = {first_byte_surprise:.2} bits, avg = {last_avg:.2} bits"
        );
    }
}
