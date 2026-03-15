// DataCortex — Context Mixing Engine
// Core bit-level prediction and mixing for lossless text compression.

use std::collections::HashMap;

/// 256-state bit history machine.
/// Each state encodes a compact representation of the bit sequence
/// observed in a given context, without storing the full history.
pub struct StateTable {
    /// State transition table: next_state[current_state][bit]
    next_state: [[u8; 2]; 256],
    /// Number of zeros seen in this state
    n0: [u16; 256],
    /// Number of ones seen in this state
    n1: [u16; 256],
}

impl StateTable {
    pub fn new() -> Self {
        let mut table = StateTable {
            next_state: [[0u8; 2]; 256],
            n0: [0u16; 256],
            n1: [0u16; 256],
        };
        table.build_transitions();
        table
    }

    /// Build the state transition table.
    /// States are organized so that states with similar bit histories
    /// are numerically close, improving cache locality.
    fn build_transitions(&mut self) {
        // State 0: initial state (no bits seen)
        self.n0[0] = 0;
        self.n1[0] = 0;

        let mut state_idx = 1u8;

        // Build states for all (n0, n1) pairs where n0 + n1 <= 15
        for total in 1..=15u16 {
            for n1 in 0..=total {
                let n0 = total - n1;
                if state_idx == 255 {
                    break;
                }

                self.n0[state_idx as usize] = n0;
                self.n1[state_idx as usize] = n1;

                // Transition on bit 0: increment n0 count
                let next_n0 = (n0 + 1).min(15);
                let next_n1_on_0 = n1;
                self.next_state[state_idx as usize][0] =
                    self.find_or_create_state(next_n0, next_n1_on_0, &mut state_idx);

                // Transition on bit 1: increment n1 count
                let next_n0_on_1 = n0;
                let next_n1 = (n1 + 1).min(15);
                self.next_state[state_idx as usize][1] =
                    self.find_or_create_state(next_n0_on_1, next_n1, &mut state_idx);

                state_idx = state_idx.saturating_add(1);
            }
        }
    }

    fn find_or_create_state(&self, n0: u16, n1: u16, _counter: &mut u8) -> u8 {
        // Find existing state with matching counts
        for i in 0..=255u8 {
            if self.n0[i as usize] == n0 && self.n1[i as usize] == n1 {
                return i;
            }
        }
        0 // fallback to initial state
    }

    /// Get the next state after observing a bit
    #[inline(always)]
    pub fn next(&self, state: u8, bit: u8) -> u8 {
        self.next_state[state as usize][bit as usize]
    }

    /// Get the (n0, n1) counts for a state
    #[inline(always)]
    pub fn counts(&self, state: u8) -> (u16, u16) {
        (self.n0[state as usize], self.n1[state as usize])
    }
}

/// Adaptive state-to-probability map.
/// Maps each state to a 12-bit probability [1, 4095] representing P(bit=1).
/// Uses 1/n adaptive learning rate for fast convergence.
pub struct StateMap {
    /// Probability table: prob[state] is 12-bit P(bit=1)
    prob: [u16; 256],
    /// Update count per state (for 1/n learning rate)
    count: [u16; 256],
}

impl StateMap {
    pub fn new() -> Self {
        StateMap {
            // Initialize all states to 50% probability (2048 out of 4096)
            prob: [2048u16; 256],
            // Start with count=2 to avoid division by zero and overshoot
            count: [2u16; 256],
        }
    }

    /// Get the predicted probability for a state (12-bit: [1, 4095])
    #[inline(always)]
    pub fn predict(&self, state: u8) -> u16 {
        self.prob[state as usize]
    }

    /// Update the probability after observing a bit.
    /// Uses adaptive 1/n learning rate: larger updates initially,
    /// converging as more data is observed.
    #[inline(always)]
    pub fn update(&mut self, state: u8, bit: u8) {
        let s = state as usize;
        let target = if bit == 1 { 4095u16 } else { 1u16 };
        let count = self.count[s];

        // prob += (target - prob) / count
        let delta = (target as i32 - self.prob[s] as i32) / count as i32;
        self.prob[s] = (self.prob[s] as i32 + delta).clamp(1, 4095) as u16;

        // Increment count, cap at 512 to maintain some adaptivity
        if self.count[s] < 512 {
            self.count[s] += 1;
        }
    }
}

/// Lossy hash table mapping context hashes to bit history states.
/// Uses multiplicative hashing for speed. Collisions replace existing entries.
pub struct ContextMap {
    /// Table of (checksum, state) pairs
    table: Vec<(u16, u8)>,
    /// Mask for indexing (table_size - 1, must be power of 2)
    mask: usize,
    /// Associated StateMap for probability lookup
    state_map: StateMap,
}

impl ContextMap {
    /// Create a new ContextMap with the given size (must be power of 2).
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "ContextMap size must be power of 2");
        ContextMap {
            table: vec![(0u16, 0u8); size],
            mask: size - 1,
            state_map: StateMap::new(),
        }
    }

    /// Look up the state for a given context hash.
    /// Returns the state (creating a new entry if needed).
    #[inline(always)]
    pub fn get_state(&self, hash: u64) -> u8 {
        let index = (hash as usize) & self.mask;
        let checksum = (hash >> 16) as u16;
        let entry = &self.table[index];

        if entry.0 == checksum {
            entry.1
        } else {
            0 // new context, return initial state
        }
    }

    /// Update the state for a context after observing a bit.
    #[inline(always)]
    pub fn update(&mut self, hash: u64, bit: u8, state_table: &StateTable) {
        let index = (hash as usize) & self.mask;
        let checksum = (hash >> 16) as u16;
        let entry = &mut self.table[index];

        if entry.0 == checksum {
            // Existing context: advance state
            let new_state = state_table.next(entry.1, bit);
            entry.1 = new_state;
            self.state_map.update(entry.1, bit);
        } else {
            // New context or collision: replace
            entry.0 = checksum;
            entry.1 = state_table.next(0, bit);
            self.state_map.update(entry.1, bit);
        }
    }

    /// Get the predicted probability for a context hash.
    #[inline(always)]
    pub fn predict(&self, hash: u64) -> u16 {
        let state = self.get_state(hash);
        self.state_map.predict(state)
    }

    /// Number of entries in the table.
    pub fn size(&self) -> usize {
        self.table.len()
    }
}

// --- Logistic mixing functions ---

/// Squash table: maps log-odds to probability.
/// squash(d) = 1 / (1 + exp(-d/256)) scaled to 12-bit [1, 4095]
static SQUASH_TABLE: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
    1546, 2048, 2550, 2995, 3349, 3608, 3786, 3902, 3976, 4023,
    4051, 4069, 4080, 4086, 4090, 4093, 4094, 4095,
];

/// Map log-odds to probability (12-bit).
/// Input: d in [-2048, 2047]
/// Output: p in [1, 4095]
#[inline(always)]
pub fn squash(d: i32) -> i32 {
    if d < -2047 {
        return 1;
    }
    if d > 2047 {
        return 4095;
    }
    let w = d & 127;
    let i = ((d + 2048) >> 7) as usize;
    let i = i.min(31);
    (SQUASH_TABLE[i] * (128 - w) + SQUASH_TABLE[i + 1] * w + 64) >> 7
}

/// Map probability to log-odds (inverse of squash).
/// Input: p in [1, 4095]
/// Output: d in [-2048, 2047]
#[inline(always)]
pub fn stretch(p: i32) -> i32 {
    assert!(p >= 1 && p <= 4095, "probability out of range: {}", p);
    // Lookup table for stretch would go here
    // For now, approximate: stretch(p) = ln(p / (4096 - p)) * 256
    let p_f = p as f64 / 4096.0;
    let d = (p_f / (1.0 - p_f)).ln() * 256.0;
    d.round() as i32
}

/// Logistic mixer: combines predictions from multiple models.
/// Uses gradient descent on log-odds with configurable learning rate.
pub struct LogisticMixer {
    /// Weight sets indexed by context
    weights: Vec<Vec<i32>>,
    /// Number of models (inputs)
    num_models: usize,
    /// Number of weight sets (contexts)
    num_contexts: usize,
    /// Learning rate (η)
    learning_rate: i32,
}

impl LogisticMixer {
    pub fn new(num_models: usize, num_contexts: usize, learning_rate: i32) -> Self {
        // Initialize all weights to 0 (equal mixing in log-odds space)
        let weights = vec![vec![0i32; num_models]; num_contexts];
        LogisticMixer {
            weights,
            num_models,
            num_contexts,
            learning_rate,
        }
    }

    /// Mix predictions from multiple models.
    /// Input: model predictions (12-bit probabilities), context index
    /// Output: mixed prediction (12-bit probability)
    pub fn mix(&self, predictions: &[u16], context: usize) -> u16 {
        assert_eq!(predictions.len(), self.num_models);
        let ctx = context % self.num_contexts;
        let w = &self.weights[ctx];

        // Weighted sum in log-odds space
        let mut sum: i64 = 0;
        for i in 0..self.num_models {
            let stretched = stretch(predictions[i] as i32) as i64;
            sum += stretched * w[i] as i64;
        }

        // Scale down and squash back to probability
        let mixed = squash((sum >> 8) as i32);
        mixed.clamp(1, 4095) as u16
    }

    /// Update weights after observing a bit.
    /// Gradient descent: w_i += η * (bit - p) * stretch(p_i)
    pub fn update(&mut self, predictions: &[u16], context: usize, bit: u8, mixed_p: u16) {
        let ctx = context % self.num_contexts;
        let error = (bit as i32) * 4096 - mixed_p as i32; // Error signal

        for i in 0..self.num_models {
            let stretched = stretch(predictions[i] as i32);
            let delta = (error as i64 * stretched as i64 * self.learning_rate as i64) >> 20;
            self.weights[ctx][i] += delta as i32;
        }
    }

    pub fn num_models(&self) -> usize {
        self.num_models
    }

    pub fn num_contexts(&self) -> usize {
        self.num_contexts
    }
}

/// Adaptive Probability Map (APM).
/// Refines a prediction based on a secondary context.
/// Maps (context, input_probability) to a refined probability.
pub struct AdaptiveProbabilityMap {
    /// Table: apm[context][probability_bucket] = refined_probability
    table: Vec<Vec<u16>>,
    /// Number of contexts
    num_contexts: usize,
    /// Number of probability buckets
    num_buckets: usize,
    /// Blend ratio with input (0-256, where 256 = 100% APM, 0 = 100% input)
    blend: u16,
}

impl AdaptiveProbabilityMap {
    pub fn new(num_contexts: usize, blend_percent: u16) -> Self {
        let num_buckets = 33; // 33 buckets covering [0, 4096] in steps of ~128
        let mut table = vec![vec![0u16; num_buckets]; num_contexts];

        // Initialize to identity mapping (APM output = input)
        for ctx in 0..num_contexts {
            for bucket in 0..num_buckets {
                table[ctx][bucket] = (bucket as u16 * 128).min(4095).max(1);
            }
        }

        let blend = (blend_percent * 256 / 100).min(256);

        AdaptiveProbabilityMap {
            table,
            num_contexts,
            num_buckets,
            blend,
        }
    }

    /// Refine a prediction using the APM context.
    /// Input: probability (12-bit), context index
    /// Output: refined probability (12-bit), blended with input
    #[inline(always)]
    pub fn refine(&self, probability: u16, context: usize) -> u16 {
        let ctx = context % self.num_contexts;
        let bucket = (probability as usize * (self.num_buckets - 1)) / 4096;
        let bucket = bucket.min(self.num_buckets - 1);

        let apm_prob = self.table[ctx][bucket];

        // Blend: output = (blend * apm + (256 - blend) * input) / 256
        let blended = (self.blend as u32 * apm_prob as u32
            + (256 - self.blend as u32) * probability as u32)
            / 256;

        blended.clamp(1, 4095) as u16
    }

    /// Update the APM table after observing a bit.
    pub fn update(&mut self, probability: u16, context: usize, bit: u8) {
        let ctx = context % self.num_contexts;
        let bucket = (probability as usize * (self.num_buckets - 1)) / 4096;
        let bucket = bucket.min(self.num_buckets - 1);

        let target = if bit == 1 { 4095u16 } else { 1u16 };
        let current = self.table[ctx][bucket];

        // Simple 1/16 learning rate update
        let delta = (target as i32 - current as i32) >> 4;
        self.table[ctx][bucket] = (current as i32 + delta).clamp(1, 4095) as u16;
    }
}

/// Format hint for format-aware context models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FormatHint {
    Generic,
    Json,
    Ndjson,
    Markdown,
    Csv,
    Code,
    Log,
}

/// Compression mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Max,
    Balanced,
    Fast,
}

impl Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Mode::Max),
            1 => Some(Mode::Balanced),
            2 => Some(Mode::Fast),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Mode::Max => 0,
            Mode::Balanced => 1,
            Mode::Fast => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_table_initial_state_is_zero() {
        let st = StateTable::new();
        assert_eq!(st.counts(0), (0, 0));
    }

    #[test]
    fn state_map_initial_probability_is_2048() {
        let sm = StateMap::new();
        assert_eq!(sm.predict(0), 2048);
    }

    #[test]
    fn state_map_converges_toward_one_on_repeated_ones() {
        let mut sm = StateMap::new();
        for _ in 0..100 {
            sm.update(0, 1);
        }
        assert!(sm.predict(0) > 3500, "Should converge toward 4095");
    }

    #[test]
    fn state_map_converges_toward_zero_on_repeated_zeros() {
        let mut sm = StateMap::new();
        for _ in 0..100 {
            sm.update(0, 0);
        }
        assert!(sm.predict(0) < 500, "Should converge toward 1");
    }

    #[test]
    fn context_map_returns_initial_for_unseen_context() {
        let cm = ContextMap::new(1024);
        let state = cm.get_state(0xDEADBEEF);
        assert_eq!(state, 0);
    }

    #[test]
    fn squash_is_monotonic() {
        let mut prev = squash(-2048);
        for d in -2047..=2047 {
            let p = squash(d);
            assert!(p >= prev, "squash must be monotonic: squash({}) = {} < squash({}) = {}", d, p, d - 1, prev);
            prev = p;
        }
    }

    #[test]
    fn squash_bounds() {
        assert_eq!(squash(-10000), 1);
        assert_eq!(squash(10000), 4095);
        let mid = squash(0);
        assert!(mid > 1900 && mid < 2200, "squash(0) should be near 2048, got {}", mid);
    }

    #[test]
    fn mixer_equal_weights_gives_average() {
        let mixer = LogisticMixer::new(2, 1, 2);
        let preds = [1000u16, 3000u16];
        let mixed = mixer.mix(&preds, 0);
        // With zero weights, should be near squash(0) ≈ 2048
        assert!(mixed > 1800 && mixed < 2200, "Equal zero weights should give ~2048, got {}", mixed);
    }

    #[test]
    fn apm_identity_when_blend_is_zero() {
        let apm = AdaptiveProbabilityMap::new(1, 0);
        let input = 2048u16;
        let output = apm.refine(input, 0);
        assert_eq!(output, input, "With 0% blend, APM should pass through input");
    }

    #[test]
    fn mode_roundtrip() {
        for mode in [Mode::Max, Mode::Balanced, Mode::Fast] {
            assert_eq!(Mode::from_u8(mode.to_u8()), Some(mode));
        }
    }
}
