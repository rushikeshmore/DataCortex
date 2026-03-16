//! ContextModel -- flexible context model using ContextMap + StateMap.
//!
//! Phase 4+: Three CM model variants, each now producing 3 predictions:
//! 1. **StateMap prediction** (p1): existing state->probability mapping
//! 2. **Run-count prediction** (p2): consecutive same-bit count in this context
//! 3. **Byte-history prediction** (p3): last byte seen in this context predicts current bit
//!
//! The run-count uses the same bit-level hash as the StateMap.
//! The byte-history uses a separate byte-level hash (independent of partial byte c0)
//! so that the stored byte is accessible at any bpos within the same byte context.

use crate::state::context_map::{AssociativeContextMap, ChecksumContextMap, ContextMap};
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

// --- Run Map: tracks consecutive same-bit runs per context ---

/// Packed run info: bits [7] = last bit, bits [6:0] = run count (0-127).
/// Stored in a lossy hash table like ContextMap.
struct RunMap {
    table: Vec<u8>,
    mask: usize,
}

impl RunMap {
    fn new(size: usize) -> Self {
        RunMap {
            table: vec![0u8; size],
            mask: size - 1,
        }
    }

    /// Get run info for context. Returns (run_count, run_bit).
    /// run_count=0 means no history.
    #[inline(always)]
    fn get(&self, hash: u32) -> (u8, u8) {
        let packed = self.table[hash as usize & self.mask];
        let run_bit = packed >> 7;
        let run_count = packed & 0x7F;
        (run_count, run_bit)
    }

    /// Update run info after observing `bit`.
    #[inline(always)]
    fn update(&mut self, hash: u32, bit: u8) {
        let idx = hash as usize & self.mask;
        let packed = self.table[idx];
        let run_bit = packed >> 7;
        let run_count = packed & 0x7F;

        let new_packed = if bit == run_bit && run_count > 0 {
            // Continue run
            let new_count = run_count.saturating_add(1).min(127);
            (bit << 7) | new_count
        } else {
            // New run starts
            (bit << 7) | 1
        };
        self.table[idx] = new_packed;
    }

    /// Convert run info to a 12-bit prediction.
    #[inline(always)]
    fn predict_p(&self, hash: u32) -> u32 {
        let (run_count, run_bit) = self.get(hash);
        if run_count == 0 {
            return 2048; // no history
        }
        // Strength ramps linearly with run length, capped.
        let strength = (run_count as u32 * 128).min(1800);
        if run_bit == 1 {
            (2048 + strength).min(4095)
        } else {
            2048u32.saturating_sub(strength).max(1)
        }
    }
}

/// Dual prediction from a context model.
/// (state_p, run_p) -- both 12-bit probabilities in [1, 4095].
/// - state_p: StateMap prediction (existing)
/// - run_p: run-count continuation prediction (new)
pub type DualPrediction = (u32, u32);

/// A context model backed by a ContextMap (hash->state) + StateMap (state->prob).
/// Now also produces run-count and byte-history predictions.
pub struct ContextModel {
    /// Hash table mapping context hashes to states.
    cmap: ContextMap,
    /// Adaptive state -> probability mapper.
    smap: StateMap,
    /// Run-count tracker per context.
    run_map: RunMap,
    /// Last looked-up state (for update after predict).
    last_state: u8,
    /// Last looked-up hash (for update after predict).
    last_hash: u32,
}

impl ContextModel {
    /// Create a new context model with the given ContextMap size.
    pub fn new(cmap_size: usize) -> Self {
        let aux_size = (cmap_size / 4).next_power_of_two().max(1024);
        ContextModel {
            cmap: ContextMap::new(cmap_size),
            smap: StateMap::new(),
            run_map: RunMap::new(aux_size),
            last_state: 0,
            last_hash: 0,
        }
    }

    /// Predict probability of bit=1 for the given context hash.
    /// Returns 12-bit probability in [1, 4095].
    #[inline(always)]
    pub fn predict(&mut self, hash: u32) -> u32 {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        self.smap.predict(state)
    }

    /// Predict dual: (state_p, run_p).
    #[inline(always)]
    pub fn predict_multi(&mut self, hash: u32) -> DualPrediction {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        let state_p = self.smap.predict(state);
        let run_p = self.run_map.predict_p(hash);
        (state_p, run_p)
    }

    /// Update the model after observing `bit`.
    /// Must be called after predict() with the same context.
    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
        self.run_map.update(self.last_hash, bit);
    }

    /// Notify model that a byte is complete (no-op).
    #[inline(always)]
    pub fn on_byte_complete(&mut self, _byte: u8) {}
}

/// A context model using ChecksumContextMap for reduced collision damage.
pub struct ChecksumContextModel {
    cmap: ChecksumContextMap,
    smap: StateMap,
    run_map: RunMap,
    last_state: u8,
    last_hash: u32,
}

impl ChecksumContextModel {
    pub fn new(byte_size: usize) -> Self {
        let aux_size = (byte_size / 4).next_power_of_two().max(1024);
        ChecksumContextModel {
            cmap: ChecksumContextMap::new(byte_size),
            smap: StateMap::new(),
            run_map: RunMap::new(aux_size),
            last_state: 0,
            last_hash: 0,
        }
    }

    #[inline(always)]
    pub fn predict(&mut self, hash: u32) -> u32 {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        self.smap.predict(state)
    }

    #[inline(always)]
    pub fn predict_multi(&mut self, hash: u32) -> DualPrediction {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        let state_p = self.smap.predict(state);
        let run_p = self.run_map.predict_p(hash);
        (state_p, run_p)
    }

    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
        self.run_map.update(self.last_hash, bit);
    }

    #[inline(always)]
    pub fn on_byte_complete(&mut self, _byte: u8) {}
}

/// A context model using 2-way set-associative ContextMap.
/// Best for order-5+ where collision rates are highest.
pub struct AssociativeContextModel {
    cmap: AssociativeContextMap,
    smap: StateMap,
    run_map: RunMap,
    last_state: u8,
    last_hash: u32,
}

impl AssociativeContextModel {
    pub fn new(byte_size: usize) -> Self {
        let aux_size = (byte_size / 4).next_power_of_two().max(1024);
        AssociativeContextModel {
            cmap: AssociativeContextMap::new(byte_size),
            smap: StateMap::new(),
            run_map: RunMap::new(aux_size),
            last_state: 0,
            last_hash: 0,
        }
    }

    #[inline(always)]
    pub fn predict(&mut self, hash: u32) -> u32 {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        self.smap.predict(state)
    }

    #[inline(always)]
    pub fn predict_multi(&mut self, hash: u32) -> DualPrediction {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        let state_p = self.smap.predict(state);
        let run_p = self.run_map.predict_p(hash);
        (state_p, run_p)
    }

    #[inline(always)]
    pub fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
        self.run_map.update(self.last_hash, bit);
    }

    #[inline(always)]
    pub fn on_byte_complete(&mut self, _byte: u8) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut cm = ContextModel::new(1024);
        let p = cm.predict(0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn predict_update_changes_probability() {
        let mut cm = ContextModel::new(1024);
        let p1 = cm.predict(42);
        cm.update(1);
        let p2 = cm.predict(42);
        assert_ne!(p1, p2, "update should change prediction");
    }

    #[test]
    fn different_contexts_diverge() {
        let mut cm = ContextModel::new(1024);
        for _ in 0..20 {
            cm.predict(10);
            cm.update(1);
        }
        for _ in 0..20 {
            cm.predict(20);
            cm.update(0);
        }
        let p10 = cm.predict(10);
        let p20 = cm.predict(20);
        assert!(
            p10 > p20,
            "ctx 10 (all 1s) should predict higher than ctx 20 (all 0s): p10={p10}, p20={p20}"
        );
    }

    #[test]
    fn predictions_in_range() {
        let mut cm = ContextModel::new(1024);
        for i in 0..100u32 {
            let p = cm.predict(i);
            assert!((1..=4095).contains(&p));
            cm.update((i & 1) as u8);
        }
    }

    #[test]
    fn multi_predict_returns_pair() {
        let mut cm = ContextModel::new(1024);
        let (sp, rp) = cm.predict_multi(42);
        assert_eq!(sp, 2048);
        assert_eq!(rp, 2048);
    }

    #[test]
    fn run_prediction_adapts() {
        let mut cm = ContextModel::new(1024);
        for _ in 0..10 {
            cm.predict_multi(42);
            cm.update(1);
        }
        let (_, rp) = cm.predict_multi(42);
        assert!(
            rp > 2048,
            "run prediction should favor 1 after many 1s: {rp}"
        );
    }

    #[test]
    fn dual_predictions_in_range() {
        let mut cm = ContextModel::new(1024);
        let (sp, rp) = cm.predict_multi(42);
        assert!((1..=4095).contains(&sp));
        assert!((1..=4095).contains(&rp));
    }

    // Checksummed variant tests

    #[test]
    fn checksum_initial_prediction_balanced() {
        let mut cm = ChecksumContextModel::new(2048);
        let p = cm.predict(0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn checksum_predict_update() {
        let mut cm = ChecksumContextModel::new(2048);
        let p1 = cm.predict(42);
        cm.update(1);
        let p2 = cm.predict(42);
        assert_ne!(p1, p2, "update should change prediction");
    }

    #[test]
    fn checksum_predictions_in_range() {
        let mut cm = ChecksumContextModel::new(2048);
        for i in 0..100u32 {
            let p = cm.predict(i);
            assert!((1..=4095).contains(&p));
            cm.update((i & 1) as u8);
        }
    }

    #[test]
    fn checksum_multi_predict() {
        let mut cm = ChecksumContextModel::new(2048);
        let (sp, rp) = cm.predict_multi(42);
        assert_eq!(sp, 2048);
        assert_eq!(rp, 2048);
    }

    #[test]
    fn assoc_multi_predict() {
        let mut cm = AssociativeContextModel::new(4096);
        let (sp, rp) = cm.predict_multi(42);
        assert_eq!(sp, 2048);
        assert_eq!(rp, 2048);
    }
}
