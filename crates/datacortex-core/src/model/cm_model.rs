//! ContextModel — flexible context model using ContextMap + StateMap.
//!
//! Phase 3: Higher-order context models (order 1-3, word model).
//! Each model maps a context hash to a state, then uses StateMap for probability.

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// A context model backed by a ContextMap (hash→state) + StateMap (state→prob).
pub struct ContextModel {
    /// Hash table mapping context hashes to states.
    cmap: ContextMap,
    /// Adaptive state → probability mapper.
    smap: StateMap,
    /// Last looked-up state (for update after predict).
    last_state: u8,
    /// Last looked-up hash (for update after predict).
    last_hash: u32,
}

impl ContextModel {
    /// Create a new context model with the given ContextMap size.
    pub fn new(cmap_size: usize) -> Self {
        ContextModel {
            cmap: ContextMap::new(cmap_size),
            smap: StateMap::new(),
            last_state: 0,
            last_hash: 0,
        }
    }

    /// Predict probability of bit=1 for the given context hash.
    /// Returns 12-bit probability in [1, 4095].
    #[inline]
    pub fn predict(&mut self, hash: u32) -> u32 {
        let state = self.cmap.get(hash);
        self.last_state = state;
        self.last_hash = hash;
        self.smap.predict(state)
    }

    /// Update the model after observing `bit`.
    /// Must be called after predict() with the same context.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        // Update StateMap probability for the state we predicted from.
        self.smap.update(self.last_state, bit);
        // Transition the state in ContextMap.
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut cm = ContextModel::new(1024);
        let p = cm.predict(0);
        assert_eq!(p, 2048); // state 0 → 2048
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
        // Train context 10 with all 1s.
        for _ in 0..20 {
            cm.predict(10);
            cm.update(1);
        }
        // Train context 20 with all 0s.
        for _ in 0..20 {
            cm.predict(20);
            cm.update(0);
        }
        // Context 10 should predict higher than context 20.
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
}
