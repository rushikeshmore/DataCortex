//! SparseModel -- skip-context model for periodic patterns.
//!
//! Phase 4: Captures patterns like column-aligned data, repeating structures,
//! and periodic byte patterns by using contexts that skip bytes.
//!
//! Two contexts:
//! - Gap-2: hash(c2, c0_partial) -- skips c1, catches every-other-byte patterns
//! - Gap-3: hash(c3, c1, c0_partial) -- skips c2, catches 3-byte periodic patterns

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// FNV-1a prime for hashing.
const FNV_PRIME: u32 = 0x01000193;
const FNV_OFFSET: u32 = 0x811C9DC5;

/// Sparse context model using skip-byte contexts.
pub struct SparseModel {
    /// Gap-2 model: context = (c2, c0_partial), skips c1.
    cmap_gap2: ContextMap,
    smap_gap2: StateMap,
    last_state_gap2: u8,
    last_hash_gap2: u32,

    /// Gap-3 model: context = (c3, c1, c0_partial), skips c2.
    cmap_gap3: ContextMap,
    smap_gap3: StateMap,
    last_state_gap3: u8,
    last_hash_gap3: u32,
}

impl SparseModel {
    /// Create a sparse model with default 16MB total (8MB per gap context).
    pub fn new() -> Self {
        Self::with_size(1 << 23) // 8MB per gap context = 16MB total
    }

    /// Create a sparse model with a custom ContextMap size per gap context (in bytes).
    /// Total memory is 2x this value.
    pub fn with_size(cmap_size: usize) -> Self {
        SparseModel {
            cmap_gap2: ContextMap::new(cmap_size),
            smap_gap2: StateMap::new(),
            last_state_gap2: 0,
            last_hash_gap2: 0,

            cmap_gap3: ContextMap::new(cmap_size),
            smap_gap3: StateMap::new(),
            last_state_gap3: 0,
            last_hash_gap3: 0,
        }
    }

    /// Predict: returns average of gap-2 and gap-3 predictions.
    #[inline]
    pub fn predict(&mut self, c0: u32, c1: u8, c2: u8, c3: u8) -> u32 {
        // Gap-2 context: skip c1
        let h2 = gap2_hash(c2, c0);
        let state2 = self.cmap_gap2.get(h2);
        self.last_state_gap2 = state2;
        self.last_hash_gap2 = h2;
        let p2 = self.smap_gap2.predict(state2);

        // Gap-3 context: skip c2
        let h3 = gap3_hash(c3, c1, c0);
        let state3 = self.cmap_gap3.get(h3);
        self.last_state_gap3 = state3;
        self.last_hash_gap3 = h3;
        let p3 = self.smap_gap3.predict(state3);

        // Blend: average in probability space
        ((p2 + p3) / 2).clamp(1, 4095)
    }

    /// Update after observing bit.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        // Update gap-2
        self.smap_gap2.update(self.last_state_gap2, bit);
        let new2 = StateTable::next(self.last_state_gap2, bit);
        self.cmap_gap2.set(self.last_hash_gap2, new2);

        // Update gap-3
        self.smap_gap3.update(self.last_state_gap3, bit);
        let new3 = StateTable::next(self.last_state_gap3, bit);
        self.cmap_gap3.set(self.last_hash_gap3, new3);
    }
}

impl Default for SparseModel {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn gap2_hash(c2: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET ^ 0xDEAD; // different seed from order hashes
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

#[inline]
fn gap3_hash(c3: u8, c1: u8, c0: u32) -> u32 {
    let mut h = FNV_OFFSET ^ 0xBEEF; // different seed
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0xFF;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut sm = SparseModel::new();
        let p = sm.predict(1, 0, 0, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn predictions_in_range() {
        let mut sm = SparseModel::new();
        for i in 0..50u32 {
            let p = sm.predict(1, i as u8, (i + 1) as u8, (i + 2) as u8);
            assert!((1..=4095).contains(&p));
            sm.update((i & 1) as u8);
        }
    }
}
