//! ContextMap — lossy hash table mapping context hashes to states.
//!
//! Phase 3: Hash table for higher-order context models.
//! Collisions replace existing entries (lossy).
//! Size must be a power of 2.

use super::state_table::StateTable;

/// Lossy hash table mapping context hashes to state bytes.
///
/// Each entry is a single byte (state from StateTable).
/// On collision, the existing entry is silently overwritten.
pub struct ContextMap {
    /// State entries indexed by (hash & mask).
    table: Vec<u8>,
    /// Bitmask for indexing: size - 1.
    mask: usize,
}

impl ContextMap {
    /// Create a new ContextMap with `size` entries.
    /// `size` must be a power of 2.
    pub fn new(size: usize) -> Self {
        debug_assert!(size.is_power_of_two(), "ContextMap size must be power of 2");
        ContextMap {
            table: vec![0u8; size],
            mask: size - 1,
        }
    }

    /// Get the current state for a context hash.
    #[inline(always)]
    pub fn get(&self, hash: u32) -> u8 {
        self.table[hash as usize & self.mask]
    }

    /// Set the state for a context hash.
    #[inline(always)]
    pub fn set(&mut self, hash: u32, state: u8) {
        self.table[hash as usize & self.mask] = state;
    }

    /// Get state, then update it after observing `bit`.
    /// Returns the probability from StateMap-like static lookup.
    /// Also transitions the state in-place.
    #[inline]
    pub fn predict_and_update(&mut self, hash: u32, bit: u8) -> u8 {
        let idx = hash as usize & self.mask;
        let state = self.table[idx];
        self.table[idx] = StateTable::next(state, bit);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_entries_are_zero() {
        let cm = ContextMap::new(1024);
        assert_eq!(cm.get(0), 0);
        assert_eq!(cm.get(999), 0);
    }

    #[test]
    fn set_and_get() {
        let mut cm = ContextMap::new(1024);
        cm.set(42, 128);
        assert_eq!(cm.get(42), 128);
    }

    #[test]
    fn hash_masking() {
        let mut cm = ContextMap::new(256);
        cm.set(0, 10);
        // 256 should map to same slot as 0
        assert_eq!(cm.get(256), 10);
    }

    #[test]
    fn predict_and_update_transitions() {
        let mut cm = ContextMap::new(1024);
        // State 0, bit 1 -> should transition
        let state = cm.predict_and_update(42, 1);
        assert_eq!(state, 0); // was state 0
        let new_state = cm.get(42);
        assert_ne!(new_state, 0); // transitioned
    }

    #[test]
    fn lossy_collision() {
        let mut cm = ContextMap::new(256);
        cm.set(5, 100);
        cm.set(5 + 256, 200); // collides with 5
        assert_eq!(cm.get(5), 200); // overwritten
    }
}
