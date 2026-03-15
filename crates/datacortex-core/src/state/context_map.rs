//! ContextMap -- hash table mapping context hashes to states.
//!
//! Phase 4: Three variants:
//! 1. Simple lossy ContextMap (1 byte per entry, for low-order models)
//! 2. ChecksumContextMap (2 bytes: checksum + state, reduces collisions)
//! 3. AssociativeContextMap (4 bytes: 2 slots with checksums, lowest collision)

use super::state_table::StateTable;

/// Simple lossy hash table mapping context hashes to state bytes.
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
    /// Returns the state before transition.
    /// Also transitions the state in-place.
    #[inline]
    pub fn predict_and_update(&mut self, hash: u32, bit: u8) -> u8 {
        let idx = hash as usize & self.mask;
        let state = self.table[idx];
        self.table[idx] = StateTable::next(state, bit);
        state
    }
}

/// Checksummed context map: 2 bytes per entry (checksum + state).
///
/// The checksum byte is derived from the upper bits of the hash.
/// On lookup, if the checksum doesn't match, the entry is treated as state 0.
pub struct ChecksumContextMap {
    /// Interleaved [checksum, state] pairs.
    table: Vec<u8>,
    /// Bitmask for indexing: (size/2) - 1.
    mask: usize,
}

impl ChecksumContextMap {
    /// Create a checksummed ContextMap.
    /// `byte_size` is the total memory in bytes (must be power of 2).
    pub fn new(byte_size: usize) -> Self {
        debug_assert!(
            byte_size.is_power_of_two(),
            "ChecksumContextMap size must be power of 2"
        );
        let entries = byte_size / 2;
        ChecksumContextMap {
            table: vec![0u8; byte_size],
            mask: entries - 1,
        }
    }

    /// Extract checksum byte from hash.
    #[inline(always)]
    fn checksum(hash: u32) -> u8 {
        ((hash >> 16) as u8) | 1 // ensure non-zero
    }

    /// Get state for hash, checking checksum.
    #[inline(always)]
    pub fn get(&self, hash: u32) -> u8 {
        let idx = (hash as usize & self.mask) * 2;
        let stored_cs = self.table[idx];
        let expected_cs = Self::checksum(hash);
        if stored_cs == expected_cs {
            self.table[idx + 1]
        } else {
            0
        }
    }

    /// Set state for hash with checksum.
    #[inline(always)]
    pub fn set(&mut self, hash: u32, state: u8) {
        let idx = (hash as usize & self.mask) * 2;
        self.table[idx] = Self::checksum(hash);
        self.table[idx + 1] = state;
    }
}

/// 2-way set-associative context map: 4 bytes per set (2 slots).
///
/// Each set has 2 slots, each with a checksum byte and state byte.
/// On lookup, both slots are checked. On write, the slot matching the
/// checksum is updated, or the least-used slot is replaced.
/// This dramatically reduces collision damage for high-order models.
pub struct AssociativeContextMap {
    /// [cs0, state0, cs1, state1] per set.
    table: Vec<u8>,
    /// Bitmask for set indexing: (size/4) - 1.
    mask: usize,
}

impl AssociativeContextMap {
    /// Create a 2-way associative ContextMap.
    /// `byte_size` is the total memory (must be power of 2, at least 8).
    pub fn new(byte_size: usize) -> Self {
        debug_assert!(
            byte_size.is_power_of_two() && byte_size >= 8,
            "AssociativeContextMap size must be power of 2 and >= 8"
        );
        let sets = byte_size / 4;
        AssociativeContextMap {
            table: vec![0u8; byte_size],
            mask: sets - 1,
        }
    }

    /// Extract checksum byte from hash.
    #[inline(always)]
    fn checksum(hash: u32) -> u8 {
        ((hash >> 16) as u8) | 1
    }

    /// Get state for hash. Checks both slots.
    #[inline(always)]
    pub fn get(&self, hash: u32) -> u8 {
        let base = (hash as usize & self.mask) * 4;
        let cs = Self::checksum(hash);

        // Check slot 0
        if self.table[base] == cs {
            return self.table[base + 1];
        }
        // Check slot 1
        if self.table[base + 2] == cs {
            return self.table[base + 3];
        }
        // Not found
        0
    }

    /// Set state for hash. Updates matching slot or replaces slot 1 (LRU-ish).
    #[inline(always)]
    pub fn set(&mut self, hash: u32, state: u8) {
        let base = (hash as usize & self.mask) * 4;
        let cs = Self::checksum(hash);

        // Check if slot 0 matches
        if self.table[base] == cs {
            self.table[base + 1] = state;
            return;
        }
        // Check if slot 1 matches
        if self.table[base + 2] == cs {
            self.table[base + 3] = state;
            return;
        }
        // Neither matches: evict slot 1 (move slot 0 to slot 1 first for LRU)
        self.table[base + 2] = self.table[base];
        self.table[base + 3] = self.table[base + 1];
        self.table[base] = cs;
        self.table[base + 1] = state;
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
        assert_eq!(cm.get(256), 10);
    }

    #[test]
    fn predict_and_update_transitions() {
        let mut cm = ContextMap::new(1024);
        let state = cm.predict_and_update(42, 1);
        assert_eq!(state, 0);
        let new_state = cm.get(42);
        assert_ne!(new_state, 0);
    }

    #[test]
    fn lossy_collision() {
        let mut cm = ContextMap::new(256);
        cm.set(5, 100);
        cm.set(5 + 256, 200);
        assert_eq!(cm.get(5), 200);
    }

    // Checksummed ContextMap tests

    #[test]
    fn checksum_new_entries_return_zero() {
        let cm = ChecksumContextMap::new(2048);
        assert_eq!(cm.get(0), 0);
        assert_eq!(cm.get(12345), 0);
    }

    #[test]
    fn checksum_set_and_get() {
        let mut cm = ChecksumContextMap::new(2048);
        cm.set(42, 128);
        assert_eq!(cm.get(42), 128);
    }

    #[test]
    fn checksum_overwrites_properly() {
        let mut cm = ChecksumContextMap::new(2048);
        cm.set(42, 100);
        cm.set(42, 200);
        assert_eq!(cm.get(42), 200);
    }

    // Associative ContextMap tests

    #[test]
    fn assoc_new_entries_return_zero() {
        let cm = AssociativeContextMap::new(4096);
        assert_eq!(cm.get(0), 0);
        assert_eq!(cm.get(12345), 0);
    }

    #[test]
    fn assoc_set_and_get() {
        let mut cm = AssociativeContextMap::new(4096);
        cm.set(42, 128);
        assert_eq!(cm.get(42), 128);
    }

    #[test]
    fn assoc_two_entries_same_set() {
        let mut cm = AssociativeContextMap::new(16); // 4 sets
        // Two different hashes that map to same set but different checksums
        cm.set(0x00010000, 100); // checksum derived from upper bits
        cm.set(0x00020000, 200); // different checksum, might be same set
        // Both should be retrievable (2-way associative)
        assert_eq!(cm.get(0x00010000), 100);
        assert_eq!(cm.get(0x00020000), 200);
    }

    #[test]
    fn assoc_overwrites_properly() {
        let mut cm = AssociativeContextMap::new(4096);
        cm.set(42, 100);
        cm.set(42, 200);
        assert_eq!(cm.get(42), 200);
    }
}
