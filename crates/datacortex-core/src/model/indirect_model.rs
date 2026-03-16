//! IndirectModel — second-order context prediction.
//!
//! Instead of "what follows this context?", predicts "what follows the byte
//! that USUALLY follows this context?"
//!
//! Maintains two tables:
//! 1. prediction_table: context_hash -> most-likely-next-byte (updated byte-by-byte)
//! 2. ContextMap: uses (predicted_byte, c0_partial, bpos) as context for bit prediction
//!
//! This captures second-order sequential patterns:
//! "after 'th' comes 'e', and after 'e' in this position comes ' '"
//!
//! Proven effective in PAQ8PX's IndirectModel: typically -0.03 to -0.08 bpb.

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// Size of the prediction table (must be power of 2).
/// 1M entries = 2MB (1 byte prediction + 1 byte count per entry).
const PRED_TABLE_SIZE: usize = 1 << 20; // 1M entries
const PRED_TABLE_MASK: usize = PRED_TABLE_SIZE - 1;

/// Size of the indirect ContextMap (must be power of 2).
/// Larger = fewer collisions for the (predicted_byte, partial) contexts.
const INDIRECT_CM_SIZE: usize = 1 << 23; // 8MB

/// FNV offset basis for hashing.
const FNV_OFFSET: u32 = 0x811C9DC5;
/// FNV prime for hashing.
const FNV_PRIME: u32 = 0x01000193;

/// Indirect context model (model #16 in the mixer).
pub struct IndirectModel {
    /// Table 1: context_hash -> predicted next byte.
    /// Each entry stores the most commonly seen next byte for that context.
    prediction_table: Vec<u8>,
    /// Count table: tracks confidence in the prediction.
    count_table: Vec<u8>,
    /// Table 2: maps (predicted_byte, c0) context hash -> state byte.
    context_map: ContextMap,
    /// StateMap: converts state byte to 12-bit probability.
    state_map: StateMap,
    /// Context hash used for prediction table lookup.
    ctx_hash: u32,
    /// Last predicted byte from the table.
    predicted_byte: u8,
    /// Last ContextMap hash used (for update).
    last_cm_hash: u32,
    /// Current partial byte (1-255).
    c0: u32,
    /// Last full byte.
    c1: u8,
    /// Second-to-last byte.
    c2: u8,
    /// Third-to-last byte.
    c3: u8,
    /// Bit position (0-7).
    bpos: u8,
}

impl IndirectModel {
    /// Create a new indirect model.
    pub fn new() -> Self {
        IndirectModel {
            prediction_table: vec![0u8; PRED_TABLE_SIZE],
            count_table: vec![0u8; PRED_TABLE_SIZE],
            context_map: ContextMap::new(INDIRECT_CM_SIZE),
            state_map: StateMap::new(),
            ctx_hash: FNV_OFFSET,
            predicted_byte: 0,
            last_cm_hash: 0,
            c0: 1,
            c1: 0,
            c2: 0,
            c3: 0,
            bpos: 0,
        }
    }

    /// Predict the probability of the next bit being 1.
    /// Returns 12-bit probability in [1, 4095].
    #[inline]
    pub fn predict(&mut self, c0: u32, bpos: u8, c1: u8) -> u32 {
        if bpos == 0 {
            // At byte boundary: look up prediction for this context.
            self.ctx_hash = indirect_hash(c1, self.c2, self.c3);
            let idx = self.ctx_hash as usize & PRED_TABLE_MASK;
            self.predicted_byte = self.prediction_table[idx];
        }

        // Build context hash from predicted_byte + c0 partial byte.
        // This gives the ContextMap a unique slot per (predicted_byte, partial_bit_pattern).
        let cm_hash = predicted_context_hash(self.predicted_byte, c0);
        self.last_cm_hash = cm_hash;

        // Look up state in ContextMap, convert to probability via StateMap.
        let state = self.context_map.get(cm_hash);
        self.state_map.predict(state)
    }

    /// Update after observing `bit`.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        // Update ContextMap state for the context we just predicted with.
        let state = self.context_map.get(self.last_cm_hash);
        self.state_map.update(state, bit);
        let new_state = StateTable::next(state, bit);
        self.context_map.set(self.last_cm_hash, new_state);

        // Track partial byte.
        self.c0 = (self.c0 << 1) | bit as u32;
        self.bpos += 1;

        if self.bpos >= 8 {
            let byte = (self.c0 & 0xFF) as u8;

            // Update prediction table: for the PREVIOUS context, record what byte actually came.
            let idx = self.ctx_hash as usize & PRED_TABLE_MASK;
            let current_pred = self.prediction_table[idx];
            let current_count = self.count_table[idx];

            if byte == current_pred {
                self.count_table[idx] = current_count.saturating_add(1);
            } else if current_count < 2 {
                self.prediction_table[idx] = byte;
                self.count_table[idx] = 1;
            } else {
                self.count_table[idx] = current_count.saturating_sub(1);
            }

            self.c3 = self.c2;
            self.c2 = self.c1;
            self.c1 = byte;
            self.c0 = 1;
            self.bpos = 0;
        }
    }
}

impl Default for IndirectModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash function for indirect model context (3 bytes).
#[inline]
fn indirect_hash(c1: u8, c2: u8, c3: u8) -> u32 {
    let mut h = FNV_OFFSET;
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Hash combining predicted byte with partial byte context (c0).
/// This gives each (predicted_byte, bit_pattern) pair a distinct context slot.
#[inline]
fn predicted_context_hash(predicted: u8, c0: u32) -> u32 {
    let mut h = 0x9E3779B9u32; // golden ratio seed for different hash space
    h ^= predicted as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0x1FF; // c0 is 1-511 during a byte (9 bits)
    h = h.wrapping_mul(FNV_PRIME);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_in_range() {
        let mut model = IndirectModel::new();
        let p = model.predict(1, 0, 0);
        assert!(
            (1..=4095).contains(&p),
            "initial prediction should be in valid range, got {p}"
        );
    }

    #[test]
    fn predictions_in_range() {
        let mut model = IndirectModel::new();
        let data = b"Hello, World! The quick brown fox.";
        for &byte in data {
            for bpos in 0..8u8 {
                let bit = (byte >> (7 - bpos)) & 1;
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let p = model.predict(
                    c0,
                    bpos,
                    if bpos == 0 {
                        byte.wrapping_sub(1)
                    } else {
                        byte
                    },
                );
                assert!(
                    (1..=4095).contains(&p),
                    "prediction out of range at bpos {bpos}: {p}"
                );
                model.update(bit);
            }
        }
    }

    #[test]
    fn prediction_table_updates() {
        let mut model = IndirectModel::new();
        let pattern = b"abcdabcdabcd";
        for &byte in pattern {
            for bpos in 0..8u8 {
                let bit = (byte >> (7 - bpos)) & 1;
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let _ = model.predict(c0, bpos, model.c1);
                model.update(bit);
            }
        }
        let idx = indirect_hash(b'c', b'b', b'a') as usize & PRED_TABLE_MASK;
        assert_eq!(
            model.prediction_table[idx], b'd',
            "prediction table should predict 'd' after 'abc'"
        );
    }

    #[test]
    fn deterministic() {
        let data = b"test determinism of indirect model";
        let mut m1 = IndirectModel::new();
        let mut m2 = IndirectModel::new();

        for &byte in data {
            for bpos in 0..8u8 {
                let bit = (byte >> (7 - bpos)) & 1;
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let p1 = m1.predict(c0, bpos, m1.c1);
                let p2 = m2.predict(c0, bpos, m2.c1);
                assert_eq!(p1, p2, "models diverged at bpos {bpos}");
                m1.update(bit);
                m2.update(bit);
            }
        }
    }
}
