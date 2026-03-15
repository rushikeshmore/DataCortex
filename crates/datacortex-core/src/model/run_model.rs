//! RunModel -- run-length context model.
//!
//! Phase 4: Detects and exploits byte-level runs (sequences of identical bytes).
//! Also tracks bit-level run lengths for finer prediction.
//!
//! Context: (last byte, run length quantized, partial byte).
//! Very effective on repetitive data (logs, JSON values, etc.)

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// Run model: predicts based on run length of identical bytes.
pub struct RunModel {
    /// Context map for run context.
    cmap: ContextMap,
    /// State map.
    smap: StateMap,
    /// Current byte-level run length.
    run_len: u32,
    /// Last complete byte.
    last_byte: u8,
    /// Previous last byte (for detecting new runs).
    prev_byte: u8,
    /// Last state.
    last_state: u8,
    /// Last hash.
    last_hash: u32,
}

impl RunModel {
    /// Create a run model with default 4MB ContextMap.
    pub fn new() -> Self {
        Self::with_size(1 << 22) // 4MB
    }

    /// Create a run model with a custom ContextMap size (in bytes).
    pub fn with_size(cmap_size: usize) -> Self {
        RunModel {
            cmap: ContextMap::new(cmap_size),
            smap: StateMap::new(),
            run_len: 0,
            last_byte: 0,
            prev_byte: 0,
            last_state: 0,
            last_hash: 0,
        }
    }

    /// Predict based on run context.
    /// `c0`: partial byte (1-255).
    /// `bpos`: bit position (0-7).
    /// `c1`: last completed byte.
    #[inline]
    pub fn predict(&mut self, c0: u32, bpos: u8, c1: u8) -> u32 {
        if bpos == 0 {
            self.update_run_state(c1);
        }

        // Context: run_len_quantized(3b) + c1(8b) + c0_partial(8b)
        let run_q = quantize_run(self.run_len);
        let mut h: u32 = 0x12345678;
        h = h.wrapping_mul(0x01000193) ^ run_q as u32;
        h = h.wrapping_mul(0x01000193) ^ c1 as u32;
        h = h.wrapping_mul(0x01000193) ^ (c0 & 0xFF);

        let state = self.cmap.get(h);
        self.last_state = state;
        self.last_hash = h;
        self.smap.predict(state)
    }

    /// Update after observing bit.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
    }

    /// Update run tracking.
    fn update_run_state(&mut self, c1: u8) {
        if c1 == self.last_byte {
            self.run_len += 1;
        } else {
            self.run_len = 1;
        }
        self.prev_byte = self.last_byte;
        self.last_byte = c1;
    }
}

impl Default for RunModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantize run length to 0-7 range.
#[inline]
fn quantize_run(len: u32) -> u8 {
    match len {
        0..=1 => 0,
        2 => 1,
        3 => 2,
        4..=5 => 3,
        6..=8 => 4,
        9..=16 => 5,
        17..=32 => 6,
        _ => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut rm = RunModel::new();
        let p = rm.predict(1, 0, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn predictions_in_range() {
        let mut rm = RunModel::new();
        for i in 0..50u32 {
            let p = rm.predict(1, 0, i as u8);
            assert!((1..=4095).contains(&p));
            rm.update((i & 1) as u8);
        }
    }
}
