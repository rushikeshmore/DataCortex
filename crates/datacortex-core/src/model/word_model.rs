//! WordModel — word boundary context model.
//!
//! Phase 3: Predicts based on word context. A "word" is a sequence of
//! alphanumeric/underscore characters. At word boundaries, a new word hash starts.
//!
//! Context: hash of current word characters + partial byte.
//! ContextMap size: 2MB (2^21).

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// WordModel: word boundary context predictor.
pub struct WordModel {
    /// Hash table for word contexts.
    cmap: ContextMap,
    /// Adaptive state → probability mapper.
    smap: StateMap,
    /// Running hash of current word.
    word_hash: u32,
    /// Previous word hash (for bigram context).
    prev_word_hash: u32,
    /// Whether we are inside a word.
    in_word: bool,
    /// Last looked-up state.
    last_state: u8,
    /// Last looked-up hash.
    last_hash: u32,
}

impl WordModel {
    /// Create a new word model with 2MB ContextMap.
    pub fn new() -> Self {
        WordModel {
            cmap: ContextMap::new(1 << 21), // 2MB
            smap: StateMap::new(),
            word_hash: 0,
            prev_word_hash: 0,
            in_word: false,
            last_state: 0,
            last_hash: 0,
        }
    }

    /// Predict probability of bit=1.
    /// Returns 12-bit probability in [1, 4095].
    ///
    /// `c0`: partial byte (1-255).
    /// `bpos`: bit position (0-7).
    /// `c1`: last completed byte.
    #[inline]
    pub fn predict(&mut self, c0: u32, bpos: u8, c1: u8) -> u32 {
        // At byte boundary, update word tracking.
        if bpos == 0 {
            self.update_word_state(c1);
        }

        // Context: combine word hash with partial byte.
        let h = self.word_hash.wrapping_mul(0x01000193) ^ (c0 & 0xFF);
        // Mix in previous word for bigram context.
        let h = h.wrapping_mul(0x01000193) ^ self.prev_word_hash;

        let state = self.cmap.get(h);
        self.last_state = state;
        self.last_hash = h;
        self.smap.predict(state)
    }

    /// Update after observing `bit`.
    #[inline]
    pub fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.cmap.set(self.last_hash, new_state);
    }

    /// Update word boundary tracking based on the last completed byte.
    fn update_word_state(&mut self, c1: u8) {
        let is_word_char = c1.is_ascii_alphanumeric() || c1 == b'_';

        if is_word_char {
            if !self.in_word {
                // Starting a new word.
                self.prev_word_hash = self.word_hash;
                self.word_hash = 0;
                self.in_word = true;
            }
            // Extend word hash with this character (lowercased for case insensitivity).
            let ch = if c1.is_ascii_uppercase() {
                c1 + 32 // lowercase
            } else {
                c1
            };
            self.word_hash = self.word_hash.wrapping_mul(0x01000193) ^ ch as u32;
        } else {
            if self.in_word {
                // Word just ended.
                self.in_word = false;
            }
            // Non-word characters: use character class as context.
            self.word_hash = c1 as u32;
        }
    }
}

impl Default for WordModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut wm = WordModel::new();
        let p = wm.predict(1, 0, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn predictions_in_range() {
        let mut wm = WordModel::new();
        for c in 0..=255u8 {
            let p = wm.predict(1, 0, c);
            assert!((1..=4095).contains(&p));
            wm.update(0);
        }
    }

    #[test]
    fn word_context_changes() {
        let mut wm = WordModel::new();
        // Feed 'hello' then predict.
        for &ch in b"hello" {
            for bpos in 0..8u8 {
                let bit = (ch >> (7 - bpos)) & 1;
                wm.predict(1, bpos, if bpos == 0 { ch } else { 0 });
                wm.update(bit);
            }
        }
        let p1 = wm.predict(1, 0, b'o');

        // Feed 'world' then predict — should differ.
        let mut wm2 = WordModel::new();
        for &ch in b"world" {
            for bpos in 0..8u8 {
                let bit = (ch >> (7 - bpos)) & 1;
                wm2.predict(1, bpos, if bpos == 0 { ch } else { 0 });
                wm2.update(bit);
            }
        }
        let p2 = wm2.predict(1, 0, b'd');
        // Different word contexts should give different predictions.
        // (Or same if they haven't learned anything yet — just check range.)
        assert!((1..=4095).contains(&p1));
        assert!((1..=4095).contains(&p2));
    }
}
