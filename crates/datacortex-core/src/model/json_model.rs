//! JsonModel -- JSON structure-aware context model.
//!
//! Phase 4: Tracks JSON parsing state and provides structure-aware predictions.
//! Gives the mixer specialized weight sets for different JSON contexts:
//! - Inside a key vs inside a value
//! - String vs number vs boolean vs null
//! - Array index position
//! - After colon vs after comma
//!
//! This model provides both a prediction and a JSON state byte that other
//! models can use as additional mixer context.

use crate::state::context_map::ContextMap;
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// JSON parser states (simplified for compression context).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum JsonState {
    /// Outside any JSON structure or at top level.
    TopLevel = 0,
    /// Inside an object, expecting key or closing brace.
    ObjectKey = 1,
    /// After colon, expecting value.
    ObjectValue = 2,
    /// Inside an array, expecting value or closing bracket.
    ArrayValue = 3,
    /// Inside a quoted string (key or value).
    String = 4,
    /// Inside a number literal.
    Number = 5,
    /// Inside a keyword (true, false, null).
    Keyword = 6,
}

/// JSON structure-aware context model.
pub struct JsonModel {
    /// Context map for JSON-state-aware prediction.
    cmap: ContextMap,
    /// State map.
    smap: StateMap,
    /// Current JSON parse state.
    state: JsonState,
    /// Whether we're in a key string (vs value string).
    in_key: bool,
    /// Nesting depth (quantized).
    depth: u8,
    /// Hash of the current key (for key->value correlation).
    key_hash: u32,
    /// Previous byte for state tracking.
    prev_byte: u8,
    /// Whether previous byte was backslash (for escape handling).
    escaped: bool,
    /// Last state for update.
    last_state: u8,
    /// Last hash for update.
    last_hash: u32,
}

impl JsonModel {
    pub fn new() -> Self {
        JsonModel {
            cmap: ContextMap::new(1 << 23), // 8MB
            smap: StateMap::new(),
            state: JsonState::TopLevel,
            in_key: false,
            depth: 0,
            key_hash: 0,
            prev_byte: 0,
            escaped: false,
            last_state: 0,
            last_hash: 0,
        }
    }

    /// Predict based on JSON structure context.
    /// `c0`: partial byte (1-255).
    /// `bpos`: bit position (0-7).
    /// `c1`: last completed byte.
    #[inline]
    pub fn predict(&mut self, c0: u32, bpos: u8, c1: u8) -> u32 {
        if bpos == 0 {
            self.update_json_state(c1);
        }

        // Context hash: json_state(3b) + in_key(1b) + depth_q(2b) + c0(8b)
        // For string contexts, also mix in key_hash
        let mut h: u32 = 0xCAFEBABE;
        h = h.wrapping_mul(0x01000193) ^ (self.state as u32);
        h = h.wrapping_mul(0x01000193) ^ (self.in_key as u32);
        h = h.wrapping_mul(0x01000193) ^ (self.depth.min(3) as u32);
        h = h.wrapping_mul(0x01000193) ^ (c0 & 0xFF);

        // For values, mix in key hash so values associated with the same key
        // share a context (e.g., all "name" values cluster together)
        if self.state == JsonState::ObjectValue || self.state == JsonState::String {
            h = h.wrapping_mul(0x01000193) ^ self.key_hash;
        }

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

    /// Return the current JSON state as a byte for mixer context.
    /// Returns 0-15 encoding the JSON parser state.
    #[inline]
    pub fn json_state_byte(&self) -> u8 {
        let state_bits = self.state as u8 & 0x7;
        let key_bit = if self.in_key { 8 } else { 0 };
        state_bits | key_bit
    }

    /// Update JSON parse state based on the last completed byte.
    fn update_json_state(&mut self, c1: u8) {
        // Handle string escaping
        if self.state == JsonState::String {
            if self.escaped {
                self.escaped = false;
                // Hash escaped char into key_hash if in key
                if self.in_key {
                    self.key_hash = self.key_hash.wrapping_mul(0x01000193) ^ c1 as u32;
                }
                self.prev_byte = c1;
                return;
            }
            if c1 == b'\\' {
                self.escaped = true;
                self.prev_byte = c1;
                return;
            }
            if c1 == b'"' {
                // End of string
                if self.in_key {
                    // Key finished — next should be colon then value
                    self.state = JsonState::ObjectKey; // waiting for colon
                } else {
                    // Value string finished
                    self.state = JsonState::ObjectValue; // will transition on comma/brace
                }
                self.prev_byte = c1;
                return;
            }
            // Regular string character — hash into key hash if in key
            if self.in_key {
                self.key_hash = self.key_hash.wrapping_mul(0x01000193) ^ c1 as u32;
            }
            self.prev_byte = c1;
            return;
        }

        // Not in string — track structural characters
        match c1 {
            b'{' => {
                self.state = JsonState::ObjectKey;
                self.depth = self.depth.saturating_add(1);
            }
            b'[' => {
                self.state = JsonState::ArrayValue;
                self.depth = self.depth.saturating_add(1);
            }
            b'}' | b']' => {
                self.depth = self.depth.saturating_sub(1);
                // Pop back to parent context
                self.state = if self.depth > 0 {
                    JsonState::ObjectValue // could be either, but ObjectValue is safe
                } else {
                    JsonState::TopLevel
                };
            }
            b'"' => {
                // Starting a string
                self.state = JsonState::String;
                // Determine if this is a key or value
                // Key if: after '{', after ',', or if prev non-ws was '{' or ','
                self.in_key = matches!(self.prev_significant_context(), b'{' | b',');
                if self.in_key {
                    self.key_hash = 0; // reset for new key
                }
            }
            b':' => {
                self.state = JsonState::ObjectValue;
            }
            b',' => {
                // After comma, context depends on container
                // Could be in object (next key) or array (next value)
                // We'll set ObjectKey and let the quote detection fix it
                self.state = JsonState::ObjectKey;
            }
            b'0'..=b'9' | b'-' => {
                if self.state != JsonState::Number {
                    self.state = JsonState::Number;
                }
            }
            b't' | b'f' | b'n' => {
                if self.state != JsonState::Keyword && self.state != JsonState::String {
                    self.state = JsonState::Keyword;
                }
            }
            _ => {
                // Whitespace or other — don't change state
            }
        }

        self.prev_byte = c1;
    }

    /// Get the previous significant (non-whitespace) byte context.
    /// Simplified: just return prev_byte since we don't store history.
    #[inline]
    fn prev_significant_context(&self) -> u8 {
        // Skip whitespace in prev_byte
        if self.prev_byte.is_ascii_whitespace() {
            // Can't look further back, assume comma context
            b','
        } else {
            self.prev_byte
        }
    }
}

impl Default for JsonModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut jm = JsonModel::new();
        let p = jm.predict(1, 0, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn predictions_in_range() {
        let mut jm = JsonModel::new();
        for c in b"{\"name\":\"Alice\",\"age\":30}" {
            for bpos in 0..8u8 {
                let bit = (c >> (7 - bpos)) & 1;
                let p = jm.predict(1, bpos, if bpos == 0 { *c } else { 0 });
                assert!((1..=4095).contains(&p));
                jm.update(bit);
            }
        }
    }

    #[test]
    fn json_state_changes() {
        let mut jm = JsonModel::new();
        // Feed opening brace
        jm.predict(1, 0, b'{');
        assert_ne!(jm.state, JsonState::TopLevel);
    }
}
