//! MatchModel — ring buffer + hash table for longest match prediction.
//!
//! Phase 3: Finds the longest match in history and predicts from match continuation.
//! Uses linear confidence ramp based on match length.
//!
//! CRITICAL V2 LESSONS:
//! - Rolling hash must NOT be cumulative
//! - Confidence ramp must be linear
//! - Length tracking must reset on mismatch

/// Size of the ring buffer (8MB).
const BUF_SIZE: usize = 8 * 1024 * 1024;

/// Size of the hash table for match finding (2M entries).
const HASH_SIZE: usize = 2 * 1024 * 1024;

/// Minimum match length before we start predicting.
const MIN_MATCH: usize = 2;

/// Maximum match length for confidence calculation.
const MAX_MATCH_FOR_CONF: usize = 64;

/// Match model: finds longest match in history, predicts from continuation.
pub struct MatchModel {
    /// Ring buffer of past bytes.
    buf: Vec<u8>,
    /// Current write position in ring buffer.
    buf_pos: usize,
    /// Total bytes written (for position validity).
    total_written: usize,
    /// Hash table: hash → position in ring buffer.
    hash_table: Vec<u32>,
    /// Current match position in ring buffer (-1 = no match).
    match_pos: i64,
    /// Current match length.
    match_len: usize,
    /// Bit position within the matched byte (0-7).
    match_bpos: u8,
    /// Rolling hash of recent bytes (non-cumulative).
    hash: u32,
    /// Last predicted probability.
    last_p: u32,
}

impl MatchModel {
    pub fn new() -> Self {
        MatchModel {
            buf: vec![0u8; BUF_SIZE],
            buf_pos: 0,
            total_written: 0,
            hash_table: vec![0u32; HASH_SIZE],
            match_pos: -1,
            match_len: 0,
            match_bpos: 0,
            hash: 0,
            last_p: 2048,
        }
    }

    /// Predict probability of bit=1 based on match continuation.
    /// Returns 12-bit probability in [1, 4095].
    ///
    /// `c0`: partial byte being decoded (1-255).
    /// `bpos`: bit position (0-7).
    /// `c1`: last completed byte.
    /// `c2`: second-to-last byte.
    /// `c3`: third-to-last byte.
    #[inline]
    pub fn predict(&mut self, _c0: u32, bpos: u8, c1: u8, c2: u8, c3: u8) -> u32 {
        if bpos == 0 {
            // At byte boundary: look for new match or extend existing one.
            self.find_match(c1, c2, c3);
        }

        if self.match_pos < 0 || self.match_len < MIN_MATCH {
            self.last_p = 2048;
            return 2048; // No match → neutral prediction
        }

        // Predict from match continuation.
        let mpos = self.match_pos as usize & (BUF_SIZE - 1);
        let match_byte = self.buf[mpos];
        let match_bit = (match_byte >> (7 - bpos)) & 1;

        // Linear confidence ramp: higher match length → more confident.
        // Steeper ramp: 150 per match byte, caps at 3500 (strong confidence).
        let conf = ((self.match_len.min(MAX_MATCH_FOR_CONF) as u32) * 150).min(3500);
        let p = if match_bit == 1 {
            2048 + conf
        } else {
            2048u32.saturating_sub(conf)
        };
        let p = p.clamp(1, 4095);
        self.last_p = p;
        p
    }

    /// Update match model after observing `bit`.
    ///
    /// `bit`: observed bit.
    /// `bpos`: bit position (0-7).
    /// `c0`: partial byte (after this bit).
    /// `c1`: last completed byte (if bpos==7, this byte just completed).
    #[inline]
    pub fn update(&mut self, bit: u8, bpos: u8, c0: u32, c1: u8, c2: u8) {
        // Check if match continues.
        if self.match_pos >= 0 {
            let mpos = self.match_pos as usize & (BUF_SIZE - 1);
            let match_bit = (self.buf[mpos] >> (7 - self.match_bpos)) & 1;
            if match_bit == bit {
                self.match_bpos += 1;
                if self.match_bpos >= 8 {
                    self.match_bpos = 0;
                    self.match_len += 1;
                    self.match_pos = (self.match_pos + 1) & (BUF_SIZE as i64 - 1);
                }
            } else {
                // Mismatch: reset match.
                self.match_pos = -1;
                self.match_len = 0;
                self.match_bpos = 0;
            }
        }

        // At byte boundary (after last bit of byte): store byte and update hash.
        if bpos == 7 {
            let byte = (c0 & 0xFF) as u8;
            self.buf[self.buf_pos] = byte;

            // Non-cumulative rolling hash: hash of last 3 bytes.
            // CRITICAL: must NOT be cumulative (V2 bug lesson).
            self.hash = hash3(byte, c1, c2);

            // Store position in hash table.
            let idx = self.hash as usize & (HASH_SIZE - 1);
            self.hash_table[idx] = self.buf_pos as u32;

            self.buf_pos = (self.buf_pos + 1) & (BUF_SIZE - 1);
            self.total_written += 1;
        }
    }

    /// Find a match using the hash of recent bytes.
    fn find_match(&mut self, c1: u8, c2: u8, c3: u8) {
        let h = hash3(c1, c2, c3);
        let idx = h as usize & (HASH_SIZE - 1);
        let candidate_pos = self.hash_table[idx] as usize;

        // Validate the candidate.
        if self.total_written < 3 {
            self.match_pos = -1;
            self.match_len = 0;
            return;
        }

        // Check if the candidate position has matching context.
        // We need the 3 bytes before candidate_pos to match c3, c2, c1.
        let bp = candidate_pos;
        let p1 = (bp.wrapping_sub(1)) & (BUF_SIZE - 1);
        let p2 = (bp.wrapping_sub(2)) & (BUF_SIZE - 1);

        if self.buf[bp] == c1 && self.buf[p1] == c2 && self.buf[p2] == c3 {
            // Match found. Set match_pos to the byte AFTER the match context.
            self.match_pos = ((bp + 1) & (BUF_SIZE - 1)) as i64;
            self.match_len = 3;
            self.match_bpos = 0;
        } else {
            self.match_pos = -1;
            self.match_len = 0;
        }
    }

    /// Return the quantized match length for mixer context.
    /// 0=no match, 1=short, 2=medium, 3=long.
    #[inline]
    pub fn match_length_quantized(&self) -> u8 {
        if self.match_pos < 0 || self.match_len < MIN_MATCH {
            0
        } else if self.match_len < 8 {
            1
        } else if self.match_len < 32 {
            2
        } else {
            3
        }
    }

    /// Return the last predicted probability (for APM context).
    #[inline]
    pub fn last_prediction(&self) -> u32 {
        self.last_p
    }
}

impl Default for MatchModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Non-cumulative hash of 3 bytes. MUST NOT accumulate across calls (V2 lesson).
#[inline]
fn hash3(b1: u8, b2: u8, b3: u8) -> u32 {
    let mut h: u32 = b3 as u32;
    h = h.wrapping_mul(0x01000193) ^ b2 as u32;
    h = h.wrapping_mul(0x01000193) ^ b1 as u32;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_model_predicts_neutral() {
        let mut mm = MatchModel::new();
        let p = mm.predict(1, 0, 0, 0, 0);
        assert_eq!(p, 2048);
    }

    #[test]
    fn prediction_in_range() {
        let mut mm = MatchModel::new();
        // Feed some bytes first.
        for i in 0..100u8 {
            for bpos in 0..8u8 {
                let bit = (i >> (7 - bpos)) & 1;
                let c0 = if bpos == 7 {
                    (i as u32) | 0x100
                } else {
                    1u32 << (bpos + 1)
                };
                mm.update(bit, bpos, c0, i.wrapping_sub(1), i.wrapping_sub(2));
            }
        }
        let p = mm.predict(1, 0, 99, 98, 97);
        assert!((1..=4095).contains(&p));
    }

    #[test]
    fn hash3_not_cumulative() {
        // Same inputs should give same hash regardless of call order.
        let h1 = hash3(10, 20, 30);
        let h2 = hash3(10, 20, 30);
        assert_eq!(h1, h2);

        // Different inputs should differ.
        let h3 = hash3(11, 20, 30);
        assert_ne!(h1, h3);
    }

    #[test]
    fn match_quantization() {
        let mm = MatchModel::new();
        assert_eq!(mm.match_length_quantized(), 0); // no match
    }
}
