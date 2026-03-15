//! MatchModel -- ring buffer + hash table for longest match prediction.
//!
//! Phase 5: Multi-candidate match finding with extended length verification.
//! Finds up to 4 candidates via different hash functions, verifies each,
//! and uses the one with the longest actual match length.
//!
//! CRITICAL V2 LESSONS:
//! - Rolling hash must NOT be cumulative
//! - Confidence ramp must be linear (not step function)
//! - Length tracking must reset on mismatch

/// Size of the ring buffer (16MB).
const BUF_SIZE: usize = 16 * 1024 * 1024;

/// Size of the hash table for match finding (8M entries).
const HASH_SIZE: usize = 8 * 1024 * 1024;

/// Minimum match length before we start predicting.
const MIN_MATCH: usize = 2;

/// Maximum match length for confidence calculation.
const MAX_MATCH_FOR_CONF: usize = 64;

/// Maximum bytes to scan forward when verifying match length.
const MAX_VERIFY_LEN: usize = 128;

/// Match model: finds longest match in history, predicts from continuation.
pub struct MatchModel {
    /// Ring buffer of past bytes.
    buf: Vec<u8>,
    /// Current write position in ring buffer.
    buf_pos: usize,
    /// Total bytes written (for position validity).
    total_written: usize,
    /// Hash table: hash -> position in ring buffer.
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
            return 2048; // No match -> neutral prediction
        }

        // Predict from match continuation.
        let mpos = self.match_pos as usize & (BUF_SIZE - 1);
        let match_byte = self.buf[mpos];
        let match_bit = (match_byte >> (7 - bpos)) & 1;

        // Improved confidence ramp: slow start, steep middle, saturating.
        // Uses a piecewise function:
        //   len 2-3: 80 per byte (tentative)
        //   len 4-8: 200 per byte (building confidence)
        //   len 9-32: 120 per byte (strong)
        //   len 33+: 60 per byte (saturating)
        let len = self.match_len.min(MAX_MATCH_FOR_CONF);
        let conf = if len <= 3 {
            (len as u32) * 80
        } else if len <= 8 {
            240 + ((len as u32 - 3) * 200)
        } else if len <= 32 {
            1240 + ((len as u32 - 8) * 120)
        } else {
            4120u32.min(1240 + 2880 + ((len as u32 - 32) * 60))
        };
        let conf = conf.min(3800);

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

            // Non-cumulative rolling hash: hash of last 4 bytes for better match finding.
            // CRITICAL: must NOT be cumulative (V2 bug lesson).
            self.hash = hash4(byte, c1, c2, self.prev_byte(3));

            // Store position in hash table using primary (4-byte) hash.
            let idx = self.hash as usize & (HASH_SIZE - 1);
            self.hash_table[idx] = self.buf_pos as u32;

            // Store using 3-byte hash for fallback match opportunities.
            let h3 = hash3(byte, c1, c2);
            let idx3 = h3 as usize & (HASH_SIZE - 1);
            // Only store if slot is empty (don't overwrite better 4-byte matches)
            if self.hash_table[idx3] == 0 || self.total_written < 4 {
                self.hash_table[idx3] = self.buf_pos as u32;
            }

            // Store using 5-byte hash for longer match opportunities.
            let c3 = self.prev_byte(3);
            let c4 = self.prev_byte(4);
            let h5 = hash5(byte, c1, c2, c3, c4);
            let idx5 = h5 as usize & (HASH_SIZE - 1);
            self.hash_table[idx5] = self.buf_pos as u32;

            self.buf_pos = (self.buf_pos + 1) & (BUF_SIZE - 1);
            self.total_written += 1;
        }
    }

    /// Get byte from N positions before current write position.
    #[inline]
    fn prev_byte(&self, n: usize) -> u8 {
        if self.total_written >= n {
            self.buf[(self.buf_pos.wrapping_sub(n)) & (BUF_SIZE - 1)]
        } else {
            0
        }
    }

    /// Verify actual match length at a candidate position.
    /// Returns the number of matching bytes starting from the candidate + 1
    /// (i.e., matching bytes after the context that was used to find it).
    #[inline]
    fn verify_match_length(&self, candidate_pos: usize) -> usize {
        let verify_start = (candidate_pos + 1) & (BUF_SIZE - 1);
        let data_start = self.buf_pos; // current write position = where next byte goes
        let max_len = self.total_written.min(MAX_VERIFY_LEN);
        let mut len = 0;
        while len < max_len {
            let mp = (verify_start + len) & (BUF_SIZE - 1);
            let dp = (data_start + len) & (BUF_SIZE - 1);
            // Don't compare beyond what we've written or wrap into the match itself.
            if mp == self.buf_pos {
                break;
            }
            if self.buf[mp] != self.buf[dp] {
                break;
            }
            len += 1;
        }
        len
    }

    /// Find the best match among multiple candidates using different hash functions.
    fn find_match(&mut self, c1: u8, c2: u8, c3: u8) {
        if self.total_written < 3 {
            self.match_pos = -1;
            self.match_len = 0;
            return;
        }

        let c4 = self.prev_byte(3); // 4th-to-last byte
        let c5 = self.prev_byte(4); // 5th-to-last byte

        // Collect up to 4 candidate positions from different hash functions.
        // Each candidate is verified for actual match quality.
        let mut best_pos: i64 = -1;
        let mut best_len: usize = 0;

        // Candidate 1: 5-byte hash (best precision)
        if self.total_written >= 5 {
            let h5 = hash5(c1, c2, c3, c4, c5);
            let idx5 = h5 as usize & (HASH_SIZE - 1);
            let cand = self.hash_table[idx5] as usize;
            self.check_candidate(cand, c1, c2, c3, &mut best_pos, &mut best_len);
        }

        // Candidate 2: 4-byte hash
        let h4 = hash4(c1, c2, c3, c4);
        let idx4 = h4 as usize & (HASH_SIZE - 1);
        let cand4 = self.hash_table[idx4] as usize;
        self.check_candidate(cand4, c1, c2, c3, &mut best_pos, &mut best_len);

        // Candidate 3: 3-byte hash (wider net)
        let h3 = hash3(c1, c2, c3);
        let idx3 = h3 as usize & (HASH_SIZE - 1);
        let cand3 = self.hash_table[idx3] as usize;
        self.check_candidate(cand3, c1, c2, c3, &mut best_pos, &mut best_len);

        // Candidate 4: alternate 4-byte hash with different mixing
        let h4b = hash4_alt(c1, c2, c3, c4);
        let idx4b = h4b as usize & (HASH_SIZE - 1);
        let cand4b = self.hash_table[idx4b] as usize;
        self.check_candidate(cand4b, c1, c2, c3, &mut best_pos, &mut best_len);

        if best_len >= MIN_MATCH {
            self.match_pos = best_pos;
            self.match_len = best_len;
            self.match_bpos = 0;
        } else {
            self.match_pos = -1;
            self.match_len = 0;
        }
    }

    /// Check a candidate position and update best match if it's better.
    #[inline]
    fn check_candidate(
        &self,
        candidate_pos: usize,
        c1: u8,
        c2: u8,
        c3: u8,
        best_pos: &mut i64,
        best_len: &mut usize,
    ) {
        let bp = candidate_pos;
        let p1 = bp.wrapping_sub(1) & (BUF_SIZE - 1);
        let p2 = bp.wrapping_sub(2) & (BUF_SIZE - 1);

        // Verify the context bytes match.
        if self.buf[bp] == c1 && self.buf[p1] == c2 && self.buf[p2] == c3 {
            // Context matches. Now verify how far the match extends forward.
            let fwd_len = self.verify_match_length(bp);
            let total_match = 3 + fwd_len; // 3 context bytes + forward extension
            if total_match > *best_len {
                *best_len = total_match;
                *best_pos = ((bp + 1) & (BUF_SIZE - 1)) as i64;
            }
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

/// Non-cumulative hash of 4 bytes for better match precision.
#[inline]
fn hash4(b1: u8, b2: u8, b3: u8, b4: u8) -> u32 {
    let mut h: u32 = b4 as u32;
    h = h.wrapping_mul(0x01000193) ^ b3 as u32;
    h = h.wrapping_mul(0x01000193) ^ b2 as u32;
    h = h.wrapping_mul(0x01000193) ^ b1 as u32;
    h
}

/// Alternate 4-byte hash with different constants for a second slot.
#[inline]
fn hash4_alt(b1: u8, b2: u8, b3: u8, b4: u8) -> u32 {
    let mut h: u32 = 0x9E3779B9; // golden ratio
    h ^= b4 as u32;
    h = h.wrapping_mul(0x01000193);
    h ^= b3 as u32;
    h = h.wrapping_mul(0x01000193);
    h ^= b2 as u32;
    h = h.wrapping_mul(0x01000193);
    h ^= b1 as u32;
    h
}

/// Non-cumulative hash of 5 bytes for longer context matching.
#[inline]
fn hash5(b1: u8, b2: u8, b3: u8, b4: u8, b5: u8) -> u32 {
    let mut h: u32 = b5 as u32;
    h = h.wrapping_mul(0x01000193) ^ b4 as u32;
    h = h.wrapping_mul(0x01000193) ^ b3 as u32;
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
    fn hash4_not_cumulative() {
        let h1 = hash4(10, 20, 30, 40);
        let h2 = hash4(10, 20, 30, 40);
        assert_eq!(h1, h2);

        let h3 = hash4(11, 20, 30, 40);
        assert_ne!(h1, h3);
    }

    #[test]
    fn hash5_not_cumulative() {
        let h1 = hash5(10, 20, 30, 40, 50);
        let h2 = hash5(10, 20, 30, 40, 50);
        assert_eq!(h1, h2);

        let h3 = hash5(11, 20, 30, 40, 50);
        assert_ne!(h1, h3);
    }

    #[test]
    fn hash4_alt_differs_from_hash4() {
        let h1 = hash4(10, 20, 30, 40);
        let h2 = hash4_alt(10, 20, 30, 40);
        assert_ne!(h1, h2, "alt hash should differ from primary");
    }

    #[test]
    fn match_quantization() {
        let mm = MatchModel::new();
        assert_eq!(mm.match_length_quantized(), 0); // no match
    }
}
