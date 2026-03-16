//! ISSE — Indirect Secondary Symbol Estimation model.
//!
//! Provides model #19 for the mixer: a 3-level ISSE chain using
//! CROSS-CONTEXT hashes that combine dimensions the main n-gram
//! models don't capture.
//!
//! The main order-0..9 models use pure n-gram context (c1, c2, ..., c9).
//! This ISSE model uses:
//!   Level 0 (ICM): word-position context (position within current word + bpos)
//!   Level 1 (ISSE): byte-class bigram transition context
//!   Level 2 (ISSE): sparse skip-2 context (c1, c3 — skipping c2)
//!
//! These contexts are orthogonal to the main models, giving the mixer
//! genuinely new information rather than duplicating existing signals.
//!
//! Architecture from ZPAQ (Matt Mahoney):
//!   ICM: context_hash → bit_history_state → probability (StateMap)
//!   ISSE: (bit_history_state, p_in) → p_out via learned weights (w0, w1)

use crate::mixer::logistic::{squash, stretch};
use crate::state::state_map::StateMap;
use crate::state::state_table::StateTable;

/// Hash table size per level (2^21 = 2M entries, 1 byte each = 2MB).
/// Total: 3 levels * 2MB = 6MB.
const HT_SIZE: usize = 1 << 21;
const HT_MASK: usize = HT_SIZE - 1;

/// Number of bit history states.
const NUM_STATES: usize = 256;

/// FNV hash prime.
const FNV_PRIME: u32 = 0x01000193;

/// Weight pair for ISSE.
#[derive(Clone, Copy)]
struct WeightPair {
    w0: i32,
    w1: i32,
}

/// ZPAQ scaling constants.
const W_SHIFT: i32 = 16;
const W_UNITY: i32 = 1 << W_SHIFT;
const BIAS_SCALE: i64 = 64;
const W_CLAMP: i64 = 524287;

/// ICM level — base predictor.
struct IcmLevel {
    ht: Vec<u8>,
    smap: StateMap,
    last_hash: u32,
    last_state: u8,
}

impl IcmLevel {
    fn new() -> Self {
        IcmLevel {
            ht: vec![0u8; HT_SIZE],
            smap: StateMap::new(),
            last_hash: 0,
            last_state: 0,
        }
    }

    #[inline]
    fn predict(&mut self, ctx_hash: u32) -> u32 {
        self.last_hash = ctx_hash;
        let state = self.ht[ctx_hash as usize & HT_MASK];
        self.last_state = state;
        self.smap.predict(state)
    }

    #[inline]
    fn update(&mut self, bit: u8) {
        self.smap.update(self.last_state, bit);
        let new_state = StateTable::next(self.last_state, bit);
        self.ht[self.last_hash as usize & HT_MASK] = new_state;
    }
}

/// ISSE level — refines input probability.
struct IsseLevel {
    ht: Vec<u8>,
    weights: [WeightPair; NUM_STATES],
    last_hash: u32,
    last_state: u8,
    last_d_in: i32,
    last_p_out: i32,
}

impl IsseLevel {
    fn new() -> Self {
        let mut weights = [WeightPair { w0: W_UNITY, w1: 0 }; NUM_STATES];

        // Initialize w1 bias from state table (ZPAQ style).
        for (s, wt) in weights.iter_mut().enumerate() {
            let state_p = StateTable::prob(s as u8);
            let state_d = stretch(state_p as u32);
            wt.w1 = state_d * 256;
        }

        IsseLevel {
            ht: vec![0u8; HT_SIZE],
            weights,
            last_hash: 0,
            last_state: 0,
            last_d_in: 0,
            last_p_out: 2048,
        }
    }

    #[inline]
    fn predict(&mut self, p_in: u32, ctx_hash: u32) -> u32 {
        self.last_hash = ctx_hash;
        let state = self.ht[ctx_hash as usize & HT_MASK];
        self.last_state = state;

        let d_in = stretch(p_in);
        self.last_d_in = d_in;

        let wt = &self.weights[state as usize];
        let d_out = (wt.w0 as i64 * d_in as i64 + wt.w1 as i64 * BIAS_SCALE) >> W_SHIFT;
        let p_out = squash(d_out as i32).clamp(1, 4095) as i32;
        self.last_p_out = p_out;
        p_out as u32
    }

    #[inline]
    fn update(&mut self, bit: u8) {
        let err = (bit as i32) * 32767 - self.last_p_out * 8;
        let wt = &mut self.weights[self.last_state as usize];

        let delta_w0 = (err as i64 * self.last_d_in as i64 + (1i64 << 12)) >> 13;
        wt.w0 = (wt.w0 as i64 + delta_w0).clamp(-W_CLAMP, W_CLAMP) as i32;

        let delta_w1 = (err + 16) >> 5;
        wt.w1 = (wt.w1 as i64 + delta_w1 as i64).clamp(-W_CLAMP, W_CLAMP) as i32;

        let new_state = StateTable::next(self.last_state, bit);
        self.ht[self.last_hash as usize & HT_MASK] = new_state;
    }
}

/// ISSE model: 3-level chain with cross-context hashes.
///
/// Uses contexts orthogonal to the main n-gram models:
/// - Word-position context (captures intra-word patterns)
/// - Byte-class transition context (captures character class patterns)
/// - Sparse skip-2 context (captures periodic patterns)
///
/// Memory: 3 * 2MB = 6MB.
pub struct IsseChain {
    icm: IcmLevel,
    isse1: IsseLevel,
    isse2: IsseLevel,
    /// Word position: distance since last space/newline/punctuation (0-255).
    word_pos: u8,
}

impl IsseChain {
    pub fn new() -> Self {
        IsseChain {
            icm: IcmLevel::new(),
            isse1: IsseLevel::new(),
            isse2: IsseLevel::new(),
            word_pos: 0,
        }
    }

    /// Produce a prediction for the mixer.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, c0: u32, c1: u8, c2: u8, c3: u8, bpos: u8) -> u32 {
        // Level 0 (ICM): word-position context.
        // Context = (word_pos, c0_partial, bpos).
        // This captures patterns like "3rd character in a word is usually lowercase".
        let h0 = word_pos_hash(self.word_pos, c0, bpos);
        let p0 = self.icm.predict(h0);

        // Level 1 (ISSE): byte-class bigram transition.
        // Context = (class(c1), class(c2), c0_partial, bpos).
        // Captures character class transitions (letter->digit, punct->letter, etc.)
        let h1 = class_transition_hash(c1, c2, c0, bpos);
        let p1 = self.isse1.predict(p0, h1);

        // Level 2 (ISSE): sparse skip-2 context.
        // Context = (c1, c3, c0_partial, bpos) — skips c2.
        // Captures periodic/skip patterns the sequential models miss.
        let h2 = sparse_skip2_hash(c1, c3, c0, bpos);
        let p2 = self.isse2.predict(p1, h2);

        p2.clamp(1, 4095)
    }

    /// Update after observing bit.
    #[inline]
    pub fn update(&mut self, bit: u8, c0: u32, bpos: u8) {
        self.isse2.update(bit);
        self.isse1.update(bit);
        self.icm.update(bit);

        // Track word position (update after byte boundary).
        if bpos == 7 {
            let byte = ((c0 << 1 | bit as u32) & 0xFF) as u8;
            if is_word_boundary(byte) {
                self.word_pos = 0;
            } else {
                self.word_pos = self.word_pos.saturating_add(1);
            }
        }
    }
}

impl Default for IsseChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if byte is a word boundary.
#[inline]
fn is_word_boundary(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\n'
            | b'\r'
            | b'\t'
            | b'.'
            | b','
            | b';'
            | b':'
            | b'!'
            | b'?'
            | b'('
            | b')'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'<'
            | b'>'
            | b'"'
            | b'\''
            | b'/'
            | b'='
    )
}

/// Byte classifier (0-7).
#[inline]
fn classify(b: u8) -> u8 {
    match b {
        0..=31 => 0,
        b' ' => 1,
        b'0'..=b'9' => 2,
        b'A'..=b'Z' => 3,
        b'a'..=b'z' => 4,
        b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~' => 5,
        128..=255 => 6,
        _ => 7,
    }
}

// --- Hash functions with unique seeds ---

const SEED_WP: u32 = 0xA5A5A5A5;
const SEED_CT: u32 = 0x5A5A5A5A;
const SEED_SK: u32 = 0x3C3C3C3C;

/// Word-position context hash.
#[inline]
fn word_pos_hash(word_pos: u8, c0: u32, bpos: u8) -> u32 {
    let mut h = SEED_WP;
    h ^= word_pos as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0x1FF;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= bpos as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Class-transition context hash.
#[inline]
fn class_transition_hash(c1: u8, c2: u8, c0: u32, bpos: u8) -> u32 {
    let mut h = SEED_CT;
    h ^= classify(c1) as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= classify(c2) as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0x1FF;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= bpos as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

/// Sparse skip-2 context hash (c1, c3 — skips c2).
#[inline]
fn sparse_skip2_hash(c1: u8, c3: u8, c0: u32, bpos: u8) -> u32 {
    let mut h = SEED_SK;
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c0 & 0x1FF;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= bpos as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_in_range() {
        let mut chain = IsseChain::new();
        let p = chain.predict(1, 0, 0, 0, 0);
        assert!(
            (1..=4095).contains(&p),
            "initial prediction out of range: {p}"
        );
    }

    #[test]
    fn prediction_always_in_range() {
        let mut chain = IsseChain::new();
        for bpos in 0..8u8 {
            let p = chain.predict(1, 65, 66, 67, bpos);
            assert!((1..=4095).contains(&p), "out of range: {p}");
            chain.update(1, 1, bpos);
        }
    }

    #[test]
    fn adapts_to_ones() {
        let mut chain = IsseChain::new();
        let mut last_p = 0u32;
        for i in 0..200 {
            let p = chain.predict(1, 0, 0, 0, 0);
            if i > 100 {
                last_p = p;
            }
            chain.update(1, 1, 0);
        }
        assert!(last_p > 2200, "should adapt toward 1: got {last_p}");
    }

    #[test]
    fn adapts_to_zeros() {
        let mut chain = IsseChain::new();
        let mut last_p = 0u32;
        for i in 0..200 {
            let p = chain.predict(1, 0, 0, 0, 0);
            if i > 100 {
                last_p = p;
            }
            chain.update(0, 1, 0);
        }
        assert!(last_p < 1800, "should adapt toward 0: got {last_p}");
    }

    #[test]
    fn different_contexts_diverge() {
        let mut chain = IsseChain::new();
        for _ in 0..100 {
            chain.predict(1, 65, 0, 0, 0);
            chain.update(1, 1, 0);
        }
        for _ in 0..100 {
            chain.predict(1, 66, 0, 0, 0);
            chain.update(0, 1, 0);
        }
        let p_a = chain.predict(1, 65, 0, 0, 0);
        let p_b = chain.predict(1, 66, 0, 0, 0);
        assert!(
            p_a > p_b,
            "trained contexts should diverge: p_a={p_a}, p_b={p_b}"
        );
    }

    #[test]
    fn deterministic() {
        let mut ch1 = IsseChain::new();
        let mut ch2 = IsseChain::new();
        let data = b"ISSE determinism";
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
                let p1 = ch1.predict(c0, byte, 0, 0, bpos);
                let p2 = ch2.predict(c0, byte, 0, 0, bpos);
                assert_eq!(p1, p2, "chains diverged at bpos {bpos}");
                ch1.update(bit, c0, bpos);
                ch2.update(bit, c0, bpos);
            }
        }
    }

    #[test]
    fn word_boundary_detection() {
        assert!(is_word_boundary(b' '));
        assert!(is_word_boundary(b'\n'));
        assert!(is_word_boundary(b'.'));
        assert!(!is_word_boundary(b'a'));
        assert!(!is_word_boundary(b'5'));
    }
}
