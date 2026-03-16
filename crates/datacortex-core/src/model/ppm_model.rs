//! PPM (Prediction by Partial Matching) byte-level predictor — PPMd variant.
//!
//! A fundamentally DIFFERENT prediction paradigm from CM:
//! - CM: hash-based, lossy collisions, bit-level, fixed context orders
//! - PPM: byte-level, adaptive order with escape/exclusion, checksum-validated
//!
//! Full-range PPM with orders 0-12. Lower orders (0-3) provide a good fallback
//! on small files. Higher orders (10-12) go BEYOND what any CM model covers,
//! providing unique prediction signal on large files. All orders use checksum
//! validation to avoid hash collisions.
//!
//! The PPMd Method D escape estimation (Shkarin formula) and full exclusion
//! mechanism provide a different error profile from CM's bit-level hash models.
//!
//! CRITICAL: PPM updates at BYTE level, not bit level. Only update after all
//! 8 bits decoded. Byte probabilities are cached and converted to bit
//! predictions on each bit.

/// Maximum context order (up to 12 preceding bytes).
const MAX_ORDER: usize = 12;

/// Maximum symbols stored per context entry.
const MAX_SYMS: usize = 48;

/// FNV offset basis.
const FNV_OFFSET: u32 = 0x811C_9DC5;
/// FNV prime.
const FNV_PRIME: u32 = 0x0100_0193;

/// Number of orders: 0..=MAX_ORDER = 13.
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// A flat PPM entry with checksum validation.
#[derive(Clone, Copy)]
struct PpmEntry {
    /// Context checksum (upper 16 bits of hash). 0 = empty slot.
    checksum: u16,
    /// Symbols observed in this context.
    syms: [u8; MAX_SYMS],
    /// Counts for each symbol.
    counts: [u16; MAX_SYMS],
    /// Number of distinct symbols stored.
    len: u8,
    /// Sum of all counts.
    total: u16,
}

impl PpmEntry {
    const EMPTY: Self = PpmEntry {
        checksum: 0,
        syms: [0; MAX_SYMS],
        counts: [0; MAX_SYMS],
        len: 0,
        total: 0,
    };

    #[inline]
    fn increment(&mut self, symbol: u8) {
        let n = self.len as usize;
        for i in 0..n {
            if self.syms[i] == symbol {
                self.counts[i] = self.counts[i].saturating_add(1);
                self.total = self.total.saturating_add(1);
                return;
            }
        }
        if n < MAX_SYMS {
            self.syms[n] = symbol;
            self.counts[n] = 1;
            self.len += 1;
            self.total = self.total.saturating_add(1);
        }
    }

    fn halve(&mut self) {
        let mut write = 0usize;
        let mut new_total: u16 = 0;
        for read in 0..self.len as usize {
            let c = self.counts[read] >> 1;
            if c > 0 {
                self.syms[write] = self.syms[read];
                self.counts[write] = c;
                new_total = new_total.saturating_add(c);
                write += 1;
            }
        }
        self.len = write as u8;
        self.total = new_total;
    }
}

/// PPM table sizes configuration.
#[derive(Debug, Clone)]
pub struct PpmConfig {
    /// Table sizes per order (0..=MAX_ORDER). Each must be a power of 2.
    pub sizes: [usize; NUM_ORDERS],
}

impl PpmConfig {
    /// Default (~90MB): original sizes.
    pub fn default_sizes() -> Self {
        PpmConfig {
            sizes: [
                1,       // order 0:  1 entry (unigram)
                1 << 8,  // order 1:  256 entries
                1 << 16, // order 2:  64K entries
                1 << 18, // order 3:  256K entries
                1 << 19, // order 4:  512K entries
                1 << 19, // order 5:  512K entries
                1 << 19, // order 6:  512K entries
                1 << 18, // order 7:  256K entries
                1 << 18, // order 8:  256K entries
                1 << 17, // order 9:  128K entries
                1 << 17, // order 10: 128K entries
                1 << 16, // order 11: 64K entries
                1 << 16, // order 12: 64K entries
            ],
        }
    }

    /// Scaled 4x (~360MB): 4x entries at orders 3-12 for fewer collisions.
    pub fn scaled_4x() -> Self {
        PpmConfig {
            sizes: [
                1,       // order 0:  1 entry (unigram)
                1 << 8,  // order 1:  256 entries
                1 << 16, // order 2:  64K entries
                1 << 20, // order 3:  1M entries (was 256K)
                1 << 21, // order 4:  2M entries (was 512K)
                1 << 21, // order 5:  2M entries (was 512K)
                1 << 21, // order 6:  2M entries (was 512K)
                1 << 20, // order 7:  1M entries (was 256K)
                1 << 20, // order 8:  1M entries (was 256K)
                1 << 19, // order 9:  512K entries (was 128K)
                1 << 19, // order 10: 512K entries (was 128K)
                1 << 18, // order 11: 256K entries (was 64K)
                1 << 18, // order 12: 256K entries (was 64K)
            ],
        }
    }
}

/// PPM model with checksum-validated hash tables at orders 0-12.
///
/// Memory budget depends on config:
/// - Default: ~90MB total
/// - Scaled 4x: ~360MB total
/// - Order 0: 1 entry (global unigram)
/// - Order 1: 256 entries
/// - Order 2: 64K entries
/// - Orders 3+: configurable
pub struct PpmModel {
    /// Hash tables for orders 0..=MAX_ORDER.
    tables: Vec<Box<[PpmEntry]>>,
    /// Table masks (size - 1) per order.
    masks: [usize; NUM_ORDERS],

    /// Cached byte probability distribution (256 entries, scaled to sum ~2^20).
    byte_probs: [u32; 256],
    /// Whether byte_probs has been computed.
    probs_valid: bool,
    /// Context bytes: last MAX_ORDER bytes. [0] = most recent.
    context: [u8; MAX_ORDER],
    /// Number of bytes seen so far.
    bytes_seen: usize,
}

fn make_table(size: usize) -> Box<[PpmEntry]> {
    vec![PpmEntry::EMPTY; size].into_boxed_slice()
}

impl PpmModel {
    /// Create a new PPM model with default sizes (~90MB).
    pub fn new() -> Self {
        Self::with_config(PpmConfig::default_sizes())
    }

    /// Create a PPM model with the given configuration.
    pub fn with_config(config: PpmConfig) -> Self {
        let mut tables = Vec::with_capacity(NUM_ORDERS);
        let mut masks = [0usize; NUM_ORDERS];
        for (i, &size) in config.sizes.iter().enumerate() {
            tables.push(make_table(size));
            masks[i] = size - 1;
        }

        PpmModel {
            tables,
            masks,
            byte_probs: [0u32; 256],
            probs_valid: false,
            context: [0u8; MAX_ORDER],
            bytes_seen: 0,
        }
    }

    /// Predict bit probability. Returns 12-bit probability [1, 4095].
    #[inline]
    pub fn predict_bit(&mut self, bpos: u8, c0: u32) -> u32 {
        if !self.probs_valid {
            self.compute_byte_probs();
            self.probs_valid = true;
        }

        let bit_pos = 7 - bpos;
        let mask = 1u8 << bit_pos;

        let mut sum_one: u64 = 0;
        let mut sum_zero: u64 = 0;

        if bpos == 0 {
            for b in 0..256usize {
                let p = self.byte_probs[b] as u64;
                if (b as u8) & mask != 0 {
                    sum_one += p;
                } else {
                    sum_zero += p;
                }
            }
        } else {
            let partial = (c0 & ((1u32 << bpos) - 1)) as u8;
            let shift = 8 - bpos;
            let base = (partial as usize) << shift;
            let count = 1usize << shift;

            for i in 0..count {
                let b = base | i;
                let p = self.byte_probs[b] as u64;
                if (b as u8) & mask != 0 {
                    sum_one += p;
                } else {
                    sum_zero += p;
                }
            }
        }

        let total = sum_one + sum_zero;
        if total == 0 {
            return 2048;
        }

        let p = ((sum_one << 12) / total) as u32;
        p.clamp(1, 4095)
    }

    /// Update PPM model after a full byte has been decoded.
    #[inline]
    pub fn update_byte(&mut self, byte: u8) {
        let max_usable_order = self.bytes_seen.min(MAX_ORDER);

        for order in 0..=max_usable_order {
            let (hash, chk) = self.context_hash_and_checksum(order);
            let idx = hash as usize & self.masks[order];
            let entry = &mut self.tables[order][idx];

            if entry.checksum == 0 || entry.checksum == chk {
                entry.checksum = chk;
                entry.increment(byte);
                if entry.total > 4000 {
                    entry.halve();
                }
            } else {
                // Hash collision. Replace weak entries.
                if entry.total < 4 {
                    *entry = PpmEntry::EMPTY;
                    entry.checksum = chk;
                    entry.increment(byte);
                }
            }
        }

        // Shift context ring.
        for i in (1..MAX_ORDER).rev() {
            self.context[i] = self.context[i - 1];
        }
        self.context[0] = byte;
        self.bytes_seen += 1;
        self.probs_valid = false;
    }

    /// Compute byte probability distribution using PPMd Method D with exclusion.
    fn compute_byte_probs(&mut self) {
        let max_usable_order = self.bytes_seen.min(MAX_ORDER);

        let mut excluded = [false; 256];
        let mut probs = [0u64; 256];
        let mut remaining_mass: u64 = 1 << 20;

        // Scan from highest order down to 0.
        for order in (0..=max_usable_order).rev() {
            let (hash, chk) = self.context_hash_and_checksum(order);
            let idx = hash as usize & self.masks[order];
            let entry = &self.tables[order][idx];

            // Skip if empty or checksum mismatch (hash collision).
            if entry.checksum != chk || entry.total == 0 || entry.len == 0 {
                continue;
            }

            let mut effective_total: u32 = 0;
            let mut effective_distinct: u32 = 0;

            let n = entry.len as usize;
            for i in 0..n {
                if !excluded[entry.syms[i] as usize] {
                    effective_total += entry.counts[i] as u32;
                    effective_distinct += 1;
                }
            }

            if effective_total == 0 || effective_distinct == 0 {
                continue;
            }

            // PPMd Method D escape.
            let escape_d = effective_distinct.div_ceil(2);
            let denominator = effective_total + escape_d;

            let symbol_mass = (remaining_mass * effective_total as u64) / denominator as u64;
            let escape_frac = remaining_mass - symbol_mass;

            for i in 0..n {
                let sym = entry.syms[i];
                if !excluded[sym as usize] {
                    let sym_prob = (symbol_mass * entry.counts[i] as u64) / effective_total as u64;
                    probs[sym as usize] += sym_prob;
                    excluded[sym as usize] = true;
                }
            }

            remaining_mass = escape_frac;
            if remaining_mass == 0 {
                break;
            }
        }

        // Order -1: uniform for remaining unseen symbols.
        if remaining_mass > 0 {
            let mut unseen: u32 = 0;
            for e in &excluded {
                if !e {
                    unseen += 1;
                }
            }
            if unseen > 0 {
                let per_sym = remaining_mass / unseen as u64;
                let mut leftover = remaining_mass - per_sym * unseen as u64;
                for i in 0..256 {
                    if !excluded[i] {
                        probs[i] += per_sym;
                        if leftover > 0 {
                            probs[i] += 1;
                            leftover -= 1;
                        }
                    }
                }
            }
        }

        for (i, &p) in probs.iter().enumerate() {
            self.byte_probs[i] = p as u32;
        }
    }

    /// Compute context hash and 16-bit checksum for a given order.
    #[inline]
    fn context_hash_and_checksum(&self, order: usize) -> (u32, u16) {
        if order == 0 {
            // Order 0: single context, fixed checksum.
            return (0, 1);
        }
        let mut h = FNV_OFFSET;
        for i in 0..order {
            h ^= self.context[i] as u32;
            h = h.wrapping_mul(FNV_PRIME);
        }
        let chk = ((h >> 16) as u16) | 1; // ensure non-zero
        (h, chk)
    }
}

impl Default for PpmModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_balanced() {
        let mut model = PpmModel::new();
        let p = model.predict_bit(0, 1);
        assert!(
            (1900..=2100).contains(&p),
            "initial prediction should be near 2048, got {p}"
        );
    }

    #[test]
    fn prediction_always_in_range() {
        let mut model = PpmModel::new();
        let data = b"Hello, World! This is a test of the PPM model for prediction.";
        for &byte in data {
            for bpos in 0..8u8 {
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let p = model.predict_bit(bpos, c0);
                assert!(
                    (1..=4095).contains(&p),
                    "prediction out of range at bpos {bpos}: {p}"
                );
            }
            model.update_byte(byte);
        }
    }

    #[test]
    fn adapts_to_repeated_bytes() {
        let mut model = PpmModel::new();
        let byte = b'A';
        for _ in 0..100 {
            model.update_byte(byte);
        }
        let p = model.predict_bit(0, 1);
        // Bit 7 of 'A' (0x41) is 0, so P(bit=1) should be low.
        assert!(
            p < 1500,
            "after 100 'A' bytes, P(bit7=1) should be low, got {p}"
        );
    }

    #[test]
    fn adapts_to_repeated_pattern() {
        let mut model = PpmModel::new();
        let pattern = b"abcdefgh";
        for _ in 0..200 {
            for &byte in pattern {
                model.update_byte(byte);
            }
        }
        for &byte in b"abcdefg" {
            model.update_byte(byte);
        }
        model.compute_byte_probs();
        let p_h = model.byte_probs[b'h' as usize];
        assert!(
            p_h > 100_000,
            "after 'abcdefg', P('h') should be significant, got {p_h} / 1048576"
        );
    }

    #[test]
    fn byte_probs_sum_correctly() {
        let mut model = PpmModel::new();
        let data = b"the quick brown fox jumps over the lazy dog the cat sat on the mat";
        for &byte in data.iter() {
            model.update_byte(byte);
        }
        model.compute_byte_probs();
        let total: u64 = model.byte_probs.iter().map(|&p| p as u64).sum();
        assert!(
            (1_000_000..=1_100_000).contains(&total),
            "byte_probs should sum to ~1M, got {total}"
        );
    }

    #[test]
    fn exclusion_works() {
        let mut model = PpmModel::new();
        for _ in 0..100 {
            model.update_byte(b'a');
            model.update_byte(b'b');
        }
        model.update_byte(b'a');
        model.compute_byte_probs();
        let p_b = model.byte_probs[b'b' as usize];
        let p_a = model.byte_probs[b'a' as usize];
        assert!(
            p_b > p_a * 2,
            "after 'a', P('b')={p_b} should be >> P('a')={p_a}"
        );
    }

    #[test]
    fn deterministic() {
        let data = b"test determinism of ppm model with enough context abcabc";
        let mut m1 = PpmModel::new();
        let mut m2 = PpmModel::new();

        for &byte in data.iter() {
            for bpos in 0..8u8 {
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let p1 = m1.predict_bit(bpos, c0);
                let p2 = m2.predict_bit(bpos, c0);
                assert_eq!(p1, p2, "models diverged at bpos {bpos}");
            }
            m1.update_byte(byte);
            m2.update_byte(byte);
        }
    }

    #[test]
    fn solo_bpb_alice29_prefix() {
        let data = include_bytes!("../../../../corpus/alice29.txt");
        let prefix = &data[..10_000.min(data.len())];

        let mut model = PpmModel::new();
        let mut total_bits: f64 = 0.0;

        for &byte in prefix {
            let mut c0 = 1u32;
            for bpos in 0..8u8 {
                let p = model.predict_bit(bpos, c0);
                let bit = (byte >> (7 - bpos)) & 1;
                let prob_of_bit = if bit == 1 {
                    p as f64 / 4096.0
                } else {
                    1.0 - p as f64 / 4096.0
                };
                total_bits += -prob_of_bit.max(1e-9).log2();
                c0 = (c0 << 1) | bit as u32;
            }
            model.update_byte(byte);
        }

        let bpb = total_bits / prefix.len() as f64;
        eprintln!("PPM solo bpb on 10KB alice29 (orders 0-{MAX_ORDER}): {bpb:.3}");
        assert!(bpb < 6.0, "PPM solo bpb too high: {bpb:.3}");
    }

    #[test]
    fn ppm_entry_increment_and_halve() {
        let mut entry = PpmEntry::EMPTY;
        entry.checksum = 1;
        entry.increment(b'a');
        entry.increment(b'a');
        entry.increment(b'b');
        assert_eq!(entry.len, 2);
        assert_eq!(entry.total, 3);

        entry.halve();
        assert_eq!(entry.len, 1);
        assert_eq!(entry.counts[0], 1);
        assert_eq!(entry.total, 1);
    }
}
