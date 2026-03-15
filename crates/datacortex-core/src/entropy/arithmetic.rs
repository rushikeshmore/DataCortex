//! Binary Arithmetic Coder — PAQ8-style, 12-bit precision, carry-free.
//!
//! Encodes/decodes one bit at a time given a 12-bit probability of bit=1.
//! Uses 32-bit range [low, high] with byte-wise normalization.
//!
//! Probabilities must be in [1, 4095]. 0 and 4096 are forbidden.

/// Precision bits for probability (12-bit).
const PROB_BITS: u32 = 12;

/// Maximum probability value (exclusive upper bound for scaling).
const PROB_SCALE: u32 = 1 << PROB_BITS; // 4096

// ─── Encoder ────────────────────────────────────────────────────────────

/// Binary arithmetic encoder. Accumulates compressed bytes.
pub struct ArithmeticEncoder {
    low: u32,
    high: u32,
    output: Vec<u8>,
}

impl ArithmeticEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        ArithmeticEncoder {
            low: 0,
            high: 0xFFFF_FFFF,
            output: Vec::new(),
        }
    }

    /// Encode a single bit with probability `p` of bit=1 (12-bit, [1, 4095]).
    #[inline(always)]
    pub fn encode(&mut self, bit: u8, p: u32) {
        debug_assert!(
            (1..=4095).contains(&p),
            "probability {p} out of range [1,4095]"
        );

        let range = self.high - self.low;
        // mid divides the range: [low, low+mid) = bit 0, [low+mid, high] = bit 1
        let mid = self.low
            + (range >> PROB_BITS) * (PROB_SCALE - p)
            + (((range & (PROB_SCALE - 1)) * (PROB_SCALE - p)) >> PROB_BITS);

        if bit != 0 {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }

        // Normalize: output matching top bytes.
        while (self.low ^ self.high) < 0x0100_0000 {
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 0xFF;
        }
    }

    /// Flush the encoder — write remaining state bytes.
    /// Must be called after encoding all bits.
    pub fn finish(mut self) -> Vec<u8> {
        // Write 4 more bytes to flush the state.
        self.output.push((self.low >> 24) as u8);
        self.output.push((self.low >> 16) as u8);
        self.output.push((self.low >> 8) as u8);
        self.output.push(self.low as u8);
        self.output
    }
}

impl Default for ArithmeticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Decoder ────────────────────────────────────────────────────────────

/// Binary arithmetic decoder. Reads bits from compressed data.
pub struct ArithmeticDecoder<'a> {
    low: u32,
    high: u32,
    code: u32, // current code value from input
    data: &'a [u8],
    pos: usize,
}

impl<'a> ArithmeticDecoder<'a> {
    /// Create a new decoder from compressed data.
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = ArithmeticDecoder {
            low: 0,
            high: 0xFFFF_FFFF,
            code: 0,
            data,
            pos: 0,
        };
        // Load initial 4 bytes into code.
        for _ in 0..4 {
            dec.code = (dec.code << 8) | dec.read_byte() as u32;
        }
        dec
    }

    /// Decode a single bit given probability `p` of bit=1 (12-bit, [1, 4095]).
    #[inline(always)]
    pub fn decode(&mut self, p: u32) -> u8 {
        debug_assert!(
            (1..=4095).contains(&p),
            "probability {p} out of range [1,4095]"
        );

        let range = self.high - self.low;
        let mid = self.low
            + (range >> PROB_BITS) * (PROB_SCALE - p)
            + (((range & (PROB_SCALE - 1)) * (PROB_SCALE - p)) >> PROB_BITS);

        let bit = if self.code > mid { 1u8 } else { 0u8 };

        if bit != 0 {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }

        // Normalize: shift out matching top bytes, read new byte.
        while (self.low ^ self.high) < 0x0100_0000 {
            self.low <<= 8;
            self.high = (self.high << 8) | 0xFF;
            self.code = (self.code << 8) | self.read_byte() as u32;
        }

        bit
    }

    /// Read the next byte from compressed data (0 if past end).
    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_single_bit_0() {
        let mut enc = ArithmeticEncoder::new();
        enc.encode(0, 2048); // p=0.5
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        let bit = dec.decode(2048);
        assert_eq!(bit, 0);
    }

    #[test]
    fn encode_decode_single_bit_1() {
        let mut enc = ArithmeticEncoder::new();
        enc.encode(1, 2048);
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        let bit = dec.decode(2048);
        assert_eq!(bit, 1);
    }

    #[test]
    fn encode_decode_sequence() {
        let bits: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let probs: Vec<u32> = vec![2048, 1000, 3000, 500, 2048, 100, 3900, 2048];

        let mut enc = ArithmeticEncoder::new();
        for (&bit, &p) in bits.iter().zip(probs.iter()) {
            enc.encode(bit, p);
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        for (i, (&expected_bit, &p)) in bits.iter().zip(probs.iter()).enumerate() {
            let decoded = dec.decode(p);
            assert_eq!(
                decoded, expected_bit,
                "mismatch at bit {i}: expected {expected_bit}, got {decoded}"
            );
        }
    }

    #[test]
    fn encode_decode_all_zeros() {
        let n = 100;
        let mut enc = ArithmeticEncoder::new();
        for _ in 0..n {
            enc.encode(0, 2048);
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        for i in 0..n {
            let bit = dec.decode(2048);
            assert_eq!(bit, 0, "mismatch at bit {i}");
        }
    }

    #[test]
    fn encode_decode_all_ones() {
        let n = 100;
        let mut enc = ArithmeticEncoder::new();
        for _ in 0..n {
            enc.encode(1, 2048);
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        for i in 0..n {
            let bit = dec.decode(2048);
            assert_eq!(bit, 1, "mismatch at bit {i}");
        }
    }

    #[test]
    fn high_probability_compresses() {
        // All 1s with high P(1) should compress well.
        let n = 1000;
        let mut enc = ArithmeticEncoder::new();
        for _ in 0..n {
            enc.encode(1, 4000); // P(1)≈0.98
        }
        let compressed = enc.finish();

        // 1000 bits at high probability should compress to much less than 125 bytes.
        assert!(
            compressed.len() < 50,
            "expected good compression, got {} bytes for {} bits at p=4000",
            compressed.len(),
            n
        );

        // Verify roundtrip.
        let mut dec = ArithmeticDecoder::new(&compressed);
        for i in 0..n {
            assert_eq!(dec.decode(4000), 1, "mismatch at bit {i}");
        }
    }

    #[test]
    fn extreme_probabilities() {
        // Test with probabilities near the bounds.
        let bits = [0, 1, 0, 1, 1, 0];
        let probs = [1, 4095, 1, 4095, 1, 4095];

        let mut enc = ArithmeticEncoder::new();
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            enc.encode(b, p);
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        for (i, (&expected, &p)) in bits.iter().zip(probs.iter()).enumerate() {
            let decoded = dec.decode(p);
            assert_eq!(decoded, expected, "mismatch at bit {i}");
        }
    }

    #[test]
    fn byte_roundtrip() {
        // Encode a full byte (8 bits) and decode it.
        let byte_val: u8 = 0xA5; // 10100101
        let mut enc = ArithmeticEncoder::new();
        for bpos in 0..8 {
            let bit = (byte_val >> (7 - bpos)) & 1;
            enc.encode(bit, 2048);
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        let mut decoded_byte: u8 = 0;
        for bpos in 0..8 {
            let bit = dec.decode(2048);
            decoded_byte |= bit << (7 - bpos);
        }
        assert_eq!(decoded_byte, byte_val);
    }

    #[test]
    fn varying_probabilities_per_bit() {
        // Simulate a model that adapts probabilities.
        let data: Vec<u8> = (0u32..50).map(|i| ((i * 7 + 13) & 0xFF) as u8).collect();

        let mut enc = ArithmeticEncoder::new();
        let mut p: u32 = 2048;
        for &byte in &data {
            for bpos in 0..8 {
                let bit = (byte >> (7 - bpos)) & 1;
                enc.encode(bit, p);
                // Simple adaptation.
                if bit == 1 {
                    p = (p + 100).min(4095);
                } else {
                    p = if p > 101 { p - 100 } else { 1 };
                }
            }
        }
        let compressed = enc.finish();

        let mut dec = ArithmeticDecoder::new(&compressed);
        let mut p: u32 = 2048;
        for (i, &byte) in data.iter().enumerate() {
            let mut decoded: u8 = 0;
            for bpos in 0..8 {
                let bit = dec.decode(p);
                decoded |= bit << (7 - bpos);
                if bit == 1 {
                    p = (p + 100).min(4095);
                } else {
                    p = if p > 101 { p - 100 } else { 1 };
                }
            }
            assert_eq!(decoded, byte, "byte mismatch at index {i}");
        }
    }
}
