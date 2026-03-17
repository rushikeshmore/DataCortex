//! LZP (Lempel-Ziv Prediction) preprocessing transform.
//!
//! LZP is a simplified LZ variant designed for preprocessing before entropy coding.
//! Unlike LZ77 (which stores distance+length pairs), LZP uses a hash table keyed by
//! recent context to predict the SINGLE most likely match position. The decompressor
//! rebuilds the identical hash table, so no match positions need to be stored.
//!
//! Based on research from:
//! - bsc/libbsc: uses LZP before BWT (hash size 10-28 bits, min match 4-255)
//! - kanzi: LZP as pre-transform ("one possible match, offset not emitted")
//! - Charles Bloom's original LZP paper (DCC 1996)
//!
//! The transform replaces matched byte sequences with (escape, length) tokens,
//! making the data smaller and more predictable for the CM engine.
//!
//! CRITICAL: Compressor and decompressor must build IDENTICAL hash tables
//! using the EXACT same hash function and update order.

/// Minimum match length before we emit a match token.
const MIN_MATCH: usize = 4;

/// Maximum match length we encode in a single token.
const MAX_MATCH: usize = 255 + MIN_MATCH - 1; // encoded len 1..=255 -> match 4..=258

/// Hash table size (1M entries = 20 bits). Each entry is a u32 position.
/// This is 4MB of memory.
const HASH_BITS: usize = 20;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;

/// Escape byte used to signal a match or a literal escape.
/// Encoding:
///   0xFE 0x00           = literal byte 0xFE
///   0xFE len (1..=255)  = match of (len + 3) bytes = 4..=258 bytes
const ESCAPE: u8 = 0xFE;

/// Number of context bytes used for hashing.
const CTX_ORDER: usize = 4;

/// FNV-1a constants for context hashing.
const FNV_OFFSET: u32 = 0x811C9DC5;
const FNV_PRIME: u32 = 0x01000193;

/// Hash 4 bytes of context to produce a hash table index.
/// MUST be deterministic and non-cumulative.
#[inline]
fn context_hash(c1: u8, c2: u8, c3: u8, c4: u8) -> usize {
    let mut h = FNV_OFFSET;
    h ^= c4 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c3 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c2 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= c1 as u32;
    h = h.wrapping_mul(FNV_PRIME);
    (h as usize) & HASH_MASK
}

/// Get context hash for position `pos` in data (pos >= CTX_ORDER).
#[inline]
fn hash_at(data: &[u8], pos: usize) -> usize {
    context_hash(data[pos - 1], data[pos - 2], data[pos - 3], data[pos - 4])
}

/// Forward LZP transform: replace matched sequences with escape tokens.
///
/// Returns None if the transform doesn't reduce size (shouldn't be applied).
///
/// Algorithm:
/// For each position `pos` (starting after CTX_ORDER bytes of context):
///   1. Compute context hash from preceding 4 bytes
///   2. Look up predicted position from hash table
///   3. If predicted position yields a match of >= MIN_MATCH bytes:
///      - Emit ESCAPE + encoded_length
///      - Update hash table for the FIRST position only (the context lookup position)
///      - Advance pos by match_len
///   4. Otherwise (literal):
///      - Update hash table for this position
///      - Emit the literal byte (with escape handling)
///      - Advance pos by 1
///
/// CRITICAL INVARIANT: Hash table is updated ONCE per context lookup,
/// at the position where the lookup occurred. This is the same for both
/// match and non-match cases.
pub fn preprocess(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 16 {
        return None;
    }

    let mut hash_table = vec![0u32; HASH_SIZE];
    let mut output = Vec::with_capacity(data.len());
    let len = data.len();

    // Copy the first CTX_ORDER bytes as literals (with escape handling).
    for &byte in data.iter().take(CTX_ORDER.min(len)) {
        emit_literal(&mut output, byte);
    }

    let mut pos = CTX_ORDER;

    while pos < len {
        let idx = hash_at(data, pos);
        let predicted_pos = hash_table[idx] as usize;

        // ALWAYS update hash table at this position (before deciding match/literal).
        // This is the KEY to keeping compressor and decompressor in sync:
        // one hash update per "step", regardless of match or literal.
        hash_table[idx] = pos as u32;

        // Try to match at predicted position.
        let match_len = if predicted_pos >= CTX_ORDER && predicted_pos < pos {
            compute_match_length(data, predicted_pos, pos, MAX_MATCH.min(len - pos))
        } else {
            0
        };

        if match_len >= MIN_MATCH {
            // Emit match token.
            let encoded_len = (match_len - MIN_MATCH + 1) as u8; // 1..=255
            output.push(ESCAPE);
            output.push(encoded_len);
            pos += match_len;
        } else {
            // Emit literal.
            emit_literal(&mut output, data[pos]);
            pos += 1;
        }
    }

    // Only return if we actually reduced size.
    if output.len() < data.len() {
        Some(output)
    } else {
        None
    }
}

/// Reverse LZP transform: expand escape tokens back to original bytes.
///
/// The decompressor rebuilds the identical hash table and knows where
/// each match came from. Same update rule: one hash update per step
/// (before each literal or match expansion).
pub fn reverse(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut hash_table = vec![0u32; HASH_SIZE];
    let mut output = Vec::with_capacity(data.len() * 2);
    let mut pos: usize = 0;
    let len = data.len();

    while pos < len {
        let out_pos = output.len();

        if data[pos] == ESCAPE {
            if pos + 1 >= len {
                // Truncated — emit as literal.
                output.push(ESCAPE);
                pos += 1;
                continue;
            }

            let next = data[pos + 1];
            if next == 0x00 {
                // Literal escape: output 0xFE byte.
                // Update hash table (mirroring compressor's "always update" before emit).
                if out_pos >= CTX_ORDER {
                    let idx = hash_at_output(&output, out_pos);
                    hash_table[idx] = out_pos as u32;
                }
                output.push(ESCAPE);
                pos += 2;
            } else {
                // Match token.
                let match_len = next as usize + MIN_MATCH - 1; // 4..=258

                // Update hash table (mirroring compressor's "always update" before match).
                if out_pos >= CTX_ORDER {
                    let idx = hash_at_output(&output, out_pos);
                    let predicted_pos = hash_table[idx] as usize;
                    hash_table[idx] = out_pos as u32;

                    // Copy match_len bytes from the predicted position.
                    for i in 0..match_len {
                        let src = predicted_pos + i;
                        let byte = if src < output.len() {
                            output[src]
                        } else {
                            0 // shouldn't happen in valid data
                        };
                        output.push(byte);
                    }
                } else {
                    // Not enough context (shouldn't happen in valid data).
                    output.extend(std::iter::repeat_n(0u8, match_len));
                }

                pos += 2;
            }
        } else {
            // Regular literal byte.
            // Update hash table (mirroring compressor's "always update" before literal).
            if out_pos >= CTX_ORDER {
                let idx = hash_at_output(&output, out_pos);
                hash_table[idx] = out_pos as u32;
            }
            output.push(data[pos]);
            pos += 1;
        }
    }

    output
}

/// Get context hash for position `pos` in the output buffer.
#[inline]
fn hash_at_output(output: &[u8], pos: usize) -> usize {
    context_hash(
        output[pos - 1],
        output[pos - 2],
        output[pos - 3],
        output[pos - 4],
    )
}

/// Compute how many bytes match starting from `predicted_pos` and `current_pos` in data.
#[inline]
fn compute_match_length(
    data: &[u8],
    predicted_pos: usize,
    current_pos: usize,
    max_len: usize,
) -> usize {
    let mut len = 0;
    while len < max_len
        && predicted_pos + len < current_pos // don't read past current position
        && data[predicted_pos + len] == data[current_pos + len]
    {
        len += 1;
    }
    len
}

/// Emit a literal byte to the output, escaping 0xFE.
#[inline]
fn emit_literal(output: &mut Vec<u8>, byte: u8) {
    if byte == ESCAPE {
        output.push(ESCAPE);
        output.push(0x00);
    } else {
        output.push(byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let data = b"Hello, World! Hello, World! Hello, World! Hello, World!";
        if let Some(compressed) = preprocess(data) {
            let decompressed = reverse(&compressed);
            assert_eq!(
                decompressed,
                data.to_vec(),
                "roundtrip failed for simple test"
            );
            assert!(
                compressed.len() < data.len(),
                "LZP should compress repetitive data: {} vs {}",
                compressed.len(),
                data.len()
            );
        }
    }

    #[test]
    fn roundtrip_with_escape_byte() {
        let mut data = vec![0u8; 100];
        data[10] = ESCAPE;
        data[20] = ESCAPE;
        data[30] = ESCAPE;
        for i in 0..50 {
            data[50 + i] = data[i];
        }
        if let Some(compressed) = preprocess(&data) {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, data, "roundtrip with escape byte failed");
        }
    }

    #[test]
    fn roundtrip_all_bytes() {
        let mut data: Vec<u8> = (0..=255).collect();
        data.extend_from_slice(&(0..=255).collect::<Vec<u8>>());
        if let Some(compressed) = preprocess(&data) {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, data, "all-bytes roundtrip failed");
        }
    }

    #[test]
    fn roundtrip_long_repetitive() {
        let data = "The quick brown fox jumps over the lazy dog. ".repeat(100);
        let data = data.as_bytes();
        if let Some(compressed) = preprocess(data) {
            let decompressed = reverse(&compressed);
            assert_eq!(
                decompressed,
                data.to_vec(),
                "long repetitive roundtrip failed"
            );
            let ratio = compressed.len() as f64 / data.len() as f64;
            assert!(
                ratio < 0.8,
                "LZP should compress repetitive text well: ratio={ratio:.3}"
            );
        } else {
            panic!("LZP should compress repetitive data");
        }
    }

    #[test]
    fn no_compression_for_random() {
        let mut data = vec![0u8; 1000];
        let mut rng: u32 = 12345;
        for byte in data.iter_mut() {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (rng >> 16) as u8;
        }
        let result = preprocess(&data);
        if let Some(compressed) = result {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, data, "random data roundtrip failed");
        }
    }

    #[test]
    fn empty_input() {
        assert_eq!(preprocess(b""), None);
        assert_eq!(reverse(b""), Vec::<u8>::new());
    }

    #[test]
    fn short_input() {
        assert_eq!(preprocess(b"hi"), None);
    }

    #[test]
    fn context_hash_deterministic() {
        let h1 = context_hash(10, 20, 30, 40);
        let h2 = context_hash(10, 20, 30, 40);
        assert_eq!(h1, h2, "hash must be deterministic");

        let h3 = context_hash(11, 20, 30, 40);
        assert_ne!(h1, h3, "different input should give different hash");
    }

    #[test]
    fn escape_literal_roundtrip() {
        let data = vec![ESCAPE; 50];
        let mut long_data = data.clone();
        long_data.extend_from_slice(&[0u8; 200]);
        long_data.extend_from_slice(&data);

        if let Some(compressed) = preprocess(&long_data) {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, long_data, "escape literal roundtrip failed");
        }
    }

    #[test]
    fn roundtrip_xml_like() {
        // XML-like content similar to enwik8
        let data = r#"<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Test Article</title>
    <text>This is a test article about something. This is a test article about something else.</text>
  </page>
  <page>
    <title>Another Article</title>
    <text>This is a test article about another topic. This is a test article about yet another topic.</text>
  </page>
</mediawiki>"#;
        let data = data.as_bytes();
        if let Some(compressed) = preprocess(data) {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, data.to_vec(), "XML-like roundtrip failed");
        }
    }

    #[test]
    fn roundtrip_1kb_repeated_pattern() {
        // A realistic test: 1KB repeated text with some variation
        let mut data = Vec::new();
        for i in 0..20 {
            data.extend_from_slice(
                format!("Line {}: The quick brown fox jumps over the lazy dog.\n", i).as_bytes(),
            );
        }
        if let Some(compressed) = preprocess(&data) {
            let decompressed = reverse(&compressed);
            assert_eq!(decompressed, data, "1KB repeated pattern roundtrip failed");
        }
    }
}
