//! WRT — Word Reduce Transform for text preprocessing.
//!
//! Replaces common English words (with trailing space) with single-byte codes
//! (128-255), effectively expanding the effective context order for CM compression.
//! With WRT, order-7 sees ~7 words of context instead of 7 characters.
//!
//! - Forward: scan input, match words greedily (longest first), replace with codes.
//! - Escape: bytes 128-254 in original data become 0xFF + original_byte.
//!   Original 0xFF becomes 0xFF 0xFF.
//! - Reverse: dictionary bytes (128-254) expand to words. 0xFF + byte = literal.
//!
//! Dictionary version 1: top 128 common English words with trailing space.
//! The dictionary is fixed in code — no metadata needed beyond version byte.

/// Escape byte: marks a literal high byte in the WRT stream.
const ESCAPE: u8 = 0xFF;

/// Dictionary version (stored in transform metadata for forward compat).
pub const WRT_VERSION: u8 = 1;

/// Number of dictionary entries (128 words, mapped to bytes 128-254).
/// Byte 255 (0xFF) is reserved as the escape marker.
const DICT_SIZE: usize = 127;

/// Dictionary: top 127 common English words WITH trailing space.
/// Sorted by frequency. Mapped to bytes 128..254.
/// These are case-sensitive, matching lowercase forms.
///
/// Sources: Brown corpus, Gutenberg top-1000, enwik8 word frequency.
/// Selected to maximize coverage on English text while keeping trailing space
/// for clean word-boundary matching.
const DICTIONARY: [&str; DICT_SIZE] = [
    "the ",     // 0x80 = 128
    "of ",      // 0x81
    "and ",     // 0x82
    "in ",      // 0x83
    "to ",      // 0x84
    "a ",       // 0x85
    "is ",      // 0x86
    "was ",     // 0x87
    "it ",      // 0x88
    "for ",     // 0x89
    "that ",    // 0x8A
    "as ",      // 0x8B
    "on ",      // 0x8C
    "with ",    // 0x8D
    "he ",      // 0x8E
    "be ",      // 0x8F
    "at ",      // 0x90
    "by ",      // 0x91
    "this ",    // 0x92
    "had ",     // 0x93
    "not ",     // 0x94
    "are ",     // 0x95
    "but ",     // 0x96
    "from ",    // 0x97
    "or ",      // 0x98
    "have ",    // 0x99
    "an ",      // 0x9A
    "they ",    // 0x9B
    "which ",   // 0x9C
    "one ",     // 0x9D
    "you ",     // 0x9E
    "were ",    // 0x9F
    "her ",     // 0xA0
    "all ",     // 0xA1
    "she ",     // 0xA2
    "there ",   // 0xA3
    "would ",   // 0xA4
    "their ",   // 0xA5
    "we ",      // 0xA6
    "him ",     // 0xA7
    "been ",    // 0xA8
    "has ",     // 0xA9
    "when ",    // 0xAA
    "who ",     // 0xAB
    "will ",    // 0xAC
    "no ",      // 0xAD
    "more ",    // 0xAE
    "if ",      // 0xAF
    "out ",     // 0xB0
    "so ",      // 0xB1
    "up ",      // 0xB2
    "said ",    // 0xB3
    "what ",    // 0xB4
    "its ",     // 0xB5
    "about ",   // 0xB6
    "than ",    // 0xB7
    "into ",    // 0xB8
    "them ",    // 0xB9
    "can ",     // 0xBA
    "only ",    // 0xBB
    "other ",   // 0xBC
    "new ",     // 0xBD
    "some ",    // 0xBE
    "could ",   // 0xBF
    "time ",    // 0xC0
    "these ",   // 0xC1
    "two ",     // 0xC2
    "may ",     // 0xC3
    "then ",    // 0xC4
    "do ",      // 0xC5
    "first ",   // 0xC6
    "any ",     // 0xC7
    "my ",      // 0xC8
    "now ",     // 0xC9
    "such ",    // 0xCA
    "like ",    // 0xCB
    "our ",     // 0xCC
    "over ",    // 0xCD
    "man ",     // 0xCE
    "me ",      // 0xCF
    "even ",    // 0xD0
    "most ",    // 0xD1
    "made ",    // 0xD2
    "after ",   // 0xD3
    "also ",    // 0xD4
    "did ",     // 0xD5
    "many ",    // 0xD6
    "before ",  // 0xD7
    "must ",    // 0xD8
    "through ", // 0xD9
    "back ",    // 0xDA
    "years ",   // 0xDB
    "where ",   // 0xDC
    "much ",    // 0xDD
    "your ",    // 0xDE
    "way ",     // 0xDF
    "well ",    // 0xE0
    "down ",    // 0xE1
    "should ",  // 0xE2
    "because ", // 0xE3
    "each ",    // 0xE4
    "just ",    // 0xE5
    "those ",   // 0xE6
    "people ",  // 0xE7
    "how ",     // 0xE8
    "too ",     // 0xE9
    "little ",  // 0xEA
    "state ",   // 0xEB
    "good ",    // 0xEC
    "very ",    // 0xED
    "make ",    // 0xEE
    "world ",   // 0xEF
    "still ",   // 0xF0
    "own ",     // 0xF1
    "see ",     // 0xF2
    "men ",     // 0xF3
    "work ",    // 0xF4
    "long ",    // 0xF5
    "get ",     // 0xF6
    "here ",    // 0xF7
    "between ", // 0xF8
    "both ",    // 0xF9
    "life ",    // 0xFA
    "being ",   // 0xFB
    "under ",   // 0xFC
    "never ",   // 0xFD
    "day ",     // 0xFE
];

/// Maximum word length in the dictionary (including trailing space).
/// Used to limit the lookahead window during forward transform.
const MAX_WORD_LEN: usize = 8; // "through " and "because " and "between " are 8

/// Forward WRT transform: replace common words with single-byte codes.
///
/// Returns the transformed data. The dictionary version byte is stored
/// as transform metadata (1 byte) by the caller.
pub fn forward(input: &[u8]) -> Vec<u8> {
    // Pre-estimate output size: usually slightly smaller than input.
    let mut output = Vec::with_capacity(input.len());
    let mut pos = 0;
    let len = input.len();

    while pos < len {
        let byte = input[pos];

        // If current byte is ASCII lowercase or start of a potential word,
        // try to match a dictionary word (greedy longest match).
        if byte.is_ascii_lowercase() {
            if let Some((code, word_len)) = match_word(input, pos) {
                output.push(code);
                pos += word_len;
                continue;
            }
        }

        // No word match. Emit the byte, escaping high bytes.
        if byte >= 128 {
            output.push(ESCAPE);
            output.push(byte);
        } else {
            output.push(byte);
        }
        pos += 1;
    }

    output
}

/// Reverse WRT transform: expand single-byte codes back to words.
pub fn reverse(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() * 2);
    let mut pos = 0;
    let len = input.len();

    while pos < len {
        let byte = input[pos];

        if byte == ESCAPE {
            // Escaped literal: next byte is a literal high byte.
            pos += 1;
            if pos < len {
                output.push(input[pos]);
            }
            // If stream ends after escape, the escape is lost (corrupt data).
            // This shouldn't happen with valid WRT data.
        } else if byte >= 128 {
            // Dictionary code: expand to word.
            let idx = (byte - 128) as usize;
            if idx < DICT_SIZE {
                output.extend_from_slice(DICTIONARY[idx].as_bytes());
            } else {
                // Should not happen with valid data (byte 255 = ESCAPE handled above).
                // But defensive: pass through.
                output.push(byte);
            }
        } else {
            output.push(byte);
        }
        pos += 1;
    }

    output
}

/// Try to match a dictionary word at position `pos` in `input`.
/// Returns (code_byte, word_length) if a match is found.
/// Greedy: matches the LONGEST word first.
#[inline]
fn match_word(input: &[u8], pos: usize) -> Option<(u8, usize)> {
    let remaining = input.len() - pos;
    let max_check = remaining.min(MAX_WORD_LEN);

    // Try longest words first for greedy matching.
    // We check each dictionary entry against the input at this position.
    // For speed, we use a simple scan — the dictionary is small (127 entries).
    let mut best: Option<(u8, usize)> = None;

    for (idx, &word) in DICTIONARY.iter().enumerate() {
        let word_bytes = word.as_bytes();
        let wlen = word_bytes.len();
        if wlen > max_check {
            continue;
        }
        // Check if input[pos..pos+wlen] matches the word.
        if &input[pos..pos + wlen] == word_bytes {
            match best {
                None => best = Some(((128 + idx) as u8, wlen)),
                Some((_, prev_len)) => {
                    if wlen > prev_len {
                        best = Some(((128 + idx) as u8, wlen));
                    }
                }
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_roundtrip() {
        let input = b"the quick brown fox";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "basic roundtrip failed");
    }

    #[test]
    fn multiple_words() {
        let input = b"the man and the woman were in the house";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // "the " appears 3 times, should be replaced
        assert!(encoded.len() < input.len(), "WRT should reduce size");
    }

    #[test]
    fn no_match_passthrough() {
        let input = b"xyz abc 123";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // No dictionary words, so output should be same as input.
        assert_eq!(encoded, input);
    }

    #[test]
    fn escape_high_bytes() {
        // Input containing bytes >= 128 must be escaped and roundtripped.
        let input: Vec<u8> = (0..=255).collect();
        let encoded = forward(&input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "high byte roundtrip failed");
    }

    #[test]
    fn escape_0xff() {
        let input = vec![0xFF, 0xFF, 0xFF];
        let encoded = forward(&input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "0xFF roundtrip failed");
    }

    #[test]
    fn empty_input() {
        let encoded = forward(b"");
        assert!(encoded.is_empty());
        let decoded = reverse(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn word_at_end_no_trailing_space() {
        // "the" without trailing space should NOT match "the " dictionary entry.
        let input = b"the";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // Should NOT be replaced (no trailing space).
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn word_at_end_with_trailing_space() {
        let input = b"the ";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // Should be replaced with 1 byte.
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0x80);
    }

    #[test]
    fn case_sensitive() {
        // "The " should NOT match (capital T).
        let input = b"The quick";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // "The " is not in dictionary, so it passes through.
    }

    #[test]
    fn greedy_longest_match() {
        // "through " (8 bytes) should match over "the " (4 bytes) if present.
        let input = b"through the ";
        let encoded = forward(input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input);
        // "through " = 1 byte, "the " = 1 byte => 2 bytes total
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn mixed_binary_and_text() {
        let mut input = Vec::new();
        input.extend_from_slice(b"the ");
        input.push(0x80); // high byte
        input.push(0xFF); // escape byte
        input.extend_from_slice(b"and ");
        input.push(200);
        let encoded = forward(&input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "mixed binary/text roundtrip failed");
    }

    #[test]
    fn all_dictionary_words_roundtrip() {
        // Build input with every dictionary word.
        let mut input = Vec::new();
        for word in &DICTIONARY {
            input.extend_from_slice(word.as_bytes());
        }
        let encoded = forward(&input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "all-words roundtrip failed");
        // Each word should be replaced by 1 byte.
        assert_eq!(
            encoded.len(),
            DICT_SIZE,
            "expected {} bytes, got {}",
            DICT_SIZE,
            encoded.len()
        );
    }

    #[test]
    fn random_binary_roundtrip() {
        // Generate pseudo-random binary data with all byte values.
        let mut input = Vec::with_capacity(10000);
        let mut seed: u32 = 12345;
        for _ in 0..10000 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            input.push((seed >> 16) as u8);
        }
        let encoded = forward(&input);
        let decoded = reverse(&encoded);
        assert_eq!(decoded, input, "random binary roundtrip failed");
    }

    #[test]
    fn enwik8_prefix_roundtrip() {
        // Test on real English text if available.
        let path = "../../benchmarks/corpus-large/enwik8";
        if let Ok(data) = std::fs::read(path) {
            let prefix = &data[..data.len().min(100_000)];
            let encoded = forward(prefix);
            let decoded = reverse(&encoded);
            assert_eq!(decoded, prefix, "enwik8 prefix roundtrip failed");
            // WRT should reduce size on English text.
            eprintln!(
                "WRT enwik8 100KB: {} -> {} bytes ({:.1}% of original)",
                prefix.len(),
                encoded.len(),
                encoded.len() as f64 / prefix.len() as f64 * 100.0
            );
        }
    }

    #[test]
    fn dictionary_entries_are_valid() {
        for (i, word) in DICTIONARY.iter().enumerate() {
            assert!(
                word.ends_with(' '),
                "dictionary entry {i} ('{}') must end with space",
                word
            );
            assert!(
                word.len() <= MAX_WORD_LEN,
                "dictionary entry {i} ('{}') exceeds MAX_WORD_LEN={MAX_WORD_LEN}",
                word
            );
            assert!(
                word.as_bytes().iter().all(|&b| b < 128),
                "dictionary entry {i} ('{}') contains non-ASCII",
                word
            );
        }
    }

    #[test]
    fn no_overlapping_codes() {
        // Ensure no two dictionary entries map to the same code.
        // This is guaranteed by construction (sequential mapping), but verify.
        let mut codes: std::collections::HashSet<u8> = std::collections::HashSet::new();
        for i in 0..DICT_SIZE {
            let code = 128 + i as u8;
            assert!(codes.insert(code), "duplicate code {code}");
            assert_ne!(code, ESCAPE, "code {code} collides with ESCAPE");
        }
    }
}
