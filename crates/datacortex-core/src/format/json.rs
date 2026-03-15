//! JSON key interning — replace repeated keys with short references.
//!
//! Forward: scan JSON, find key strings, replace with \x00 + index.
//! Reverse: scan for \x00 markers, expand from key dictionary.
//!
//! Escape scheme (null byte cannot appear in valid JSON):
//!   \x00 + idx (0..=252)     → key dictionary reference
//!   \x00 + 0xFD + u16 LE    → extended key reference (253..65535)
//!   \x00 + 0xFE              → literal null byte (escape)

use super::transform::TransformResult;
use std::collections::HashMap;

const ESCAPE: u8 = 0x00;
const ESCAPE_EXTENDED: u8 = 0xFD;
const ESCAPE_LITERAL: u8 = 0xFE;
const MAX_SHORT_INDEX: u8 = 0xFC;

/// A key occurrence found during scanning.
struct KeyOccurrence {
    start: usize, // position of opening quote
    end: usize,   // position after closing quote
    content: Vec<u8>,
}

/// Scan JSON bytes and find all key string positions.
/// A key is a quoted string followed by ':' (after optional whitespace).
fn find_keys(data: &[u8]) -> Vec<KeyOccurrence> {
    let mut keys = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if data[pos] == b'"' {
            let start = pos;
            pos += 1;
            let mut content = Vec::new();
            let mut escaped = false;

            while pos < data.len() {
                if escaped {
                    content.push(data[pos]);
                    escaped = false;
                } else if data[pos] == b'\\' {
                    content.push(data[pos]);
                    escaped = true;
                } else if data[pos] == b'"' {
                    pos += 1;
                    break;
                } else {
                    content.push(data[pos]);
                }
                pos += 1;
            }

            let end = pos;

            // Check if followed by ':' (after optional whitespace) → key.
            let mut check = pos;
            while check < data.len() && data[check].is_ascii_whitespace() {
                check += 1;
            }
            if check < data.len() && data[check] == b':' {
                keys.push(KeyOccurrence {
                    start,
                    end,
                    content,
                });
            }
        } else {
            pos += 1;
        }
    }

    keys
}

/// Build frequency-sorted key dictionary. Only includes keys appearing > 1 time.
fn build_dictionary(keys: &[KeyOccurrence]) -> Vec<Vec<u8>> {
    let mut freq: HashMap<Vec<u8>, usize> = HashMap::new();
    for k in keys {
        *freq.entry(k.content.clone()).or_default() += 1;
    }

    let mut entries: Vec<(Vec<u8>, usize)> =
        freq.into_iter().filter(|(_, count)| *count > 1).collect();

    // Most frequent first = smallest index.
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    entries.into_iter().map(|(k, _)| k).collect()
}

/// Forward transform: intern repeated JSON keys.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    let keys = find_keys(data);
    if keys.is_empty() {
        return None;
    }

    let dict = build_dictionary(&keys);
    if dict.is_empty() {
        return None;
    }

    let lookup: HashMap<&[u8], usize> = dict
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_slice(), i))
        .collect();

    let mut output = Vec::with_capacity(data.len());
    let mut last_end = 0;

    for key in &keys {
        if let Some(&idx) = lookup.get(key.content.as_slice()) {
            // Copy bytes before this key, escaping null bytes.
            escape_copy(&data[last_end..key.start], &mut output);

            // Write key reference.
            output.push(ESCAPE);
            if idx <= MAX_SHORT_INDEX as usize {
                output.push(idx as u8);
            } else {
                output.push(ESCAPE_EXTENDED);
                output.extend_from_slice(&(idx as u16).to_le_bytes());
            }

            last_end = key.end;
        } else {
            // Key not in dictionary — copy verbatim with null escaping.
            escape_copy(&data[last_end..key.end], &mut output);
            last_end = key.end;
        }
    }

    // Copy remaining bytes.
    escape_copy(&data[last_end..], &mut output);

    // Only apply if preprocessed data is smaller (metadata stored separately in header).
    let metadata = serialize_dict(&dict);
    if output.len() >= data.len() {
        return None;
    }

    Some(TransformResult {
        data: output,
        metadata,
    })
}

/// Reverse transform: expand key references back to original strings.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    let dict = deserialize_dict(metadata);
    let mut output = Vec::with_capacity(data.len() * 2);
    let mut pos = 0;

    while pos < data.len() {
        if data[pos] == ESCAPE {
            pos += 1;
            if pos >= data.len() {
                break;
            }
            match data[pos] {
                ESCAPE_LITERAL => {
                    output.push(ESCAPE);
                    pos += 1;
                }
                ESCAPE_EXTENDED => {
                    pos += 1;
                    if pos + 2 <= data.len() {
                        let idx = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                        pos += 2;
                        if idx < dict.len() {
                            output.push(b'"');
                            output.extend_from_slice(&dict[idx]);
                            output.push(b'"');
                        }
                    }
                }
                idx if idx <= MAX_SHORT_INDEX => {
                    let idx = idx as usize;
                    if idx < dict.len() {
                        output.push(b'"');
                        output.extend_from_slice(&dict[idx]);
                        output.push(b'"');
                    }
                    pos += 1;
                }
                _ => {
                    // Unknown escape — pass through.
                    pos += 1;
                }
            }
        } else {
            output.push(data[pos]);
            pos += 1;
        }
    }

    output
}

/// Copy bytes, escaping null bytes as \x00\xFE.
fn escape_copy(src: &[u8], dst: &mut Vec<u8>) {
    for &b in src {
        if b == ESCAPE {
            dst.push(ESCAPE);
            dst.push(ESCAPE_LITERAL);
        } else {
            dst.push(b);
        }
    }
}

fn serialize_dict(dict: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1); // version
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    for key in dict {
        out.extend_from_slice(&(key.len() as u16).to_le_bytes());
        out.extend_from_slice(key);
    }
    out
}

fn deserialize_dict(data: &[u8]) -> Vec<Vec<u8>> {
    if data.len() < 3 {
        return vec![];
    }
    let mut pos = 0;
    let _version = data[pos];
    pos += 1;
    let num = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
    pos += 2;
    let mut dict = Vec::with_capacity(num);
    for _ in 0..num {
        if pos + 2 > data.len() {
            break;
        }
        let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + len > data.len() {
            break;
        }
        dict.push(data[pos..pos + len].to_vec());
        pos += len;
    }
    dict
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_keys_simple() {
        let data = br#"{"name": "Alice", "age": 30}"#;
        let keys = find_keys(data);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].content, b"name");
        assert_eq!(keys[1].content, b"age");
    }

    #[test]
    fn find_keys_nested() {
        let data = br#"{"a": {"b": 1, "c": 2}, "a": {"b": 3}}"#;
        let keys = find_keys(data);
        // Keys: "a", "b", "c", "a", "b"
        assert_eq!(keys.len(), 5);
    }

    #[test]
    fn find_keys_escaped_quotes() {
        let data = br#"{"key\"name": "val"}"#;
        let keys = find_keys(data);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].content, br#"key\"name"#.to_vec());
    }

    #[test]
    fn roundtrip_simple() {
        let data = br#"{"name": "Alice", "age": 30, "name": "Bob", "age": 25}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_nested() {
        let data = br#"{"id": 1, "data": {"id": 2, "type": "x"}, "id": 3, "type": "y"}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_ndjson_lines() {
        let data = br#"{"ts":"a","val":1}
{"ts":"b","val":2}
{"ts":"c","val":3}
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn no_transform_unique_keys() {
        let data = br#"{"a": 1, "b": 2, "c": 3}"#;
        assert!(
            preprocess(data).is_none(),
            "unique keys should not be interned"
        );
    }

    #[test]
    fn dict_roundtrip() {
        let dict = vec![b"name".to_vec(), b"age".to_vec(), b"city".to_vec()];
        let serialized = serialize_dict(&dict);
        let deserialized = deserialize_dict(&serialized);
        assert_eq!(deserialized, dict);
    }

    #[test]
    fn size_reduction() {
        let data = br#"{"name":"Alice","age":30,"name":"Bob","age":25,"name":"Carol","age":35}"#;
        let result = preprocess(data).expect("should produce transform");
        // Interned data should be smaller than original.
        assert!(
            result.data.len() + result.metadata.len() < data.len(),
            "interned={} + meta={} should be < original={}",
            result.data.len(),
            result.metadata.len(),
            data.len()
        );
    }
}
