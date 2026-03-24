//! Format detection and preprocessing pipeline.
//!
//! Phase 0: heuristic detection.
//! Phase 1: format-aware preprocessing (JSON key interning) + detection.

pub mod json;
pub mod json_array;
pub mod ndjson;
pub mod transform;
pub mod value_dict;

use crate::dcx::{FormatHint, Mode};
use transform::{
    TRANSFORM_JSON_ARRAY_COLUMNAR, TRANSFORM_JSON_KEY_INTERN,
    TRANSFORM_NDJSON_COLUMNAR, TRANSFORM_VALUE_DICT, TransformChain,
};

/// Detect file format from content bytes.
pub fn detect_format(data: &[u8]) -> FormatHint {
    if data.is_empty() {
        return FormatHint::Generic;
    }

    let trimmed = trim_leading_whitespace(data);

    if starts_with_byte(trimmed, b'{') || starts_with_byte(trimmed, b'[') {
        if is_ndjson(data) {
            return FormatHint::Ndjson;
        }
        return FormatHint::Json;
    }

    FormatHint::Generic
}

/// Detect format from file extension (fallback).
pub fn detect_from_extension(path: &str) -> Option<FormatHint> {
    let ext = path.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "json" => Some(FormatHint::Json),
        "ndjson" | "jsonl" => Some(FormatHint::Ndjson),
        _ => None,
    }
}

/// Apply format-aware preprocessing transforms.
/// Returns (preprocessed_data, transform_chain).
///
/// NDJSON columnar: ALL modes (grouping similar values helps both zstd and CM).
/// Key interning: Balanced/Max only (hurts Fast mode due to zstd redundancy).
/// For NDJSON, columnar is applied FIRST — if it succeeds, key interning is skipped
/// (keys are already removed from the data stream by the columnar transform).
pub fn preprocess(data: &[u8], format: FormatHint, mode: Mode) -> (Vec<u8>, TransformChain) {
    let mut chain = TransformChain::new();
    let mut current = data.to_vec();

    // Track whether a columnar transform was applied (for value dict chaining).
    let mut columnar_applied = false;

    // NDJSON columnar reorg: ALL modes (dramatic improvement for uniform NDJSON).
    if format == FormatHint::Ndjson {
        if let Some(result) = ndjson::preprocess(&current) {
            chain.push(TRANSFORM_NDJSON_COLUMNAR, result.metadata);
            current = result.data;
            columnar_applied = true;
        }
    }

    // JSON array columnar reorg: ALL modes.
    if !columnar_applied && format == FormatHint::Json {
        if let Some(result) = json_array::preprocess(&current) {
            chain.push(TRANSFORM_JSON_ARRAY_COLUMNAR, result.metadata);
            current = result.data;
            columnar_applied = true;
        }
    }

    // Value dictionary: chain AFTER any columnar transform.
    // Replaces repeated multi-byte values with single-byte codes.
    // Only applies to columnar data (uses \x00/\x01 separators).
    if columnar_applied {
        if let Some(result) = value_dict::preprocess(&current) {
            chain.push(TRANSFORM_VALUE_DICT, result.metadata);
            current = result.data;
        }
    }

    if columnar_applied {
        return (current, chain);
    }

    // JSON key interning: Balanced/Max only (hurts Fast mode due to zstd redundancy).
    if matches!(mode, Mode::Balanced | Mode::Max)
        && matches!(format, FormatHint::Json | FormatHint::Ndjson)
        && let Some(result) = json::preprocess(&current)
    {
        chain.push(TRANSFORM_JSON_KEY_INTERN, result.metadata);
        current = result.data;
    }

    (current, chain)
}

/// Reverse preprocessing transforms (applied in reverse order).
pub fn reverse_preprocess(data: &[u8], chain: &TransformChain) -> Vec<u8> {
    let mut current = data.to_vec();

    // Apply in reverse order.
    for record in chain.records.iter().rev() {
        match record.id {
            TRANSFORM_JSON_KEY_INTERN => {
                current = json::reverse(&current, &record.metadata);
            }
            TRANSFORM_NDJSON_COLUMNAR => {
                current = ndjson::reverse(&current, &record.metadata);
            }
            TRANSFORM_JSON_ARRAY_COLUMNAR => {
                current = json_array::reverse(&current, &record.metadata);
            }
            TRANSFORM_VALUE_DICT => {
                current = value_dict::reverse(&current, &record.metadata);
            }
            _ => {} // Unknown/legacy transform — skip.
        }
    }

    current
}

// --- Detection helpers (unchanged from Phase 0) ---

fn trim_leading_whitespace(data: &[u8]) -> &[u8] {
    let start = data
        .iter()
        .position(|&b| !b.is_ascii_whitespace())
        .unwrap_or(data.len());
    &data[start..]
}

fn starts_with_byte(data: &[u8], byte: u8) -> bool {
    data.first() == Some(&byte)
}

fn is_ndjson(data: &[u8]) -> bool {
    let mut json_lines = 0;
    let mut total_lines = 0;

    for line in data.split(|&b| b == b'\n') {
        let trimmed = trim_leading_whitespace(line);
        if trimmed.is_empty() {
            continue;
        }
        total_lines += 1;
        if starts_with_byte(trimmed, b'{') {
            json_lines += 1;
        }
    }

    total_lines >= 2 && json_lines as f64 / total_lines as f64 > 0.8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_json() {
        assert_eq!(detect_format(b"  {\"key\": \"value\"}"), FormatHint::Json);
        assert_eq!(detect_format(b"[1, 2, 3]"), FormatHint::Json);
    }

    #[test]
    fn detect_ndjson() {
        let data = b"{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n";
        assert_eq!(detect_format(data), FormatHint::Ndjson);
    }

    #[test]
    fn detect_generic_fallback() {
        assert_eq!(detect_format(b""), FormatHint::Generic);
        assert_eq!(detect_format(b"just some random text"), FormatHint::Generic);
    }

    #[test]
    fn extension_detection() {
        assert_eq!(detect_from_extension("test.json"), Some(FormatHint::Json));
        assert_eq!(
            detect_from_extension("data.ndjson"),
            Some(FormatHint::Ndjson)
        );
        assert_eq!(detect_from_extension("file.txt"), None);
    }

    #[test]
    fn preprocess_json_key_interning() {
        let data = br#"{"name":"Alice","age":30,"name":"Bob","age":25}"#;
        let (preprocessed, chain) = preprocess(data, FormatHint::Json, Mode::Balanced);
        assert!(!chain.is_empty(), "should have applied key interning");
        assert!(
            preprocessed.len() < data.len(),
            "preprocessed should be smaller"
        );

        // Reverse and verify.
        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn preprocess_ndjson_columnar() {
        let data = br#"{"ts":"a","val":1}
{"ts":"b","val":2}
{"ts":"c","val":3}
"#;
        let (preprocessed, chain) = preprocess(data, FormatHint::Ndjson, Mode::Balanced);
        assert!(!chain.is_empty());
        // Should use columnar transform (ID 2), not key interning.
        assert_eq!(
            chain.records[0].id,
            transform::TRANSFORM_NDJSON_COLUMNAR,
            "NDJSON should use columnar transform"
        );

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn preprocess_ndjson_columnar_fast_mode() {
        // Columnar should apply for ALL modes, including Fast.
        let data = br#"{"ts":"a","val":1}
{"ts":"b","val":2}
{"ts":"c","val":3}
"#;
        let (preprocessed, chain) = preprocess(data, FormatHint::Ndjson, Mode::Fast);
        assert!(!chain.is_empty());
        assert_eq!(chain.records[0].id, transform::TRANSFORM_NDJSON_COLUMNAR);

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());

        // Verify columnar data groups values.
        let cols: Vec<&[u8]> = preprocessed.split(|&b| b == 0x00).collect();
        assert_eq!(cols.len(), 2, "should have 2 columns");
    }

    #[test]
    fn preprocess_json_array_columnar() {
        let data = br#"{"data": [{"id": 1, "type": "a"}, {"id": 2, "type": "b"}, {"id": 3, "type": "c"}, {"id": 4, "type": "d"}, {"id": 5, "type": "e"}], "meta": {"count": 5}}"#;
        let (preprocessed, chain) = preprocess(data, FormatHint::Json, Mode::Balanced);
        assert!(!chain.is_empty());
        assert_eq!(
            chain.records[0].id,
            transform::TRANSFORM_JSON_ARRAY_COLUMNAR,
            "JSON with array should use array columnar transform"
        );

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn preprocess_json_array_too_few_falls_through() {
        // Only 3 elements — below MIN_ROWS, should fall through to key interning.
        let data = br#"{"data": [{"id": 1, "type": "a"}, {"id": 2, "type": "a"}, {"id": 3, "type": "a"}], "meta": {"count": 3}, "data2": [{"id": 1, "type": "a"}, {"id": 2, "type": "a"}, {"id": 3, "type": "a"}]}"#;
        let (preprocessed, chain) = preprocess(data, FormatHint::Json, Mode::Balanced);
        // Should fall through to key interning (not array columnar).
        if !chain.is_empty() {
            assert_ne!(
                chain.records[0].id,
                transform::TRANSFORM_JSON_ARRAY_COLUMNAR,
                "3 elements should NOT trigger array columnar"
            );
        }

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn preprocess_non_json_passthrough() {
        let data = b"just some plain text with no JSON keys";
        let (preprocessed, chain) = preprocess(data, FormatHint::Generic, Mode::Fast);
        assert!(chain.is_empty());
        assert_eq!(preprocessed, data.to_vec());
    }
}
