//! Format detection and preprocessing pipeline.
//!
//! Phase 0: heuristic detection.
//! Phase 1: format-aware preprocessing (JSON key interning) + detection.

pub mod json;
pub mod json_array;
pub mod ndjson;
pub mod schema;
pub mod transform;
pub mod typed_encoding;
pub mod value_dict;

use crate::dcx::{FormatHint, Mode};
use transform::{
    TRANSFORM_JSON_ARRAY_COLUMNAR, TRANSFORM_JSON_KEY_INTERN, TRANSFORM_NDJSON_COLUMNAR,
    TRANSFORM_NESTED_FLATTEN, TRANSFORM_TYPED_ENCODING, TRANSFORM_VALUE_DICT, TransformChain,
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

    // Track whether a uniform columnar transform was applied (for value dict chaining).
    // Uniform columnar = data is \x00/\x01-separated, downstream transforms are compatible.
    let mut columnar_applied = false;
    // Track whether ANY ndjson transform was applied (uniform or grouped).
    let mut ndjson_transform_applied = false;

    // NDJSON columnar reorg: ALL modes (dramatic improvement for uniform NDJSON).
    // Strategy 1 (uniform, version=1) produces \x00/\x01 separated columnar data.
    // Strategy 2 (grouped, version=2) produces a different format with per-group data.
    // Only Strategy 1 output is compatible with downstream typed_encoding/value_dict.
    if format == FormatHint::Ndjson {
        if let Some(result) = ndjson::preprocess(&current) {
            let is_uniform_columnar = !result.metadata.is_empty() && result.metadata[0] == 1;
            chain.push(TRANSFORM_NDJSON_COLUMNAR, result.metadata);
            current = result.data;
            ndjson_transform_applied = true;
            columnar_applied = is_uniform_columnar;
        }
    }

    // JSON array columnar reorg: ALL modes.
    // Strategy 1 (uniform, version=1) produces \x00/\x01 separated columnar data.
    // Strategy 2 (grouped, version=2) produces a different format with per-group data.
    // Only Strategy 1 output is compatible with downstream typed_encoding/value_dict/nested_flatten.
    let mut json_array_applied = false;
    if !columnar_applied && !ndjson_transform_applied && format == FormatHint::Json {
        if let Some(result) = json_array::preprocess(&current) {
            let is_uniform = !result.metadata.is_empty() && result.metadata[0] == 1;
            chain.push(TRANSFORM_JSON_ARRAY_COLUMNAR, result.metadata);
            current = result.data;
            json_array_applied = true;
            columnar_applied = is_uniform;
        }
    }

    // Nested flatten: decompose nested objects into sub-columns.
    // Works on any \x00/\x01 columnar data. Only for non-NDJSON paths because
    // the NDJSON uniform path already handles its own nested flatten internally.
    if columnar_applied && !ndjson_transform_applied {
        // Extract num_rows from the json_array metadata (offset 1, u32 LE).
        let ja_meta = &chain.records.last().unwrap().metadata;
        if ja_meta.len() >= 5 {
            let num_rows = u32::from_le_bytes(ja_meta[1..5].try_into().unwrap()) as usize;
            if let Some((flat_data, nested_groups)) =
                ndjson::flatten_nested_columns(&current, num_rows)
            {
                // Build metadata: num_rows + total_flat_cols + serialized nested info.
                let total_flat_cols = flat_data.split(|&b| b == 0x00).count() as u16;

                // Verify roundtrip: unflatten must produce the exact original columnar
                // data. Nested objects with varying sub-key sets or key ordering can
                // cause the compact reconstruction to reorder keys, breaking byte-exact
                // roundtrip. Only apply if the unflatten is provably lossless.
                let unflattened = ndjson::unflatten_nested_columns(
                    &flat_data,
                    &nested_groups,
                    num_rows,
                    total_flat_cols as usize,
                );
                if unflattened == current {
                    let mut nested_meta = Vec::new();
                    nested_meta.extend_from_slice(&(num_rows as u32).to_le_bytes());
                    nested_meta.extend_from_slice(&total_flat_cols.to_le_bytes());
                    nested_meta
                        .extend_from_slice(&ndjson::serialize_nested_info(&nested_groups));
                    chain.push(TRANSFORM_NESTED_FLATTEN, nested_meta);
                    current = flat_data;
                }
                // else: roundtrip not exact — skip nested flatten (data stays columnar
                // without sub-column decomposition, still benefits from typed encoding
                // and value dict on the outer columns).
            }
        }
    }

    // Typed encoding: Fast mode ONLY. CM mode doesn't benefit (gotcha #35 confirmed).
    // Binary encoding disrupts CM's learned text patterns. But zstd benefits from
    // smaller raw data (delta varints, boolean bitmaps).
    if columnar_applied && mode == Mode::Fast {
        if let Some(result) = typed_encoding::preprocess(&current) {
            chain.push(TRANSFORM_TYPED_ENCODING, result.metadata);
            current = result.data;
        }
    }

    // Value dictionary: chain AFTER any columnar transform.
    // Replaces repeated multi-byte values with single-byte codes.
    // Only applies to columnar data (uses \x00/\x01 separators).
    // NOTE: value dict only operates on \x00/\x01-separated data.
    // If typed encoding was applied, the data is now binary (no separators),
    // so value dict will naturally not apply (it won't find separators to split on,
    // or its size check will fail).
    if columnar_applied {
        if let Some(result) = value_dict::preprocess(&current) {
            chain.push(TRANSFORM_VALUE_DICT, result.metadata);
            current = result.data;
        }
    }

    if columnar_applied || ndjson_transform_applied || json_array_applied {
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
            TRANSFORM_TYPED_ENCODING => {
                current = typed_encoding::reverse(&current, &record.metadata);
            }
            TRANSFORM_NESTED_FLATTEN => {
                // Metadata: num_rows (u32 LE) + total_flat_cols (u16 LE) + nested_info.
                if record.metadata.len() >= 6 {
                    let num_rows =
                        u32::from_le_bytes(record.metadata[0..4].try_into().unwrap()) as usize;
                    let total_flat_cols =
                        u16::from_le_bytes(record.metadata[4..6].try_into().unwrap()) as usize;
                    if let Some((nested_groups, _)) =
                        ndjson::deserialize_nested_info(&record.metadata[6..])
                    {
                        current = ndjson::unflatten_nested_columns(
                            &current,
                            &nested_groups,
                            num_rows,
                            total_flat_cols,
                        );
                    }
                }
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

    #[test]
    fn test_json_array_nested_flatten_roundtrip() {
        // JSON array with nested objects — should apply json_array columnar + nested flatten.
        let mut json = String::from(r#"{"data": ["#);
        for i in 0..10 {
            if i > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!(
                r#"{{"id": {}, "name": "item_{}", "meta": {{"score": {}, "active": {}, "tag": "t{}"}}}}"#,
                i, i, i * 10, if i % 2 == 0 { "true" } else { "false" }, i
            ));
        }
        json.push_str(r#"], "total": 10}"#);

        let data = json.as_bytes();
        let (preprocessed, chain) = preprocess(data, FormatHint::Json, Mode::Fast);
        assert!(!chain.is_empty());
        assert_eq!(
            chain.records[0].id,
            transform::TRANSFORM_JSON_ARRAY_COLUMNAR,
            "should apply json_array columnar first"
        );

        // Check that nested flatten was applied.
        let has_nested_flatten = chain
            .records
            .iter()
            .any(|r| r.id == transform::TRANSFORM_NESTED_FLATTEN);
        assert!(
            has_nested_flatten,
            "should apply nested flatten for objects with nested fields"
        );

        // Verify byte-exact roundtrip.
        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_json_array_nested_flatten_improves_ratio() {
        // Build a dataset where nested flatten demonstrably helps:
        // many rows with a nested object having repeated/similar values.
        let mut json = String::from(r#"{"items": ["#);
        for i in 0..50 {
            if i > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!(
                r#"{{"id": {}, "user": {{"name": "user_{}", "role": "admin", "level": {}, "verified": true, "email": "user_{}@test.com"}}}}"#,
                i, i, i % 5, i
            ));
        }
        json.push_str(r#"]}"#);

        let data = json.as_bytes();

        // Preprocess WITH nested flatten (current code).
        let (preprocessed_with, chain_with) = preprocess(data, FormatHint::Json, Mode::Fast);
        assert!(
            chain_with
                .records
                .iter()
                .any(|r| r.id == transform::TRANSFORM_NESTED_FLATTEN),
            "nested flatten should activate"
        );

        // Verify roundtrip.
        let restored = reverse_preprocess(&preprocessed_with, &chain_with);
        assert_eq!(restored, data.to_vec());

        // The preprocessed data should have more columns (sub-columns from nested objects).
        let num_cols_with = preprocessed_with.split(|&b| b == 0x00).count();
        // Without nested flatten, json_array produces 2 columns (id, user).
        // With nested flatten, user is decomposed into 5 sub-columns, so total = 1 + 5 = 6.
        assert!(
            num_cols_with > 2,
            "nested flatten should produce more columns: got {}",
            num_cols_with
        );
    }

    #[test]
    fn test_ndjson_unaffected() {
        // NDJSON with nested objects — should use NDJSON path, NOT the standalone nested flatten.
        let mut ndjson = String::new();
        for i in 0..10 {
            ndjson.push_str(&format!(
                r#"{{"id":{},"user":{{"name":"u{}","level":{}}}}}"#,
                i,
                i,
                i % 3
            ));
            ndjson.push('\n');
        }

        let data = ndjson.as_bytes();
        let (preprocessed, chain) = preprocess(data, FormatHint::Ndjson, Mode::Fast);
        assert!(!chain.is_empty());
        assert_eq!(
            chain.records[0].id,
            transform::TRANSFORM_NDJSON_COLUMNAR,
            "NDJSON should use its own columnar transform"
        );

        // Should NOT have standalone TRANSFORM_NESTED_FLATTEN in chain.
        let has_standalone_nested = chain
            .records
            .iter()
            .any(|r| r.id == transform::TRANSFORM_NESTED_FLATTEN);
        assert!(
            !has_standalone_nested,
            "NDJSON path should NOT use standalone nested flatten (it handles nesting internally)"
        );

        // Verify roundtrip.
        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_ndjson_large_delta_integer_roundtrip() {
        // Regression: NDJSON with integers spanning the epoch-timestamp range
        // (e.g. 2147483647 = i32::MAX) caused schema misclassification and
        // CRC-32 mismatch on decompression in Fast mode.
        let edges: &[i64] = &[
            0, -1, 1, -2147483648, 2147483647, -9007199254740991, 9007199254740991,
        ];
        let mut ndjson = String::new();
        for i in 0..203 {
            ndjson.push_str(&format!(
                "{{\"val\":{},\"idx\":{}}}\n",
                edges[i % 7],
                i
            ));
        }

        let data = ndjson.as_bytes();

        // Full pipeline roundtrip (ndjson columnar + typed encoding in Fast mode).
        let (preprocessed, chain) = preprocess(data, FormatHint::Ndjson, Mode::Fast);

        // Verify typed encoding was applied.
        assert!(
            chain
                .records
                .iter()
                .any(|r| r.id == transform::TRANSFORM_TYPED_ENCODING),
            "typed encoding should be applied in Fast mode"
        );

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec(), "byte-exact roundtrip failed");
    }

    #[test]
    fn test_nested_flatten_varying_subkeys_roundtrip() {
        // Regression test for npm_search.json roundtrip bug:
        // JSON array of uniform objects where nested dicts have VARYING sub-keys
        // across rows (e.g., some rows have "license", some don't; "links" has
        // 5 different schemas). The nested flatten must verify its roundtrip is
        // byte-exact before applying, because compact reconstruction reorders
        // keys to discovery order instead of preserving the original order.
        let mut json = String::from(r#"{"objects":["#);
        for i in 0..250 {
            if i > 0 {
                json.push(',');
            }
            // Nested dict with optional key (missing for first 6 rows)
            let license = if i >= 6 {
                r#","license":"MIT""#
            } else {
                ""
            };
            // Nested dict with varying key sets across rows
            let links = match i % 5 {
                0 => format!(r#"{{"homepage":"h{i}","repository":"r{i}","bugs":"b{i}","npm":"n{i}"}}"#),
                1 => format!(r#"{{"homepage":"h{i}","npm":"n{i}","repository":"r{i}"}}"#),
                2 => format!(r#"{{"npm":"n{i}"}}"#),
                3 => format!(r#"{{"bugs":"b{i}","homepage":"h{i}","npm":"n{i}"}}"#),
                _ => format!(r#"{{"npm":"n{i}","repository":"r{i}"}}"#),
            };
            let publisher = if i % 3 == 0 {
                format!(r#"{{"email":"u{i}@t.com","username":"u{i}","actor":"a{i}"}}"#)
            } else {
                format!(r#"{{"email":"u{i}@t.com","username":"u{i}"}}"#)
            };
            json.push_str(&format!(
                r#"{{"dl":{{"m":{},"w":{}}},"dep":"{}","sc":{},"pkg":{{"name":"p{i}","kw":["j","t"],"ver":"{i}.0","pub":{publisher},"mnt":[{{"u":"u{i}"}}]{license},"links":{links}}},"score":{{"f":0.5,"d":{{"q":0.8}}}},"flags":{{"x":0}}}}"#,
                1000 * (i + 1),
                250 * (i + 1),
                i * 5,
                1697.0894 + i as f64 * 0.1,
            ));
        }
        json.push_str(r#"],"total":250}"#);

        let data = json.as_bytes();

        for mode in [Mode::Fast, Mode::Balanced] {
            let (preprocessed, chain) = preprocess(data, FormatHint::Json, mode);
            assert!(!chain.is_empty(), "should apply transforms in {mode} mode");
            let restored = reverse_preprocess(&preprocessed, &chain);
            assert_eq!(
                restored.len(),
                data.len(),
                "length mismatch in {mode} mode",
            );
            assert_eq!(restored, data.to_vec(), "roundtrip failed in {mode} mode");
        }
    }
}
