//! Format detection and preprocessing pipeline.
//!
//! Phase 0: heuristic detection.
//! Phase 1: format-aware preprocessing (JSON key interning) + detection.

pub mod csv;
pub mod json;
pub mod lzp;
pub mod ndjson;
pub mod transform;
pub mod wrt;

use crate::dcx::{FormatHint, Mode};
use transform::{
    TRANSFORM_CSV_COLUMNAR, TRANSFORM_JSON_KEY_INTERN, TRANSFORM_LZP, TRANSFORM_NDJSON_COLUMNAR,
    TRANSFORM_WRT, TransformChain,
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

    if starts_with_byte(trimmed, b'#') || has_markdown_indicators(data) {
        return FormatHint::Markdown;
    }

    if is_csv(data) {
        return FormatHint::Csv;
    }

    if is_log(data) {
        return FormatHint::Log;
    }

    if is_code(data) {
        return FormatHint::Code;
    }

    FormatHint::Generic
}

/// Detect format from file extension (fallback).
pub fn detect_from_extension(path: &str) -> Option<FormatHint> {
    let ext = path.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "json" => Some(FormatHint::Json),
        "md" | "markdown" => Some(FormatHint::Markdown),
        "ndjson" | "jsonl" => Some(FormatHint::Ndjson),
        "csv" | "tsv" => Some(FormatHint::Csv),
        "log" => Some(FormatHint::Log),
        "rs" | "py" | "js" | "ts" | "go" | "c" | "cpp" | "h" | "java" | "rb" | "zig" => {
            Some(FormatHint::Code)
        }
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

    // NDJSON columnar reorg: ALL modes (dramatic improvement for uniform NDJSON).
    if format == FormatHint::Ndjson {
        if let Some(result) = ndjson::preprocess(&current) {
            chain.push(TRANSFORM_NDJSON_COLUMNAR, result.metadata);
            current = result.data;
            // Keys are gone from the data stream — skip key interning.
            return (current, chain);
        }
    }

    // CSV columnar reorg: ALL modes (dramatic improvement for tabular CSV data).
    if format == FormatHint::Csv {
        if let Some(result) = csv::preprocess(&current) {
            chain.push(TRANSFORM_CSV_COLUMNAR, result.metadata);
            current = result.data;
            return (current, chain);
        }
    }

    // JSON key interning: Balanced/Max only (hurts Fast mode due to zstd redundancy).
    if matches!(mode, Mode::Balanced | Mode::Max)
        && matches!(format, FormatHint::Json | FormatHint::Ndjson)
        && let Some(result) = json::preprocess(&current)
    {
        chain.push(TRANSFORM_JSON_KEY_INTERN, result.metadata);
        current = result.data;
    }

    // WRT (Word Reduce Transform): disabled for now.
    // Testing showed WRT regresses on XML-heavy content (enwik8: 2.14 -> 2.18 bpb)
    // because WRT codes break natural byte patterns that order models exploit,
    // and confuse the XML state tracker. Keep byte_class fix for future use,
    // but don't apply WRT in preprocessing.
    // TODO: Re-enable WRT only for pure text (Markdown, Log) after per-format A/B test.

    // LZP (Lempel-Ziv Prediction): DISABLED.
    // A/B testing showed LZP HURTS context mixing (2.00 -> 2.20 bpb on enwik8 10MB).
    // The CM engine's match model already captures the same redundancy more efficiently.
    // LZP escape tokens break context models and remove bytes that order models learn from.
    // Keep the code for potential future use with different coders (BWT, Fast mode).

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
            TRANSFORM_CSV_COLUMNAR => {
                current = csv::reverse(&current, &record.metadata);
            }
            TRANSFORM_WRT => {
                current = wrt::reverse(&current);
            }
            TRANSFORM_LZP => {
                current = lzp::reverse(&current);
            }
            _ => {} // Unknown transform — skip.
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

fn has_markdown_indicators(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(4096)];
    let text = String::from_utf8_lossy(sample);

    let mut indicators = 0;
    if text.contains("\n# ") || text.contains("\n## ") || text.contains("\n### ") {
        indicators += 2;
    }
    if text.contains("](") || text.contains("![") {
        indicators += 1;
    }
    if text.contains("```") || text.contains("**") || text.contains("__") {
        indicators += 1;
    }
    if text.contains("\n- ") || text.contains("\n* ") {
        indicators += 1;
    }

    indicators >= 2
}

fn is_csv(data: &[u8]) -> bool {
    // Delegate to the CSV module's delimiter detection, which handles
    // quoted fields with embedded delimiters correctly.
    csv::detect_csv(data)
}

fn is_log(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(4096)];
    let text = String::from_utf8_lossy(sample);
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).take(10).collect();

    if lines.len() < 3 {
        return false;
    }

    let mut log_lines = 0;
    for line in &lines {
        let upper = line.to_uppercase();
        if upper.contains("[INFO]")
            || upper.contains("[ERROR]")
            || upper.contains("[WARN]")
            || upper.contains("[DEBUG]")
            || upper.contains("INFO ")
            || upper.contains("ERROR ")
            || upper.contains("WARN ")
            || upper.contains("DEBUG ")
            || looks_like_timestamp_prefix(line)
        {
            log_lines += 1;
        }
    }

    log_lines as f64 / lines.len() as f64 > 0.5
}

fn looks_like_timestamp_prefix(line: &str) -> bool {
    let bytes = line.as_bytes();
    if bytes.len() >= 10 {
        let start = if bytes[0] == b'[' { 1 } else { 0 };
        if start + 10 <= bytes.len() {
            let s = &bytes[start..start + 10];
            return s[4] == b'-' && s[7] == b'-' && s[0..4].iter().all(|b| b.is_ascii_digit());
        }
    }
    false
}

fn is_code(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(4096)];
    let text = String::from_utf8_lossy(sample);

    let mut indicators = 0;
    if text.contains("fn ")
        || text.contains("func ")
        || text.contains("function ")
        || text.contains("def ")
    {
        indicators += 2;
    }
    if text.contains("pub ") || text.contains("private ") || text.contains("public ") {
        indicators += 1;
    }
    if text.contains("struct ") || text.contains("class ") || text.contains("impl ") {
        indicators += 1;
    }
    if text.contains("use ") || text.contains("import ") || text.contains("#include") {
        indicators += 1;
    }
    if text.contains("return ") || text.contains("if ") || text.contains("for ") {
        indicators += 1;
    }
    let brace_count = text.matches('{').count() + text.matches('}').count();
    if brace_count > 10 {
        indicators += 1;
    }

    indicators >= 3
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
    fn detect_markdown() {
        let data = b"# Title\n\n## Section\n\nSome text with **bold** and [links](url).\n\n- item 1\n- item 2\n";
        assert_eq!(detect_format(data), FormatHint::Markdown);
    }

    #[test]
    fn detect_log() {
        let data = b"2024-01-15 10:30:00 INFO Starting server\n2024-01-15 10:30:01 DEBUG Listening on :8080\n2024-01-15 10:30:02 ERROR Connection refused\n";
        assert_eq!(detect_format(data), FormatHint::Log);
    }

    #[test]
    fn detect_code() {
        let data = b"use std::io;\n\nfn main() {\n    let x = 42;\n    if x > 0 {\n        println!(\"positive\");\n    }\n    return;\n}\n\nstruct Foo {\n    bar: i32,\n}\n\nimpl Foo {\n    pub fn new() -> Self { Foo { bar: 0 } }\n}\n";
        assert_eq!(detect_format(data), FormatHint::Code);
    }

    #[test]
    fn detect_csv() {
        let data = b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCharlie,35,LA\n";
        assert_eq!(detect_format(data), FormatHint::Csv);
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
            detect_from_extension("README.md"),
            Some(FormatHint::Markdown)
        );
        assert_eq!(
            detect_from_extension("data.ndjson"),
            Some(FormatHint::Ndjson)
        );
        assert_eq!(detect_from_extension("data.csv"), Some(FormatHint::Csv));
        assert_eq!(detect_from_extension("server.log"), Some(FormatHint::Log));
        assert_eq!(detect_from_extension("main.rs"), Some(FormatHint::Code));
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
    fn preprocess_csv_columnar() {
        let data =
            b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\nDave,28,CHI\nEve,32,BOS\n";
        let (preprocessed, chain) = preprocess(data, FormatHint::Csv, Mode::Balanced);
        assert!(!chain.is_empty());
        assert_eq!(
            chain.records[0].id,
            transform::TRANSFORM_CSV_COLUMNAR,
            "CSV should use columnar transform"
        );

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn preprocess_csv_columnar_fast_mode() {
        // Columnar should apply for ALL modes, including Fast.
        let data =
            b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\nDave,28,CHI\nEve,32,BOS\n";
        let (preprocessed, chain) = preprocess(data, FormatHint::Csv, Mode::Fast);
        assert!(!chain.is_empty());
        assert_eq!(chain.records[0].id, transform::TRANSFORM_CSV_COLUMNAR);

        let restored = reverse_preprocess(&preprocessed, &chain);
        assert_eq!(restored, data.to_vec());

        // Verify columnar data groups values.
        let cols: Vec<&[u8]> = preprocessed.split(|&b| b == 0x00).collect();
        assert_eq!(cols.len(), 3, "should have 3 columns");
    }

    #[test]
    fn preprocess_non_json_passthrough() {
        let data = b"just some plain text with no JSON keys";
        let (preprocessed, chain) = preprocess(data, FormatHint::Generic, Mode::Fast);
        assert!(chain.is_empty());
        assert_eq!(preprocessed, data.to_vec());
    }
}
