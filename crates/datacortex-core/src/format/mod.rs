//! Format detection — identify file type from content.
//!
//! Phase 0: heuristic detection from first bytes + structure.
//! Phase 1+: full format-aware preprocessing pipelines.

use crate::dcx::FormatHint;

/// Detect file format from content bytes.
pub fn detect_format(data: &[u8]) -> FormatHint {
    if data.is_empty() {
        return FormatHint::Generic;
    }

    // Skip leading whitespace for structural detection.
    let trimmed = trim_leading_whitespace(data);

    // JSON: starts with { or [
    if starts_with_byte(trimmed, b'{') || starts_with_byte(trimmed, b'[') {
        // Check if NDJSON: multiple JSON objects separated by newlines.
        if is_ndjson(data) {
            return FormatHint::Ndjson;
        }
        return FormatHint::Json;
    }

    // Markdown: starts with # or has markdown indicators.
    if starts_with_byte(trimmed, b'#') || has_markdown_indicators(data) {
        return FormatHint::Markdown;
    }

    // CSV: consistent comma-separated lines with similar column counts.
    if is_csv(data) {
        return FormatHint::Csv;
    }

    // Log files: lines start with timestamps or log levels.
    if is_log(data) {
        return FormatHint::Log;
    }

    // Source code: common keywords and syntax patterns.
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

/// NDJSON: multiple lines, each starting with { (after optional whitespace).
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

/// Markdown: look for heading markers, links, emphasis.
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

/// CSV: consistent column counts across lines with comma or tab separators.
fn is_csv(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(4096)];
    let lines: Vec<&[u8]> = sample
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .take(10)
        .collect();

    if lines.len() < 3 {
        return false;
    }

    // Check comma-separated consistency.
    let counts: Vec<usize> = lines
        .iter()
        .map(|l| l.iter().filter(|&&b| b == b',').count())
        .collect();
    let first = counts[0];
    first >= 2 && counts.iter().all(|&c| c == first)
}

/// Log: lines start with timestamps or log level keywords.
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

/// Check if line starts with something that looks like a timestamp.
fn looks_like_timestamp_prefix(line: &str) -> bool {
    let bytes = line.as_bytes();
    // 2024-01-15 or [2024-01-15 patterns
    if bytes.len() >= 10 {
        let start = if bytes[0] == b'[' { 1 } else { 0 };
        if start + 10 <= bytes.len() {
            let s = &bytes[start..start + 10];
            return s[4] == b'-' && s[7] == b'-' && s[0..4].iter().all(|b| b.is_ascii_digit());
        }
    }
    false
}

/// Source code: look for common syntax patterns.
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
    // Braces density check (code has lots of { })
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
}
