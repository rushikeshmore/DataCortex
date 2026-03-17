//! Log file columnar reorg — lossless transform that reorders row-oriented
//! log data into column-oriented layout.
//!
//! Row-oriented (before):
//!   2026-03-15T10:30:00.001Z INFO  [datacortex::server] Starting server v0.3.0
//!   2026-03-15T10:30:00.002Z INFO  [datacortex::server] Listening on 0.0.0.0:8080
//!   2026-03-15T10:30:00.003Z DEBUG [datacortex::format] Detecting format...
//!
//! Column-oriented (after):
//!   [timestamps] 2026-03-15T10:30:00.001Z\x012026-03-15T10:30:00.002Z\x01...\x00
//!   [levels]     INFO \x01INFO \x01DEBUG\x01...\x00
//!   [modules]    datacortex::server\x01datacortex::server\x01datacortex::format\x01...\x00
//!   [messages]   Starting server v0.3.0\x01Listening on 0.0.0.0:8080\x01Detecting format...
//!
//! When similar values are adjacent, both LZ (zstd) and CM compress dramatically better.
//! Timestamps differ by milliseconds -> nearly identical strings grouped together.
//! Log levels are ~5 distinct values -> near-zero entropy when grouped.
//! Module paths repeat constantly -> excellent repetition when grouped.
//!
//! Separators:
//!   \x00 = column separator
//!   \x01 = value separator within a column
//!
//! The template captures the fixed structural separators between fields.
//! Variable-width whitespace (e.g. padded levels) is stored as part of the
//! level column values, preserving exact formatting.

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;

/// Known log levels (uppercase).
const LOG_LEVELS: &[&str] = &[
    "TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL",
];

/// A parsed log line with its structural components.
///
/// Reconstruction pattern:
///   timestamp + sep_ts_level + level_with_pad + "[" + module + "]" + sep_bracket_msg + message
///
/// Where:
///   - `sep_ts_level`: fixed whitespace between timestamp and level (e.g., " ")
///   - `level_with_pad`: the level keyword + any trailing whitespace up to `[`
///     (e.g., "INFO " or "DEBUG" or "WARN ")
///   - `sep_bracket_msg`: fixed whitespace between `]` and message (e.g., " ")
#[derive(Debug)]
struct ParsedLogLine<'a> {
    timestamp: &'a [u8],
    /// The level keyword + any trailing whitespace before '['.
    /// e.g., "INFO ", "INFO  ", "DEBUG ", "WARN  ", "ERROR "
    level_with_pad: &'a [u8],
    module: &'a [u8],
    message: &'a [u8],
    /// Whitespace between timestamp end and level start.
    sep_ts_level: &'a [u8],
    /// Whitespace between ']' and message start.
    sep_bracket_msg: &'a [u8],
}

/// Try to parse a log line into structured components.
///
/// Supports these common patterns:
///   TIMESTAMP LEVEL [MODULE] MESSAGE
///   TIMESTAMP LEVEL  [MODULE] MESSAGE  (padded levels)
///
/// Timestamp patterns recognized:
///   - ISO 8601: `2026-03-15T10:30:00.001Z`
///   - ISO 8601 with space: `2026-03-15 10:30:00.001`
///   - ISO 8601 with timezone offset: `2026-03-15T10:30:00.001+05:30`
///
/// Returns None if the line doesn't match the expected pattern.
fn parse_log_line(line: &[u8]) -> Option<ParsedLogLine<'_>> {
    if line.len() < 20 {
        return None;
    }

    // Step 1: Find timestamp end.
    let ts_end = find_timestamp_end(line)?;
    let timestamp = &line[..ts_end];

    // Step 2: Skip whitespace between timestamp and level.
    let mut pos = ts_end;
    let sep_ts_start = pos;
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= line.len() || pos == sep_ts_start {
        return None;
    }
    let sep_ts_level = &line[sep_ts_start..pos];

    // Step 3: Match log level keyword.
    let level_kw_start = pos;
    let level_kw_end = find_level_end(line, pos)?;
    pos = level_kw_end;

    // Step 4: Capture level + trailing whitespace up to '['.
    // The level_with_pad includes the keyword and any padding spaces.
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= line.len() || line[pos] != b'[' {
        return None;
    }
    let level_with_pad = &line[level_kw_start..pos];
    pos += 1; // skip '['

    // Step 5: Find module (content inside brackets).
    let module_start = pos;
    while pos < line.len() && line[pos] != b']' {
        pos += 1;
    }
    if pos >= line.len() {
        return None;
    }
    let module = &line[module_start..pos];
    pos += 1; // skip ']'

    // Step 6: Whitespace between ']' and message.
    let sep_bm_start = pos;
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let sep_bracket_msg = &line[sep_bm_start..pos];

    let message = &line[pos..];

    Some(ParsedLogLine {
        timestamp,
        level_with_pad,
        module,
        message,
        sep_ts_level,
        sep_bracket_msg,
    })
}

/// Find the end of the timestamp in a log line.
///
/// Scans from the start looking for an ISO 8601-like timestamp.
/// Returns the byte position after the timestamp ends.
fn find_timestamp_end(line: &[u8]) -> Option<usize> {
    if line.len() < 10 {
        return None;
    }
    // Check YYYY-MM-DD.
    if !line[0..4].iter().all(|b| b.is_ascii_digit())
        || line[4] != b'-'
        || !line[5..7].iter().all(|b| b.is_ascii_digit())
        || line[7] != b'-'
        || !line[8..10].iter().all(|b| b.is_ascii_digit())
    {
        return None;
    }

    let mut pos = 10;
    if pos >= line.len() {
        return None;
    }
    if line[pos] != b'T' && line[pos] != b' ' {
        return None;
    }

    // For space separator, we need to check it's followed by time digits (HH:),
    // otherwise the space might be the separator between date and level.
    if line[pos] == b' ' {
        // Check if next chars look like HH:MM
        if pos + 5 < line.len()
            && line[pos + 1].is_ascii_digit()
            && line[pos + 2].is_ascii_digit()
            && line[pos + 3] == b':'
        {
            // It's a time part.
        } else {
            // Space is the separator, not part of timestamp.
            return Some(pos);
        }
    }

    pos += 1;

    // Scan through time portion: HH:MM:SS[.fractional][Z|+HH:MM|-HH:MM]
    while pos < line.len() {
        match line[pos] {
            b'0'..=b'9' | b':' | b'.' => pos += 1,
            b'Z' => {
                pos += 1;
                break;
            }
            b'+' | b'-' if pos > 19 => {
                // Timezone offset like +05:30 or -08:00.
                pos += 1;
                while pos < line.len() && (line[pos].is_ascii_digit() || line[pos] == b':') {
                    pos += 1;
                }
                break;
            }
            _ => break,
        }
    }

    // Sanity check: timestamp should be at least 19 chars (YYYY-MM-DDTHH:MM:SS).
    if pos < 19 {
        return None;
    }

    Some(pos)
}

/// Find the end of a log level keyword starting at `start`.
/// Returns the position after the level keyword, or None if not a valid level.
fn find_level_end(line: &[u8], start: usize) -> Option<usize> {
    let mut end = start;
    while end < line.len() && !line[end].is_ascii_whitespace() && line[end] != b'[' {
        end += 1;
    }
    if end == start {
        return None;
    }

    let word = std::str::from_utf8(&line[start..end]).ok()?;
    let upper = word.to_uppercase();
    if LOG_LEVELS.iter().any(|&lvl| upper == lvl) {
        Some(end)
    } else {
        None
    }
}

/// Forward transform: log file columnar reorg.
///
/// Returns None if the data is not suitable (not a structured log file,
/// inconsistent format, too few lines, etc).
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    // Split into lines.
    let mut lines: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            lines.push(&data[start..i]);
            start = i + 1;
        }
    }
    if start < data.len() {
        lines.push(&data[start..]);
    }

    // Filter out empty lines.
    let non_empty: Vec<&[u8]> = lines.iter().copied().filter(|l| !l.is_empty()).collect();
    if non_empty.len() < 5 {
        return None;
    }

    // Parse all lines — all must match the pattern.
    let mut parsed: Vec<ParsedLogLine<'_>> = Vec::with_capacity(non_empty.len());
    for &line in &non_empty {
        match parse_log_line(line) {
            Some(p) => parsed.push(p),
            None => return None,
        }
    }

    // Verify the two fixed separators are consistent across all lines.
    // sep_ts_level (between timestamp and level) and sep_bracket_msg (between ] and message)
    // must be the same for all lines. level_with_pad can vary (different levels).
    let ref_sep_ts = parsed[0].sep_ts_level;
    let ref_sep_bm = parsed[0].sep_bracket_msg;
    for p in &parsed[1..] {
        if p.sep_ts_level != ref_sep_ts || p.sep_bracket_msg != ref_sep_bm {
            return None;
        }
    }

    let num_rows = parsed.len();
    let num_cols: u16 = 4; // timestamp, level_with_pad, module, message

    // Build column data: 4 columns separated by \x00.
    let mut col_data = Vec::with_capacity(data.len());

    // Column 0: Timestamps.
    for (i, p) in parsed.iter().enumerate() {
        col_data.extend_from_slice(p.timestamp);
        if i < num_rows - 1 {
            col_data.push(VAL_SEP);
        }
    }
    col_data.push(COL_SEP);

    // Column 1: Levels (with padding whitespace).
    for (i, p) in parsed.iter().enumerate() {
        col_data.extend_from_slice(p.level_with_pad);
        if i < num_rows - 1 {
            col_data.push(VAL_SEP);
        }
    }
    col_data.push(COL_SEP);

    // Column 2: Modules.
    for (i, p) in parsed.iter().enumerate() {
        col_data.extend_from_slice(p.module);
        if i < num_rows - 1 {
            col_data.push(VAL_SEP);
        }
    }
    col_data.push(COL_SEP);

    // Column 3: Messages.
    for (i, p) in parsed.iter().enumerate() {
        col_data.extend_from_slice(p.message);
        if i < num_rows - 1 {
            col_data.push(VAL_SEP);
        }
    }

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&num_cols.to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    // Store the two fixed separator templates.
    // sep_ts_level: whitespace between timestamp and level.
    metadata.extend_from_slice(&(ref_sep_ts.len() as u16).to_le_bytes());
    metadata.extend_from_slice(ref_sep_ts);
    // sep_bracket_msg: whitespace between ']' and message.
    metadata.extend_from_slice(&(ref_sep_bm.len() as u16).to_le_bytes());
    metadata.extend_from_slice(ref_sep_bm);

    Some(TransformResult {
        data: col_data,
        metadata,
    })
}

/// Reverse transform: reconstruct log file from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 10 {
        return data.to_vec();
    }

    let mut pos = 0;
    let _version = metadata[pos];
    pos += 1;
    let num_rows = u32::from_le_bytes(metadata[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let num_cols = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let has_trailing_newline = metadata[pos] != 0;
    pos += 1;

    if num_cols != 4 || num_rows == 0 {
        return data.to_vec();
    }

    // Read separator templates.
    let mut seps: Vec<Vec<u8>> = Vec::new();
    for _ in 0..2 {
        if pos + 2 > metadata.len() {
            return data.to_vec();
        }
        let sep_len = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + sep_len > metadata.len() {
            return data.to_vec();
        }
        seps.push(metadata[pos..pos + sep_len].to_vec());
        pos += sep_len;
    }

    let sep_ts_level = &seps[0]; // between timestamp and level
    let sep_bracket_msg = &seps[1]; // between ']' and message

    // Parse column data: split by \x00 into 4 columns.
    let col_chunks: Vec<&[u8]> = data.split(|&b| b == COL_SEP).collect();
    if col_chunks.len() != 4 {
        return data.to_vec();
    }

    let mut columns: Vec<Vec<&[u8]>> = Vec::with_capacity(4);
    for chunk in &col_chunks {
        let vals: Vec<&[u8]> = chunk.split(|&b| b == VAL_SEP).collect();
        if vals.len() != num_rows {
            return data.to_vec();
        }
        columns.push(vals);
    }

    // Reconstruct each line:
    //   timestamp + sep_ts_level + level_with_pad + "[" + module + "]" + sep_bracket_msg + message
    let mut output = Vec::with_capacity(data.len() * 2);

    #[allow(clippy::needless_range_loop)]
    for row in 0..num_rows {
        output.extend_from_slice(columns[0][row]); // timestamp
        output.extend_from_slice(sep_ts_level); // " "
        output.extend_from_slice(columns[1][row]); // level_with_pad (e.g., "INFO " or "DEBUG")
        output.push(b'['); // opening bracket
        output.extend_from_slice(columns[2][row]); // module
        output.push(b']'); // closing bracket
        output.extend_from_slice(sep_bracket_msg); // " "
        output.extend_from_slice(columns[3][row]); // message

        if row < num_rows - 1 || has_trailing_newline {
            output.push(b'\n');
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_log_line() {
        let line = b"2026-03-15T10:30:00.001Z INFO  [datacortex::server] Starting server v0.3.0";
        let parsed = parse_log_line(line).expect("should parse");
        assert_eq!(parsed.timestamp, b"2026-03-15T10:30:00.001Z");
        assert_eq!(parsed.level_with_pad, b"INFO  ");
        assert_eq!(parsed.module, b"datacortex::server");
        assert_eq!(parsed.message, b"Starting server v0.3.0");
    }

    #[test]
    fn parse_debug_level() {
        let line = b"2026-03-15T10:30:01.235Z DEBUG [datacortex::format] Detecting format for 1048576 bytes...";
        let parsed = parse_log_line(line).expect("should parse");
        assert_eq!(parsed.level_with_pad, b"DEBUG "); // 5-char level + 1 space
        assert_eq!(parsed.module, b"datacortex::format");
    }

    #[test]
    fn parse_error_level() {
        let line = b"2026-03-15T10:34:12.345Z ERROR [datacortex::handler] POST /compress client=10.0.0.51 error=\"too large\"";
        let parsed = parse_log_line(line).expect("should parse");
        assert_eq!(parsed.level_with_pad, b"ERROR "); // 5-char level + 1 space
    }

    #[test]
    fn parse_warn_level() {
        let line = b"2026-03-15T10:30:04.200Z WARN  [datacortex::handler] POST /compress client=192.168.1.102 size=0 error=\"empty input\"";
        let parsed = parse_log_line(line).expect("should parse");
        assert_eq!(parsed.level_with_pad, b"WARN  "); // 4-char level + 2 spaces (padded to 5+1)
    }

    #[test]
    fn parse_no_timestamp_returns_none() {
        let line = b"Just some random text without a timestamp";
        assert!(parse_log_line(line).is_none());
    }

    #[test]
    fn parse_no_level_returns_none() {
        let line = b"2026-03-15T10:30:00.001Z [datacortex::server] message";
        assert!(parse_log_line(line).is_none());
    }

    #[test]
    fn parse_no_module_returns_none() {
        let line = b"2026-03-15T10:30:00.001Z INFO message without module";
        assert!(parse_log_line(line).is_none());
    }

    #[test]
    fn roundtrip_simple() {
        let data = b"2026-03-15T10:30:00.001Z INFO  [datacortex::server] Starting server v0.3.0\n\
2026-03-15T10:30:00.002Z INFO  [datacortex::server] Listening on 0.0.0.0:8080\n\
2026-03-15T10:30:00.003Z INFO  [datacortex::config] Configuration loaded\n\
2026-03-15T10:30:00.003Z INFO  [datacortex::config] mode: balanced\n\
2026-03-15T10:30:00.003Z INFO  [datacortex::config] max_memory_mb: 4096\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_no_trailing_newline() {
        let data = b"2026-03-15T10:30:00.001Z INFO [datacortex::server] Starting server\n\
2026-03-15T10:30:00.002Z INFO [datacortex::server] Listening\n\
2026-03-15T10:30:00.003Z DEBUG [datacortex::format] Detecting\n\
2026-03-15T10:30:00.004Z WARN [datacortex::handler] Warning\n\
2026-03-15T10:30:00.005Z ERROR [datacortex::handler] Error occurred";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_mixed_levels() {
        let data = b"2026-03-15T10:30:00.001Z INFO [app::server] Server started\n\
2026-03-15T10:30:01.234Z DEBUG [app::handler] Processing request\n\
2026-03-15T10:30:01.235Z DEBUG [app::format] Format detected: JSON\n\
2026-03-15T10:30:01.890Z INFO [app::handler] Request completed\n\
2026-03-15T10:30:02.001Z WARN [app::handler] Slow query detected\n\
2026-03-15T10:30:02.500Z ERROR [app::handler] Connection refused\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_padded_levels() {
        // Levels padded to 5 chars + space (common in Rust tracing/env_logger).
        let data = b"2026-03-15T10:30:00.001Z INFO  [app::server] Server started\n\
2026-03-15T10:30:01.234Z DEBUG [app::handler] Processing request\n\
2026-03-15T10:30:01.235Z DEBUG [app::format] Format detected: JSON\n\
2026-03-15T10:30:01.890Z INFO  [app::handler] Request completed\n\
2026-03-15T10:30:02.001Z WARN  [app::handler] Slow query detected\n\
2026-03-15T10:30:02.500Z ERROR [app::handler] Connection refused\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn column_layout_groups_similar_values() {
        let data = b"2026-03-15T10:30:00.001Z INFO [app::server] msg1\n\
2026-03-15T10:30:00.002Z INFO [app::server] msg2\n\
2026-03-15T10:30:00.003Z INFO [app::server] msg3\n\
2026-03-15T10:30:00.004Z INFO [app::server] msg4\n\
2026-03-15T10:30:00.005Z INFO [app::server] msg5\n";

        let result = preprocess(data).unwrap();
        let cols: Vec<&[u8]> = result.data.split(|&b| b == COL_SEP).collect();
        assert_eq!(cols.len(), 4);

        // Column 0 = timestamps (nearly identical, differ by last digit).
        let timestamps: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(timestamps.len(), 5);
        assert_eq!(timestamps[0], b"2026-03-15T10:30:00.001Z");
        assert_eq!(timestamps[4], b"2026-03-15T10:30:00.005Z");

        // Column 1 = levels (all "INFO ").
        let levels: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(levels.len(), 5);
        assert!(levels.iter().all(|&l| l == b"INFO "));

        // Column 2 = modules (all "app::server").
        let modules: Vec<&[u8]> = cols[2].split(|&b| b == VAL_SEP).collect();
        assert_eq!(modules.len(), 5);
        assert!(modules.iter().all(|&m| m == b"app::server"));
    }

    #[test]
    fn too_few_lines_returns_none() {
        let data = b"2026-03-15T10:30:00.001Z INFO [app::server] msg1\n\
2026-03-15T10:30:00.002Z INFO [app::server] msg2\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn roundtrip_with_timestamp_space_separator() {
        // Some logs use space instead of T between date and time.
        let data = b"2026-03-15 10:30:00.001 INFO [app::server] msg1\n\
2026-03-15 10:30:00.002 INFO [app::server] msg2\n\
2026-03-15 10:30:00.003 DEBUG [app::format] msg3\n\
2026-03-15 10:30:00.004 WARN [app::handler] msg4\n\
2026-03-15 10:30:00.005 ERROR [app::handler] msg5\n";

        let result = preprocess(data).expect("should parse space-separated timestamps");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_large_log() {
        // Generate a realistic log with many lines — all single-space after level.
        let mut log = String::new();
        let modules = [
            "app::server",
            "app::handler",
            "app::format",
            "app::engine",
            "app::mixer",
        ];
        let levels = ["INFO", "DEBUG", "WARN", "ERROR"];
        for i in 0..200 {
            let ms = format!("{:03}", i % 1000);
            let sec = format!("{:02}", (i / 1000) % 60);
            let module = modules[i % modules.len()];
            let level = levels[i % levels.len()];
            log.push_str(&format!(
                "2026-03-15T10:30:{sec}.{ms}Z {level} [{module}] Request {i} processed in {}ms\n",
                i * 3 + 17
            ));
        }
        let data = log.as_bytes();
        let result = preprocess(data).expect("should parse large log");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }
}
