//! NDJSON columnar reorg — lossless transform that reorders row-oriented
//! NDJSON data into column-oriented layout.
//!
//! Row-oriented (before):
//!   {"ts":"2026-03-15T10:30:00.001Z","type":"page_view","user":"usr_a1b2c3d4"}
//!   {"ts":"2026-03-15T10:30:00.234Z","type":"api_call","user":"usr_a1b2c3d4"}
//!
//! Column-oriented (after):
//!   [ts values] "2026-03-15T10:30:00.001Z" \x01 "2026-03-15T10:30:00.234Z" \x00
//!   [type values] "page_view" \x01 "api_call" \x00
//!   [user values] "usr_a1b2c3d4" \x01 "usr_a1b2c3d4"
//!
//! When similar values are adjacent, both LZ (zstd) and CM compress dramatically better.
//!
//! Separators:
//!   \x00 = column separator (cannot appear in valid JSON text)
//!   \x01 = value separator within a column (cannot appear in valid JSON)
//!
//! Metadata stores the template parts for reconstruction.

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;

/// Extract the raw value bytes from a JSON line at a given position.
/// `pos` should point to the first byte of the value (after `:`).
/// Returns (value_bytes, end_position).
///
/// Handles: strings, numbers, booleans, null, nested objects, nested arrays.
fn extract_value(line: &[u8], mut pos: usize) -> Option<(Vec<u8>, usize)> {
    // Skip whitespace after colon.
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= line.len() {
        return None;
    }

    let start = pos;
    match line[pos] {
        b'"' => {
            // String value — scan to closing unescaped quote.
            pos += 1;
            let mut escaped = false;
            while pos < line.len() {
                if escaped {
                    escaped = false;
                } else if line[pos] == b'\\' {
                    escaped = true;
                } else if line[pos] == b'"' {
                    pos += 1;
                    return Some((line[start..pos].to_vec(), pos));
                }
                pos += 1;
            }
            None // Unterminated string.
        }
        b'{' => {
            // Nested object — match braces, respecting strings.
            let mut depth = 1;
            pos += 1;
            while pos < line.len() && depth > 0 {
                match line[pos] {
                    b'"' => {
                        // Skip over string contents.
                        pos += 1;
                        let mut escaped = false;
                        while pos < line.len() {
                            if escaped {
                                escaped = false;
                            } else if line[pos] == b'\\' {
                                escaped = true;
                            } else if line[pos] == b'"' {
                                break;
                            }
                            pos += 1;
                        }
                    }
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                pos += 1;
            }
            Some((line[start..pos].to_vec(), pos))
        }
        b'[' => {
            // Nested array — match brackets, respecting strings.
            let mut depth = 1;
            pos += 1;
            while pos < line.len() && depth > 0 {
                match line[pos] {
                    b'"' => {
                        pos += 1;
                        let mut escaped = false;
                        while pos < line.len() {
                            if escaped {
                                escaped = false;
                            } else if line[pos] == b'\\' {
                                escaped = true;
                            } else if line[pos] == b'"' {
                                break;
                            }
                            pos += 1;
                        }
                    }
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    _ => {}
                }
                pos += 1;
            }
            Some((line[start..pos].to_vec(), pos))
        }
        _ => {
            // Number, boolean, null — scan until , or } or ] or whitespace.
            while pos < line.len() {
                match line[pos] {
                    b',' | b'}' | b']' => break,
                    _ if line[pos].is_ascii_whitespace() => break,
                    _ => pos += 1,
                }
            }
            if pos == start {
                None
            } else {
                Some((line[start..pos].to_vec(), pos))
            }
        }
    }
}

/// Parse a single JSON line into (template_parts, values).
///
/// Template parts are the structural bytes between values:
///   Part 0 = everything from { up to and including the : before value 0
///   Part 1 = everything from after value 0 up to and including : before value 1
///   ...
///   Part N = everything from after the last value to end of line (including } and \n)
///
/// Returns None if the line is not a flat-ish JSON object (we handle nested values
/// as opaque blobs, but the top-level structure must be key:value pairs).
type ParsedLine = (Vec<Vec<u8>>, Vec<Vec<u8>>);

fn parse_line(line: &[u8]) -> Option<ParsedLine> {
    let mut pos = 0;

    // Skip leading whitespace.
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= line.len() || line[pos] != b'{' {
        return None;
    }

    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut part_start = 0;

    pos += 1; // Skip opening {.

    loop {
        // Skip whitespace.
        while pos < line.len() && line[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= line.len() {
            return None;
        }

        // Check for closing brace (end of object).
        if line[pos] == b'}' {
            // Capture the final part: everything from part_start to end of line.
            parts.push(line[part_start..].to_vec());
            break;
        }

        // Expect a key string.
        if line[pos] != b'"' {
            return None;
        }
        // Skip over the key string.
        pos += 1;
        let mut escaped = false;
        while pos < line.len() {
            if escaped {
                escaped = false;
            } else if line[pos] == b'\\' {
                escaped = true;
            } else if line[pos] == b'"' {
                pos += 1;
                break;
            }
            pos += 1;
        }

        // Skip whitespace, expect colon.
        while pos < line.len() && line[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= line.len() || line[pos] != b':' {
            return None;
        }
        pos += 1; // Skip colon.

        // Everything from part_start up to here is a "template part".
        parts.push(line[part_start..pos].to_vec());

        // Extract the value.
        let (value, value_end) = extract_value(line, pos)?;
        values.push(value);
        pos = value_end;

        // Mark the start of the next part.
        part_start = pos;

        // Skip whitespace.
        while pos < line.len() && line[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= line.len() {
            return None;
        }

        // Expect comma or closing brace.
        if line[pos] == b',' {
            pos += 1;
        } else if line[pos] == b'}' {
            // Will be caught at the top of the loop next iteration — but we
            // need to NOT advance pos, so the } check above catches it.
            // Actually, let's just handle it here.
            parts.push(line[part_start..].to_vec());
            break;
        } else {
            return None; // Unexpected character.
        }
    }

    if values.is_empty() {
        return None;
    }

    Some((parts, values))
}

/// Forward transform: NDJSON columnar reorg.
///
/// Returns None if the data is not suitable (not uniform schema, too few lines, etc).
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    // Split into lines. For NDJSON, each non-empty line is a JSON object.
    // We need to preserve the exact line endings.
    let mut lines: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            lines.push(&data[start..i]);
            start = i + 1;
        }
    }
    // Handle last line if no trailing newline.
    if start < data.len() {
        lines.push(&data[start..]);
    }

    // Filter out empty lines — but we need at least 2 non-empty lines for columnar to help.
    let non_empty: Vec<&[u8]> = lines.iter().copied().filter(|l| !l.is_empty()).collect();
    if non_empty.len() < 2 {
        return None;
    }

    // Parse the first line to get the template.
    let (template_parts, first_values) = parse_line(non_empty[0])?;
    let num_cols = first_values.len();

    // Validate that template parts count = num_cols + 1.
    if template_parts.len() != num_cols + 1 {
        return None;
    }

    // Collect all columns. columns[col_idx] = vec of value bytes per row.
    let mut columns: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_cols);
    for v in &first_values {
        columns.push(vec![v.clone()]);
    }

    // Parse remaining lines — must all have the same template.
    for &line in &non_empty[1..] {
        let (parts, values) = parse_line(line)?;
        if values.len() != num_cols {
            return None; // Schema mismatch.
        }
        if parts.len() != template_parts.len() {
            return None;
        }
        // Verify template parts match (keys and structure must be identical).
        for (a, b) in parts.iter().zip(template_parts.iter()) {
            if a != b {
                return None; // Different key names or structure.
            }
        }
        for (col, val) in values.iter().enumerate() {
            columns[col].push(val.clone());
        }
    }

    let num_rows = non_empty.len();

    // Build column data: values separated by \x01, columns separated by \x00.
    let mut col_data = Vec::with_capacity(data.len());
    for (ci, col) in columns.iter().enumerate() {
        for (ri, val) in col.iter().enumerate() {
            col_data.extend_from_slice(val);
            if ri < num_rows - 1 {
                col_data.push(VAL_SEP);
            }
        }
        if ci < num_cols - 1 {
            col_data.push(COL_SEP);
        }
    }

    // Build metadata: version + num_rows + num_cols + trailing_newline + template parts.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });
    metadata.extend_from_slice(&(template_parts.len() as u16).to_le_bytes());
    for part in &template_parts {
        metadata.extend_from_slice(&(part.len() as u16).to_le_bytes());
        metadata.extend_from_slice(part);
    }

    // Only apply if the transform is a net win: the column data + metadata should
    // be smaller than the original input. Metadata stores keys once, which amortizes
    // with more rows. With few rows or many long keys, skip the transform.
    if col_data.len() + metadata.len() >= data.len() {
        return None;
    }

    Some(TransformResult {
        data: col_data,
        metadata,
    })
}

/// Reverse transform: reconstruct NDJSON from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    // Parse metadata.
    if metadata.len() < 10 {
        // Not enough metadata — return data as-is (shouldn't happen).
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
    let num_parts = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        if pos + 2 > metadata.len() {
            return data.to_vec();
        }
        let part_len = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + part_len > metadata.len() {
            return data.to_vec();
        }
        parts.push(metadata[pos..pos + part_len].to_vec());
        pos += part_len;
    }

    if parts.len() != num_cols + 1 || num_rows == 0 || num_cols == 0 {
        return data.to_vec();
    }

    // Parse column data: split by \x00 into columns, each column split by \x01 into values.
    let col_chunks: Vec<&[u8]> = data.split(|&b| b == COL_SEP).collect();
    if col_chunks.len() != num_cols {
        return data.to_vec();
    }

    let mut columns: Vec<Vec<&[u8]>> = Vec::with_capacity(num_cols);
    for chunk in &col_chunks {
        let vals: Vec<&[u8]> = chunk.split(|&b| b == VAL_SEP).collect();
        if vals.len() != num_rows {
            return data.to_vec();
        }
        columns.push(vals);
    }

    // Reconstruct each line (row indexes across multiple column vectors).
    let mut output = Vec::with_capacity(data.len() * 2);
    #[allow(clippy::needless_range_loop)]
    for row in 0..num_rows {
        // Part 0: e.g., {"timestamp":
        output.extend_from_slice(&parts[0]);
        // Value 0
        output.extend_from_slice(columns[0][row]);
        for col in 1..num_cols {
            // Part col: e.g., ,"event_type":
            output.extend_from_slice(&parts[col]);
            // Value col
            output.extend_from_slice(columns[col][row]);
        }
        // Final part: e.g., }\n or just }
        // The last template part includes everything after the last value to end of line.
        // But it does NOT include the newline — we add newlines between rows ourselves.
        output.extend_from_slice(&parts[num_cols]);

        // Add newline between rows (and after last row if trailing newline).
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
    fn extract_value_string() {
        let line = br#""hello","next""#;
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, b"\"hello\"");
        assert_eq!(end, 7);
    }

    #[test]
    fn extract_value_number() {
        let line = b"42,next";
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, b"42");
        assert_eq!(end, 2);
    }

    #[test]
    fn extract_value_bool() {
        let line = b"true,next";
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, b"true");
        assert_eq!(end, 4);
    }

    #[test]
    fn extract_value_null() {
        let line = b"null,next";
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, b"null");
        assert_eq!(end, 4);
    }

    #[test]
    fn extract_value_object() {
        let line = br#"{"a":1,"b":"x"},next"#;
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, br#"{"a":1,"b":"x"}"#.to_vec());
        assert_eq!(end, 15);
    }

    #[test]
    fn extract_value_array() {
        let line = b"[1,2,3],next";
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, b"[1,2,3]");
        assert_eq!(end, 7);
    }

    #[test]
    fn extract_value_string_with_escapes() {
        let line = br#""he\"llo",next"#;
        let (val, end) = extract_value(line, 0).unwrap();
        assert_eq!(val, br#""he\"llo""#.to_vec());
        assert_eq!(end, 9);
    }

    #[test]
    fn parse_line_simple() {
        let line = br#"{"a":1,"b":"x"}"#;
        let (parts, values) = parse_line(line).unwrap();
        assert_eq!(parts.len(), 3); // {"a": , ,"b": , }
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], b"1");
        assert_eq!(values[1], b"\"x\"");
        assert_eq!(parts[0], br#"{"a":"#.to_vec());
        assert_eq!(parts[1], br#","b":"#.to_vec());
        assert_eq!(parts[2], b"}");
    }

    #[test]
    fn roundtrip_simple() {
        let data = br#"{"a":1,"b":"x"}
{"a":2,"b":"y"}
{"a":3,"b":"z"}
"#;
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
        let data = br#"{"a":1,"b":"x"}
{"a":2,"b":"y"}
{"a":3,"b":"z"}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_nested_values() {
        let data = br#"{"id":1,"meta":{"x":10,"y":20}}
{"id":2,"meta":{"x":30,"y":40}}
{"id":3,"meta":{"x":50,"y":60}}
{"id":4,"meta":{"x":70,"y":80}}
{"id":5,"meta":{"x":90,"y":100}}
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_mixed_types() {
        let data = br#"{"s":"hello","n":42,"b":true,"x":null,"a":[1,2]}
{"s":"world","n":99,"b":false,"x":null,"a":[3,4]}
{"s":"foo","n":7,"b":true,"x":null,"a":[5,6]}
{"s":"bar","n":13,"b":false,"x":null,"a":[7,8]}
{"s":"baz","n":21,"b":true,"x":null,"a":[9,0]}
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn schema_mismatch_returns_none() {
        // Different keys on different lines.
        let data = br#"{"a":1,"b":2}
{"a":1,"c":3}
"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn different_num_keys_returns_none() {
        let data = br#"{"a":1,"b":2}
{"a":1}
"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn single_line_returns_none() {
        let data = br#"{"a":1,"b":2}
"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn column_layout_groups_similar_values() {
        let data = br#"{"type":"page_view","user":"alice"}
{"type":"api_call","user":"alice"}
{"type":"click","user":"bob"}
"#;
        let result = preprocess(data).unwrap();

        // The columnar data should have type values grouped, then user values grouped.
        let col_data = &result.data;
        let cols: Vec<&[u8]> = col_data.split(|&b| b == COL_SEP).collect();
        assert_eq!(cols.len(), 2);

        // Column 0 = type values.
        let type_vals: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(type_vals.len(), 3);
        assert_eq!(type_vals[0], br#""page_view""#);
        assert_eq!(type_vals[1], br#""api_call""#);
        assert_eq!(type_vals[2], br#""click""#);

        // Column 1 = user values.
        let user_vals: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(user_vals.len(), 3);
        assert_eq!(user_vals[0], br#""alice""#);
        assert_eq!(user_vals[1], br#""alice""#);
        assert_eq!(user_vals[2], br#""bob""#);
    }

    #[test]
    fn roundtrip_string_with_escaped_chars() {
        let data = br#"{"msg":"he said \"hi\"","val":1}
{"msg":"line\nbreak","val":2}
{"msg":"tab\there","val":3}
{"msg":"back\\slash","val":4}
{"msg":"normal text","val":5}
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_negative_and_float_numbers() {
        let data = br#"{"x":-3.14,"y":0}
{"x":2.718,"y":-1}
{"x":0.001,"y":999}
{"x":-100,"y":-200}
{"x":42.0,"y":7}
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    /// Test that the transform+reverse is lossless even for tiny inputs
    /// by repeating them to pass the size check threshold.
    #[test]
    fn reverse_roundtrip_small_data() {
        // Verify parse_line works on small lines.
        let (parts, vals) = parse_line(br#"{"x":-3.14,"y":0}"#).unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(parts.len(), 3);

        // 2-row data might fail the size check, so repeat to get enough rows.
        let big_data = br#"{"x":-3.14,"y":0}
{"x":2.718,"y":-1}
"#
        .repeat(20);
        let result = preprocess(&big_data).expect("should produce transform with 40 rows");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, big_data);
    }
}
