//! Nested JSON array columnar reorg — lossless transform for JSON files
//! containing arrays of objects with consistent schema.
//!
//! Targets JSON-API style responses:
//! ```json
//! {
//!   "data": [
//!     {"id": 1, "type": "repo", "name": "datacortex"},
//!     {"id": 2, "type": "repo", "name": "codecortex"},
//!     ...
//!   ],
//!   "meta": {...},
//!   "links": {...}
//! }
//! ```
//!
//! The array elements have repeated keys and similar value patterns.
//! This transform:
//! 1. Finds the largest array of objects with consistent top-level keys
//! 2. Applies columnar reorg to that array (same approach as NDJSON)
//! 3. Preserves the wrapper structure (everything outside the array)
//!
//! Minimum row threshold: 5 objects. Below that, metadata overhead > savings.
//!
//! Output format:
//!   [columnar data for the array]
//!
//! Metadata stores:
//!   - version (1 byte)
//!   - num_rows (u32 LE)
//!   - num_cols (u16 LE)
//!   - prefix: bytes before the array content (e.g., `{"data": [`)
//!   - suffix: bytes after the array content (e.g., `], "meta": {...}}`)
//!   - template_parts: structural template from the first object
//!   - row_separators: the bytes between consecutive objects (usually `,\n    `)
//!
//! Separators (same as NDJSON):
//!   \x00 = column separator
//!   \x01 = value separator within a column

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;
const MIN_ROWS: usize = 5;

/// Find the position and span of the largest array of objects in the JSON.
///
/// Returns (array_content_start, array_content_end, prefix_end, suffix_start)
/// where:
///   - prefix = data[..array_content_start] (everything before first `{` of first element)
///   - array content = data[array_content_start..array_content_end]
///   - suffix = data[suffix_start..] (everything from `]` onward)
///
/// We look for `[` followed by `{` patterns, respecting string quoting.
fn find_object_array(data: &[u8]) -> Option<ArraySpan> {
    let mut best: Option<ArraySpan> = None;

    let mut pos = 0;
    while pos < data.len() {
        match data[pos] {
            b'"' => {
                pos = skip_string(data, pos)?;
            }
            b'[' => {
                let bracket_pos = pos;
                pos += 1;
                // Skip whitespace.
                while pos < data.len() && data[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < data.len() && data[pos] == b'{' {
                    // This might be an array of objects. Try to parse it.
                    if let Some(span) = try_parse_array(data, bracket_pos) {
                        if span.num_elements >= MIN_ROWS
                            && best
                                .as_ref()
                                .is_none_or(|b| span.num_elements > b.num_elements)
                        {
                            best = Some(span);
                        }
                    }
                    // Continue scanning from after this bracket (we might find a bigger one).
                    // Don't advance past the array — elements inside may also match.
                }
            }
            _ => {
                pos += 1;
            }
        }
    }

    best
}

struct ArraySpan {
    /// Individual element byte ranges (start of `{` to after `}`).
    elements: Vec<(usize, usize)>,
    /// Separators between elements (bytes between end of one `}` and start of next `{`).
    separators: Vec<Vec<u8>>,
    num_elements: usize,
}

/// Try to parse an array of objects starting at `bracket_pos`.
fn try_parse_array(data: &[u8], bracket_pos: usize) -> Option<ArraySpan> {
    let mut pos = bracket_pos + 1; // skip `[`

    // Skip whitespace between `[` and first `{`.
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= data.len() || data[pos] != b'{' {
        return None;
    }

    let mut elements = Vec::new();
    let mut separators = Vec::new();

    loop {
        if pos >= data.len() || data[pos] != b'{' {
            break;
        }

        // Parse one object.
        let elem_start = pos;
        pos = skip_object(data, pos)?;
        let elem_end = pos;
        elements.push((elem_start, elem_end));

        // Check for comma or closing bracket.
        let sep_start = pos;
        // Skip whitespace.
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= data.len() {
            return None;
        }

        if data[pos] == b',' {
            pos += 1; // skip comma
            // Continue to whitespace before next element.
            while pos < data.len() && data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            // Separator = everything from after `}` to before next `{` (includes comma + ws).
            separators.push(data[sep_start..pos].to_vec());
        } else if data[pos] == b']' {
            return Some(ArraySpan {
                num_elements: elements.len(),
                elements,
                separators,
            });
        } else {
            return None; // unexpected character
        }
    }

    None
}

/// Skip over a JSON string starting at position `pos` (which should be `"`).
/// Returns position after the closing `"`.
fn skip_string(data: &[u8], pos: usize) -> Option<usize> {
    if pos >= data.len() || data[pos] != b'"' {
        return None;
    }
    let mut i = pos + 1;
    let mut escaped = false;
    while i < data.len() {
        if escaped {
            escaped = false;
        } else if data[i] == b'\\' {
            escaped = true;
        } else if data[i] == b'"' {
            return Some(i + 1);
        }
        i += 1;
    }
    None // unterminated
}

/// Skip over a JSON object starting at position `pos` (which should be `{`).
/// Returns position after the closing `}`.
fn skip_object(data: &[u8], pos: usize) -> Option<usize> {
    if pos >= data.len() || data[pos] != b'{' {
        return None;
    }
    let mut i = pos + 1;
    let mut depth = 1;
    while i < data.len() && depth > 0 {
        match data[i] {
            b'"' => {
                i = skip_string(data, i)?;
                continue;
            }
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'[' => {
                // skip nested arrays
                i = skip_array(data, i)?;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    Some(i)
}

/// Skip over a JSON array starting at position `pos` (which should be `[`).
/// Returns position after the closing `]`.
fn skip_array(data: &[u8], pos: usize) -> Option<usize> {
    if pos >= data.len() || data[pos] != b'[' {
        return None;
    }
    let mut i = pos + 1;
    let mut depth = 1;
    while i < data.len() && depth > 0 {
        match data[i] {
            b'"' => {
                i = skip_string(data, i)?;
                continue;
            }
            b'[' => depth += 1,
            b']' => depth -= 1,
            b'{' => {
                i = skip_object(data, i)?;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    Some(i)
}

/// Extract values from a JSON object element.
/// Reuses the same approach as the NDJSON transform's parse_line:
/// template_parts = structural parts (keys, braces, colons)
/// values = the actual values between keys.
type ParsedElement = (Vec<Vec<u8>>, Vec<Vec<u8>>);

fn parse_element(element: &[u8]) -> Option<ParsedElement> {
    let mut pos = 0;

    // Skip leading whitespace.
    while pos < element.len() && element[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= element.len() || element[pos] != b'{' {
        return None;
    }

    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut part_start = 0;

    pos += 1; // Skip opening {.

    loop {
        // Skip whitespace.
        while pos < element.len() && element[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= element.len() {
            return None;
        }

        // Check for closing brace.
        if element[pos] == b'}' {
            parts.push(element[part_start..].to_vec());
            break;
        }

        // Expect a key string.
        if element[pos] != b'"' {
            return None;
        }
        pos = skip_string(element, pos)?;

        // Skip whitespace, expect colon.
        while pos < element.len() && element[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= element.len() || element[pos] != b':' {
            return None;
        }
        pos += 1; // Skip colon.

        // Skip whitespace between colon and value — include in template.
        while pos < element.len() && element[pos].is_ascii_whitespace() {
            pos += 1;
        }

        // Everything from part_start up to here is a template part
        // (includes key, colon, and post-colon whitespace).
        parts.push(element[part_start..pos].to_vec());

        // Extract the value (no whitespace skipping — already consumed above).
        let (value, value_end) = extract_value_no_ws(element, pos)?;
        values.push(value);
        pos = value_end;

        part_start = pos;

        // Skip whitespace.
        while pos < element.len() && element[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= element.len() {
            return None;
        }

        if element[pos] == b',' {
            pos += 1;
        } else if element[pos] == b'}' {
            parts.push(element[part_start..].to_vec());
            break;
        } else {
            return None;
        }
    }

    if values.is_empty() {
        return None;
    }

    Some((parts, values))
}

/// Extract value bytes from an element at a given position.
/// Does NOT skip leading whitespace — caller must have already consumed it.
/// Handles strings, numbers, booleans, null, nested objects, nested arrays.
fn extract_value_no_ws(data: &[u8], pos: usize) -> Option<(Vec<u8>, usize)> {
    if pos >= data.len() {
        return None;
    }

    let start = pos;
    match data[pos] {
        b'"' => {
            let end = skip_string(data, pos)?;
            Some((data[start..end].to_vec(), end))
        }
        b'{' => {
            let end = skip_object(data, pos)?;
            Some((data[start..end].to_vec(), end))
        }
        b'[' => {
            let end = skip_array(data, pos)?;
            Some((data[start..end].to_vec(), end))
        }
        _ => {
            // Number, boolean, null.
            let mut p = pos;
            while p < data.len() {
                match data[p] {
                    b',' | b'}' | b']' => break,
                    _ if data[p].is_ascii_whitespace() => break,
                    _ => p += 1,
                }
            }
            if p == start {
                None
            } else {
                Some((data[start..p].to_vec(), p))
            }
        }
    }
}

/// Forward transform: nested JSON array columnar reorg.
///
/// Returns None if:
/// - No suitable array of objects found
/// - Fewer than MIN_ROWS elements
/// - Inconsistent schema across elements
/// - Transform doesn't save space
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let span = find_object_array(data)?;

    // Parse the first element to get the template.
    let first_elem = &data[span.elements[0].0..span.elements[0].1];
    let (template_parts, first_values) = parse_element(first_elem)?;
    let num_cols = first_values.len();

    if template_parts.len() != num_cols + 1 {
        return None;
    }

    // Collect columns.
    let mut columns: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_cols);
    for v in &first_values {
        columns.push(vec![v.clone()]);
    }

    // Parse remaining elements — must have same template.
    for &(elem_start, elem_end) in &span.elements[1..] {
        let elem = &data[elem_start..elem_end];
        let (parts, values) = parse_element(elem)?;
        if values.len() != num_cols {
            return None; // schema mismatch
        }
        if parts.len() != template_parts.len() {
            return None;
        }
        // Verify template parts match.
        for (a, b) in parts.iter().zip(template_parts.iter()) {
            if a != b {
                return None; // different keys or structure
            }
        }
        for (col, val) in values.iter().enumerate() {
            columns[col].push(val.clone());
        }
    }

    let num_rows = span.elements.len();

    // Build column data.
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

    // Prefix = everything before the array bracket `[`.
    // Actually we need: everything before first element content.
    // prefix = data[0..bracket_open] + `[` + pre_first
    let prefix = &data[..span.elements[0].0];
    let suffix_start = span.elements[num_rows - 1].1;
    // suffix = post_last + `]` + everything after
    let suffix = &data[suffix_start..];

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());

    // Prefix.
    metadata.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(prefix);

    // Suffix.
    metadata.extend_from_slice(&(suffix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(suffix);

    // Template parts.
    metadata.extend_from_slice(&(template_parts.len() as u16).to_le_bytes());
    for part in &template_parts {
        metadata.extend_from_slice(&(part.len() as u16).to_le_bytes());
        metadata.extend_from_slice(part);
    }

    // Row separators (num_rows - 1 of them).
    // If all separators are identical (common case), store just one + a flag.
    let all_same = span.separators.windows(2).all(|w| w[0] == w[1]);
    if all_same && !span.separators.is_empty() {
        metadata.push(1); // flag: uniform separators
        let sep = &span.separators[0];
        metadata.extend_from_slice(&(sep.len() as u16).to_le_bytes());
        metadata.extend_from_slice(sep);
    } else {
        metadata.push(0); // flag: per-row separators
        for sep in &span.separators {
            metadata.extend_from_slice(&(sep.len() as u16).to_le_bytes());
            metadata.extend_from_slice(sep);
        }
    }

    // Size check: transform should be a net win.
    if col_data.len() + metadata.len() >= data.len() {
        return None;
    }

    Some(TransformResult {
        data: col_data,
        metadata,
    })
}

/// Reverse transform: reconstruct JSON from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 7 {
        return data.to_vec();
    }

    let mut mpos = 0;
    let _version = metadata[mpos];
    mpos += 1;
    let num_rows = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let num_cols = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    if num_rows == 0 || num_cols == 0 {
        return data.to_vec();
    }

    // Read prefix.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let prefix_len = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    if mpos + prefix_len > metadata.len() {
        return data.to_vec();
    }
    let prefix = &metadata[mpos..mpos + prefix_len];
    mpos += prefix_len;

    // Read suffix.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let suffix_len = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    if mpos + suffix_len > metadata.len() {
        return data.to_vec();
    }
    let suffix = &metadata[mpos..mpos + suffix_len];
    mpos += suffix_len;

    // Read template parts.
    if mpos + 2 > metadata.len() {
        return data.to_vec();
    }
    let num_parts = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let part_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        if mpos + part_len > metadata.len() {
            return data.to_vec();
        }
        parts.push(metadata[mpos..mpos + part_len].to_vec());
        mpos += part_len;
    }

    if parts.len() != num_cols + 1 {
        return data.to_vec();
    }

    // Read separator info.
    if mpos >= metadata.len() {
        return data.to_vec();
    }
    let sep_uniform = metadata[mpos] != 0;
    mpos += 1;

    let separators: Vec<Vec<u8>> = if sep_uniform {
        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let sep_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        if mpos + sep_len > metadata.len() {
            return data.to_vec();
        }
        let sep = metadata[mpos..mpos + sep_len].to_vec();
        vec![sep; num_rows.saturating_sub(1)]
    } else {
        let mut seps = Vec::with_capacity(num_rows.saturating_sub(1));
        for _ in 0..num_rows.saturating_sub(1) {
            if mpos + 2 > metadata.len() {
                return data.to_vec();
            }
            let sep_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
            mpos += 2;
            if mpos + sep_len > metadata.len() {
                return data.to_vec();
            }
            seps.push(metadata[mpos..mpos + sep_len].to_vec());
            mpos += sep_len;
        }
        seps
    };

    // Parse column data.
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

    // Reconstruct.
    let mut output = Vec::with_capacity(data.len() * 2);

    // Prefix (everything before the first element).
    output.extend_from_slice(prefix);

    // Reconstruct each element.
    for row in 0..num_rows {
        // Part 0 (e.g., {"id":)
        output.extend_from_slice(&parts[0]);
        // Value 0
        output.extend_from_slice(columns[0][row]);
        for col in 1..num_cols {
            output.extend_from_slice(&parts[col]);
            output.extend_from_slice(columns[col][row]);
        }
        // Final part (e.g., })
        output.extend_from_slice(&parts[num_cols]);

        // Separator between elements.
        if row < num_rows - 1 && row < separators.len() {
            output.extend_from_slice(&separators[row]);
        }
    }

    // Suffix (everything after the last element).
    output.extend_from_slice(suffix);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_string_basic() {
        let data = br#""hello" rest"#;
        assert_eq!(skip_string(data, 0), Some(7));
    }

    #[test]
    fn skip_string_escaped() {
        let data = br#""he\"llo" rest"#;
        assert_eq!(skip_string(data, 0), Some(9));
    }

    #[test]
    fn skip_object_basic() {
        let data = br#"{"a": 1, "b": "x"} rest"#;
        assert_eq!(skip_object(data, 0), Some(18));
    }

    #[test]
    fn skip_object_nested() {
        let data = br#"{"a": {"b": [1,2]}} rest"#;
        assert_eq!(skip_object(data, 0), Some(19));
    }

    #[test]
    fn find_array_basic() {
        let data = br#"{"data": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]}"#;
        let span = find_object_array(data);
        assert!(span.is_some());
        let span = span.unwrap();
        assert_eq!(span.num_elements, 5);
    }

    #[test]
    fn find_array_too_few() {
        let data = br#"{"data": [{"id": 1}, {"id": 2}, {"id": 3}]}"#;
        assert!(find_object_array(data).is_none());
    }

    #[test]
    fn parse_element_simple() {
        let elem = br#"{"id": 1, "name": "test"}"#;
        let (parts, values) = parse_element(elem).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], b"1");
        assert_eq!(values[1], br#""test""#);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn roundtrip_simple_array() {
        let data = br#"{"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}, {"id": 3, "name": "gamma"}, {"id": 4, "name": "delta"}, {"id": 5, "name": "epsilon"}]}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_with_wrapper() {
        let data = br#"{"data": [{"id": 1, "type": "repo"}, {"id": 2, "type": "repo"}, {"id": 3, "type": "repo"}, {"id": 4, "type": "repo"}, {"id": 5, "type": "repo"}], "meta": {"total": 5}}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_pretty_printed() {
        let data = br#"{
  "data": [
    {"id": 1, "type": "a", "val": 10},
    {"id": 2, "type": "b", "val": 20},
    {"id": 3, "type": "c", "val": 30},
    {"id": 4, "type": "d", "val": 40},
    {"id": 5, "type": "e", "val": 50}
  ],
  "meta": {"count": 5}
}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_nested_values() {
        let data = br#"{"repos": [{"id": 1, "meta": {"stars": 10}}, {"id": 2, "meta": {"stars": 20}}, {"id": 3, "meta": {"stars": 30}}, {"id": 4, "meta": {"stars": 40}}, {"id": 5, "meta": {"stars": 50}}]}"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn too_few_elements_returns_none() {
        let data = br#"{"data": [{"id": 1}, {"id": 2}, {"id": 3}]}"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn schema_mismatch_returns_none() {
        let data = br#"{"data": [{"id": 1, "a": 1}, {"id": 2, "b": 2}, {"id": 3, "a": 3}, {"id": 4, "a": 4}, {"id": 5, "a": 5}]}"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn no_array_returns_none() {
        let data = br#"{"key": "value", "num": 42}"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn column_layout_groups_values() {
        let data = br#"{"items": [{"type": "a", "score": 10}, {"type": "b", "score": 20}, {"type": "c", "score": 30}, {"type": "d", "score": 40}, {"type": "e", "score": 50}]}"#;
        let result = preprocess(data).unwrap();

        let cols: Vec<&[u8]> = result.data.split(|&b| b == COL_SEP).collect();
        assert_eq!(cols.len(), 2);

        // Column 0 = type values.
        let type_vals: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(type_vals.len(), 5);
        assert_eq!(type_vals[0], br#""a""#);
        assert_eq!(type_vals[4], br#""e""#);

        // Column 1 = score values.
        let score_vals: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(score_vals.len(), 5);
        assert_eq!(score_vals[0], b"10");
        assert_eq!(score_vals[4], b"50");
    }

    #[test]
    fn roundtrip_top_level_array() {
        // Top-level JSON array (no wrapper object).
        let data = br#"[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}, {"id": 4, "name": "d"}, {"id": 5, "name": "e"}]"#;
        let result = preprocess(data).expect("should handle top-level arrays");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_larger_dataset() {
        // Build a JSON-API response with 20 objects.
        let mut json = String::from(r#"{"data": ["#);
        for i in 0..20 {
            if i > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!(
                r#"{{"id": {}, "type": "item", "name": "item_{}", "score": {}, "active": {}}}"#,
                i,
                i,
                i * 10 + 5,
                if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json.push_str(r#"], "meta": {"total": 20, "page": 1}}"#);

        let data = json.as_bytes();
        let result = preprocess(data).expect("should transform 20-element array");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }
}
