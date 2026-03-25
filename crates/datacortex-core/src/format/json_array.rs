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

use super::ndjson;
use super::transform::TransformResult;
use std::collections::HashMap;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION_UNIFORM: u8 = 1;
const METADATA_VERSION_GROUPED: u8 = 2;
const MIN_ROWS: usize = 5;
/// Sentinel for absent keys in grouped nested flatten.
/// Safe to use here because grouped data is NOT processed by typed_encoding.
const ABSENT_KEY: &[u8] = b"\x02";
/// Minimum elements in a schema group for it to be columnarized (not residual).
const MIN_GROUP_ELEMENTS: usize = 3;

/// A schema group: template parts + list of (element_index, parsed_values).
type SchemaGroup = (Vec<Vec<u8>>, Vec<(usize, Vec<Vec<u8>>)>);

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
/// Strategy 1 (uniform): All elements share the same schema. Output is \x00/\x01 columnar.
/// Strategy 2 (grouped): Elements have diverse schemas. Groups by schema, columnarizes each.
///
/// Returns None if:
/// - No suitable array of objects found
/// - Fewer than MIN_ROWS elements
/// - Transform doesn't save space
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let span = find_object_array(data)?;

    // Try Strategy 1 (uniform) first.
    if let Some(result) = preprocess_uniform(data, &span) {
        return Some(result);
    }

    // Strategy 1 failed (schema mismatch). Try Strategy 2 (grouped).
    preprocess_grouped(data, &span)
}

/// Build uniform columnar data + metadata for a set of elements sharing the same template.
/// Used by both Strategy 1 (full array) and Strategy 2 (per-group).
fn build_uniform_columnar(
    template_parts: &[Vec<u8>],
    columns: &[Vec<Vec<u8>>],
    num_rows: usize,
) -> (Vec<u8>, Vec<u8>) {
    let num_cols = columns.len();

    // Build column data: values separated by \x01, columns separated by \x00.
    let mut col_data = Vec::new();
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

    // Build metadata: version + num_rows + num_cols + template parts.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_UNIFORM);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.extend_from_slice(&(template_parts.len() as u16).to_le_bytes());
    for part in template_parts {
        metadata.extend_from_slice(&(part.len() as u16).to_le_bytes());
        metadata.extend_from_slice(part);
    }

    (col_data, metadata)
}

/// Strategy 1: Uniform schema — all elements must have the same template.
fn preprocess_uniform(data: &[u8], span: &ArraySpan) -> Option<TransformResult> {
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
    let num_cols = first_values.len();

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

    // Build the full metadata with prefix/suffix/separators.
    let prefix = &data[..span.elements[0].0];
    let suffix_start = span.elements[num_rows - 1].1;
    let suffix = &data[suffix_start..];

    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_UNIFORM);
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

/// Flatten nested objects in a group's columnar data, using ABSENT_KEY sentinel
/// for keys that are absent from some rows. This is safe because grouped data
/// is NOT processed by typed_encoding (only uniform columnar is).
///
/// Returns Some((flattened_data, nested_groups)) if any columns were flattened.
fn flatten_group_nested(
    col_data: &[u8],
    num_rows: usize,
) -> Option<(Vec<u8>, Vec<ndjson::NestedGroupInfo>)> {
    let columns: Vec<&[u8]> = col_data.split(|&b| b == COL_SEP).collect();
    if columns.is_empty() || num_rows == 0 {
        return None;
    }

    let mut nested_groups: Vec<ndjson::NestedGroupInfo> = Vec::new();
    let mut output_columns: Vec<Vec<Vec<u8>>> = Vec::new();

    for (col_idx, &col_chunk) in columns.iter().enumerate() {
        let values: Vec<&[u8]> = col_chunk.split(|&b| b == VAL_SEP).collect();
        if values.len() != num_rows {
            return None;
        }

        // Check if ALL non-null values start with '{' (nested object).
        let mut all_objects = true;
        let mut has_non_null = false;
        for val in &values {
            if *val == b"null" {
                continue;
            }
            has_non_null = true;
            if !val.starts_with(b"{") {
                all_objects = false;
                break;
            }
        }

        if !all_objects || !has_non_null {
            let col_values: Vec<Vec<u8>> = values.iter().map(|v| v.to_vec()).collect();
            output_columns.push(col_values);
            continue;
        }

        // Parse all nested objects and collect all unique sub-keys.
        let mut all_sub_keys: Vec<Vec<u8>> = Vec::new();
        let mut nested_template: Vec<Vec<u8>> = Vec::new();
        type KvPairs = Vec<(Vec<u8>, Vec<u8>)>;
        let mut parsed_rows: Vec<Option<KvPairs>> = Vec::with_capacity(num_rows);
        let mut parse_failed = false;

        for val in &values {
            if *val == b"null" {
                parsed_rows.push(None);
                continue;
            }
            if nested_template.is_empty() {
                match ndjson::parse_nested_object_with_template(val) {
                    Some((template, kv_pairs)) => {
                        for (key, _) in &kv_pairs {
                            if !all_sub_keys.iter().any(|k| k == key) {
                                all_sub_keys.push(key.clone());
                            }
                        }
                        nested_template = template;
                        parsed_rows.push(Some(kv_pairs));
                    }
                    None => {
                        parse_failed = true;
                        break;
                    }
                }
            } else {
                match ndjson::parse_nested_object_kv(val) {
                    Some(kv_pairs) => {
                        for (key, _) in &kv_pairs {
                            if !all_sub_keys.iter().any(|k| k == key) {
                                all_sub_keys.push(key.clone());
                            }
                        }
                        parsed_rows.push(Some(kv_pairs));
                    }
                    None => {
                        parse_failed = true;
                        break;
                    }
                }
            }
        }

        if parse_failed || all_sub_keys.is_empty() {
            let col_values: Vec<Vec<u8>> = values.iter().map(|v| v.to_vec()).collect();
            output_columns.push(col_values);
            continue;
        }

        // Build sub-columns using ABSENT_KEY sentinel for missing keys.
        let num_sub_keys = all_sub_keys.len();
        let mut sub_columns: Vec<Vec<Vec<u8>>> = vec![Vec::with_capacity(num_rows); num_sub_keys];

        for parsed in &parsed_rows {
            match parsed {
                Some(kv_pairs) => {
                    for (sk_idx, sk) in all_sub_keys.iter().enumerate() {
                        let found = kv_pairs.iter().find(|(k, _)| k == sk);
                        match found {
                            Some((_, v)) => sub_columns[sk_idx].push(v.clone()),
                            None => sub_columns[sk_idx].push(ABSENT_KEY.to_vec()),
                        }
                    }
                }
                None => {
                    for sc in sub_columns.iter_mut() {
                        sc.push(b"null".to_vec());
                    }
                }
            }
        }

        nested_groups.push(ndjson::NestedGroupInfo {
            original_col_index: col_idx as u16,
            sub_keys: all_sub_keys,
            nested_template,
        });

        for sc in sub_columns {
            output_columns.push(sc);
        }
    }

    if nested_groups.is_empty() {
        return None;
    }

    // Build the flattened columnar data.
    let num_out_cols = output_columns.len();
    let mut out = Vec::new();
    for (ci, col) in output_columns.iter().enumerate() {
        for (ri, val) in col.iter().enumerate() {
            out.extend_from_slice(val);
            if ri < num_rows - 1 {
                out.push(VAL_SEP);
            }
        }
        if ci < num_out_cols - 1 {
            out.push(COL_SEP);
        }
    }

    Some((out, nested_groups))
}

/// Unflatten nested columns for grouped json_array, handling ABSENT_KEY sentinel.
/// Reconstructs original columnar data from flattened data + nested info.
fn unflatten_group_nested(
    flat_data: &[u8],
    nested_groups: &[ndjson::NestedGroupInfo],
    num_rows: usize,
    total_flat_cols: usize,
) -> Vec<u8> {
    let flat_columns: Vec<&[u8]> = flat_data.split(|&b| b == COL_SEP).collect();
    if flat_columns.len() != total_flat_cols {
        return flat_data.to_vec();
    }

    let mut flat_col_values: Vec<Vec<&[u8]>> = Vec::with_capacity(total_flat_cols);
    for chunk in &flat_columns {
        let vals: Vec<&[u8]> = chunk.split(|&b| b == VAL_SEP).collect();
        if vals.len() != num_rows {
            return flat_data.to_vec();
        }
        flat_col_values.push(vals);
    }

    let original_num_cols = total_flat_cols
        - nested_groups
            .iter()
            .map(|g| g.sub_keys.len())
            .sum::<usize>()
        + nested_groups.len();

    let mut original_col_map: Vec<Option<usize>> = vec![None; original_num_cols];
    for (gi, group) in nested_groups.iter().enumerate() {
        if (group.original_col_index as usize) < original_num_cols {
            original_col_map[group.original_col_index as usize] = Some(gi);
        }
    }

    let mut output_columns: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut flat_idx = 0;

    for entry in original_col_map.iter().take(original_num_cols) {
        if let Some(gi) = entry {
            let group = &nested_groups[*gi];
            let num_sub = group.sub_keys.len();

            let mut merged_col: Vec<Vec<u8>> = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                // Check if all sub-columns are null for this row (true null row).
                let all_null = (0..num_sub).all(|si| {
                    flat_idx + si < flat_col_values.len()
                        && flat_col_values[flat_idx + si][row] == b"null"
                });
                if all_null {
                    merged_col.push(b"null".to_vec());
                } else if !group.nested_template.is_empty()
                    && group.nested_template.len() == num_sub + 1
                {
                    // Template-based reconstruction with absent key handling.
                    let has_absent = (0..num_sub).any(|si| {
                        flat_idx + si < flat_col_values.len()
                            && flat_col_values[flat_idx + si][row] == ABSENT_KEY
                    });
                    if !has_absent {
                        // Fast path: no absent keys, use template directly.
                        let mut obj = Vec::new();
                        obj.extend_from_slice(&group.nested_template[0]);
                        if flat_idx < flat_col_values.len() {
                            obj.extend_from_slice(flat_col_values[flat_idx][row]);
                        }
                        for si in 1..num_sub {
                            obj.extend_from_slice(&group.nested_template[si]);
                            if flat_idx + si < flat_col_values.len() {
                                obj.extend_from_slice(flat_col_values[flat_idx + si][row]);
                            }
                        }
                        obj.extend_from_slice(&group.nested_template[num_sub]);
                        merged_col.push(obj);
                    } else {
                        // Some keys absent: skip their template parts.
                        // Template[0] = opening (e.g., '{\n  "first_key": ')
                        // Template[i>0] = separator + key (e.g., ',\n  "key_i": ')
                        // Template[num_sub] = closing (e.g., '\n}')
                        let mut obj = Vec::new();
                        let mut first_written = false;
                        for si in 0..num_sub {
                            if flat_idx + si >= flat_col_values.len() {
                                break;
                            }
                            let val = flat_col_values[flat_idx + si][row];
                            if val == ABSENT_KEY {
                                continue;
                            }
                            if !first_written {
                                if si == 0 {
                                    // First key is present: use template[0] directly.
                                    obj.extend_from_slice(&group.nested_template[0]);
                                } else {
                                    // First key(s) absent: use opening brace from
                                    // template[0] + key part from template[si] sans comma.
                                    let t0 = &group.nested_template[0];
                                    let ti = &group.nested_template[si];
                                    // t0 = '{\n        "first_key": '
                                    // ti = ',\n        "key_si": '
                                    // Want: '{\n        "key_si": '
                                    if let Some(brace_pos) = t0.iter().position(|&b| b == b'{') {
                                        obj.extend_from_slice(&t0[..brace_pos + 1]);
                                        if let Some(comma_pos) = ti.iter().position(|&b| b == b',')
                                        {
                                            obj.extend_from_slice(&ti[comma_pos + 1..]);
                                        } else {
                                            obj.extend_from_slice(ti);
                                        }
                                    } else {
                                        obj.extend_from_slice(ti);
                                    }
                                }
                                first_written = true;
                            } else {
                                obj.extend_from_slice(&group.nested_template[si]);
                            }
                            obj.extend_from_slice(val);
                        }
                        obj.extend_from_slice(&group.nested_template[num_sub]);
                        merged_col.push(obj);
                    }
                } else {
                    // Compact reconstruction.
                    let mut obj = Vec::new();
                    obj.push(b'{');
                    let mut first = true;
                    for si in 0..num_sub {
                        if flat_idx + si >= flat_col_values.len() {
                            break;
                        }
                        let val = flat_col_values[flat_idx + si][row];
                        if val == b"null" || val == ABSENT_KEY {
                            continue;
                        }
                        if !first {
                            obj.push(b',');
                        }
                        first = false;
                        obj.push(b'"');
                        obj.extend_from_slice(&group.sub_keys[si]);
                        obj.push(b'"');
                        obj.push(b':');
                        obj.extend_from_slice(val);
                    }
                    obj.push(b'}');
                    merged_col.push(obj);
                }
            }
            output_columns.push(merged_col);
            flat_idx += num_sub;
        } else {
            if flat_idx < flat_col_values.len() {
                let col: Vec<Vec<u8>> = flat_col_values[flat_idx]
                    .iter()
                    .map(|v| v.to_vec())
                    .collect();
                output_columns.push(col);
            }
            flat_idx += 1;
        }
    }

    // Rebuild columnar data.
    let num_out_cols = output_columns.len();
    let mut out = Vec::new();
    for (ci, col) in output_columns.iter().enumerate() {
        for (ri, val) in col.iter().enumerate() {
            out.extend_from_slice(val);
            if ri < num_rows - 1 {
                out.push(VAL_SEP);
            }
        }
        if ci < num_out_cols - 1 {
            out.push(COL_SEP);
        }
    }

    out
}

/// Strategy 2: Group-by-schema — group elements by template, columnarize each group.
///
/// Metadata format (version=2):
///   version: u8 = 2
///   prefix_len: u32 LE
///   prefix: bytes (JSON before the array content)
///   suffix_len: u32 LE
///   suffix: bytes (JSON after the array content)
///   separator_len: u16 LE
///   separator: bytes (most common separator between elements)
///   num_groups: u16 LE
///   for each group:
///     num_elements: u32 LE
///     element_indices: [u32 LE * num_elements]
///     group_metadata_len: u32 LE
///     group_metadata: bytes (Strategy 1 uniform columnar metadata)
///     group_data_len: u32 LE
///     group_data: bytes (columnar data for this group)
///   residual_count: u32 LE
///   residual_indices: [u32 LE * residual_count]
///   residual_data_len: u32 LE
///   residual_data: bytes (raw JSON elements joined by separator)
///
/// The element_indices + residual_indices together form a permutation of 0..N,
/// allowing byte-exact reconstruction of the original array order.
fn preprocess_grouped(data: &[u8], span: &ArraySpan) -> Option<TransformResult> {
    let num_elements = span.elements.len();
    if num_elements < MIN_ROWS {
        return None;
    }

    // Parse all elements and group by template (key set).
    let mut parsed: Vec<Option<ParsedElement>> = Vec::with_capacity(num_elements);
    for &(start, end) in &span.elements {
        parsed.push(parse_element(&data[start..end]));
    }

    // Group by template. Key = serialized template parts.
    let mut group_map: HashMap<Vec<u8>, SchemaGroup> = HashMap::new();
    let mut residual_indices: Vec<usize> = Vec::new();

    for (idx, parsed_elem) in parsed.into_iter().enumerate() {
        if let Some((parts, values)) = parsed_elem {
            // Build a hashable key from the template parts.
            let mut key = Vec::new();
            for part in &parts {
                key.extend_from_slice(&(part.len() as u32).to_le_bytes());
                key.extend_from_slice(part);
            }
            group_map
                .entry(key)
                .or_insert_with(|| (parts, Vec::new()))
                .1
                .push((idx, values));
        } else {
            residual_indices.push(idx);
        }
    }

    // Separate groups into qualifying (>= MIN_GROUP_ELEMENTS) and residual.
    let mut groups: Vec<SchemaGroup> = Vec::new();
    for (_key, (template_parts, rows)) in group_map {
        if rows.len() >= MIN_GROUP_ELEMENTS {
            groups.push((template_parts, rows));
        } else {
            for (idx, _) in &rows {
                residual_indices.push(*idx);
            }
        }
    }

    // Need at least 1 qualifying group.
    if groups.is_empty() {
        return None;
    }

    // Sort groups by first element index for deterministic output.
    groups.sort_by_key(|(_, rows)| rows[0].0);
    residual_indices.sort_unstable();

    // Build per-group columnar data and metadata.
    struct GroupOutput {
        element_indices: Vec<u32>,
        col_data: Vec<u8>,
        group_metadata: Vec<u8>,
        /// Serialized nested flatten info for this group, if nested flatten was applied.
        nested_meta: Option<Vec<u8>>,
    }

    let mut group_outputs: Vec<GroupOutput> = Vec::with_capacity(groups.len());

    for (template_parts, rows) in &groups {
        let num_cols = template_parts.len() - 1;
        let mut columns: Vec<Vec<Vec<u8>>> = (0..num_cols).map(|_| Vec::new()).collect();
        let mut element_indices: Vec<u32> = Vec::with_capacity(rows.len());

        for (idx, values) in rows {
            element_indices.push(*idx as u32);
            for (col, val) in values.iter().enumerate() {
                columns[col].push(val.clone());
            }
        }

        let (col_data, group_metadata) =
            build_uniform_columnar(template_parts, &columns, rows.len());

        // Try nested flatten on this group's columnar data.
        // Uses json_array-specific flatten that handles absent keys with sentinel.
        let num_rows = rows.len();
        let (final_col_data, nested_meta) =
            if let Some((flattened, nested_groups)) = flatten_group_nested(&col_data, num_rows) {
                if flattened.len() < col_data.len() {
                    // Verify roundtrip: unflatten must produce the exact original.
                    let total_flat_cols = flattened.split(|&b| b == COL_SEP).count() as u16;
                    let unflattened = unflatten_group_nested(
                        &flattened,
                        &nested_groups,
                        num_rows,
                        total_flat_cols as usize,
                    );
                    if unflattened == col_data {
                        // Nested flatten is byte-exact — serialize the nested info.
                        let serialized = ndjson::serialize_nested_info(&nested_groups);
                        let mut meta = Vec::new();
                        meta.extend_from_slice(&(num_rows as u32).to_le_bytes());
                        meta.extend_from_slice(&total_flat_cols.to_le_bytes());
                        meta.extend_from_slice(&serialized);
                        (flattened, Some(meta))
                    } else {
                        // Roundtrip not exact — skip nested flatten for this group.
                        (col_data, None)
                    }
                } else {
                    (col_data, None)
                }
            } else {
                (col_data, None)
            };

        group_outputs.push(GroupOutput {
            element_indices,
            col_data: final_col_data,
            group_metadata,
            nested_meta,
        });
    }

    // Determine the most common separator (for reconstructing the array).
    // All separators should be the same in well-formatted JSON (e.g., ",\n    ").
    let separator = if span.separators.is_empty() {
        b",".to_vec()
    } else {
        // Use the first separator as representative (most JSON is uniform).
        span.separators[0].clone()
    };

    // Build residual data: raw JSON elements joined by a unique delimiter.
    // We use \x00 as delimiter since it can't appear in JSON.
    let mut residual_data = Vec::new();
    for (i, &idx) in residual_indices.iter().enumerate() {
        let (start, end) = span.elements[idx];
        residual_data.extend_from_slice(&data[start..end]);
        if i < residual_indices.len() - 1 {
            residual_data.push(0x00); // delimiter between residual elements
        }
    }

    // Build the combined metadata.
    let prefix = &data[..span.elements[0].0];
    let suffix_start = span.elements[num_elements - 1].1;
    let suffix = &data[suffix_start..];

    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_GROUPED);

    // Prefix.
    metadata.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(prefix);

    // Suffix.
    metadata.extend_from_slice(&(suffix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(suffix);

    // Separator.
    metadata.extend_from_slice(&(separator.len() as u16).to_le_bytes());
    metadata.extend_from_slice(&separator);

    // Per-element separators: store all original separators for byte-exact roundtrip.
    // Flag: 1 = all same (just store count), 0 = store individually.
    let all_same = span.separators.windows(2).all(|w| w[0] == w[1]);
    if all_same && !span.separators.is_empty() && span.separators[0] == separator {
        metadata.push(1); // all separators match the stored one
    } else {
        metadata.push(0); // per-element separators
        metadata.extend_from_slice(&(span.separators.len() as u32).to_le_bytes());
        for sep in &span.separators {
            metadata.extend_from_slice(&(sep.len() as u16).to_le_bytes());
            metadata.extend_from_slice(sep);
        }
    }

    // Number of groups.
    metadata.extend_from_slice(&(group_outputs.len() as u16).to_le_bytes());

    // Per-group: indices, metadata, data.
    for group in &group_outputs {
        metadata.extend_from_slice(&(group.element_indices.len() as u32).to_le_bytes());
        for &idx in &group.element_indices {
            metadata.extend_from_slice(&idx.to_le_bytes());
        }
        metadata.extend_from_slice(&(group.group_metadata.len() as u32).to_le_bytes());
        metadata.extend_from_slice(&group.group_metadata);
        metadata.extend_from_slice(&(group.col_data.len() as u32).to_le_bytes());
        metadata.extend_from_slice(&group.col_data);
    }

    // Residual.
    metadata.extend_from_slice(&(residual_indices.len() as u32).to_le_bytes());
    for &idx in &residual_indices {
        metadata.extend_from_slice(&(idx as u32).to_le_bytes());
    }
    metadata.extend_from_slice(&(residual_data.len() as u32).to_le_bytes());
    metadata.extend_from_slice(&residual_data);

    // For grouped, the "data" is empty — everything is in metadata.
    // This is different from NDJSON grouped which puts data in the data blob.
    // But for the pipeline, the data blob is what gets compressed. So we should
    // put the group data + residual data in the data blob for better compression.
    //
    // Actually, let's follow the NDJSON pattern: group data blobs + residual in data,
    // structural metadata in metadata.

    // Rebuild: move group data + residual data into the data blob.
    let mut data_out = Vec::new();
    for group in &group_outputs {
        data_out.extend_from_slice(&(group.col_data.len() as u32).to_le_bytes());
        data_out.extend_from_slice(&group.col_data);
    }
    // Residual data.
    data_out.extend_from_slice(&residual_data);

    // Rebuild metadata WITHOUT the group data and residual data.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_GROUPED);

    // Prefix.
    metadata.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(prefix);

    // Suffix.
    metadata.extend_from_slice(&(suffix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(suffix);

    // Separator.
    metadata.extend_from_slice(&(separator.len() as u16).to_le_bytes());
    metadata.extend_from_slice(&separator);

    // Per-element separators.
    if all_same && !span.separators.is_empty() && span.separators[0] == separator {
        metadata.push(1);
    } else {
        metadata.push(0);
        metadata.extend_from_slice(&(span.separators.len() as u32).to_le_bytes());
        for sep in &span.separators {
            metadata.extend_from_slice(&(sep.len() as u16).to_le_bytes());
            metadata.extend_from_slice(sep);
        }
    }

    // Number of groups.
    metadata.extend_from_slice(&(group_outputs.len() as u16).to_le_bytes());

    // Per-group: indices + group_metadata + nested info (data is in the data blob).
    for group in &group_outputs {
        metadata.extend_from_slice(&(group.element_indices.len() as u32).to_le_bytes());
        for &idx in &group.element_indices {
            metadata.extend_from_slice(&idx.to_le_bytes());
        }
        metadata.extend_from_slice(&(group.group_metadata.len() as u32).to_le_bytes());
        metadata.extend_from_slice(&group.group_metadata);
        // Nested flatten info for this group.
        if let Some(ref nested) = group.nested_meta {
            metadata.push(1u8); // has_nested = true
            metadata.extend_from_slice(&(nested.len() as u32).to_le_bytes());
            metadata.extend_from_slice(nested);
        } else {
            metadata.push(0u8); // has_nested = false
        }
    }

    // Residual.
    metadata.extend_from_slice(&(residual_indices.len() as u32).to_le_bytes());
    for &idx in &residual_indices {
        metadata.extend_from_slice(&(idx as u32).to_le_bytes());
    }
    metadata.extend_from_slice(&(residual_data.len() as u32).to_le_bytes());

    // Size check: transform should be a net win.
    if data_out.len() + metadata.len() >= data.len() {
        return None;
    }

    Some(TransformResult {
        data: data_out,
        metadata,
    })
}

/// Reverse transform: reconstruct JSON from columnar layout + metadata.
/// Dispatches to Strategy 1 (uniform) or Strategy 2 (grouped) based on version byte.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.is_empty() {
        return data.to_vec();
    }
    match metadata[0] {
        METADATA_VERSION_UNIFORM => reverse_uniform(data, metadata),
        METADATA_VERSION_GROUPED => reverse_grouped(data, metadata),
        _ => data.to_vec(),
    }
}

/// Reverse Strategy 1: uniform schema.
fn reverse_uniform(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 7 {
        return data.to_vec();
    }

    let mut mpos = 1; // Skip version byte.
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

/// Parse Strategy 1 (uniform) group metadata into (parts, num_rows, num_cols).
/// Used by reverse_grouped to decode per-group metadata.
fn parse_group_metadata(metadata: &[u8]) -> Option<(Vec<Vec<u8>>, usize, usize)> {
    if metadata.len() < 9 {
        return None;
    }
    let mut pos = 1; // Skip version byte.
    let num_rows = u32::from_le_bytes(metadata[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let num_cols = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let num_parts = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    let mut parts = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        if pos + 2 > metadata.len() {
            return None;
        }
        let part_len = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + part_len > metadata.len() {
            return None;
        }
        parts.push(metadata[pos..pos + part_len].to_vec());
        pos += part_len;
    }

    if parts.len() != num_cols + 1 || num_rows == 0 || num_cols == 0 {
        return None;
    }

    Some((parts, num_rows, num_cols))
}

/// Reverse Strategy 2: grouped schema.
fn reverse_grouped(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 2 {
        return data.to_vec();
    }

    let mut mpos = 1; // Skip version byte.

    // Read prefix.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let prefix_len = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    if mpos + prefix_len > metadata.len() {
        return data.to_vec();
    }
    let prefix = metadata[mpos..mpos + prefix_len].to_vec();
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
    let suffix = metadata[mpos..mpos + suffix_len].to_vec();
    mpos += suffix_len;

    // Read separator.
    if mpos + 2 > metadata.len() {
        return data.to_vec();
    }
    let sep_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;
    if mpos + sep_len > metadata.len() {
        return data.to_vec();
    }
    let default_separator = metadata[mpos..mpos + sep_len].to_vec();
    mpos += sep_len;

    // Read per-element separator info.
    if mpos >= metadata.len() {
        return data.to_vec();
    }
    let sep_flag = metadata[mpos];
    mpos += 1;

    let per_element_separators: Option<Vec<Vec<u8>>> = if sep_flag == 1 {
        // All separators are the same as default_separator.
        None
    } else {
        // Per-element separators stored.
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let sep_count = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;
        let mut seps = Vec::with_capacity(sep_count);
        for _ in 0..sep_count {
            if mpos + 2 > metadata.len() {
                return data.to_vec();
            }
            let s_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
            mpos += 2;
            if mpos + s_len > metadata.len() {
                return data.to_vec();
            }
            seps.push(metadata[mpos..mpos + s_len].to_vec());
            mpos += s_len;
        }
        Some(seps)
    };

    // Read number of groups.
    if mpos + 2 > metadata.len() {
        return data.to_vec();
    }
    let num_groups = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    // We need to figure out the total number of elements to allocate slots.
    // We'll collect all element_indices and residual_indices first.
    // Actually, we process groups + residual and collect (index, element_bytes).

    // Track all element slots: index -> reconstructed element bytes.
    let mut element_slots: Vec<(usize, Vec<u8>)> = Vec::new();

    let mut dpos: usize = 0;

    for _ in 0..num_groups {
        // Read element indices for this group.
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let group_count = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;

        let mut element_indices = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            if mpos + 4 > metadata.len() {
                return data.to_vec();
            }
            let idx = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
            mpos += 4;
            element_indices.push(idx);
        }

        // Read group metadata.
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let gm_len = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;
        if mpos + gm_len > metadata.len() {
            return data.to_vec();
        }
        let group_metadata = &metadata[mpos..mpos + gm_len];
        mpos += gm_len;

        // Read nested flatten info for this group.
        if mpos >= metadata.len() {
            return data.to_vec();
        }
        let has_nested = metadata[mpos];
        mpos += 1;

        let nested_info: Option<(usize, u16, Vec<ndjson::NestedGroupInfo>)> = if has_nested == 1 {
            if mpos + 4 > metadata.len() {
                return data.to_vec();
            }
            let nested_meta_len =
                u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
            mpos += 4;
            if mpos + nested_meta_len > metadata.len() {
                return data.to_vec();
            }
            let nested_meta_bytes = &metadata[mpos..mpos + nested_meta_len];
            mpos += nested_meta_len;

            // Parse nested meta: num_rows (u32 LE) + total_flat_cols (u16 LE) + nested_info
            if nested_meta_bytes.len() < 6 {
                return data.to_vec();
            }
            let nested_num_rows =
                u32::from_le_bytes(nested_meta_bytes[0..4].try_into().unwrap()) as usize;
            let total_flat_cols = u16::from_le_bytes(nested_meta_bytes[4..6].try_into().unwrap());
            match ndjson::deserialize_nested_info(&nested_meta_bytes[6..]) {
                Some((groups, _)) => Some((nested_num_rows, total_flat_cols, groups)),
                None => return data.to_vec(),
            }
        } else {
            None
        };

        // Read group data from the data blob.
        if dpos + 4 > data.len() {
            return data.to_vec();
        }
        let gd_len = u32::from_le_bytes(data[dpos..dpos + 4].try_into().unwrap()) as usize;
        dpos += 4;
        if dpos + gd_len > data.len() {
            return data.to_vec();
        }
        let group_data_raw = &data[dpos..dpos + gd_len];
        dpos += gd_len;

        // If nested flatten was applied, unflatten first.
        let group_data_owned: Vec<u8>;
        let group_data: &[u8] =
            if let Some((nested_num_rows, total_flat_cols, ref nested_groups)) = nested_info {
                group_data_owned = unflatten_group_nested(
                    group_data_raw,
                    nested_groups,
                    nested_num_rows,
                    total_flat_cols as usize,
                );
                &group_data_owned
            } else {
                group_data_raw
            };

        // Decode this group using the uniform metadata parser.
        let (parts, num_rows, num_cols) = match parse_group_metadata(group_metadata) {
            Some(v) => v,
            None => return data.to_vec(),
        };

        if num_rows != group_count {
            return data.to_vec();
        }

        // Split columnar data.
        let col_chunks: Vec<&[u8]> = group_data.split(|&b| b == COL_SEP).collect();
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

        // Reconstruct each element for this group.
        for (row_within_group, &original_idx) in element_indices.iter().enumerate() {
            let mut elem = Vec::new();
            elem.extend_from_slice(&parts[0]);
            elem.extend_from_slice(columns[0][row_within_group]);
            for col in 1..num_cols {
                elem.extend_from_slice(&parts[col]);
                elem.extend_from_slice(columns[col][row_within_group]);
            }
            elem.extend_from_slice(&parts[num_cols]);

            element_slots.push((original_idx, elem));
        }
    }

    // Read residual.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let residual_count = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;

    let mut residual_indices = Vec::with_capacity(residual_count);
    for _ in 0..residual_count {
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let idx = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;
        residual_indices.push(idx);
    }

    // Read residual data length.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let residual_data_len =
        u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let _ = mpos; // metadata fully consumed

    // Remaining data is residual.
    let residual_data = &data[dpos..dpos + residual_data_len.min(data.len() - dpos)];

    if residual_count > 0 && !residual_data.is_empty() {
        // Split residual data by \x00 delimiter.
        let residual_elements: Vec<&[u8]> = residual_data.split(|&b| b == 0x00).collect();
        if residual_elements.len() != residual_count {
            return data.to_vec();
        }
        for (i, &idx) in residual_indices.iter().enumerate() {
            element_slots.push((idx, residual_elements[i].to_vec()));
        }
    }

    // Sort by original index.
    element_slots.sort_by_key(|(idx, _)| *idx);

    let total_elements = element_slots.len();

    // Determine per-element separators.
    // There are (total_elements - 1) separators between elements.
    let separators: Vec<&[u8]> = if let Some(ref per_elem) = per_element_separators {
        per_elem.iter().map(|s| s.as_slice()).collect()
    } else {
        vec![default_separator.as_slice(); total_elements.saturating_sub(1)]
    };

    // Reconstruct the full JSON.
    let mut output = Vec::with_capacity(data.len() * 2);
    output.extend_from_slice(&prefix);

    for (i, (_idx, elem)) in element_slots.iter().enumerate() {
        output.extend_from_slice(elem);
        if i < total_elements - 1 && i < separators.len() {
            output.extend_from_slice(separators[i]);
        }
    }

    output.extend_from_slice(&suffix);

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

    #[test]
    fn test_grouped_json_array_diverse_schemas() {
        // Array with 2 different schemas: schema A (id, name, score) and schema B (id, tag, active).
        // Each group has enough elements (>= MIN_GROUP_ELEMENTS=3) to be columnarized.
        let mut json = String::from(r#"{"data": ["#);
        for i in 0..30 {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 3 == 0 {
                // Schema B: 10 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "tag": "tag_{}", "active": {}}}"#,
                    i,
                    i,
                    if i % 2 == 0 { "true" } else { "false" }
                ));
            } else {
                // Schema A: 20 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "name": "item_{}", "score": {}}}"#,
                    i,
                    i,
                    i * 10
                ));
            }
        }
        json.push_str(r#"], "meta": {"count": 30}}"#);

        let data = json.as_bytes();
        let result =
            preprocess(data).expect("should produce grouped transform for diverse schemas");

        // Verify it's Strategy 2 (grouped).
        assert_eq!(
            result.metadata[0], METADATA_VERSION_GROUPED,
            "should use grouped strategy"
        );

        // Verify roundtrip.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_grouped_json_array_roundtrip() {
        // Byte-exact roundtrip on a diverse array with pretty-printing.
        let data = br#"{"statuses": [
    {"id": 1, "text": "hello", "user": "alice"},
    {"id": 2, "text": "world", "user": "bob"},
    {"id": 3, "text": "foo", "user": "charlie"},
    {"id": 4, "text": "bar", "retweet": true},
    {"id": 5, "text": "baz", "retweet": false},
    {"id": 6, "text": "qux", "retweet": true},
    {"id": 7, "text": "quux", "user": "dave"},
    {"id": 8, "text": "corge", "user": "eve"},
    {"id": 9, "text": "grault", "user": "frank"},
    {"id": 10, "text": "garply", "user": "grace"}
], "count": 10}"#;
        let result = preprocess(data).expect("should produce grouped transform");

        // Verify byte-exact roundtrip.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_grouped_json_array_with_residuals() {
        // Array where some elements don't fit any group (residuals).
        // Schema A: 20 elements, Schema B: 8 elements, Schema C: 2 elements (residual).
        let mut json = String::from(r#"{"items": ["#);
        for i in 0..30 {
            if i > 0 {
                json.push_str(", ");
            }
            if i == 10 || i == 20 {
                // Schema C: only 2 elements — below MIN_GROUP_ELEMENTS, will be residual.
                json.push_str(&format!(
                    r#"{{"id": {}, "special": true, "data": "unique_{}", "extra": {}}}"#,
                    i,
                    i,
                    i * 100
                ));
            } else if i % 3 == 0 {
                // Schema B: ~8 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "category": "cat_{}", "weight": {}}}"#,
                    i,
                    i,
                    i as f64 * 1.5
                ));
            } else {
                // Schema A: ~20 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "name": "item_{}", "value": {}}}"#,
                    i,
                    i,
                    i * 10
                ));
            }
        }
        json.push_str(r#"]}"#);

        let data = json.as_bytes();
        let result = preprocess(data).expect("should produce grouped transform with residuals");

        // Verify it's grouped.
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);

        // Verify byte-exact roundtrip.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_twitter_json_roundtrip() {
        // Compress corpus/json-bench/twitter.json, decompress, verify match.
        let twitter_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/twitter.json"
        );
        let data = match std::fs::read(twitter_path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping test_twitter_json_roundtrip: twitter.json not found");
                return;
            }
        };

        let result =
            preprocess(&data).expect("twitter.json should be transformable with grouped strategy");

        // Should be grouped (twitter has diverse schemas).
        assert_eq!(
            result.metadata[0], METADATA_VERSION_GROUPED,
            "twitter.json should use grouped strategy"
        );

        // Verify byte-exact roundtrip.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            restored.len(),
            data.len(),
            "roundtrip length mismatch: got {}, expected {}",
            restored.len(),
            data.len()
        );
        assert_eq!(restored, data, "twitter.json roundtrip is not byte-exact");

        // Verify size improvement.
        let compressed_size = result.data.len() + result.metadata.len();
        assert!(
            compressed_size < data.len(),
            "grouped transform should be smaller: {} vs {}",
            compressed_size,
            data.len()
        );
    }

    #[test]
    fn test_grouped_with_nested_flatten() {
        // Grouped array where at least one schema has nested objects.
        // Schema A (with nested): id, name, user:{role, level, verified}
        // Schema B (flat): id, tag, active
        let mut json = String::from(r#"{"data": ["#);
        for i in 0..30 {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 3 == 0 {
                // Schema B: flat, 10 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "tag": "tag_{}", "active": {}}}"#,
                    i,
                    i,
                    if i % 2 == 0 { "true" } else { "false" }
                ));
            } else {
                // Schema A: nested, 20 elements
                json.push_str(&format!(
                    r#"{{"id": {}, "name": "user_{}", "user": {{"role": "admin", "level": {}, "verified": {}}}}}"#,
                    i, i, i % 5, if i % 2 == 0 { "true" } else { "false" }
                ));
            }
        }
        json.push_str(r#"], "meta": {"count": 30}}"#);

        let data = json.as_bytes();
        let result =
            preprocess(data).expect("should produce grouped transform with nested flatten");

        // Should be grouped.
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);

        // Verify byte-exact roundtrip.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn test_twitter_json_improved() {
        // Verify twitter.json gets better pre-transform ratio with nested flatten in groups.
        let twitter_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/json-bench/twitter.json"
        );
        let data = match std::fs::read(twitter_path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping test_twitter_json_improved: twitter.json not found");
                return;
            }
        };

        let result = preprocess(&data).expect("twitter.json should transform");

        // Verify roundtrip first.
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data, "roundtrip must be byte-exact");

        // The pre-transform output should be smaller than before (nested flatten helps).
        let pre_transform_size = result.data.len() + result.metadata.len();
        let ratio = data.len() as f64 / pre_transform_size as f64;
        eprintln!(
            "twitter.json: original={} pre_transform={} ratio={:.2}x",
            data.len(),
            pre_transform_size,
            ratio
        );
        // With nested flatten, pre-transform ratio should be > 1.0 (it's a net win).
        assert!(
            ratio > 1.0,
            "pre-transform should be smaller than original: ratio={:.2}x",
            ratio,
        );
    }
}
