//! NDJSON columnar reorg — lossless transform that reorders row-oriented
//! NDJSON data into column-oriented layout.
//!
//! Two strategies:
//!
//! **Strategy 1 (uniform):** All rows share the same schema (keys in same order).
//!   Row-oriented (before):
//!     {"ts":"2026-03-15T10:30:00.001Z","type":"page_view","user":"usr_a1b2c3d4"}
//!     {"ts":"2026-03-15T10:30:00.234Z","type":"api_call","user":"usr_a1b2c3d4"}
//!   Column-oriented (after):
//!     [ts values] "2026-03-15T10:30:00.001Z" \x01 "2026-03-15T10:30:00.234Z" \x00
//!     [type values] "page_view" \x01 "api_call" \x00
//!     [user values] "usr_a1b2c3d4" \x01 "usr_a1b2c3d4"
//!
//! **Strategy 2 (grouped):** Rows have diverse schemas (e.g., GitHub Archive events).
//!   Groups rows by schema, applies Strategy 1 per group, stores residual rows raw.
//!   Metadata version byte distinguishes: 1 = uniform, 2 = grouped.
//!
//! Separators:
//!   \x00 = column separator (cannot appear in valid JSON text)
//!   \x01 = value separator within a column (cannot appear in valid JSON)

use super::transform::TransformResult;
use std::collections::HashMap;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION_UNIFORM: u8 = 1;
const METADATA_VERSION_GROUPED: u8 = 2;

/// Minimum rows in a schema group for it to be columnarized (not residual).
const MIN_GROUP_ROWS: usize = 5;

/// A schema group: template parts + list of (row_index, parsed_values).
type SchemaGroup = (Vec<Vec<u8>>, Vec<(usize, Vec<Vec<u8>>)>);

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
            if depth != 0 || pos > line.len() {
                return None; // Unterminated object.
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
            if depth != 0 || pos > line.len() {
                return None; // Unterminated array.
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

/// Split NDJSON data into lines (without newline characters).
fn split_lines(data: &[u8]) -> Vec<&[u8]> {
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
    lines
}

/// Build columnar data from parsed lines that share the same template.
/// Returns (col_data, metadata) for a uniform group.
fn build_uniform_columnar(
    template_parts: &[Vec<u8>],
    columns: &[Vec<Vec<u8>>],
    num_rows: usize,
    has_trailing_newline: bool,
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

    // Build metadata: version + num_rows + num_cols + trailing_newline + template parts.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_UNIFORM);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });
    metadata.extend_from_slice(&(template_parts.len() as u16).to_le_bytes());
    for part in template_parts {
        metadata.extend_from_slice(&(part.len() as u16).to_le_bytes());
        metadata.extend_from_slice(part);
    }

    (col_data, metadata)
}

/// Strategy 1: Uniform schema — all rows must have the same template.
/// Returns None if schemas differ.
fn preprocess_uniform(non_empty: &[&[u8]], has_trailing_newline: bool) -> Option<(Vec<u8>, Vec<u8>)> {
    if non_empty.len() < 2 {
        return None;
    }

    let (template_parts, first_values) = parse_line(non_empty[0])?;
    let num_cols = first_values.len();
    if template_parts.len() != num_cols + 1 {
        return None;
    }

    let mut columns: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_cols);
    for v in &first_values {
        columns.push(vec![v.clone()]);
    }

    for &line in &non_empty[1..] {
        let (parts, values) = parse_line(line)?;
        if values.len() != num_cols || parts.len() != template_parts.len() {
            return None;
        }
        for (a, b) in parts.iter().zip(template_parts.iter()) {
            if a != b {
                return None;
            }
        }
        for (col, val) in values.iter().enumerate() {
            columns[col].push(val.clone());
        }
    }

    Some(build_uniform_columnar(&template_parts, &columns, non_empty.len(), has_trailing_newline))
}

/// Strategy 2: Group-by-schema — group rows by template, columnarize each group.
///
/// Metadata format (version=2):
///   version: u8 = 2
///   has_trailing_newline: u8
///   total_rows: u32 LE
///   num_groups: u16 LE
///   for each group:
///     num_rows: u32 LE
///     row_indices: [u32 LE * num_rows]
///     group_metadata_len: u32 LE
///     group_metadata: [bytes]  (Strategy 1 metadata for this group)
///   residual_count: u32 LE
///   residual_indices: [u32 LE * residual_count]
///
/// Data format:
///   for each group:
///     data_len: u32 LE
///     data: [bytes]  (columnar data for this group)
///   residual_data: [bytes]  (raw lines joined by \n)
fn preprocess_grouped(non_empty: &[&[u8]], has_trailing_newline: bool) -> Option<(Vec<u8>, Vec<u8>)> {
    if non_empty.len() < MIN_GROUP_ROWS {
        return None;
    }

    // Parse all lines and group by template (the template parts identify the schema).
    // We use the template parts as the group key.
    let mut parsed: Vec<Option<ParsedLine>> = Vec::with_capacity(non_empty.len());
    for &line in non_empty {
        parsed.push(parse_line(line));
    }

    // Group rows by template. Key = template parts (as bytes for hashing).
    // We store groups as: template_key -> (template_parts, vec of (row_index, values)).
    let mut group_map: HashMap<Vec<u8>, SchemaGroup> = HashMap::new();
    let mut residual_indices: Vec<usize> = Vec::new();

    for (idx, parsed_line) in parsed.into_iter().enumerate() {
        if let Some((parts, values)) = parsed_line {
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
            // Unparseable line goes to residual.
            residual_indices.push(idx);
        }
    }

    // Separate groups into qualifying (>= MIN_GROUP_ROWS) and residual.
    let mut groups: Vec<SchemaGroup> = Vec::new();
    for (_key, (template_parts, rows)) in group_map {
        if rows.len() >= MIN_GROUP_ROWS {
            groups.push((template_parts, rows));
        } else {
            // Too few rows — send to residual.
            for (idx, _) in &rows {
                residual_indices.push(*idx);
            }
        }
    }

    // Need at least 1 qualifying group for this to be useful.
    if groups.is_empty() {
        return None;
    }

    // Sort groups by their first row index for deterministic output.
    groups.sort_by_key(|(_, rows)| rows[0].0);
    residual_indices.sort_unstable();

    // Build per-group columnar data and metadata.
    struct GroupOutput {
        row_indices: Vec<u32>,
        col_data: Vec<u8>,
        group_metadata: Vec<u8>,
    }

    let mut group_outputs: Vec<GroupOutput> = Vec::with_capacity(groups.len());

    for (template_parts, rows) in &groups {
        let num_cols = template_parts.len() - 1;
        let mut columns: Vec<Vec<Vec<u8>>> = (0..num_cols).map(|_| Vec::new()).collect();
        let mut row_indices: Vec<u32> = Vec::with_capacity(rows.len());

        for (idx, values) in rows {
            row_indices.push(*idx as u32);
            for (col, val) in values.iter().enumerate() {
                columns[col].push(val.clone());
            }
        }

        // Build columnar data for this group (trailing_newline=false for sub-groups).
        let (col_data, group_metadata) =
            build_uniform_columnar(template_parts, &columns, rows.len(), false);

        group_outputs.push(GroupOutput {
            row_indices,
            col_data,
            group_metadata,
        });
    }

    // Build the combined data blob.
    let mut data_out = Vec::new();
    for group in &group_outputs {
        data_out.extend_from_slice(&(group.col_data.len() as u32).to_le_bytes());
        data_out.extend_from_slice(&group.col_data);
    }

    // Append residual lines (raw, separated by \n).
    let residual_start = data_out.len();
    for (i, &idx) in residual_indices.iter().enumerate() {
        data_out.extend_from_slice(non_empty[idx]);
        if i < residual_indices.len() - 1 {
            data_out.push(b'\n');
        }
    }
    let _residual_len = data_out.len() - residual_start;

    // Build the combined metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION_GROUPED);
    metadata.push(if has_trailing_newline { 1 } else { 0 });
    metadata.extend_from_slice(&(non_empty.len() as u32).to_le_bytes());
    metadata.extend_from_slice(&(group_outputs.len() as u16).to_le_bytes());

    for group in &group_outputs {
        metadata.extend_from_slice(&(group.row_indices.len() as u32).to_le_bytes());
        for &idx in &group.row_indices {
            metadata.extend_from_slice(&idx.to_le_bytes());
        }
        metadata.extend_from_slice(&(group.group_metadata.len() as u32).to_le_bytes());
        metadata.extend_from_slice(&group.group_metadata);
    }

    metadata.extend_from_slice(&(residual_indices.len() as u32).to_le_bytes());
    for &idx in &residual_indices {
        metadata.extend_from_slice(&(idx as u32).to_le_bytes());
    }

    Some((data_out, metadata))
}

/// Forward transform: NDJSON columnar reorg.
///
/// Tries Strategy 1 (uniform) first, then Strategy 2 (grouped) if schemas differ.
/// Returns None if data is not suitable for columnar transform.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');
    let lines = split_lines(data);
    let non_empty: Vec<&[u8]> = lines.into_iter().filter(|l| !l.is_empty()).collect();

    if non_empty.len() < 2 {
        return None;
    }

    // Strategy 1: try uniform schema first.
    if let Some((col_data, metadata)) = preprocess_uniform(&non_empty, has_trailing_newline) {
        if col_data.len() + metadata.len() < data.len() {
            return Some(TransformResult {
                data: col_data,
                metadata,
            });
        }
    }

    // Strategy 2: group by schema.
    if let Some((grouped_data, grouped_metadata)) =
        preprocess_grouped(&non_empty, has_trailing_newline)
    {
        if grouped_data.len() + grouped_metadata.len() < data.len() {
            return Some(TransformResult {
                data: grouped_data,
                metadata: grouped_metadata,
            });
        }
    }

    None
}

/// Reverse transform: reconstruct NDJSON from columnar layout + metadata.
/// Dispatches to the appropriate decoder based on metadata version byte.
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

    reverse_uniform_from_parts(data, &parts, num_rows, num_cols, has_trailing_newline)
}

/// Core uniform reverse: given parsed parts, reconstruct lines from columnar data.
fn reverse_uniform_from_parts(
    data: &[u8],
    parts: &[Vec<u8>],
    num_rows: usize,
    num_cols: usize,
    has_trailing_newline: bool,
) -> Vec<u8> {
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

    let mut output = Vec::with_capacity(data.len() * 2);
    #[allow(clippy::needless_range_loop)]
    for row in 0..num_rows {
        output.extend_from_slice(&parts[0]);
        output.extend_from_slice(columns[0][row]);
        for col in 1..num_cols {
            output.extend_from_slice(&parts[col]);
            output.extend_from_slice(columns[col][row]);
        }
        output.extend_from_slice(&parts[num_cols]);

        if row < num_rows - 1 || has_trailing_newline {
            output.push(b'\n');
        }
    }

    output
}

/// Parse Strategy 1 metadata and return (parts, num_rows, num_cols, has_trailing_newline).
/// Used by reverse_grouped to decode per-group metadata.
fn parse_uniform_metadata(metadata: &[u8]) -> Option<(Vec<Vec<u8>>, usize, usize, bool)> {
    if metadata.len() < 10 {
        return None;
    }
    let mut pos = 1; // Skip version byte.
    let num_rows = u32::from_le_bytes(metadata[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let num_cols = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let has_trailing_newline = metadata[pos] != 0;
    pos += 1;
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

    Some((parts, num_rows, num_cols, has_trailing_newline))
}

/// Reverse Strategy 2: grouped schema.
fn reverse_grouped(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 8 {
        return data.to_vec();
    }

    let mut mpos = 1; // Skip version byte.
    let has_trailing_newline = metadata[mpos] != 0;
    mpos += 1;
    let total_rows =
        u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let num_groups =
        u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    // Allocate output slots.
    let mut output_lines: Vec<Option<Vec<u8>>> = vec![None; total_rows];

    // Data cursor.
    let mut dpos: usize = 0;

    for _ in 0..num_groups {
        // Read row indices for this group.
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let group_row_count =
            u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;

        let mut row_indices = Vec::with_capacity(group_row_count);
        for _ in 0..group_row_count {
            if mpos + 4 > metadata.len() {
                return data.to_vec();
            }
            let idx = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
            mpos += 4;
            row_indices.push(idx);
        }

        // Read group metadata.
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let gm_len =
            u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;
        if mpos + gm_len > metadata.len() {
            return data.to_vec();
        }
        let group_metadata = &metadata[mpos..mpos + gm_len];
        mpos += gm_len;

        // Read group data from the data blob.
        if dpos + 4 > data.len() {
            return data.to_vec();
        }
        let gd_len =
            u32::from_le_bytes(data[dpos..dpos + 4].try_into().unwrap()) as usize;
        dpos += 4;
        if dpos + gd_len > data.len() {
            return data.to_vec();
        }
        let group_data = &data[dpos..dpos + gd_len];
        dpos += gd_len;

        // Decode this group using Strategy 1 reverse.
        let (parts, num_rows, num_cols, _trailing) =
            match parse_uniform_metadata(group_metadata) {
                Some(v) => v,
                None => return data.to_vec(),
            };

        if num_rows != group_row_count {
            return data.to_vec();
        }

        // Split columnar data into columns and values.
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

        // Reconstruct each line for this group.
        for (row_within_group, &original_idx) in row_indices.iter().enumerate() {
            let mut line = Vec::new();
            line.extend_from_slice(&parts[0]);
            line.extend_from_slice(columns[0][row_within_group]);
            for col in 1..num_cols {
                line.extend_from_slice(&parts[col]);
                line.extend_from_slice(columns[col][row_within_group]);
            }
            line.extend_from_slice(&parts[num_cols]);

            if original_idx < total_rows {
                output_lines[original_idx] = Some(line);
            }
        }
    }

    // Read residual indices from metadata.
    if mpos + 4 > metadata.len() {
        return data.to_vec();
    }
    let residual_count =
        u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
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

    // Remaining data is residual lines.
    let residual_data = &data[dpos..];
    if residual_count > 0 {
        let residual_lines: Vec<&[u8]> = if residual_data.is_empty() {
            vec![]
        } else {
            residual_data.split(|&b| b == b'\n').collect()
        };
        // There should be exactly residual_count lines.
        if residual_lines.len() != residual_count {
            return data.to_vec();
        }
        for (i, &idx) in residual_indices.iter().enumerate() {
            if idx < total_rows {
                output_lines[idx] = Some(residual_lines[i].to_vec());
            }
        }
    }

    // Assemble final output.
    let mut output = Vec::with_capacity(data.len() * 2);
    for (i, slot) in output_lines.iter().enumerate() {
        match slot {
            Some(line) => output.extend_from_slice(line),
            None => {
                // Should not happen — missing row. Return data as-is.
                return data.to_vec();
            }
        }
        if i < total_rows - 1 || has_trailing_newline {
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
    fn schema_mismatch_too_few_returns_none() {
        // Different keys on different lines, but each group has < MIN_GROUP_ROWS.
        let data = br#"{"a":1,"b":2}
{"a":1,"c":3}
"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn different_num_keys_too_few_returns_none() {
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

    // --- Strategy 2 (grouped) tests ---

    #[test]
    fn grouped_roundtrip_two_schemas() {
        // Two different schemas, each with >= MIN_GROUP_ROWS rows.
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(
                format!(r#"{{"id":{},"type":"push","repo":"r{}"}}"#, i, i).as_bytes(),
            );
            data.push(b'\n');
        }
        for i in 10..20 {
            data.extend_from_slice(
                format!(
                    r#"{{"id":{},"type":"watch","repo":"r{}","org":"o{}"}}"#,
                    i, i, i
                )
                .as_bytes(),
            );
            data.push(b'\n');
        }
        let result = preprocess(&data).expect("should produce grouped transform");
        // Should be version 2 (grouped).
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data);
    }

    #[test]
    fn grouped_roundtrip_interleaved_schemas() {
        // Interleaved schemas: alternating between two different key sets.
        let mut data = Vec::new();
        for i in 0..20 {
            if i % 2 == 0 {
                data.extend_from_slice(
                    format!(r#"{{"id":{},"type":"push","repo":"r{}"}}"#, i, i).as_bytes(),
                );
            } else {
                data.extend_from_slice(
                    format!(
                        r#"{{"id":{},"type":"watch","repo":"r{}","org":"o{}"}}"#,
                        i, i, i
                    )
                    .as_bytes(),
                );
            }
            data.push(b'\n');
        }
        let result = preprocess(&data).expect("should produce grouped transform");
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data);
    }

    #[test]
    fn grouped_roundtrip_with_residuals() {
        // Two large groups + a few unique-schema rows (residuals).
        let mut data = Vec::new();
        // Group A: 8 rows.
        for i in 0..8 {
            data.extend_from_slice(
                format!(r#"{{"a":{},"b":"val{}"}}"#, i, i).as_bytes(),
            );
            data.push(b'\n');
        }
        // 2 unique rows (will be residual).
        data.extend_from_slice(br#"{"x":1,"y":2,"z":3}"#);
        data.push(b'\n');
        data.extend_from_slice(br#"{"p":"q"}"#);
        data.push(b'\n');
        // Group B: 6 rows.
        for i in 0..6 {
            data.extend_from_slice(
                format!(r#"{{"c":{},"d":"val{}","e":true}}"#, i, i).as_bytes(),
            );
            data.push(b'\n');
        }
        let result = preprocess(&data).expect("should produce grouped transform");
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(&data),
        );
        assert_eq!(restored, data);
    }

    #[test]
    fn grouped_roundtrip_no_trailing_newline() {
        let mut data = Vec::new();
        for i in 0..6 {
            data.extend_from_slice(
                format!(r#"{{"id":{},"type":"push"}}"#, i).as_bytes(),
            );
            data.push(b'\n');
        }
        for i in 0..6 {
            data.extend_from_slice(
                format!(r#"{{"id":{},"type":"watch","org":"o{}"}}"#, i, i).as_bytes(),
            );
            if i < 5 {
                data.push(b'\n');
            }
            // Last line: no trailing newline.
        }
        let result = preprocess(&data).expect("should produce grouped transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data);
    }

    #[test]
    fn uniform_still_preferred_over_grouped() {
        // All rows same schema — should use Strategy 1 (version 1), not Strategy 2.
        let data = br#"{"a":1,"b":"x"}
{"a":2,"b":"y"}
{"a":3,"b":"z"}
{"a":4,"b":"w"}
{"a":5,"b":"v"}
"#;
        let result = preprocess(data).expect("should produce transform");
        assert_eq!(
            result.metadata[0], METADATA_VERSION_UNIFORM,
            "uniform schema should use Strategy 1"
        );
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn grouped_gharchive_simulation() {
        // Simulates GitHub Archive: most rows have 7 keys, some have 8.
        let mut data = Vec::new();
        for i in 0..50 {
            if i % 5 == 0 {
                // 8-key rows (with org).
                data.extend_from_slice(
                    format!(
                        r#"{{"id":"{}","type":"WatchEvent","actor":{{"id":{}}},"repo":{{"id":{}}},"payload":{{}},"public":true,"created_at":"2026-03-20T12:00:00Z","org":{{"id":{}}}}}"#,
                        i, i, i, i
                    )
                    .as_bytes(),
                );
            } else {
                // 7-key rows (no org).
                data.extend_from_slice(
                    format!(
                        r#"{{"id":"{}","type":"PushEvent","actor":{{"id":{}}},"repo":{{"id":{}}},"payload":{{}},"public":true,"created_at":"2026-03-20T12:00:00Z"}}"#,
                        i, i, i
                    )
                    .as_bytes(),
                );
            }
            data.push(b'\n');
        }
        let result = preprocess(&data).expect("should produce grouped transform");
        assert_eq!(result.metadata[0], METADATA_VERSION_GROUPED);
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data);
    }
}
