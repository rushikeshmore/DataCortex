//! logfmt columnar reorg — lossless transform that reorders row-oriented
//! logfmt data into column-oriented layout.
//!
//! logfmt is THE structured logging format used by Heroku, Grafana Loki, etc:
//!   ts=2026-03-15T10:30:00.001Z level=info msg="Server started" host=web-01 port=8080
//!   ts=2026-03-15T10:30:00.234Z level=info msg="Request received" host=web-01 method=GET
//!
//! Column-oriented (after):
//!   [ts values]    2026-03-15T10:30:00.001Z\x012026-03-15T10:30:00.234Z\x00
//!   [level values] info\x01info\x00
//!   [msg values]   Server started\x01Request received\x00
//!   [host values]  web-01\x01web-01\x00
//!   ...
//!
//! Timestamps grouped together (differ by ms) -> massive compression.
//! Levels (~5 values) -> near-zero entropy when grouped.
//! Hosts (small set, high repetition) -> excellent repetition.
//!
//! Variable keys: if some lines have extra keys, a "missing" marker (\x03) is used.
//!
//! Separators:
//!   \x00 = column separator
//!   \x01 = value separator within a column
//!   \x03 = missing value marker

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const MISSING: u8 = 0x03;
const METADATA_VERSION: u8 = 1;

/// A single key=value pair from a logfmt line.
#[derive(Debug)]
struct KvPair<'a> {
    key: &'a [u8],
    value: &'a [u8],
}

/// Parse a logfmt line into key-value pairs.
///
/// Handles:
///   key=value              (unquoted)
///   key="value with spaces" (quoted, with \" escapes)
///   key=                   (empty value)
fn parse_logfmt_line(line: &[u8]) -> Option<Vec<KvPair<'_>>> {
    if line.is_empty() {
        return None;
    }

    let mut pairs = Vec::new();
    let mut pos = 0;

    while pos < line.len() {
        // Skip whitespace between pairs.
        while pos < line.len() && line[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= line.len() {
            break;
        }

        // Parse key: everything up to '='.
        let key_start = pos;
        while pos < line.len() && line[pos] != b'=' && !line[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= line.len() || line[pos] != b'=' {
            return None; // Not a valid key=value pair.
        }
        let key = &line[key_start..pos];
        if key.is_empty() {
            return None;
        }
        pos += 1; // Skip '='.

        // Parse value.
        if pos >= line.len() || line[pos].is_ascii_whitespace() {
            // Empty value.
            pairs.push(KvPair {
                key,
                value: &line[pos..pos],
            });
            continue;
        }

        if line[pos] == b'"' {
            // Quoted value — scan to closing unescaped quote.
            let val_start = pos;
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
            let value = &line[val_start..pos];
            pairs.push(KvPair { key, value });
        } else {
            // Unquoted value — scan to whitespace.
            let val_start = pos;
            while pos < line.len() && !line[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let value = &line[val_start..pos];
            pairs.push(KvPair { key, value });
        }
    }

    if pairs.len() >= 3 { Some(pairs) } else { None }
}

/// Check if data looks like logfmt.
///
/// Returns true if 80%+ of first 20 non-empty lines have >= 3 key=value pairs
/// with a consistent key set.
pub fn detect_logfmt(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(8192)];
    let mut logfmt_lines = 0;
    let mut total_lines = 0;

    for line in sample.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        total_lines += 1;
        if total_lines > 20 {
            break;
        }
        if parse_logfmt_line(line).is_some() {
            logfmt_lines += 1;
        }
    }

    total_lines >= 5 && logfmt_lines as f64 / total_lines as f64 > 0.8
}

/// Forward transform: logfmt columnar reorg.
///
/// Returns None if the data is not suitable (not logfmt, too few lines, etc).
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

    // Filter empty lines.
    let non_empty: Vec<&[u8]> = lines.iter().copied().filter(|l| !l.is_empty()).collect();
    if non_empty.len() < 5 {
        return None;
    }

    // Parse all lines.
    let mut parsed: Vec<Vec<KvPair<'_>>> = Vec::with_capacity(non_empty.len());
    for &line in &non_empty {
        match parse_logfmt_line(line) {
            Some(pairs) => parsed.push(pairs),
            None => return None,
        }
    }

    // Build the union key set (preserving order from first line, then appending new keys).
    let mut key_order: Vec<Vec<u8>> = Vec::new();
    let mut key_set: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    for pairs in &parsed {
        for pair in pairs {
            let k = pair.key.to_vec();
            if key_set.insert(k.clone()) {
                key_order.push(k);
            }
        }
    }

    let num_cols = key_order.len();
    if num_cols < 3 {
        return None;
    }

    // Check key consistency: at least 70% of keys should appear in 80%+ of lines.
    let threshold = (non_empty.len() as f64 * 0.8) as usize;
    let mut common_keys = 0;
    for key in &key_order {
        let count = parsed
            .iter()
            .filter(|pairs| pairs.iter().any(|p| p.key == key.as_slice()))
            .count();
        if count >= threshold {
            common_keys += 1;
        }
    }
    if common_keys < (num_cols as f64 * 0.7) as usize {
        return None;
    }

    let num_rows = parsed.len();

    // Build a key->index map.
    let key_index: std::collections::HashMap<&[u8], usize> = key_order
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_slice(), i))
        .collect();

    // Build columns: for each row, look up the value for each key (or MISSING).
    let mut columns: Vec<Vec<Vec<u8>>> = (0..num_cols)
        .map(|_| Vec::with_capacity(num_rows))
        .collect();

    for pairs in &parsed {
        // Build a map for this line's values.
        let mut line_vals: Vec<Option<&[u8]>> = vec![None; num_cols];
        for pair in pairs {
            if let Some(&idx) = key_index.get(pair.key) {
                line_vals[idx] = Some(pair.value);
            }
        }
        for (ci, val) in line_vals.into_iter().enumerate() {
            match val {
                Some(v) => columns[ci].push(v.to_vec()),
                None => columns[ci].push(vec![MISSING]),
            }
        }
    }

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

    // Build metadata: version + num_rows + num_cols + trailing_newline + key names.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    // Store key names (for reconstruction).
    for key in &key_order {
        metadata.extend_from_slice(&(key.len() as u16).to_le_bytes());
        metadata.extend_from_slice(key);
    }

    // Store per-row key presence bitmask for variable keys.
    // For each row, store which keys are present (as a bitfield).
    // This is needed to reconstruct the exact key order per line.
    // Actually, we also need the key ORDER per line for exact reconstruction.
    // Store per-row: count of pairs, then index of each key in order.
    for pairs in &parsed {
        metadata.push(pairs.len() as u8);
        for pair in pairs {
            let idx = key_index[pair.key];
            metadata.push(idx as u8);
        }
    }

    // Verify roundtrip before committing — logfmt has edge cases with
    // variable keys, quoted values with special chars, etc.
    let result = TransformResult {
        data: col_data,
        metadata,
    };
    let restored = reverse(&result.data, &result.metadata);
    if restored != data {
        return None;
    }

    Some(result)
}

/// Reverse transform: reconstruct logfmt from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 8 {
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

    if num_rows == 0 || num_cols == 0 {
        return data.to_vec();
    }

    // Read key names.
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(num_cols);
    for _ in 0..num_cols {
        if pos + 2 > metadata.len() {
            return data.to_vec();
        }
        let klen = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + klen > metadata.len() {
            return data.to_vec();
        }
        keys.push(metadata[pos..pos + klen].to_vec());
        pos += klen;
    }

    // Read per-row key order.
    let mut row_key_orders: Vec<Vec<usize>> = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        if pos >= metadata.len() {
            return data.to_vec();
        }
        let count = metadata[pos] as usize;
        pos += 1;
        let mut order = Vec::with_capacity(count);
        for _ in 0..count {
            if pos >= metadata.len() {
                return data.to_vec();
            }
            order.push(metadata[pos] as usize);
            pos += 1;
        }
        row_key_orders.push(order);
    }

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

    // Reconstruct each line.
    let mut output = Vec::with_capacity(data.len() * 2);

    for row in 0..num_rows {
        let key_order = &row_key_orders[row];
        let mut first = true;
        for &ki in key_order {
            if ki >= num_cols {
                return data.to_vec();
            }
            let val = columns[ki][row];
            // Skip MISSING values.
            if val == [MISSING] {
                continue;
            }
            if !first {
                output.push(b' ');
            }
            first = false;
            output.extend_from_slice(&keys[ki]);
            output.push(b'=');
            output.extend_from_slice(val);
        }

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
    fn parse_simple_logfmt() {
        let line = b"ts=2026-03-15T10:30:00.001Z level=info msg=\"Server started\" host=web-01";
        let pairs = parse_logfmt_line(line).unwrap();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0].key, b"ts");
        assert_eq!(pairs[0].value, b"2026-03-15T10:30:00.001Z");
        assert_eq!(pairs[1].key, b"level");
        assert_eq!(pairs[1].value, b"info");
        assert_eq!(pairs[2].key, b"msg");
        assert_eq!(pairs[2].value, b"\"Server started\"");
        assert_eq!(pairs[3].key, b"host");
        assert_eq!(pairs[3].value, b"web-01");
    }

    #[test]
    fn parse_with_escaped_quotes() {
        let line = br#"key=value msg="he said \"hello\"" other=1"#;
        let pairs = parse_logfmt_line(line).unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[1].value, br#""he said \"hello\"""#);
    }

    #[test]
    fn parse_with_empty_value() {
        let line = b"key1=val1 key2= key3=val3";
        let pairs = parse_logfmt_line(line).unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[1].value, b"");
    }

    #[test]
    fn detect_logfmt_positive() {
        let data = b"ts=2026-03-15T10:30:00.001Z level=info msg=\"Server started\" host=web-01\n\
                     ts=2026-03-15T10:30:00.002Z level=info msg=\"Request\" host=web-01\n\
                     ts=2026-03-15T10:30:00.003Z level=debug msg=\"Cache hit\" host=web-02\n\
                     ts=2026-03-15T10:30:00.004Z level=warn msg=\"Slow\" host=web-01\n\
                     ts=2026-03-15T10:30:00.005Z level=error msg=\"Fail\" host=web-03\n";
        assert!(detect_logfmt(data));
    }

    #[test]
    fn detect_logfmt_negative() {
        let data = b"just some plain text\nwithout any key=value pairs\nnot logfmt at all\n";
        assert!(!detect_logfmt(data));
    }

    #[test]
    fn roundtrip_simple() {
        let data = b"ts=2026-03-15T10:30:00.001Z level=info msg=\"Server started\" host=web-01\n\
ts=2026-03-15T10:30:00.002Z level=info msg=\"Request\" host=web-01\n\
ts=2026-03-15T10:30:00.003Z level=debug msg=\"Cache hit\" host=web-02\n\
ts=2026-03-15T10:30:00.004Z level=warn msg=\"Slow\" host=web-01\n\
ts=2026-03-15T10:30:00.005Z level=error msg=\"Fail\" host=web-03\n";

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
        let data = b"ts=1 level=info msg=\"a\" host=h1\n\
ts=2 level=info msg=\"b\" host=h2\n\
ts=3 level=debug msg=\"c\" host=h3\n\
ts=4 level=warn msg=\"d\" host=h4\n\
ts=5 level=error msg=\"e\" host=h5";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_variable_keys() {
        // Some lines have extra keys that others don't.
        let data = b"ts=1 level=info msg=\"a\" host=h1\n\
ts=2 level=info msg=\"b\" host=h1 method=GET\n\
ts=3 level=debug msg=\"c\" host=h2\n\
ts=4 level=warn msg=\"d\" host=h1 method=POST status=200\n\
ts=5 level=error msg=\"e\" host=h3\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn column_layout_groups_values() {
        let data = b"ts=1 level=info host=web-01\n\
ts=2 level=info host=web-01\n\
ts=3 level=debug host=web-02\n\
ts=4 level=info host=web-01\n\
ts=5 level=warn host=web-03\n";

        let result = preprocess(data).unwrap();
        let cols: Vec<&[u8]> = result.data.split(|&b| b == COL_SEP).collect();
        assert_eq!(cols.len(), 3);

        // Column 0 = ts values.
        let ts_vals: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(ts_vals.len(), 5);
        assert_eq!(ts_vals[0], b"1");
        assert_eq!(ts_vals[4], b"5");

        // Column 1 = level values (high repetition).
        let level_vals: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(level_vals.len(), 5);
    }

    #[test]
    fn too_few_lines_returns_none() {
        let data = b"ts=1 level=info msg=\"a\"\nts=2 level=info msg=\"b\"\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn roundtrip_large() {
        let mut log = String::new();
        let hosts = ["web-01", "web-02", "api-01"];
        let levels = ["info", "debug", "warn", "error"];
        for i in 0..200 {
            let host = hosts[i % hosts.len()];
            let level = levels[i % levels.len()];
            log.push_str(&format!(
                "ts=2026-03-15T10:30:00.{:03}Z level={level} msg=\"Request {i}\" host={host} duration_ms={}\n",
                i % 1000, i * 3 + 17
            ));
        }
        let data = log.as_bytes();
        let result = preprocess(data).expect("should parse large logfmt");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }
}
