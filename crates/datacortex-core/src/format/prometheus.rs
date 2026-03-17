//! Prometheus/OpenMetrics columnar reorg — lossless transform that groups
//! metric lines by metric name and applies columnar layout on labels.
//!
//! Prometheus exposition format (every Kubernetes cluster exports this):
//!   # HELP http_requests_total Total HTTP requests
//!   # TYPE http_requests_total counter
//!   http_requests_total{method="GET",endpoint="/api/v1/users",status="200"} 15234 1710499800001
//!   http_requests_total{method="POST",endpoint="/api/v1/users",status="201"} 4567 1710499800001
//!   # HELP http_request_duration_seconds Request duration
//!   # TYPE http_request_duration_seconds histogram
//!   http_request_duration_seconds_bucket{le="0.01"} 24054
//!
//! Transform approach:
//! - Comment/HELP/TYPE lines stored as-is in a "comments" stream
//! - Metric data lines grouped by metric name
//! - Within each group: metric name stored once, label values columnar, numeric values columnar
//! - This groups identical label keys together (method= values all adjacent)
//!
//! Layout:
//!   [comment lines joined by \x01] \x00 [metric group 0] \x00 [metric group 1] \x00 ...
//!
//! Each metric group:
//!   [label_col_0 values joined by \x01] \x02 [label_col_1] \x02 ... \x02 [numeric values joined by \x01]
//!
//! Separators:
//!   \x00 = group separator (comments vs metric groups, and between metric groups)
//!   \x01 = value separator within a column
//!   \x02 = column separator within a metric group

use super::transform::TransformResult;

const GROUP_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const COL_SEP: u8 = 0x02;
const METADATA_VERSION: u8 = 1;

/// A parsed Prometheus metric line.
#[derive(Debug)]
struct MetricLine<'a> {
    /// The metric name (e.g., "http_requests_total")
    name: &'a [u8],
    /// Label key-value pairs, in order.
    labels: Vec<(&'a [u8], &'a [u8])>,
    /// The numeric value + optional timestamp as raw bytes.
    value_part: &'a [u8],
}

/// Parse a Prometheus metric data line.
///
/// Format: metric_name{label1="val1",label2="val2"} value [timestamp]
/// Or:     metric_name value [timestamp]  (no labels)
fn parse_metric_line(line: &[u8]) -> Option<MetricLine<'_>> {
    if line.is_empty() || line[0] == b'#' {
        return None;
    }

    // Find metric name end (first { or space).
    let mut pos = 0;
    while pos < line.len() && line[pos] != b'{' && !line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos == 0 || pos >= line.len() {
        return None;
    }

    let name = &line[..pos];

    let mut labels = Vec::new();

    if pos < line.len() && line[pos] == b'{' {
        pos += 1; // skip '{'

        // Parse label pairs.
        while pos < line.len() && line[pos] != b'}' {
            // Skip whitespace/commas.
            while pos < line.len() && (line[pos] == b',' || line[pos].is_ascii_whitespace()) {
                pos += 1;
            }
            if pos >= line.len() || line[pos] == b'}' {
                break;
            }

            // Parse label key.
            let key_start = pos;
            while pos < line.len() && line[pos] != b'=' {
                pos += 1;
            }
            if pos >= line.len() {
                return None;
            }
            let key = &line[key_start..pos];
            pos += 1; // skip '='

            // Parse label value (must be quoted).
            if pos >= line.len() || line[pos] != b'"' {
                return None;
            }
            pos += 1; // skip opening quote
            let val_start = pos;
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
            if pos >= line.len() {
                return None;
            }
            let val = &line[val_start..pos];
            pos += 1; // skip closing quote

            labels.push((key, val));
        }

        if pos >= line.len() || line[pos] != b'}' {
            return None;
        }
        pos += 1; // skip '}'
    }

    // Skip whitespace before value.
    while pos < line.len() && line[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos >= line.len() {
        return None;
    }

    let value_part = &line[pos..];

    Some(MetricLine {
        name,
        labels,
        value_part,
    })
}

/// Check if data looks like Prometheus exposition format.
pub fn detect_prometheus(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(8192)];
    let mut metric_lines = 0;
    let mut comment_lines = 0;
    let mut total_lines = 0;

    for line in sample.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        total_lines += 1;
        if total_lines > 30 {
            break;
        }
        if line.starts_with(b"# HELP ") || line.starts_with(b"# TYPE ") {
            comment_lines += 1;
        } else if parse_metric_line(line).is_some() {
            metric_lines += 1;
        }
    }

    // Need at least some metric lines and some HELP/TYPE comments.
    metric_lines >= 5 && comment_lines >= 2
}

/// Forward transform: Prometheus columnar reorg.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    // Split into lines.
    let mut raw_lines: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            raw_lines.push(&data[start..i]);
            start = i + 1;
        }
    }
    if start < data.len() {
        raw_lines.push(&data[start..]);
    }

    // Classify lines: comments/blanks vs metric data.
    let mut comment_lines: Vec<&[u8]> = Vec::new();
    let mut metric_data: Vec<MetricLine<'_>> = Vec::new();
    // Track the original line indices for ordering reconstruction.
    // We store: for each raw line, either 'C' (comment/blank) or 'M{group_name}'.
    let mut line_types: Vec<u8> = Vec::new(); // 0=comment/blank, 1=metric

    for &line in &raw_lines {
        if line.is_empty() || line.starts_with(b"#") {
            comment_lines.push(line);
            line_types.push(0);
        } else if let Some(ml) = parse_metric_line(line) {
            line_types.push(1);
            metric_data.push(ml);
        } else {
            // Unrecognized line — treat as comment.
            comment_lines.push(line);
            line_types.push(0);
        }
    }

    if metric_data.len() < 5 {
        return None;
    }

    // Group metric data by metric name (preserving order within groups).
    let mut groups: Vec<(Vec<u8>, Vec<usize>)> = Vec::new(); // (name, indices into metric_data)
    let mut name_to_group: std::collections::HashMap<Vec<u8>, usize> =
        std::collections::HashMap::new();

    for (i, ml) in metric_data.iter().enumerate() {
        let name = ml.name.to_vec();
        if let Some(&gi) = name_to_group.get(&name) {
            groups[gi].1.push(i);
        } else {
            let gi = groups.len();
            name_to_group.insert(name.clone(), gi);
            groups.push((name, vec![i]));
        }
    }

    // Build output: [comments] \x00 [group0] \x00 [group1] ...
    let mut col_data = Vec::with_capacity(data.len());

    // Comments stream.
    for (i, &cl) in comment_lines.iter().enumerate() {
        col_data.extend_from_slice(cl);
        if i < comment_lines.len() - 1 {
            col_data.push(VAL_SEP);
        }
    }
    col_data.push(GROUP_SEP);

    // For each metric group, build columnar layout.
    for (gi, (_, indices)) in groups.iter().enumerate() {
        let group_metrics: Vec<&MetricLine<'_>> =
            indices.iter().map(|&i| &metric_data[i]).collect();
        let num_rows = group_metrics.len();

        // Find the label keys for this group (union of all labels).
        let mut label_keys: Vec<Vec<u8>> = Vec::new();
        let mut label_set: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for ml in &group_metrics {
            for &(key, _) in &ml.labels {
                let k = key.to_vec();
                if label_set.insert(k.clone()) {
                    label_keys.push(k);
                }
            }
        }

        // Build label columns.
        for (li, lkey) in label_keys.iter().enumerate() {
            for (ri, ml) in group_metrics.iter().enumerate() {
                let val = ml
                    .labels
                    .iter()
                    .find(|&&(k, _)| k == lkey.as_slice())
                    .map(|&(_, v)| v)
                    .unwrap_or(b"");
                col_data.extend_from_slice(val);
                if ri < num_rows - 1 {
                    col_data.push(VAL_SEP);
                }
            }
            if li < label_keys.len() - 1 || !label_keys.is_empty() {
                col_data.push(COL_SEP);
            }
        }

        // If no labels at all, still need the COL_SEP before values.
        if label_keys.is_empty() {
            col_data.push(COL_SEP);
        }

        // Value column (numeric value + optional timestamp).
        for (ri, ml) in group_metrics.iter().enumerate() {
            col_data.extend_from_slice(ml.value_part);
            if ri < num_rows - 1 {
                col_data.push(VAL_SEP);
            }
        }

        if gi < groups.len() - 1 {
            col_data.push(GROUP_SEP);
        }
    }

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(raw_lines.len() as u32).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    // Number of comment lines.
    metadata.extend_from_slice(&(comment_lines.len() as u32).to_le_bytes());

    // Number of metric groups.
    metadata.extend_from_slice(&(groups.len() as u16).to_le_bytes());

    // For each group: metric name, num_rows, label_keys.
    for (name, indices) in &groups {
        metadata.extend_from_slice(&(name.len() as u16).to_le_bytes());
        metadata.extend_from_slice(name);
        metadata.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        // Label keys for this group.
        let group_metrics: Vec<&MetricLine<'_>> =
            indices.iter().map(|&i| &metric_data[i]).collect();
        let mut label_keys: Vec<Vec<u8>> = Vec::new();
        let mut label_set: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for ml in &group_metrics {
            for &(key, _) in &ml.labels {
                let k = key.to_vec();
                if label_set.insert(k.clone()) {
                    label_keys.push(k);
                }
            }
        }
        metadata.extend_from_slice(&(label_keys.len() as u16).to_le_bytes());
        for lk in &label_keys {
            metadata.extend_from_slice(&(lk.len() as u16).to_le_bytes());
            metadata.extend_from_slice(lk);
        }
    }

    // Store line_types for reconstruction order.
    // This is a sequence: for each original line, 0=comment(pop from comments) or 1=metric(pop from metrics).
    metadata.extend_from_slice(&line_types);

    // Verify roundtrip before committing — Prometheus format has many edge cases
    // (empty lines, metrics without labels, varying label sets per group).
    // If reconstruction doesn't match, bail out.
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

/// Reverse transform: reconstruct Prometheus text from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 12 {
        return data.to_vec();
    }

    let mut mpos = 0;
    let _version = metadata[mpos];
    mpos += 1;
    let total_lines = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let has_trailing_newline = metadata[mpos] != 0;
    mpos += 1;
    let _num_comments = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let num_groups = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    if total_lines == 0 {
        return data.to_vec();
    }

    // Read group metadata.
    struct GroupMeta {
        name: Vec<u8>,
        num_rows: usize,
        label_keys: Vec<Vec<u8>>,
    }
    let mut group_metas: Vec<GroupMeta> = Vec::with_capacity(num_groups);

    for _ in 0..num_groups {
        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let name_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        if mpos + name_len > metadata.len() {
            return data.to_vec();
        }
        let name = metadata[mpos..mpos + name_len].to_vec();
        mpos += name_len;

        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let num_rows = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;

        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let num_label_keys =
            u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;

        let mut label_keys = Vec::with_capacity(num_label_keys);
        for _ in 0..num_label_keys {
            if mpos + 2 > metadata.len() {
                return data.to_vec();
            }
            let lk_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
            mpos += 2;
            if mpos + lk_len > metadata.len() {
                return data.to_vec();
            }
            label_keys.push(metadata[mpos..mpos + lk_len].to_vec());
            mpos += lk_len;
        }

        group_metas.push(GroupMeta {
            name,
            num_rows,
            label_keys,
        });
    }

    // Read line_types.
    if mpos + total_lines > metadata.len() {
        return data.to_vec();
    }
    let line_types = &metadata[mpos..mpos + total_lines];

    // Parse data: split by GROUP_SEP.
    let data_groups: Vec<&[u8]> = data.split(|&b| b == GROUP_SEP).collect();
    // First group is comments, rest are metric groups.
    if data_groups.len() != num_groups + 1 {
        return data.to_vec();
    }

    // Parse comments.
    let comment_lines: Vec<&[u8]> = if data_groups[0].is_empty() {
        Vec::new()
    } else {
        data_groups[0].split(|&b| b == VAL_SEP).collect()
    };

    // Parse metric groups.
    struct ParsedGroup<'a> {
        label_columns: Vec<Vec<&'a [u8]>>,
        values: Vec<&'a [u8]>,
    }

    let mut parsed_groups: Vec<ParsedGroup<'_>> = Vec::with_capacity(num_groups);

    for (gi, gm) in group_metas.iter().enumerate() {
        let group_data = data_groups[gi + 1];
        let cols: Vec<&[u8]> = group_data.split(|&b| b == COL_SEP).collect();

        let expected_cols = if gm.label_keys.is_empty() {
            // Just the empty prefix + value column
            2
        } else {
            gm.label_keys.len() + 1 // label cols + value col
        };

        if cols.len() != expected_cols {
            return data.to_vec();
        }

        let mut label_columns: Vec<Vec<&[u8]>> = Vec::new();
        for col_data in cols.iter().take(gm.label_keys.len()) {
            let vals: Vec<&[u8]> = col_data.split(|&b| b == VAL_SEP).collect();
            if vals.len() != gm.num_rows {
                return data.to_vec();
            }
            label_columns.push(vals);
        }

        let value_col_idx = if gm.label_keys.is_empty() {
            1
        } else {
            gm.label_keys.len()
        };
        let values: Vec<&[u8]> = cols[value_col_idx].split(|&b| b == VAL_SEP).collect();
        if values.len() != gm.num_rows {
            return data.to_vec();
        }

        parsed_groups.push(ParsedGroup {
            label_columns,
            values,
        });
    }

    // Reconstruct lines using line_types ordering.
    let mut output = Vec::with_capacity(data.len() * 2);
    let mut comment_idx = 0;
    let mut group_row_indices: Vec<usize> = vec![0; num_groups];
    let mut group_cursor = 0; // which metric group we're currently pulling from

    // We need to know which group each metric line belongs to.
    // The metric lines appear in the same order as the groups: all of group 0 first,
    // then group 1, etc. So we use a sequential cursor.

    for (li, &lt) in line_types.iter().enumerate() {
        if lt == 0 {
            // Comment/blank line.
            if comment_idx < comment_lines.len() {
                output.extend_from_slice(comment_lines[comment_idx]);
                comment_idx += 1;
            }
        } else {
            // Metric line — find the right group and row.
            while group_cursor < num_groups
                && group_row_indices[group_cursor] >= group_metas[group_cursor].num_rows
            {
                group_cursor += 1;
            }
            if group_cursor >= num_groups {
                return data.to_vec();
            }

            let gm = &group_metas[group_cursor];
            let pg = &parsed_groups[group_cursor];
            let row = group_row_indices[group_cursor];

            // Reconstruct: name{key1="val1",key2="val2"} value [timestamp]
            output.extend_from_slice(&gm.name);

            if !gm.label_keys.is_empty() {
                output.push(b'{');
                for (li_idx, lk) in gm.label_keys.iter().enumerate() {
                    if li_idx > 0 {
                        output.push(b',');
                    }
                    output.extend_from_slice(lk);
                    output.extend_from_slice(b"=\"");
                    output.extend_from_slice(pg.label_columns[li_idx][row]);
                    output.push(b'"');
                }
                output.push(b'}');
            }

            output.push(b' ');
            output.extend_from_slice(pg.values[row]);

            group_row_indices[group_cursor] += 1;
        }

        // Add newline between lines.
        if li < total_lines - 1 || has_trailing_newline {
            output.push(b'\n');
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_metric_with_labels() {
        let line = br#"http_requests_total{method="GET",endpoint="/api/v1/users",status="200"} 15234 1710499800001"#;
        let ml = parse_metric_line(line).unwrap();
        assert_eq!(ml.name, b"http_requests_total");
        assert_eq!(ml.labels.len(), 3);
        assert_eq!(ml.labels[0].0, b"method");
        assert_eq!(ml.labels[0].1, b"GET");
        assert_eq!(ml.labels[2].0, b"status");
        assert_eq!(ml.labels[2].1, b"200");
        assert_eq!(ml.value_part, b"15234 1710499800001");
    }

    #[test]
    fn parse_metric_no_labels() {
        let line = b"process_resident_memory_bytes 123456789";
        let ml = parse_metric_line(line).unwrap();
        assert_eq!(ml.name, b"process_resident_memory_bytes");
        assert!(ml.labels.is_empty());
        assert_eq!(ml.value_part, b"123456789");
    }

    #[test]
    fn parse_comment_returns_none() {
        assert!(parse_metric_line(b"# HELP http_requests_total Total requests").is_none());
        assert!(parse_metric_line(b"# TYPE http_requests_total counter").is_none());
    }

    #[test]
    fn detect_prometheus_positive() {
        let data = b"# HELP http_requests_total Total requests\n\
# TYPE http_requests_total counter\n\
http_requests_total{method=\"GET\",status=\"200\"} 100\n\
http_requests_total{method=\"POST\",status=\"200\"} 50\n\
http_requests_total{method=\"GET\",status=\"404\"} 10\n\
http_requests_total{method=\"GET\",status=\"500\"} 5\n\
http_requests_total{method=\"DELETE\",status=\"200\"} 20\n";
        assert!(detect_prometheus(data));
    }

    #[test]
    fn detect_prometheus_negative() {
        let data = b"just some plain text\nwithout metrics\n";
        assert!(!detect_prometheus(data));
    }

    #[test]
    fn roundtrip_simple() {
        let data = b"# HELP http_requests_total Total requests\n\
# TYPE http_requests_total counter\n\
http_requests_total{method=\"GET\",status=\"200\"} 100\n\
http_requests_total{method=\"POST\",status=\"200\"} 50\n\
http_requests_total{method=\"GET\",status=\"404\"} 10\n\
http_requests_total{method=\"GET\",status=\"500\"} 5\n\
http_requests_total{method=\"DELETE\",status=\"200\"} 20\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_multiple_metrics() {
        let data = b"# HELP http_requests_total Total requests\n\
# TYPE http_requests_total counter\n\
http_requests_total{method=\"GET\"} 100\n\
http_requests_total{method=\"POST\"} 50\n\
http_requests_total{method=\"PUT\"} 25\n\
http_requests_total{method=\"DELETE\"} 10\n\
http_requests_total{method=\"PATCH\"} 5\n\
\n\
# HELP process_memory_bytes Memory usage\n\
# TYPE process_memory_bytes gauge\n\
process_memory_bytes 123456789\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_histogram() {
        let data = b"# HELP http_duration_seconds Duration\n\
# TYPE http_duration_seconds histogram\n\
http_duration_seconds_bucket{le=\"0.01\"} 100\n\
http_duration_seconds_bucket{le=\"0.025\"} 200\n\
http_duration_seconds_bucket{le=\"0.05\"} 350\n\
http_duration_seconds_bucket{le=\"0.1\"} 500\n\
http_duration_seconds_bucket{le=\"+Inf\"} 600\n\
http_duration_seconds_sum 45.123\n\
http_duration_seconds_count 600\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_with_timestamps() {
        let data = b"# HELP up Target up\n\
# TYPE up gauge\n\
up{instance=\"web-01:9090\"} 1 1710499800001\n\
up{instance=\"web-02:9090\"} 1 1710499800001\n\
up{instance=\"api-01:9090\"} 0 1710499800001\n\
up{instance=\"api-02:9090\"} 1 1710499800001\n\
up{instance=\"db-01:9090\"} 1 1710499800001\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn too_few_metrics_returns_none() {
        let data = b"# HELP m Total\n# TYPE m counter\nm{a=\"1\"} 1\nm{a=\"2\"} 2\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }
}
