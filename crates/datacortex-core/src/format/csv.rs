//! CSV columnar reorg — lossless transform that reorders row-oriented
//! CSV/TSV/PSV data into column-oriented layout.
//!
//! Row-oriented (before):
//!   name,age,city,score
//!   Alice,30,NYC,95
//!   Bob,25,SF,87
//!   Carol,35,LA,92
//!
//! Column-oriented (after):
//!   Alice\x01Bob\x01Carol\x00
//!   30\x0125\x0135\x00
//!   NYC\x01SF\x01LA\x00
//!   95\x0187\x0192
//!
//! When similar values are adjacent, both LZ (zstd) and CM compress dramatically better.
//!
//! Separators:
//!   \x00 = column separator (between columns in the output)
//!   \x01 = value separator within a column (between values in same column)
//!
//! Metadata stores headers, delimiter, and row/column counts for reconstruction.

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;

/// Supported CSV delimiters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    Comma,
    Tab,
    Pipe,
}

impl Delimiter {
    fn as_byte(self) -> u8 {
        match self {
            Delimiter::Comma => b',',
            Delimiter::Tab => b'\t',
            Delimiter::Pipe => b'|',
        }
    }
}

/// Split a CSV line by delimiter, respecting quoted fields.
///
/// Handles RFC 4180-style quoting:
///   - Fields may be enclosed in double quotes
///   - Embedded quotes are escaped as ""
///   - Delimiters inside quotes are literal (not field separators)
///   - Newlines inside quotes are literal (caller handles line splitting)
///
/// Returns the raw field bytes (including quotes if present).
fn split_csv_line(line: &[u8], delim: u8) -> Vec<Vec<u8>> {
    let mut fields: Vec<Vec<u8>> = Vec::new();
    let mut field = Vec::new();
    let mut in_quotes = false;
    let mut i = 0;

    while i < line.len() {
        let b = line[i];
        if in_quotes {
            if b == b'"' {
                field.push(b);
                // Check for escaped quote ("").
                if i + 1 < line.len() && line[i + 1] == b'"' {
                    field.push(b'"');
                    i += 2;
                    continue;
                }
                // End of quoted field.
                in_quotes = false;
                i += 1;
                continue;
            }
            field.push(b);
            i += 1;
        } else if b == b'"' && field.is_empty() {
            // Start of quoted field (only at beginning of field).
            in_quotes = true;
            field.push(b);
            i += 1;
        } else if b == delim {
            fields.push(std::mem::take(&mut field));
            i += 1;
        } else {
            field.push(b);
            i += 1;
        }
    }
    fields.push(field);
    fields
}

/// Detect the delimiter used in CSV data.
///
/// Strategy: for each candidate delimiter (comma, tab, pipe), count the number
/// of occurrences per line across the first 20 non-empty lines. Pick the delimiter
/// where all lines have the same count and count >= 1 (meaning >= 2 columns).
///
/// Returns None if no consistent delimiter is found.
fn detect_delimiter(data: &[u8]) -> Option<Delimiter> {
    let candidates = [Delimiter::Comma, Delimiter::Tab, Delimiter::Pipe];
    let mut lines: Vec<&[u8]> = Vec::new();

    // Collect first 20 non-empty lines.
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = strip_cr(&data[start..i]);
            if !line.is_empty() {
                lines.push(line);
                if lines.len() >= 20 {
                    break;
                }
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = strip_cr(&data[start..]);
        if !line.is_empty() && lines.len() < 20 {
            lines.push(line);
        }
    }

    if lines.len() < 3 {
        return None;
    }

    let mut best: Option<(Delimiter, usize)> = None;

    for delim in candidates {
        let db = delim.as_byte();
        // Count fields per line using proper CSV splitting.
        let counts: Vec<usize> = lines.iter().map(|l| split_csv_line(l, db).len()).collect();
        let first = counts[0];
        if first >= 2 && counts.iter().all(|&c| c == first) {
            // Prefer delimiters with more columns (more specific).
            if best.is_none() || first > best.unwrap().1 {
                best = Some((delim, first));
            }
        }
    }

    best.map(|(d, _)| d)
}

/// Strip trailing \r from a line (handle CRLF).
fn strip_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

/// Check if data looks like CSV/TSV/PSV.
///
/// Used by the format detection module. Returns true if a consistent delimiter
/// is found with >= 2 columns across >= 3 lines.
pub fn detect_csv(data: &[u8]) -> bool {
    detect_delimiter(data).is_some()
}

/// Forward transform: CSV columnar reorg.
///
/// Returns None if the data is not suitable (not CSV, too few rows, etc).
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    let delim = detect_delimiter(data)?;
    let db = delim.as_byte();

    let has_trailing_newline = data.last() == Some(&b'\n');

    // Split into lines, preserving CRLF awareness.
    let mut raw_lines: Vec<&[u8]> = Vec::new();
    let mut line_has_cr: Vec<bool> = Vec::new();
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line_end = if i > 0 && data[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            let has_cr = i > 0 && data[i - 1] == b'\r';
            raw_lines.push(&data[start..line_end]);
            line_has_cr.push(has_cr);
            start = i + 1;
        }
    }
    // Handle last line if no trailing newline.
    if start < data.len() {
        let trailing = &data[start..];
        let line = strip_cr(trailing);
        line_has_cr.push(trailing.len() != line.len());
        raw_lines.push(line);
    }

    // Filter out empty lines.
    let mut lines: Vec<&[u8]> = Vec::new();
    let mut lines_cr: Vec<bool> = Vec::new();
    for (i, &l) in raw_lines.iter().enumerate() {
        if !l.is_empty() {
            lines.push(l);
            lines_cr.push(line_has_cr[i]);
        }
    }

    // Need at least header + 5 data rows = 6 lines for columnar to help.
    if lines.len() < 6 {
        return None;
    }

    // Parse header line.
    let header_fields = split_csv_line(lines[0], db);
    let num_cols = header_fields.len();
    if num_cols < 2 {
        return None;
    }

    // Determine if first line is a header (heuristic: if all fields are
    // non-numeric and non-empty, it's likely a header).
    let has_header = header_fields.iter().all(|f| {
        let s = String::from_utf8_lossy(f);
        let trimmed = s.trim().trim_matches('"');
        !trimmed.is_empty() && trimmed.parse::<f64>().is_err()
    });

    // Parse all data rows.
    let data_start = if has_header { 1 } else { 0 };
    let mut columns: Vec<Vec<Vec<u8>>> = (0..num_cols).map(|_| Vec::new()).collect();

    for &line in &lines[data_start..] {
        let fields = split_csv_line(line, db);
        if fields.len() != num_cols {
            return None; // Inconsistent column count — bail.
        }
        for (col, field) in fields.into_iter().enumerate() {
            columns[col].push(field);
        }
    }

    let num_rows = lines.len() - data_start;
    if num_rows < 5 {
        return None;
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

    // Check if all lines use CRLF (if any do). Store as a single flag
    // rather than per-line — CSV files are consistently one or the other.
    let uses_crlf = lines_cr.iter().any(|&c| c);

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.push(db); // delimiter byte
    metadata.push(if has_header { 1 } else { 0 });
    metadata.push(if has_trailing_newline { 1 } else { 0 });
    metadata.push(if uses_crlf { 1 } else { 0 });

    // Store headers (or first-row fields if no header).
    if has_header {
        for field in &header_fields {
            metadata.extend_from_slice(&(field.len() as u16).to_le_bytes());
            metadata.extend_from_slice(field);
        }
    }

    // Unlike NDJSON (which removes keys and shrinks raw size), CSV columnar's
    // benefit is entirely in downstream compressibility — similar values adjacent
    // compress dramatically better with both zstd and CM. The raw transformed size
    // may be slightly larger due to metadata overhead, but this is always recovered
    // (and then some) by the downstream compressor. The minimum row count (5) is
    // sufficient as a quality gate.

    Some(TransformResult {
        data: col_data,
        metadata,
    })
}

/// Reverse transform: reconstruct CSV from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 11 {
        return data.to_vec();
    }

    let mut pos = 0;
    let _version = metadata[pos];
    pos += 1;
    let num_rows = u32::from_le_bytes(metadata[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let num_cols = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let delim = metadata[pos];
    pos += 1;
    let has_header = metadata[pos] != 0;
    pos += 1;
    let has_trailing_newline = metadata[pos] != 0;
    pos += 1;
    let uses_crlf = metadata[pos] != 0;
    pos += 1;

    let newline: &[u8] = if uses_crlf { b"\r\n" } else { b"\n" };

    // Read headers if present.
    let mut headers: Vec<Vec<u8>> = Vec::new();
    if has_header {
        for _ in 0..num_cols {
            if pos + 2 > metadata.len() {
                return data.to_vec();
            }
            let hlen = u16::from_le_bytes(metadata[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            if pos + hlen > metadata.len() {
                return data.to_vec();
            }
            headers.push(metadata[pos..pos + hlen].to_vec());
            pos += hlen;
        }
    }

    if num_rows == 0 || num_cols == 0 {
        return data.to_vec();
    }

    // Parse column data: split by \x00 into columns, each split by \x01 into values.
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

    // Reconstruct CSV.
    let mut output = Vec::with_capacity(data.len() * 2);

    // Write header line.
    if has_header {
        for (i, h) in headers.iter().enumerate() {
            if i > 0 {
                output.push(delim);
            }
            output.extend_from_slice(h);
        }
        output.extend_from_slice(newline);
    }

    // Write data rows.
    #[allow(clippy::needless_range_loop)]
    for row in 0..num_rows {
        for col in 0..num_cols {
            if col > 0 {
                output.push(delim);
            }
            output.extend_from_slice(columns[col][row]);
        }
        if row < num_rows - 1 || has_trailing_newline {
            output.extend_from_slice(newline);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_comma_delimiter() {
        let data = b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\n";
        assert_eq!(detect_delimiter(data), Some(Delimiter::Comma));
    }

    #[test]
    fn detect_tab_delimiter() {
        let data = b"name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tSF\nCarol\t35\tLA\n";
        assert_eq!(detect_delimiter(data), Some(Delimiter::Tab));
    }

    #[test]
    fn detect_pipe_delimiter() {
        let data = b"name|age|city\nAlice|30|NYC\nBob|25|SF\nCarol|35|LA\n";
        assert_eq!(detect_delimiter(data), Some(Delimiter::Pipe));
    }

    #[test]
    fn split_simple_fields() {
        let fields = split_csv_line(b"Alice,30,NYC", b',');
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[1], b"30");
        assert_eq!(fields[2], b"NYC");
    }

    #[test]
    fn split_quoted_field_with_comma() {
        let fields = split_csv_line(br#""Smith, John",42,NYC"#, b',');
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], br#""Smith, John""#);
        assert_eq!(fields[1], b"42");
        assert_eq!(fields[2], b"NYC");
    }

    #[test]
    fn split_quoted_field_with_escaped_quote() {
        let fields = split_csv_line(br#""She said ""hello""",42"#, b',');
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], br#""She said ""hello""""#);
        assert_eq!(fields[1], b"42");
    }

    #[test]
    fn split_empty_fields() {
        let fields = split_csv_line(b"a,,c,", b',');
        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0], b"a");
        assert_eq!(fields[1], b"");
        assert_eq!(fields[2], b"c");
        assert_eq!(fields[3], b"");
    }

    #[test]
    fn roundtrip_simple_csv() {
        let data =
            b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\nDave,28,CHI\nEve,32,BOS\n";
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
        let data = b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\nDave,28,CHI\nEve,32,BOS";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_tab_delimited() {
        let data = b"name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tSF\nCarol\t35\tLA\nDave\t28\tCHI\nEve\t32\tBOS\n";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_pipe_delimited() {
        let data =
            b"name|age|city\nAlice|30|NYC\nBob|25|SF\nCarol|35|LA\nDave|28|CHI\nEve|32|BOS\n";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_quoted_fields() {
        let data = br#"name,age,city
"Smith, Alice",30,"New York, NY"
"Doe, Bob",25,"San Francisco, CA"
"Lee, Carol",35,"Los Angeles, CA"
"Kim, Dave",28,"Chicago, IL"
"Park, Eve",32,"Boston, MA"
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_crlf() {
        let data = b"name,age,city\r\nAlice,30,NYC\r\nBob,25,SF\r\nCarol,35,LA\r\nDave,28,CHI\r\nEve,32,BOS\r\n";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_numeric_header() {
        // All-numeric first row should be treated as data, not header.
        let data = b"1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n16,17,18\n";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn too_few_rows_returns_none() {
        // Header + 3 data rows = 4 lines, need at least 6.
        let data = b"name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn inconsistent_columns_returns_none() {
        let data = b"name,age,city\nAlice,30\nBob,25,SF\nCarol,35,LA\nDave,28,CHI\nEve,32,BOS\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn single_column_returns_none() {
        let data = b"name\nAlice\nBob\nCarol\nDave\nEve\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn column_layout_groups_values() {
        let data =
            b"name,age,city\nAlice,30,NYC\nBob,30,NYC\nCarol,30,SF\nDave,30,SF\nEve,30,NYC\n";
        let result = preprocess(data).unwrap();

        // Column data should have 3 columns separated by \x00.
        let cols: Vec<&[u8]> = result.data.split(|&b| b == COL_SEP).collect();
        assert_eq!(cols.len(), 3);

        // Column 0 = names.
        let names: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(names.len(), 5);
        assert_eq!(names[0], b"Alice");
        assert_eq!(names[4], b"Eve");

        // Column 1 = ages (all 30 — very compressible).
        let ages: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(ages.len(), 5);
        assert!(ages.iter().all(|&a| a == b"30"));

        // Column 2 = cities.
        let cities: Vec<&[u8]> = cols[2].split(|&b| b == VAL_SEP).collect();
        assert_eq!(cities.len(), 5);
    }

    #[test]
    fn roundtrip_many_rows() {
        // Generate enough rows for the size check to pass.
        let mut csv = String::from("id,name,value,status\n");
        for i in 0..100 {
            csv.push_str(&format!("{},item_{},{:.2},active\n", i, i, i as f64 * 1.5));
        }
        let data = csv.as_bytes();
        let result = preprocess(data).expect("should produce transform with 100 rows");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_with_empty_fields() {
        let data = b"a,b,c\n1,,3\n,5,6\n7,8,\n10,11,12\n13,,15\n";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn quoted_field_with_embedded_delimiter() {
        let data = br#"name,desc,val
"A","has, comma",1
"B","no comma",2
"C","another, one",3
"D","plain",4
"E","last, entry",5
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }
}
