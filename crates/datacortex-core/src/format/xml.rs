//! XML columnar reorg — lossless transform that detects repeated sibling
//! elements and extracts their text content into columnar layout.
//!
//! XML files with repeated elements (catalog of books, RSS items, etc.):
//!   <catalog>
//!     <book id="1"><title>Rust</title><author>Alice</author><price>29.99</price></book>
//!     <book id="2"><title>Go</title><author>Bob</author><price>39.99</price></book>
//!   </catalog>
//!
//! Strategy (same as NDJSON): each element has a "template" (the structural markup)
//! and "values" (the text content between tags). All elements must share the same
//! template. Values are extracted into columns.
//!
//! Template parts for `  <book id="1"><title>Rust</title><author>Alice</author></book>\n`:
//!   Part 0: `  <book id="`
//!   Part 1: `"><title>`
//!   Part 2: `</title><author>`
//!   Part 3: `</author></book>\n`
//!
//! Values: [`1`, `Rust`, `Alice`]
//!
//! This preserves exact bytes because reconstruction is:
//!   part[0] + val[0] + part[1] + val[1] + ... + part[N]
//!
//! Separators:
//!   \x00 = column separator
//!   \x01 = value separator within a column

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;

/// Check if data looks like XML with repeated elements.
pub fn detect_xml(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(8192)];

    let trimmed = sample
        .iter()
        .position(|&b| !b.is_ascii_whitespace() && b != 0xEF && b != 0xBB && b != 0xBF)
        .unwrap_or(sample.len());

    if trimmed >= sample.len() || sample[trimmed] != b'<' {
        return false;
    }

    let text = String::from_utf8_lossy(sample);
    let has_xml_decl = text.contains("<?xml");
    let has_tags = text.matches("</").count() >= 5;

    let mut tag_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('<')
            && !t.starts_with("</")
            && !t.starts_with("<?")
            && !t.starts_with("<!")
        {
            let tag_end = t[1..]
                .find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
                .map(|p| p + 1)
                .unwrap_or(t.len());
            let tag_name = &t[1..tag_end];
            if !tag_name.is_empty() {
                *tag_counts.entry(tag_name.to_string()).or_insert(0) += 1;
            }
        }
    }

    let max_count = tag_counts.values().max().copied().unwrap_or(0);
    (has_xml_decl || has_tags) && max_count >= 5
}

/// Find the most-repeated tag name and its indentation level.
fn find_repeated_tag(data: &[u8]) -> Option<(String, usize)> {
    let text = std::str::from_utf8(data).ok()?;
    let mut tag_counts: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();

    for line in text.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if t.starts_with('<')
            && !t.starts_with("</")
            && !t.starts_with("<?")
            && !t.starts_with("<!")
        {
            let tag_end = t[1..]
                .find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
                .map(|p| p + 1)
                .unwrap_or(t.len());
            let tag_name = &t[1..tag_end];
            if !tag_name.is_empty() {
                let entry = tag_counts
                    .entry(tag_name.to_string())
                    .or_insert((0, indent));
                entry.0 += 1;
            }
        }
    }

    let mut best: Option<(String, usize, usize)> = None;
    for (tag, (count, indent)) in &tag_counts {
        if *count >= 5 && (best.is_none() || *count > best.as_ref().unwrap().2) {
            best = Some((tag.clone(), *indent, *count));
        }
    }

    best.map(|(tag, indent, _)| (tag, indent))
}

/// Find raw element spans: byte ranges from start-of-line to end-of-closing-tag-line.
fn find_element_spans(data: &[u8], tag_name: &str, tag_indent: usize) -> Vec<(usize, usize)> {
    let tag_bytes = tag_name.as_bytes();
    let closing = format!("</{}>", tag_name);
    let closing_bytes = closing.as_bytes();
    let mut spans = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let line_start = pos;
        let mut indent = 0;
        while pos < data.len() && data[pos] == b' ' {
            indent += 1;
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }

        let is_target = indent == tag_indent
            && pos + tag_bytes.len() + 1 < data.len()
            && data[pos] == b'<'
            && data[pos + 1..].starts_with(tag_bytes)
            && {
                let after = pos + 1 + tag_bytes.len();
                after < data.len()
                    && (data[after] == b' ' || data[after] == b'>' || data[after] == b'/')
            };

        if is_target {
            let mut search = pos;
            let mut elem_end = None;
            while search + closing_bytes.len() <= data.len() {
                if data[search..].starts_with(closing_bytes) {
                    let mut end = search + closing_bytes.len();
                    while end < data.len() && data[end] != b'\n' {
                        end += 1;
                    }
                    if end < data.len() {
                        end += 1;
                    }
                    elem_end = Some(end);
                    break;
                }
                search += 1;
            }
            if let Some(end) = elem_end {
                spans.push((line_start, end));
                pos = end;
                continue;
            }
        }

        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    spans
}

/// Parse a single element into (template_parts, values).
///
/// Extract text content that appears between `>"` and `"<` or between `>` and `<`.
/// The approach: scan for every `>....<` span where the inner part doesn't contain
/// `<` or `>`. Those inner spans are values. Everything else is template.
///
/// Also extract attribute values: text between `="` and `"` within tags.
///
/// For byte-perfect roundtrip: parts[0] + values[0] + parts[1] + ... + parts[N] = original
type ParsedXmlElement = (Vec<Vec<u8>>, Vec<Vec<u8>>);

fn parse_element(element: &[u8]) -> Option<ParsedXmlElement> {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut pos = 0;
    let mut part_start = 0;

    // We extract both attribute values and text content as "values".
    // Attribute value: follows =" and ends at next unescaped "
    // Text content: follows > and ends at next <

    while pos < element.len() {
        if element[pos] == b'<' {
            // We're entering a tag. Scan through the tag.
            pos += 1;
            let is_closing = pos < element.len() && element[pos] == b'/';
            let is_special = pos < element.len() && (element[pos] == b'?' || element[pos] == b'!');

            if is_special {
                // Skip processing/declaration tags entirely.
                while pos < element.len() && element[pos] != b'>' {
                    pos += 1;
                }
                if pos < element.len() {
                    pos += 1;
                }
                continue;
            }

            // Inside a tag — look for attribute values (=").
            while pos < element.len() && element[pos] != b'>' {
                if element[pos] == b'=' && pos + 1 < element.len() && element[pos + 1] == b'"' {
                    // Found attribute value start.
                    // Template part goes up to and including the opening quote.
                    pos += 2; // skip ="
                    parts.push(element[part_start..pos].to_vec());

                    // Value = everything until closing quote.
                    let val_start = pos;
                    while pos < element.len() && element[pos] != b'"' {
                        pos += 1;
                    }
                    values.push(element[val_start..pos].to_vec());
                    part_start = pos;
                } else {
                    pos += 1;
                }
            }
            if pos < element.len() {
                pos += 1; // skip '>'
            }

            // After '>', check if we have text content (not immediately followed by '<' or at end).
            if !is_closing && pos < element.len() && element[pos] != b'<' && element[pos] != b'\n' {
                // Text content until next '<'.
                parts.push(element[part_start..pos].to_vec());
                let val_start = pos;
                while pos < element.len() && element[pos] != b'<' {
                    pos += 1;
                }
                values.push(element[val_start..pos].to_vec());
                part_start = pos;
            }
        } else {
            pos += 1;
        }
    }

    // Final template part.
    parts.push(element[part_start..].to_vec());

    if values.is_empty() {
        return None;
    }
    if parts.len() != values.len() + 1 {
        return None;
    }

    Some((parts, values))
}

/// Forward transform: XML columnar reorg.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() || data.len() < 100 {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    let (tag_name, tag_indent) = find_repeated_tag(data)?;
    let spans = find_element_spans(data, &tag_name, tag_indent);

    if spans.len() < 5 {
        return None;
    }

    let prefix = data[..spans[0].0].to_vec();
    let suffix = data[spans.last().unwrap().1..].to_vec();

    // Parse first element for template.
    let first_elem = &data[spans[0].0..spans[0].1];
    let (template_parts, first_values) = parse_element(first_elem)?;
    let num_cols = first_values.len();

    if num_cols == 0 {
        return None;
    }

    let mut columns: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_cols);
    for v in &first_values {
        columns.push(vec![v.clone()]);
    }

    // Parse remaining elements — must match template.
    for &(start, end) in &spans[1..] {
        let elem = &data[start..end];
        let (parts, values) = parse_element(elem)?;
        if values.len() != num_cols || parts.len() != template_parts.len() {
            return None;
        }
        for (a, b) in parts.iter().zip(template_parts.iter()) {
            if a != b {
                return None;
            }
        }
        for (ci, val) in values.into_iter().enumerate() {
            columns[ci].push(val);
        }
    }

    let num_rows = spans.len();

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

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(num_rows as u32).to_le_bytes());
    metadata.extend_from_slice(&(num_cols as u16).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    metadata.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(&prefix);

    metadata.extend_from_slice(&(suffix.len() as u32).to_le_bytes());
    metadata.extend_from_slice(&suffix);

    metadata.extend_from_slice(&(template_parts.len() as u16).to_le_bytes());
    for part in &template_parts {
        // Use u32 for part lengths since XML templates can be large.
        metadata.extend_from_slice(&(part.len() as u32).to_le_bytes());
        metadata.extend_from_slice(part);
    }

    // Verify roundtrip before committing — XML has many edge cases
    // (multi-line elements, varying attributes, nested content).
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

/// Reverse transform: reconstruct XML from columnar layout + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 10 {
        return data.to_vec();
    }

    let mut mpos = 0;
    let _version = metadata[mpos];
    mpos += 1;
    let num_rows = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let num_cols = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;
    let _has_trailing_newline = metadata[mpos] != 0;
    mpos += 1;

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

    if num_parts != num_cols + 1 {
        return data.to_vec();
    }

    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        if mpos + 4 > metadata.len() {
            return data.to_vec();
        }
        let part_len = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
        mpos += 4;
        if mpos + part_len > metadata.len() {
            return data.to_vec();
        }
        parts.push(metadata[mpos..mpos + part_len].to_vec());
        mpos += part_len;
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

    // Reconstruct: prefix + (for each row: part[0] + val[0] + part[1] + ... + part[N]) + suffix
    let mut output = Vec::with_capacity(data.len() * 2);
    output.extend_from_slice(prefix);

    #[allow(clippy::needless_range_loop)]
    for row in 0..num_rows {
        output.extend_from_slice(&parts[0]);
        output.extend_from_slice(columns[0][row]);
        for col in 1..num_cols {
            output.extend_from_slice(&parts[col]);
            output.extend_from_slice(columns[col][row]);
        }
        output.extend_from_slice(&parts[num_cols]);
    }

    output.extend_from_slice(suffix);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_xml_positive() {
        let data = br#"<?xml version="1.0"?>
<catalog>
  <book id="1"><title>Rust</title><author>Alice</author></book>
  <book id="2"><title>Go</title><author>Bob</author></book>
  <book id="3"><title>Python</title><author>Carol</author></book>
  <book id="4"><title>Java</title><author>Dave</author></book>
  <book id="5"><title>C++</title><author>Eve</author></book>
</catalog>
"#;
        assert!(detect_xml(data));
    }

    #[test]
    fn detect_xml_negative() {
        let data = b"just some plain text\nwithout XML structure\n";
        assert!(!detect_xml(data));
    }

    #[test]
    fn parse_element_basic() {
        let elem = b"  <book id=\"1\"><title>Rust</title><author>Alice</author></book>\n";
        let (parts, values) = parse_element(elem).unwrap();
        assert_eq!(values.len(), 3); // id value, title text, author text
        assert_eq!(values[0], b"1");
        assert_eq!(values[1], b"Rust");
        assert_eq!(values[2], b"Alice");
        assert_eq!(parts.len(), 4);
    }

    #[test]
    fn roundtrip_simple() {
        let data = br#"<?xml version="1.0"?>
<catalog>
  <book id="1"><title>Rust</title><author>Alice</author><price>29.99</price></book>
  <book id="2"><title>Go</title><author>Bob</author><price>39.99</price></book>
  <book id="3"><title>Python</title><author>Carol</author><price>24.99</price></book>
  <book id="4"><title>Java</title><author>Dave</author><price>34.99</price></book>
  <book id="5"><title>C++</title><author>Eve</author><price>44.99</price></book>
</catalog>
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
    fn roundtrip_with_attributes() {
        let data = br#"<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <book id="1" lang="en"><title>Rust</title><author>Alice</author></book>
  <book id="2" lang="de"><title>Go</title><author>Bob</author></book>
  <book id="3" lang="en"><title>Python</title><author>Carol</author></book>
  <book id="4" lang="fr"><title>Java</title><author>Dave</author></book>
  <book id="5" lang="en"><title>C++</title><author>Eve</author></book>
</catalog>
"#;
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn column_layout_groups_values() {
        let data = br#"<catalog>
  <item id="1"><name>A</name><price>10</price></item>
  <item id="2"><name>B</name><price>20</price></item>
  <item id="3"><name>C</name><price>30</price></item>
  <item id="4"><name>D</name><price>40</price></item>
  <item id="5"><name>E</name><price>50</price></item>
</catalog>
"#;
        let result = preprocess(data).unwrap();
        let cols: Vec<&[u8]> = result.data.split(|&b| b == COL_SEP).collect();
        // 3 columns: id attr value, name text, price text
        assert_eq!(cols.len(), 3);

        let ids: Vec<&[u8]> = cols[0].split(|&b| b == VAL_SEP).collect();
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], b"1");

        let names: Vec<&[u8]> = cols[1].split(|&b| b == VAL_SEP).collect();
        assert_eq!(names.len(), 5);
        assert_eq!(names[0], b"A");

        let prices: Vec<&[u8]> = cols[2].split(|&b| b == VAL_SEP).collect();
        assert_eq!(prices.len(), 5);
        assert_eq!(prices[0], b"10");
    }

    #[test]
    fn too_few_elements_returns_none() {
        let data = br#"<catalog>
  <book><title>A</title></book>
  <book><title>B</title></book>
</catalog>
"#;
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn roundtrip_large() {
        let mut xml = String::from("<?xml version=\"1.0\"?>\n<catalog>\n");
        for i in 0..100 {
            xml.push_str(&format!(
                "  <book id=\"{i}\"><title>Book {i}</title><author>Author {}</author><price>{:.2}</price></book>\n",
                i % 10, 9.99 + i as f64 * 0.5
            ));
        }
        xml.push_str("</catalog>\n");
        let data = xml.as_bytes();

        let result = preprocess(data).expect("should produce transform for 100 books");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }
}
