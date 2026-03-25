//! Per-column value dictionary transform — replaces repeated multi-byte
//! values with single-byte dictionary codes.
//!
//! Operates on columnar data (post-NDJSON/CSV/XML columnar reorg):
//!   Input:  "page_view\x01page_view\x01api_call\x01page_view\x00alice\x01bob\x01alice"
//!   Output: "\x02\x01\x02\x01\x03\x01\x02\x00\x04\x01\x05\x01\x04"
//!
//! Dictionary codes start at \x02 (since \x00 = column sep, \x01 = value sep).
//! Columns with >253 unique values or where dict doesn't help are left raw.
//!
//! Metadata stores per-column dictionaries for reconstruction.

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;
/// First usable dictionary code — 0x00 and 0x01 are separators.
const DICT_CODE_START: u8 = 0x02;
/// Maximum unique values per column for single-byte codes.
const MAX_DICT_ENTRIES: usize = 253; // 0x02..0xFF = 254 codes, leave 1 for "raw" marker
/// Marker byte: this column is NOT dictionary-encoded (left raw).
const RAW_COLUMN_MARKER: u8 = 0xFF;
/// Metadata version for value dictionary transform.
const VDICT_VERSION: u8 = 1;

/// Forward transform: apply per-column value dictionary to columnar data.
///
/// Returns None if no column benefits from dictionary encoding.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() {
        return None;
    }

    // Split into columns by \x00.
    let columns: Vec<&[u8]> = split_columns(data);
    if columns.is_empty() {
        return None;
    }

    // Analyze each column: build value frequency tables.
    let mut col_analyses: Vec<ColumnAnalysis> = Vec::with_capacity(columns.len());
    let mut any_dictable = false;

    for col_data in &columns {
        let analysis = analyze_column(col_data);
        if analysis.should_dict {
            any_dictable = true;
        }
        col_analyses.push(analysis);
    }

    if !any_dictable {
        return None; // No column benefits from dictionary encoding.
    }

    // Build output: encode each column (dict or raw).
    let mut output = Vec::with_capacity(data.len());
    let mut metadata = Vec::new();

    // Metadata header.
    metadata.push(VDICT_VERSION);
    metadata.extend_from_slice(&(columns.len() as u16).to_le_bytes());

    for (ci, (col_data, analysis)) in columns.iter().zip(col_analyses.iter()).enumerate() {
        if analysis.should_dict {
            // Dictionary-encode this column.
            let dict = &analysis.dictionary;

            // Metadata: num_entries + entries (length-prefixed).
            metadata.push(dict.len() as u8);
            for entry in dict {
                // Store value length (u32) + value bytes.
                metadata.extend_from_slice(&(entry.len() as u32).to_le_bytes());
                metadata.extend_from_slice(entry);
            }

            // Encode values: replace each value with its dictionary code.
            let values = split_values(col_data);
            for (vi, val) in values.iter().enumerate() {
                // Find this value in the dictionary.
                if let Some(idx) = dict.iter().position(|d| d.as_slice() == *val) {
                    output.push(DICT_CODE_START + idx as u8);
                } else {
                    // Should not happen if analysis is correct.
                    output.push(DICT_CODE_START);
                }
                if vi < values.len() - 1 {
                    output.push(VAL_SEP);
                }
            }
        } else {
            // Leave this column raw.
            metadata.push(RAW_COLUMN_MARKER);
            output.extend_from_slice(col_data);
        }

        // Column separator (except after last column).
        if ci < columns.len() - 1 {
            output.push(COL_SEP);
        }
    }

    // Only apply if we save space.
    if output.len() + metadata.len() >= data.len() {
        return None;
    }

    Some(TransformResult {
        data: output,
        metadata,
    })
}

/// Reverse transform: reconstruct columnar data from dictionary-encoded data + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.is_empty() {
        return data.to_vec();
    }

    let mut mpos = 0;

    // Parse header.
    let _version = metadata[mpos];
    mpos += 1;
    if mpos + 2 > metadata.len() {
        return data.to_vec();
    }
    let num_cols = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
    mpos += 2;

    // Parse per-column dictionaries.
    let mut dictionaries: Vec<Option<Vec<Vec<u8>>>> = Vec::with_capacity(num_cols);
    for _ in 0..num_cols {
        if mpos >= metadata.len() {
            return data.to_vec();
        }
        let marker = metadata[mpos];
        mpos += 1;

        if marker == RAW_COLUMN_MARKER {
            dictionaries.push(None);
        } else {
            let num_entries = marker as usize;
            let mut dict = Vec::with_capacity(num_entries);
            for _ in 0..num_entries {
                if mpos + 4 > metadata.len() {
                    return data.to_vec();
                }
                let val_len =
                    u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
                mpos += 4;
                if mpos + val_len > metadata.len() {
                    return data.to_vec();
                }
                dict.push(metadata[mpos..mpos + val_len].to_vec());
                mpos += val_len;
            }
            dictionaries.push(Some(dict));
        }
    }

    // Split encoded data into columns.
    let encoded_columns = split_columns(data);

    // Reconstruct each column.
    let mut output = Vec::with_capacity(data.len() * 2); // Decoded is larger.
    for (ci, dict_opt) in dictionaries.iter().enumerate() {
        if ci >= encoded_columns.len() {
            break;
        }
        let col_data = encoded_columns[ci];

        if let Some(dict) = dict_opt {
            // Dictionary-decode: replace codes with original values.
            let values = split_values(col_data);
            for (vi, val) in values.iter().enumerate() {
                if val.len() == 1 && val[0] >= DICT_CODE_START {
                    let idx = (val[0] - DICT_CODE_START) as usize;
                    if idx < dict.len() {
                        output.extend_from_slice(&dict[idx]);
                    }
                } else {
                    // Unexpected — pass through raw.
                    output.extend_from_slice(val);
                }
                if vi < values.len() - 1 {
                    output.push(VAL_SEP);
                }
            }
        } else {
            // Raw column — copy as-is.
            output.extend_from_slice(col_data);
        }

        if ci < dictionaries.len() - 1 {
            output.push(COL_SEP);
        }
    }

    output
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Split columnar data by \x00 (column separator).
fn split_columns(data: &[u8]) -> Vec<&[u8]> {
    let mut columns = Vec::new();
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == COL_SEP {
            columns.push(&data[start..i]);
            start = i + 1;
        }
    }
    if start <= data.len() {
        columns.push(&data[start..]);
    }
    columns
}

/// Split a column's data by \x01 (value separator).
fn split_values(col: &[u8]) -> Vec<&[u8]> {
    let mut values = Vec::new();
    let mut start = 0;
    for i in 0..col.len() {
        if col[i] == VAL_SEP {
            values.push(&col[start..i]);
            start = i + 1;
        }
    }
    if start <= col.len() {
        values.push(&col[start..]);
    }
    values
}

/// Column analysis: determine if dictionary encoding helps.
struct ColumnAnalysis {
    /// Whether this column should be dictionary-encoded.
    should_dict: bool,
    /// Dictionary entries (sorted by frequency, most frequent first).
    dictionary: Vec<Vec<u8>>,
}

/// Analyze a column's values and decide if dictionary encoding is beneficial.
fn analyze_column(col_data: &[u8]) -> ColumnAnalysis {
    let values = split_values(col_data);

    if values.is_empty() {
        return ColumnAnalysis {
            should_dict: false,
            dictionary: Vec::new(),
        };
    }

    // Count unique values.
    let mut freq: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    for val in &values {
        *freq.entry(*val).or_insert(0) += 1;
    }

    let unique_count = freq.len();

    // Skip if too many unique values for single-byte codes.
    if unique_count > MAX_DICT_ENTRIES {
        return ColumnAnalysis {
            should_dict: false,
            dictionary: Vec::new(),
        };
    }

    // Calculate savings: current size vs dictionary-encoded size.
    // Current: sum of value bytes + separators.
    let current_size = col_data.len();

    // Dictionary-encoded: 1 byte per value + separators + dictionary overhead.
    let dict_overhead: usize = freq.keys().map(|k| 4 + k.len()).sum::<usize>() + 1; // 1 byte marker + entries
    let encoded_data_size = values.len() + values.len().saturating_sub(1); // 1 byte per value + separators

    let dict_total = encoded_data_size + dict_overhead;

    // Only worth it if we save at least 10% of column size.
    let min_savings = current_size / 10;
    if current_size <= dict_total + min_savings {
        return ColumnAnalysis {
            should_dict: false,
            dictionary: Vec::new(),
        };
    }

    // Build dictionary sorted by frequency (most frequent first = lower code).
    let mut sorted: Vec<(&[u8], usize)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let dictionary: Vec<Vec<u8>> = sorted.into_iter().map(|(k, _)| k.to_vec()).collect();

    ColumnAnalysis {
        should_dict: true,
        dictionary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        // 3 columns: col 0 has many repeated values to overcome dict overhead,
        // col 1 has repeated values, col 2 has all unique values (not dictable).
        let mut col0 = Vec::new();
        let mut col1 = Vec::new();
        let mut col2 = Vec::new();
        for i in 0..20 {
            if i > 0 {
                col0.push(VAL_SEP);
                col1.push(VAL_SEP);
                col2.push(VAL_SEP);
            }
            col0.extend_from_slice(if i % 3 == 0 {
                b"page_view"
            } else {
                b"api_call"
            });
            col1.extend_from_slice(if i % 2 == 0 { b"alice" } else { b"bob" });
            col2.extend_from_slice(format!("unique{i}").as_bytes());
        }
        let mut input = col0;
        input.push(COL_SEP);
        input.extend_from_slice(&col1);
        input.push(COL_SEP);
        input.extend_from_slice(&col2);

        let result = preprocess(&input);
        assert!(result.is_some(), "should apply dict transform");
        let result = result.unwrap();
        let recovered = reverse(&result.data, &result.metadata);
        assert_eq!(recovered, input, "roundtrip failed");
    }

    #[test]
    fn roundtrip_all_unique() {
        // Every value is unique — dict should NOT be applied.
        let input = b"a\x01b\x01c\x00d\x01e\x01f";
        let result = preprocess(input);
        assert!(result.is_none(), "should not apply dict when all unique");
    }

    #[test]
    fn roundtrip_single_column() {
        let input = b"hello\x01hello\x01hello\x01hello\x01hello\x01hello\x01hello\x01hello\x01hello\x01hello";
        let result = preprocess(input);
        assert!(result.is_some(), "should apply dict for repeated values");
        let result = result.unwrap();
        assert!(result.data.len() < input.len(), "should be smaller");
        let recovered = reverse(&result.data, &result.metadata);
        assert_eq!(recovered, input.to_vec(), "roundtrip failed");
    }

    #[test]
    fn saves_space() {
        // Simulate NDJSON-like column with many repeated values.
        let mut col_data = Vec::new();
        for i in 0..100 {
            if i > 0 {
                col_data.push(VAL_SEP);
            }
            let val = match i % 5 {
                0 => b"page_view".as_slice(),
                1 => b"api_call".as_slice(),
                2 => b"click".as_slice(),
                3 => b"scroll".as_slice(),
                _ => b"form_submit".as_slice(),
            };
            col_data.extend_from_slice(val);
        }

        let result = preprocess(&col_data);
        assert!(result.is_some());
        let result = result.unwrap();
        let saving_pct = (col_data.len() - result.data.len()) * 100 / col_data.len();
        assert!(
            saving_pct > 70,
            "should save >70% on repeated values, got {saving_pct}%"
        );
        let recovered = reverse(&result.data, &result.metadata);
        assert_eq!(recovered, col_data, "roundtrip failed");
    }
}
