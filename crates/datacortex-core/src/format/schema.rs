//! Schema inference engine for columnar JSON/NDJSON data.
//!
//! Takes the output of `ndjson::preprocess` (columns separated by \x00,
//! values within columns separated by \x01) and infers the type of each
//! column by examining its values.
//!
//! The inferred schema can be serialized into compact binary metadata
//! for storage in .dcx transform metadata, and deserialized by the decoder.

use std::collections::HashSet;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;

// ─── Type Definitions ────────────────────────────────────────────────────────

/// Detected timestamp format.
#[derive(Debug, Clone, PartialEq)]
pub enum TimestampFormat {
    /// ISO 8601: "2026-03-15T10:30:00.001Z" or with offset
    Iso8601,
    /// Unix epoch in seconds: 1742036400
    EpochSeconds,
    /// Unix epoch in milliseconds: 1742036400001
    EpochMillis,
}

/// Inferred type of a single column.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Integer {
        min: i64,
        max: i64,
        nullable: bool,
    },
    Float {
        nullable: bool,
    },
    Boolean {
        nullable: bool,
    },
    Timestamp {
        format: TimestampFormat,
        nullable: bool,
    },
    Uuid {
        nullable: bool,
    },
    Enum {
        cardinality: u16,
        nullable: bool,
    },
    String {
        nullable: bool,
    },
    /// Column where every value is null.
    Null,
}

/// Schema for a single column: type + null statistics.
#[derive(Debug, Clone)]
pub struct ColumnSchema {
    pub col_type: ColumnType,
    pub null_count: usize,
    pub total_count: usize,
}

/// Inferred schema for an entire columnar dataset.
#[derive(Debug, Clone)]
pub struct InferredSchema {
    pub columns: Vec<ColumnSchema>,
}

// ─── Type Tag Constants (for serialization) ──────────────────────────────────

const TAG_NULL: u8 = 0;
const TAG_INTEGER: u8 = 1;
const TAG_FLOAT: u8 = 2;
const TAG_BOOLEAN: u8 = 3;
const TAG_TIMESTAMP_ISO: u8 = 4;
const TAG_TIMESTAMP_EPOCH_S: u8 = 5;
const TAG_TIMESTAMP_EPOCH_MS: u8 = 6;
const TAG_UUID: u8 = 7;
const TAG_ENUM: u8 = 8;
const TAG_STRING: u8 = 9;

const FLAG_NULLABLE: u8 = 0x01;

// ─── Value Classification ────────────────────────────────────────────────────

/// Classification of a single value for type inference.
#[derive(Debug, Clone, PartialEq)]
enum ValueType {
    Null,
    Boolean,
    Integer(i64),
    Float,
    TimestampIso,
    TimestampEpochS,
    TimestampEpochMs,
    Uuid,
    QuotedString,
}

/// Classify a single value from the columnar data.
fn classify_value(val: &[u8]) -> ValueType {
    if val == b"null" {
        return ValueType::Null;
    }
    if val == b"true" || val == b"false" {
        return ValueType::Boolean;
    }

    // Try integer: ^-?[0-9]+$
    if is_integer(val) {
        if let Some(n) = parse_i64(val) {
            // Check if it could be an epoch timestamp.
            // Seconds range: 946684800 (2000-01-01) .. 4102444800 (2099-12-31)
            // Millis range: those * 1000
            if n >= 0 {
                let nu = n as u64;
                if (946_684_800_000..=4_102_444_800_000).contains(&nu) {
                    return ValueType::TimestampEpochMs;
                }
                if (946_684_800..=4_102_444_800).contains(&nu) {
                    return ValueType::TimestampEpochS;
                }
            }
            return ValueType::Integer(n);
        }
    }

    // Try float: ^-?[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?$ or integer with exponent
    if is_float(val) {
        return ValueType::Float;
    }

    // Quoted value checks — must start and end with "
    if val.len() >= 2 && val[0] == b'"' && val[val.len() - 1] == b'"' {
        let inner = &val[1..val.len() - 1];

        // ISO 8601 timestamp: YYYY-MM-DDTHH:MM:SS...
        if is_iso8601(inner) {
            return ValueType::TimestampIso;
        }

        // UUID: 8-4-4-4-12 hex
        if is_uuid(inner) {
            return ValueType::Uuid;
        }

        return ValueType::QuotedString;
    }

    // Unquoted, non-null, non-bool, non-numeric — treat as string.
    ValueType::QuotedString
}

/// Check if bytes represent an integer: ^-?[0-9]+$
fn is_integer(val: &[u8]) -> bool {
    if val.is_empty() {
        return false;
    }
    let start = if val[0] == b'-' { 1 } else { 0 };
    if start >= val.len() {
        return false;
    }
    val[start..].iter().all(|&b| b.is_ascii_digit())
}

/// Parse bytes as i64, returning None on overflow.
fn parse_i64(val: &[u8]) -> Option<i64> {
    // Safe: we already validated it's ASCII digits with optional leading minus.
    let s = std::str::from_utf8(val).ok()?;
    s.parse::<i64>().ok()
}

/// Check if bytes represent a float:
/// ^-?[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?$ OR integer with exponent ^-?[0-9]+[eE][+-]?[0-9]+$
fn is_float(val: &[u8]) -> bool {
    if val.is_empty() {
        return false;
    }
    let s = match std::str::from_utf8(val) {
        Ok(s) => s,
        Err(_) => return false,
    };
    // Must parse as a valid float and contain either '.' or 'e'/'E'
    if s.parse::<f64>().is_err() {
        return false;
    }
    // Distinguish from pure integer: must have decimal point or exponent.
    val.iter().any(|&b| b == b'.' || b == b'e' || b == b'E')
}

/// Check if inner bytes (without surrounding quotes) match ISO 8601.
/// Pattern: YYYY-MM-DDTHH:MM:SS with optional fractional seconds and timezone.
fn is_iso8601(inner: &[u8]) -> bool {
    // Minimum: "2026-03-15T10:30:00" = 19 chars
    if inner.len() < 19 {
        return false;
    }
    // YYYY-MM-DDTHH:MM:SS
    if !inner[0].is_ascii_digit()
        || !inner[1].is_ascii_digit()
        || !inner[2].is_ascii_digit()
        || !inner[3].is_ascii_digit()
        || inner[4] != b'-'
        || !inner[5].is_ascii_digit()
        || !inner[6].is_ascii_digit()
        || inner[7] != b'-'
        || !inner[8].is_ascii_digit()
        || !inner[9].is_ascii_digit()
        || inner[10] != b'T'
        || !inner[11].is_ascii_digit()
        || !inner[12].is_ascii_digit()
        || inner[13] != b':'
        || !inner[14].is_ascii_digit()
        || !inner[15].is_ascii_digit()
        || inner[16] != b':'
        || !inner[17].is_ascii_digit()
        || !inner[18].is_ascii_digit()
    {
        return false;
    }
    // After the base datetime, allow: nothing, .fractional, Z, +HH:MM, -HH:MM, or combos.
    let rest = &inner[19..];
    if rest.is_empty() {
        return true;
    }
    let mut pos = 0;
    // Optional fractional seconds: .NNN...
    if pos < rest.len() && rest[pos] == b'.' {
        pos += 1;
        if pos >= rest.len() || !rest[pos].is_ascii_digit() {
            return false;
        }
        while pos < rest.len() && rest[pos].is_ascii_digit() {
            pos += 1;
        }
    }
    // Optional timezone: Z or +HH:MM or -HH:MM
    if pos < rest.len() {
        match rest[pos] {
            b'Z' => {
                pos += 1;
            }
            b'+' | b'-' => {
                pos += 1;
                // Expect HH:MM (5 chars)
                if pos + 5 > rest.len() {
                    return false;
                }
                if !rest[pos].is_ascii_digit()
                    || !rest[pos + 1].is_ascii_digit()
                    || rest[pos + 2] != b':'
                    || !rest[pos + 3].is_ascii_digit()
                    || !rest[pos + 4].is_ascii_digit()
                {
                    return false;
                }
                pos += 5;
            }
            _ => return false,
        }
    }
    pos == rest.len()
}

/// Check if inner bytes (without surrounding quotes) match UUID format.
/// 8-4-4-4-12 hex characters.
fn is_uuid(inner: &[u8]) -> bool {
    // Exactly 36 chars: 8-4-4-4-12
    if inner.len() != 36 {
        return false;
    }
    let groups = [
        (0, 8),   // 8 hex
        (9, 13),  // 4 hex
        (14, 18), // 4 hex
        (19, 23), // 4 hex
        (24, 36), // 12 hex
    ];
    // Check dashes at positions 8, 13, 18, 23
    if inner[8] != b'-' || inner[13] != b'-' || inner[18] != b'-' || inner[23] != b'-' {
        return false;
    }
    for &(start, end) in &groups {
        for &b in &inner[start..end] {
            if !b.is_ascii_hexdigit() {
                return false;
            }
        }
    }
    true
}

// ─── Schema Inference ────────────────────────────────────────────────────────

/// Infer schema from columnar data (post `ndjson::preprocess` output).
///
/// Data format: columns separated by \x00, values within columns by \x01.
pub fn infer_schema(columnar_data: &[u8]) -> InferredSchema {
    if columnar_data.is_empty() {
        return InferredSchema {
            columns: Vec::new(),
        };
    }

    let col_chunks: Vec<&[u8]> = columnar_data.split(|&b| b == COL_SEP).collect();
    let mut columns = Vec::with_capacity(col_chunks.len());

    for col_data in &col_chunks {
        let values: Vec<&[u8]> = col_data.split(|&b| b == VAL_SEP).collect();
        let total_count = values.len();

        // Classify every value.
        let mut null_count: usize = 0;
        let mut classifications: Vec<ValueType> = Vec::with_capacity(total_count);

        for val in &values {
            let vt = classify_value(val);
            if vt == ValueType::Null {
                null_count += 1;
            }
            classifications.push(vt);
        }

        let non_null: Vec<&ValueType> = classifications
            .iter()
            .filter(|c| **c != ValueType::Null)
            .collect();
        let nullable = null_count > 0;

        let col_type = if non_null.is_empty() {
            // All null.
            ColumnType::Null
        } else if non_null.iter().all(|c| matches!(c, ValueType::Boolean)) {
            ColumnType::Boolean { nullable }
        } else if non_null.iter().all(|c| matches!(c, ValueType::Integer(_))) {
            let mut min = i64::MAX;
            let mut max = i64::MIN;
            for c in &non_null {
                if let ValueType::Integer(n) = c {
                    if *n < min {
                        min = *n;
                    }
                    if *n > max {
                        max = *n;
                    }
                }
            }
            ColumnType::Integer { min, max, nullable }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::Integer(_) | ValueType::Float))
        {
            // Mixed int+float => Float
            ColumnType::Float { nullable }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::TimestampIso))
        {
            ColumnType::Timestamp {
                format: TimestampFormat::Iso8601,
                nullable,
            }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::TimestampEpochS))
        {
            ColumnType::Timestamp {
                format: TimestampFormat::EpochSeconds,
                nullable,
            }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::TimestampEpochMs))
        {
            ColumnType::Timestamp {
                format: TimestampFormat::EpochMillis,
                nullable,
            }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::TimestampEpochS | ValueType::TimestampEpochMs))
        {
            // Mixed epoch seconds and millis — pick millis as the broader type.
            ColumnType::Timestamp {
                format: TimestampFormat::EpochMillis,
                nullable,
            }
        } else if non_null.iter().all(|c| {
            matches!(
                c,
                ValueType::Integer(_) | ValueType::TimestampEpochS | ValueType::TimestampEpochMs
            )
        }) {
            // Mixed integers and epoch timestamps — the epoch classification was a
            // heuristic guess.  Since not ALL values look like timestamps, treat the
            // whole column as plain integers.  Epoch timestamps are just integers
            // that happen to fall in a certain range.
            let mut min = i64::MAX;
            let mut max = i64::MIN;
            for val in &values {
                let vt = classify_value(val);
                if vt == ValueType::Null {
                    continue;
                }
                // All non-null values in this branch are numeric (Integer or
                // epoch timestamp), so parse_i64 will succeed.
                if let Some(n) = parse_i64(val) {
                    if n < min {
                        min = n;
                    }
                    if n > max {
                        max = n;
                    }
                }
            }
            ColumnType::Integer { min, max, nullable }
        } else if non_null.iter().all(|c| matches!(c, ValueType::Uuid)) {
            ColumnType::Uuid { nullable }
        } else if non_null
            .iter()
            .all(|c| matches!(c, ValueType::QuotedString))
        {
            // Check cardinality for Enum vs String.
            let mut unique_vals: HashSet<&[u8]> = HashSet::new();
            for val in &values {
                let vt = classify_value(val);
                if vt != ValueType::Null {
                    unique_vals.insert(val);
                }
            }
            let cardinality = unique_vals.len();
            if cardinality <= 256 {
                ColumnType::Enum {
                    cardinality: cardinality as u16,
                    nullable,
                }
            } else {
                ColumnType::String { nullable }
            }
        } else {
            // Mixed types that don't fit any unified category.
            ColumnType::String { nullable }
        };

        columns.push(ColumnSchema {
            col_type,
            null_count,
            total_count,
        });
    }

    InferredSchema { columns }
}

// ─── Serialization ───────────────────────────────────────────────────────────

/// Serialize schema to compact binary bytes for transform metadata.
///
/// Format:
///   Header: num_columns (u16 LE)
///   Per column:
///     byte 0: type tag
///     byte 1: flags (bit 0 = nullable)
///     [type-specific data]:
///       Integer: 8 bytes min (i64 LE) + 8 bytes max (i64 LE)
///       Enum: 2 bytes cardinality (u16 LE)
///       Others: no extra data
pub fn serialize_schema(schema: &InferredSchema) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(schema.columns.len() as u16).to_le_bytes());

    for col in &schema.columns {
        let (tag, flags, extra) = match &col.col_type {
            ColumnType::Null => (TAG_NULL, 0u8, Vec::new()),
            ColumnType::Integer { min, max, nullable } => {
                let mut extra = Vec::with_capacity(16);
                extra.extend_from_slice(&min.to_le_bytes());
                extra.extend_from_slice(&max.to_le_bytes());
                (
                    TAG_INTEGER,
                    if *nullable { FLAG_NULLABLE } else { 0 },
                    extra,
                )
            }
            ColumnType::Float { nullable } => (
                TAG_FLOAT,
                if *nullable { FLAG_NULLABLE } else { 0 },
                Vec::new(),
            ),
            ColumnType::Boolean { nullable } => (
                TAG_BOOLEAN,
                if *nullable { FLAG_NULLABLE } else { 0 },
                Vec::new(),
            ),
            ColumnType::Timestamp { format, nullable } => {
                let tag = match format {
                    TimestampFormat::Iso8601 => TAG_TIMESTAMP_ISO,
                    TimestampFormat::EpochSeconds => TAG_TIMESTAMP_EPOCH_S,
                    TimestampFormat::EpochMillis => TAG_TIMESTAMP_EPOCH_MS,
                };
                (tag, if *nullable { FLAG_NULLABLE } else { 0 }, Vec::new())
            }
            ColumnType::Uuid { nullable } => (
                TAG_UUID,
                if *nullable { FLAG_NULLABLE } else { 0 },
                Vec::new(),
            ),
            ColumnType::Enum {
                cardinality,
                nullable,
            } => {
                let mut extra = Vec::with_capacity(2);
                extra.extend_from_slice(&cardinality.to_le_bytes());
                (TAG_ENUM, if *nullable { FLAG_NULLABLE } else { 0 }, extra)
            }
            ColumnType::String { nullable } => (
                TAG_STRING,
                if *nullable { FLAG_NULLABLE } else { 0 },
                Vec::new(),
            ),
        };
        out.push(tag);
        out.push(flags);
        out.extend_from_slice(&extra);
    }

    out
}

/// Deserialize schema from transform metadata bytes.
pub fn deserialize_schema(data: &[u8]) -> InferredSchema {
    if data.len() < 2 {
        return InferredSchema {
            columns: Vec::new(),
        };
    }

    let num_columns = u16::from_le_bytes(data[0..2].try_into().unwrap()) as usize;
    let mut pos = 2;
    let mut columns = Vec::with_capacity(num_columns);

    for _ in 0..num_columns {
        if pos + 2 > data.len() {
            break;
        }
        let tag = data[pos];
        pos += 1;
        let flags = data[pos];
        pos += 1;
        let nullable = (flags & FLAG_NULLABLE) != 0;

        let col_type = match tag {
            TAG_NULL => ColumnType::Null,
            TAG_INTEGER => {
                if pos + 16 > data.len() {
                    break;
                }
                let min = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let max = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                ColumnType::Integer { min, max, nullable }
            }
            TAG_FLOAT => ColumnType::Float { nullable },
            TAG_BOOLEAN => ColumnType::Boolean { nullable },
            TAG_TIMESTAMP_ISO => ColumnType::Timestamp {
                format: TimestampFormat::Iso8601,
                nullable,
            },
            TAG_TIMESTAMP_EPOCH_S => ColumnType::Timestamp {
                format: TimestampFormat::EpochSeconds,
                nullable,
            },
            TAG_TIMESTAMP_EPOCH_MS => ColumnType::Timestamp {
                format: TimestampFormat::EpochMillis,
                nullable,
            },
            TAG_UUID => ColumnType::Uuid { nullable },
            TAG_ENUM => {
                if pos + 2 > data.len() {
                    break;
                }
                let cardinality = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                pos += 2;
                ColumnType::Enum {
                    cardinality,
                    nullable,
                }
            }
            TAG_STRING => ColumnType::String { nullable },
            _ => ColumnType::String { nullable }, // Unknown tag fallback.
        };

        columns.push(ColumnSchema {
            col_type,
            null_count: 0,  // Not stored in serialized form.
            total_count: 0, // Not stored in serialized form.
        });
    }

    InferredSchema { columns }
}

// ─── Helper trait for test convenience ───────────────────────────────────────

impl ColumnType {
    /// Extract max from Integer variant (test helper).
    #[cfg(test)]
    fn integer_max(self) -> Option<i64> {
        match self {
            ColumnType::Integer { max, .. } => Some(max),
            _ => None,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build columnar data from column value slices.
    /// Each inner slice is one column's values.
    fn build_columnar(columns: &[&[&[u8]]]) -> Vec<u8> {
        let mut out = Vec::new();
        for (ci, col) in columns.iter().enumerate() {
            for (vi, val) in col.iter().enumerate() {
                out.extend_from_slice(val);
                if vi < col.len() - 1 {
                    out.push(VAL_SEP);
                }
            }
            if ci < columns.len() - 1 {
                out.push(COL_SEP);
            }
        }
        out
    }

    #[test]
    fn test_infer_integers() {
        let data = build_columnar(&[&[b"1", b"2", b"300", b"-5"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Integer {
                min: -5,
                max: 300,
                nullable: false,
            }
        );
        assert_eq!(schema.columns[0].null_count, 0);
        assert_eq!(schema.columns[0].total_count, 4);
    }

    #[test]
    fn test_infer_mixed_integer_and_epoch_as_integer() {
        // Regression: 2147483647 (i32::MAX) falls in the epoch-seconds range
        // and was misclassified as TimestampEpochS.  When mixed with plain
        // integers, the column should be inferred as Integer, not String.
        let data = build_columnar(&[&[
            b"0",
            b"-1",
            b"1",
            b"-2147483648",
            b"2147483647",
            b"-9007199254740991",
            b"9007199254740991",
        ]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Integer {
                min: -9007199254740991,
                max: 9007199254740991,
                nullable: false,
            },
            "mixed integers with epoch-range values should infer as Integer, got {:?}",
            schema.columns[0].col_type
        );
    }

    #[test]
    fn test_infer_floats() {
        let data = build_columnar(&[&[b"3.14", b"2.718", b"1.0"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Float { nullable: false }
        );
    }

    #[test]
    fn test_infer_booleans() {
        let data = build_columnar(&[&[b"true", b"false", b"true"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Boolean { nullable: false }
        );
    }

    #[test]
    fn test_infer_timestamps() {
        let data = build_columnar(&[&[
            br#""2026-03-15T10:30:00.001Z""#.as_slice(),
            br#""2026-03-15T10:30:00.234Z""#.as_slice(),
            br#""2026-03-15T10:30:01.000Z""#.as_slice(),
        ]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Timestamp {
                format: TimestampFormat::Iso8601,
                nullable: false,
            }
        );
    }

    #[test]
    fn test_infer_timestamps_with_offset() {
        let data = build_columnar(&[&[
            br#""2026-03-15T10:30:00+05:30""#.as_slice(),
            br#""2026-03-15T10:30:00-04:00""#.as_slice(),
        ]]);
        let schema = infer_schema(&data);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Timestamp {
                format: TimestampFormat::Iso8601,
                nullable: false,
            }
        );
    }

    #[test]
    fn test_infer_uuids() {
        let data = build_columnar(&[&[
            br#""550e8400-e29b-41d4-a716-446655440000""#.as_slice(),
            br#""6ba7b810-9dad-11d1-80b4-00c04fd430c8""#.as_slice(),
            br#""f47ac10b-58cc-4372-a567-0e02b2c3d479""#.as_slice(),
        ]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Uuid { nullable: false }
        );
    }

    #[test]
    fn test_infer_enums() {
        let data = build_columnar(&[&[
            br#""page_view""#.as_slice(),
            br#""api_call""#.as_slice(),
            br#""click""#.as_slice(),
            br#""page_view""#.as_slice(),
            br#""scroll""#.as_slice(),
            br#""api_call""#.as_slice(),
        ]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        match &schema.columns[0].col_type {
            ColumnType::Enum {
                cardinality,
                nullable,
            } => {
                assert_eq!(*cardinality, 4); // page_view, api_call, click, scroll
                assert!(!nullable);
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_strings() {
        // High cardinality: every value unique, > 256 unique values.
        let vals: Vec<Vec<u8>> = (0..300)
            .map(|i| format!("\"unique_value_{}\"", i).into_bytes())
            .collect();
        let val_refs: Vec<&[u8]> = vals.iter().map(|v| v.as_slice()).collect();
        let data = build_columnar(&[&val_refs]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::String { nullable: false }
        );
    }

    #[test]
    fn test_infer_nullable() {
        let data = build_columnar(&[&[b"1", b"null", b"3", b"null", b"5"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Integer {
                min: 1,
                max: 5,
                nullable: true,
            }
        );
        assert_eq!(schema.columns[0].null_count, 2);
        assert_eq!(schema.columns[0].total_count, 5);
    }

    #[test]
    fn test_infer_mixed_int_float() {
        let data = build_columnar(&[&[b"1", b"2.5", b"3"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Float { nullable: false }
        );
    }

    #[test]
    fn test_infer_all_null() {
        let data = build_columnar(&[&[b"null", b"null", b"null"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(schema.columns[0].col_type, ColumnType::Null);
        assert_eq!(schema.columns[0].null_count, 3);
    }

    #[test]
    fn test_schema_roundtrip() {
        // Build a schema with every column type.
        let schema = InferredSchema {
            columns: vec![
                ColumnSchema {
                    col_type: ColumnType::Null,
                    null_count: 10,
                    total_count: 10,
                },
                ColumnSchema {
                    col_type: ColumnType::Integer {
                        min: -100,
                        max: 999,
                        nullable: true,
                    },
                    null_count: 2,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Float { nullable: false },
                    null_count: 0,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Boolean { nullable: true },
                    null_count: 1,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Timestamp {
                        format: TimestampFormat::Iso8601,
                        nullable: false,
                    },
                    null_count: 0,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Timestamp {
                        format: TimestampFormat::EpochSeconds,
                        nullable: true,
                    },
                    null_count: 3,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Timestamp {
                        format: TimestampFormat::EpochMillis,
                        nullable: false,
                    },
                    null_count: 0,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Uuid { nullable: false },
                    null_count: 0,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::Enum {
                        cardinality: 7,
                        nullable: true,
                    },
                    null_count: 5,
                    total_count: 50,
                },
                ColumnSchema {
                    col_type: ColumnType::String { nullable: false },
                    null_count: 0,
                    total_count: 50,
                },
            ],
        };

        let bytes = serialize_schema(&schema);
        let recovered = deserialize_schema(&bytes);

        assert_eq!(recovered.columns.len(), schema.columns.len());
        for (orig, rec) in schema.columns.iter().zip(recovered.columns.iter()) {
            assert_eq!(orig.col_type, rec.col_type);
        }
    }

    #[test]
    fn test_serialize_size() {
        // Verify serialization is compact.
        let schema = InferredSchema {
            columns: vec![
                ColumnSchema {
                    col_type: ColumnType::Integer {
                        min: 0,
                        max: 1000,
                        nullable: false,
                    },
                    null_count: 0,
                    total_count: 100,
                },
                ColumnSchema {
                    col_type: ColumnType::String { nullable: true },
                    null_count: 5,
                    total_count: 100,
                },
            ],
        };
        let bytes = serialize_schema(&schema);
        // Header: 2 bytes
        // Integer column: 2 (tag+flags) + 16 (min+max) = 18
        // String column: 2 (tag+flags) = 2
        // Total: 2 + 18 + 2 = 22
        assert_eq!(bytes.len(), 22);
    }

    #[test]
    fn test_empty_input() {
        let schema = infer_schema(b"");
        assert!(schema.columns.is_empty());
    }

    #[test]
    fn test_multi_column() {
        // Two columns: integers and booleans.
        let data = build_columnar(&[&[b"1", b"2", b"3"], &[b"true", b"false", b"true"]]);
        let schema = infer_schema(&data);
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Integer {
                min: 1,
                max: 3,
                nullable: false,
            }
        );
        assert_eq!(
            schema.columns[1].col_type,
            ColumnType::Boolean { nullable: false }
        );
    }

    #[test]
    fn test_epoch_seconds() {
        // Values in the epoch seconds range.
        let data = build_columnar(&[&[b"1742036400", b"1742036500", b"1742036600"]]);
        let schema = infer_schema(&data);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Timestamp {
                format: TimestampFormat::EpochSeconds,
                nullable: false,
            }
        );
    }

    #[test]
    fn test_epoch_millis() {
        let data = build_columnar(&[&[b"1742036400001", b"1742036400234", b"1742036401000"]]);
        let schema = infer_schema(&data);
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Timestamp {
                format: TimestampFormat::EpochMillis,
                nullable: false,
            }
        );
    }

    #[test]
    fn test_real_ndjson_corpus() {
        // Read the test corpus, run through ndjson::preprocess, then infer schema.
        let corpus = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .expect("failed to read test-ndjson.ndjson");

        let transform_result =
            crate::format::ndjson::preprocess(&corpus).expect("ndjson::preprocess failed");

        let schema = infer_schema(&transform_result.data);

        // The corpus has 20 columns (keys per JSON line):
        // timestamp, event_type, user_id, session_id, page, referrer,
        // user_agent, ip_hash, country, region, city, device, browser,
        // browser_version, os, duration_ms, is_authenticated, plan, metadata
        //
        // All 200 lines have the same schema.
        assert!(
            schema.columns.len() >= 19,
            "expected at least 19 columns, got {}",
            schema.columns.len()
        );

        // Find columns by examining the corpus key order.
        // Column 0: timestamp — ISO 8601 strings like "2026-03-15T10:30:00.081Z"
        assert_eq!(
            schema.columns[0].col_type,
            ColumnType::Timestamp {
                format: TimestampFormat::Iso8601,
                nullable: false,
            },
            "column 0 (timestamp) should be Timestamp/Iso8601"
        );

        // Column 1: event_type — low cardinality quoted strings
        match &schema.columns[1].col_type {
            ColumnType::Enum {
                cardinality,
                nullable,
            } => {
                assert!(*cardinality <= 20, "event_type cardinality should be low");
                assert!(!nullable, "event_type should not be nullable");
            }
            other => panic!("column 1 (event_type) should be Enum, got {:?}", other),
        }

        // Column 2: user_id — quoted strings like "usr_a1b2c3d4"
        match &schema.columns[2].col_type {
            ColumnType::Enum { .. } | ColumnType::String { .. } => {
                // user_id with limited users could be Enum or String.
            }
            other => panic!(
                "column 2 (user_id) should be Enum or String, got {:?}",
                other
            ),
        }

        // Column 15: duration_ms — integers
        assert_eq!(
            schema.columns[15].col_type,
            ColumnType::Integer {
                min: 0,
                max: schema.columns[15]
                    .col_type
                    .clone()
                    .integer_max()
                    .unwrap_or(0),
                nullable: false,
            },
            "column 15 (duration_ms) should be Integer"
        );

        // Column 16: is_authenticated — booleans
        assert_eq!(
            schema.columns[16].col_type,
            ColumnType::Boolean { nullable: false },
            "column 16 (is_authenticated) should be Boolean"
        );

        // Column 5: referrer — has null values
        match &schema.columns[5].col_type {
            ColumnType::Enum { nullable, .. } | ColumnType::String { nullable } => {
                assert!(*nullable, "column 5 (referrer) should be nullable");
            }
            other => panic!(
                "column 5 (referrer) should be nullable Enum/String, got {:?}",
                other
            ),
        }
    }
}
