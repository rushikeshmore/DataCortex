//! Typed encoding — type-specific binary encoding for columnar data.
//!
//! Takes columnar data (from ndjson::preprocess or json_array::preprocess),
//! runs schema inference, and applies type-specific encoders per column:
//!
//!   - Integer columns: delta + zigzag + LEB128 varint
//!   - Boolean columns: bitmap packing (8 bools per byte)
//!   - Timestamp columns: epoch-micros delta + zigzag + LEB128 varint
//!   - Enum columns: dictionary + ordinal bytes
//!   - Nullable columns: null bitmap + typed sub-encoding
//!   - Everything else: raw passthrough (kept as \x01-separated text)
//!
//! The output is a compact binary format with a header containing the schema
//! and per-column encoding metadata, followed by concatenated column data.
//!
//! Output format:
//! ```text
//! [Header]
//!   num_columns: u16 LE
//!   schema_len:  u32 LE
//!   schema_bytes: [serialized InferredSchema]
//!
//! [Per Column]
//!   encoding_type: u8 (0=Raw, 1=DeltaVarint, 2=Bitmap, 3=NullBitmap+Typed,
//!                       4=Timestamp, 5=Enum)
//!   data_length: u32 LE
//!   [column data bytes]
//! ```

use super::schema::{self, ColumnType, TimestampFormat};
use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
const VAL_SEP: u8 = 0x01;

// Encoding type tags (stored per column in the binary output).
const ENC_RAW: u8 = 0;
const ENC_DELTA_VARINT: u8 = 1;
const ENC_BITMAP: u8 = 2;
const ENC_NULL_TYPED: u8 = 3;
const ENC_TIMESTAMP: u8 = 4;
const ENC_ENUM: u8 = 5;

// ─── ZigZag + LEB128 Primitives ─────────────────────────────────────────────

/// ZigZag encode: maps signed i64 to unsigned u64.
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
#[inline]
fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// ZigZag decode: maps unsigned u64 back to signed i64.
#[inline]
fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

/// LEB128 varint encode: write a u64 into variable-length bytes.
fn leb128_encode(mut val: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if val == 0 {
            break;
        }
    }
}

/// LEB128 varint decode: read a u64 from bytes at the given position.
/// Returns (value, new_position).
fn leb128_decode(data: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if pos >= data.len() {
            return None;
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return None; // Overflow protection.
        }
    }
}

// ─── Integer Encoder (DeltaVarint) ──────────────────────────────────────────

/// Encode an integer column using delta + zigzag + LEB128 varint.
///
/// Input: slices of text bytes, each representing an i64 (e.g., b"42", b"-7").
/// Output: binary varint stream.
fn encode_integer_column(values: &[&[u8]]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(values.len() * 2);
    let mut prev: i64 = 0;

    for (i, val) in values.iter().enumerate() {
        let n = parse_i64(val);
        let delta = if i == 0 { n } else { n - prev };
        let zz = zigzag_encode(delta);
        leb128_encode(zz, &mut out);
        prev = n;
    }

    out
}

/// Decode an integer column from delta + zigzag + LEB128 varint back to text.
///
/// Returns a Vec of byte vectors, each containing the text representation
/// of the decoded i64 (e.g., vec![b"42", b"-7"]).
fn decode_integer_column(data: &[u8], count: usize) -> Vec<Vec<u8>> {
    let mut result = Vec::with_capacity(count);
    let mut pos = 0;
    let mut prev: i64 = 0;

    for i in 0..count {
        let (zz, new_pos) = match leb128_decode(data, pos) {
            Some(v) => v,
            None => {
                // If data is truncated, pad with zeros.
                for _ in i..count {
                    result.push(b"0".to_vec());
                }
                return result;
            }
        };
        pos = new_pos;
        let delta = zigzag_decode(zz);
        let n = if i == 0 { delta } else { prev + delta };
        prev = n;
        result.push(format_i64(n));
    }

    result
}

// ─── Boolean Encoder (Bitmap) ───────────────────────────────────────────────

/// Encode a boolean column as a packed bitmap.
///
/// Input: slices of text bytes (b"true" or b"false").
/// Output: count (u32 LE) + packed bytes (bit 0 of byte 0 = first value).
fn encode_boolean_column(values: &[&[u8]]) -> Vec<u8> {
    let count = values.len();
    let num_bytes = count.div_ceil(8);
    let mut out = Vec::with_capacity(4 + num_bytes);

    // Store count as u32 LE.
    out.extend_from_slice(&(count as u32).to_le_bytes());

    let mut packed = vec![0u8; num_bytes];
    for (i, val) in values.iter().enumerate() {
        if is_true(val) {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    out.extend_from_slice(&packed);

    out
}

/// Decode a boolean column from packed bitmap back to text.
///
/// Returns a Vec of byte vectors (b"true" or b"false").
fn decode_boolean_column(data: &[u8]) -> Vec<Vec<u8>> {
    if data.len() < 4 {
        return Vec::new();
    }

    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let packed = &data[4..];
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1 {
            result.push(b"true".to_vec());
        } else {
            result.push(b"false".to_vec());
        }
    }

    result
}

// ─── Timestamp Encoder ──────────────────────────────────────────────────────

/// Timestamp format byte values for the column header.
/// These encode the exact format variation so decoding reproduces byte-exact output.
const TS_FMT_Z: u8 = 0;       // "...Z" suffix
const TS_FMT_OFFSET: u8 = 1;  // "+HH:MM" or "-HH:MM" suffix
const TS_FMT_BARE: u8 = 2;    // No timezone suffix

/// Parse an ISO 8601 timestamp string (WITH surrounding quotes) to epoch microseconds.
/// Returns (epoch_micros, format_byte, tz_offset_minutes, frac_digits).
fn parse_iso8601_to_micros(val: &[u8]) -> Option<(u64, u8, i16, u8)> {
    // Must be quoted: "2026-03-15T10:30:00.081Z"
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"' {
        return None;
    }
    let inner = &val[1..val.len() - 1];
    if inner.len() < 19 {
        return None;
    }

    // Parse YYYY-MM-DDThh:mm:ss
    let year = parse_digits(inner, 0, 4)? as i64;
    if inner[4] != b'-' { return None; }
    let month = parse_digits(inner, 5, 2)? as u32;
    if inner[7] != b'-' { return None; }
    let day = parse_digits(inner, 8, 2)? as u32;
    if inner[10] != b'T' { return None; }
    let hour = parse_digits(inner, 11, 2)? as u64;
    if inner[13] != b':' { return None; }
    let minute = parse_digits(inner, 14, 2)? as u64;
    if inner[16] != b':' { return None; }
    let second = parse_digits(inner, 17, 2)? as u64;

    // Parse optional fractional seconds.
    let mut pos = 19;
    let mut frac_micros: u64 = 0;
    let mut frac_digits: u8 = 0;

    if pos < inner.len() && inner[pos] == b'.' {
        pos += 1;
        let frac_start = pos;
        while pos < inner.len() && inner[pos].is_ascii_digit() {
            pos += 1;
        }
        frac_digits = (pos - frac_start) as u8;
        if frac_digits > 0 {
            let frac_val = parse_digits(inner, frac_start, frac_digits as usize)?;
            // Normalize to microseconds (6 digits).
            frac_micros = match frac_digits {
                1 => frac_val * 100_000,
                2 => frac_val * 10_000,
                3 => frac_val * 1_000,
                4 => frac_val * 100,
                5 => frac_val * 10,
                6 => frac_val,
                _ => {
                    // More than 6 digits: truncate to 6.
                    let s = std::str::from_utf8(&inner[frac_start..frac_start + 6]).ok()?;
                    s.parse::<u64>().ok()?
                }
            };
        }
    }

    // Parse timezone.
    let mut format_byte = TS_FMT_BARE;
    let mut tz_offset_minutes: i16 = 0;

    if pos < inner.len() {
        match inner[pos] {
            b'Z' => {
                format_byte = TS_FMT_Z;
                pos += 1;
            }
            b'+' | b'-' => {
                format_byte = TS_FMT_OFFSET;
                let sign: i16 = if inner[pos] == b'+' { 1 } else { -1 };
                pos += 1;
                if pos + 5 > inner.len() { return None; }
                let tz_h = parse_digits(inner, pos, 2)? as i16;
                if inner[pos + 2] != b':' { return None; }
                let tz_m = parse_digits(inner, pos + 3, 2)? as i16;
                tz_offset_minutes = sign * (tz_h * 60 + tz_m);
                pos += 5;
            }
            _ => return None,
        }
    }

    // Ensure we consumed everything.
    if pos != inner.len() {
        return None;
    }

    // Convert date to days since epoch.
    let days = days_since_epoch(year, month, day)?;
    let total_seconds = days as u64 * 86400 + hour * 3600 + minute * 60 + second;

    // Adjust for timezone offset (convert to UTC).
    let total_seconds_utc = if tz_offset_minutes != 0 {
        (total_seconds as i64 - tz_offset_minutes as i64 * 60) as u64
    } else {
        total_seconds
    };

    let epoch_micros = total_seconds_utc * 1_000_000 + frac_micros;
    Some((epoch_micros, format_byte, tz_offset_minutes, frac_digits))
}

/// Convert epoch microseconds back to an ISO 8601 string WITH surrounding quotes.
fn micros_to_iso8601(epoch_micros: u64, format_byte: u8, tz_offset_minutes: i16, frac_digits: u8) -> Vec<u8> {
    // Adjust from UTC to local time if there's a timezone offset.
    let adjusted_micros = if tz_offset_minutes != 0 {
        (epoch_micros as i64 + tz_offset_minutes as i64 * 60 * 1_000_000) as u64
    } else {
        epoch_micros
    };

    let total_seconds = adjusted_micros / 1_000_000;
    let frac_micros = adjusted_micros % 1_000_000;

    let days = (total_seconds / 86400) as i64;
    let day_seconds = total_seconds % 86400;
    let hour = day_seconds / 3600;
    let minute = (day_seconds % 3600) / 60;
    let second = day_seconds % 60;

    let (year, month, day) = date_from_epoch_days(days);

    // Build the string.
    let mut out = Vec::with_capacity(32);
    out.push(b'"');

    // YYYY-MM-DDThh:mm:ss
    write_digits(&mut out, year as u64, 4);
    out.push(b'-');
    write_digits(&mut out, month as u64, 2);
    out.push(b'-');
    write_digits(&mut out, day as u64, 2);
    out.push(b'T');
    write_digits(&mut out, hour, 2);
    out.push(b':');
    write_digits(&mut out, minute, 2);
    out.push(b':');
    write_digits(&mut out, second, 2);

    // Fractional seconds.
    if frac_digits > 0 {
        out.push(b'.');
        let frac_val = match frac_digits {
            1 => frac_micros / 100_000,
            2 => frac_micros / 10_000,
            3 => frac_micros / 1_000,
            4 => frac_micros / 100,
            5 => frac_micros / 10,
            6 => frac_micros,
            _ => frac_micros,
        };
        write_digits(&mut out, frac_val, frac_digits as usize);
    }

    // Timezone.
    match format_byte {
        TS_FMT_Z => out.push(b'Z'),
        TS_FMT_OFFSET => {
            let abs_offset = tz_offset_minutes.unsigned_abs() as u64;
            if tz_offset_minutes >= 0 {
                out.push(b'+');
            } else {
                out.push(b'-');
            }
            write_digits(&mut out, abs_offset / 60, 2);
            out.push(b':');
            write_digits(&mut out, abs_offset % 60, 2);
        }
        _ => {} // TS_FMT_BARE: no suffix.
    }

    out.push(b'"');
    out
}

/// Encode a timestamp column.
///
/// Column header:
///   base_value: u64 LE (8 bytes) — first timestamp in epoch micros
///   format_byte: u8 — which format variation
///   tz_offset_minutes: i16 LE (2 bytes)
///   frac_digits: u8
///   count: u32 LE (4 bytes)
/// Followed by delta+zigzag+LEB128 varint stream.
fn encode_timestamp_column(values: &[&[u8]]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    // Parse first value to get format metadata.
    let first = match parse_iso8601_to_micros(values[0]) {
        Some(v) => v,
        None => return encode_raw(values), // Fallback.
    };
    let (base_micros, format_byte, tz_offset_minutes, frac_digits) = first;

    let mut out = Vec::with_capacity(16 + values.len() * 2);

    // Column header.
    out.extend_from_slice(&base_micros.to_le_bytes());
    out.push(format_byte);
    out.extend_from_slice(&tz_offset_minutes.to_le_bytes());
    out.push(frac_digits);
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());

    // Delta encode: first value has delta=0, subsequent values are delta from prev.
    let mut prev_micros = base_micros;

    for (i, val) in values.iter().enumerate() {
        let micros = if i == 0 {
            base_micros
        } else {
            match parse_iso8601_to_micros(val) {
                Some((m, _, _, _)) => m,
                None => prev_micros, // Fallback: repeat previous.
            }
        };
        let delta = micros as i64 - prev_micros as i64;
        let zz = zigzag_encode(delta);
        leb128_encode(zz, &mut out);
        prev_micros = micros;
    }

    out
}

/// Decode a timestamp column from delta+zigzag+LEB128 back to quoted ISO 8601 strings.
fn decode_timestamp_column(data: &[u8]) -> Vec<Vec<u8>> {
    // Read column header: base_value(8) + format_byte(1) + tz_offset(2) + frac_digits(1) + count(4) = 16 bytes
    if data.len() < 16 {
        return decode_raw(data);
    }

    let mut pos = 0;
    let base_micros = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let format_byte = data[pos];
    pos += 1;
    let tz_offset_minutes = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
    pos += 2;
    let frac_digits = data[pos];
    pos += 1;
    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut result = Vec::with_capacity(count);
    let mut prev_micros = base_micros;

    for i in 0..count {
        let (zz, new_pos) = match leb128_decode(data, pos) {
            Some(v) => v,
            None => {
                // Pad remaining with base value.
                for _ in i..count {
                    result.push(micros_to_iso8601(prev_micros, format_byte, tz_offset_minutes, frac_digits));
                }
                return result;
            }
        };
        pos = new_pos;
        let delta = zigzag_decode(zz);
        let micros = (prev_micros as i64 + delta) as u64;
        result.push(micros_to_iso8601(micros, format_byte, tz_offset_minutes, frac_digits));
        prev_micros = micros;
    }

    result
}

// ─── Enum Encoder ───────────────────────────────────────────────────────────

/// Encode an enum column using dictionary + ordinal bytes.
///
/// Column header:
///   dict_count: u8
///   for each entry: len (u16 LE) + bytes (includes quotes)
///   count: u32 LE
/// Followed by one ordinal byte per value.
fn encode_enum_column(values: &[&[u8]]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    // Count frequency of each unique value.
    let mut freq: Vec<(&[u8], usize)> = Vec::new();
    for val in values {
        if let Some(entry) = freq.iter_mut().find(|(v, _)| *v == *val) {
            entry.1 += 1;
        } else {
            freq.push((val, 1));
        }
    }

    // Sort by frequency descending (most common first gets index 0).
    freq.sort_by(|a, b| b.1.cmp(&a.1));

    // Limit to 256 entries.
    if freq.len() > 256 {
        return encode_raw(values); // Fallback.
    }

    let dict_count = freq.len() as u8;

    let mut out = Vec::new();

    // Write dictionary.
    out.push(dict_count);
    for (val, _) in &freq {
        let len = val.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(val);
    }

    // Write count.
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());

    // Write ordinal indices.
    for val in values {
        let idx = freq.iter().position(|(v, _)| *v == *val).unwrap_or(0);
        out.push(idx as u8);
    }

    out
}

/// Decode an enum column from dictionary + ordinal bytes back to original strings.
fn decode_enum_column(data: &[u8]) -> Vec<Vec<u8>> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut pos = 0;
    let dict_count = data[pos] as usize;
    pos += 1;

    // Read dictionary entries.
    let mut dictionary: Vec<Vec<u8>> = Vec::with_capacity(dict_count);
    for _ in 0..dict_count {
        if pos + 2 > data.len() {
            return Vec::new();
        }
        let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + len > data.len() {
            return Vec::new();
        }
        dictionary.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    // Read count.
    if pos + 4 > data.len() {
        return Vec::new();
    }
    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    // Read ordinal bytes and map to dictionary.
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        if pos >= data.len() {
            break;
        }
        let idx = data[pos] as usize;
        pos += 1;
        if idx < dictionary.len() {
            result.push(dictionary[idx].clone());
        } else {
            result.push(Vec::new());
        }
    }

    result
}

// ─── Null-Aware Encoder ─────────────────────────────────────────────────────

/// Encode a nullable column: null bitmap + typed sub-encoding.
///
/// Output format:
///   sub_type: u8 (1=DeltaVarint, 2=Bitmap, 4=Timestamp, 5=Enum)
///   count: u32 LE (total values including nulls)
///   null_bitmap_len: u32 LE
///   null_bitmap_bytes (1=present, 0=null, packed like boolean)
///   typed_data_bytes (only non-null values)
fn encode_nullable_column(values: &[&[u8]], sub_type: u8) -> Vec<u8> {
    let count = values.len();
    let bitmap_bytes = count.div_ceil(8);

    let mut out = Vec::new();
    out.push(sub_type);
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&(bitmap_bytes as u32).to_le_bytes());

    // Build null bitmap: 1 = present, 0 = null.
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut non_null_values: Vec<&[u8]> = Vec::new();

    for (i, val) in values.iter().enumerate() {
        if *val != b"null" {
            bitmap[i / 8] |= 1 << (i % 8);
            non_null_values.push(val);
        }
    }
    out.extend_from_slice(&bitmap);

    // Encode non-null values with the appropriate sub-encoder.
    match sub_type {
        ENC_DELTA_VARINT => {
            let encoded = encode_integer_column(&non_null_values);
            out.extend_from_slice(&encoded);
        }
        ENC_BITMAP => {
            let encoded = encode_boolean_column(&non_null_values);
            out.extend_from_slice(&encoded);
        }
        ENC_TIMESTAMP => {
            let encoded = encode_timestamp_column(&non_null_values);
            out.extend_from_slice(&encoded);
        }
        ENC_ENUM => {
            let encoded = encode_enum_column(&non_null_values);
            out.extend_from_slice(&encoded);
        }
        _ => {}
    }

    out
}

/// Decode a nullable column from null bitmap + typed sub-encoding back to text.
fn decode_nullable_column(data: &[u8]) -> Vec<Vec<u8>> {
    if data.len() < 9 {
        return Vec::new();
    }

    let mut pos = 0;
    let sub_type = data[pos];
    pos += 1;
    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let bitmap_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + bitmap_len > data.len() {
        return Vec::new();
    }
    let bitmap = &data[pos..pos + bitmap_len];
    pos += bitmap_len;

    // Count non-null values.
    let mut non_null_count = 0;
    for i in 0..count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 == 1 {
            non_null_count += 1;
        }
    }

    // Decode the typed sub-data.
    let typed_data = &data[pos..];
    let decoded_non_null = match sub_type {
        ENC_DELTA_VARINT => decode_integer_column(typed_data, non_null_count),
        ENC_BITMAP => decode_boolean_column(typed_data),
        ENC_TIMESTAMP => decode_timestamp_column(typed_data),
        ENC_ENUM => decode_enum_column(typed_data),
        _ => Vec::new(),
    };

    // Interleave nulls and non-null values using the bitmap.
    let mut result = Vec::with_capacity(count);
    let mut nn_idx = 0;
    for i in 0..count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 == 1 {
            if nn_idx < decoded_non_null.len() {
                result.push(decoded_non_null[nn_idx].clone());
            } else {
                result.push(b"null".to_vec());
            }
            nn_idx += 1;
        } else {
            result.push(b"null".to_vec());
        }
    }

    result
}

// ─── Top-Level Pipeline Functions ───────────────────────────────────────────

/// Forward transform: typed encoding of columnar data.
///
/// Takes columnar data (columns separated by \x00, values by \x01),
/// infers schema, applies type-specific encoding per column.
///
/// Returns None if the encoded data is not smaller than the original.
pub fn preprocess(columnar_data: &[u8]) -> Option<TransformResult> {
    if columnar_data.is_empty() {
        return None;
    }

    // 1. Split into columns by \x00.
    let columns: Vec<&[u8]> = columnar_data.split(|&b| b == COL_SEP).collect();
    if columns.is_empty() {
        return None;
    }

    // 2. Run schema inference.
    let inferred = schema::infer_schema(columnar_data);
    if inferred.columns.len() != columns.len() {
        return None;
    }

    // 3. Serialize schema for the header.
    let schema_bytes = schema::serialize_schema(&inferred);

    // 4. Encode each column.
    let mut encoded_columns: Vec<(u8, Vec<u8>)> = Vec::with_capacity(columns.len());

    for (ci, col_data) in columns.iter().enumerate() {
        let col_schema = &inferred.columns[ci];
        let values: Vec<&[u8]> = col_data.split(|&b| b == VAL_SEP).collect();

        let (enc_type, enc_data) = encode_column(&values, &col_schema.col_type);
        encoded_columns.push((enc_type, enc_data));
    }

    // 5. Build output: header + encoded columns.
    let num_columns = columns.len() as u16;
    let mut output = Vec::new();

    // Header.
    output.extend_from_slice(&num_columns.to_le_bytes());
    output.extend_from_slice(&(schema_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&schema_bytes);

    // Per-column data.
    for (enc_type, enc_data) in &encoded_columns {
        output.push(*enc_type);
        output.extend_from_slice(&(enc_data.len() as u32).to_le_bytes());
        output.extend_from_slice(enc_data);
    }

    // 6. Size check: only apply if encoded < original.
    if output.len() >= columnar_data.len() {
        return None;
    }

    Some(TransformResult {
        data: output,
        metadata: Vec::new(), // All metadata is embedded in the data stream.
    })
}

/// Reverse transform: decode typed-encoded data back to columnar format.
///
/// Reconstructs the original columnar layout with \x01 value separators
/// and \x00 column separators.
pub fn reverse(data: &[u8], _metadata: &[u8]) -> Vec<u8> {
    if data.len() < 6 {
        return data.to_vec();
    }

    let mut pos = 0;

    // Read header.
    let num_columns =
        u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let schema_len =
        u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if pos + schema_len > data.len() {
        return data.to_vec();
    }
    let schema_bytes = &data[pos..pos + schema_len];
    pos += schema_len;

    let schema = schema::deserialize_schema(schema_bytes);
    if schema.columns.len() != num_columns {
        return data.to_vec();
    }

    // Decode each column.
    let mut output = Vec::with_capacity(data.len() * 2);

    for ci in 0..num_columns {
        if pos + 5 > data.len() {
            return data.to_vec();
        }

        let enc_type = data[pos];
        pos += 1;
        let data_len =
            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + data_len > data.len() {
            return data.to_vec();
        }
        let col_data = &data[pos..pos + data_len];
        pos += data_len;

        // Decode column based on encoding type.
        let decoded_values = decode_column(enc_type, col_data);

        // Write values separated by \x01.
        for (vi, val) in decoded_values.iter().enumerate() {
            output.extend_from_slice(val);
            if vi < decoded_values.len() - 1 {
                output.push(VAL_SEP);
            }
        }

        // Column separator (except after last column).
        if ci < num_columns - 1 {
            output.push(COL_SEP);
        }
    }

    output
}

// ─── Column Encoding Dispatch ───────────────────────────────────────────────

/// Choose and apply the appropriate encoder for a column based on its type.
/// Returns (encoding_type_byte, encoded_data).
fn encode_column(values: &[&[u8]], col_type: &ColumnType) -> (u8, Vec<u8>) {
    match col_type {
        ColumnType::Integer { nullable, .. } => {
            if *nullable {
                (ENC_NULL_TYPED, encode_nullable_column(values, ENC_DELTA_VARINT))
            } else {
                (ENC_DELTA_VARINT, encode_integer_column(values))
            }
        }
        ColumnType::Boolean { nullable } => {
            if *nullable {
                (ENC_NULL_TYPED, encode_nullable_column(values, ENC_BITMAP))
            } else {
                (ENC_BITMAP, encode_boolean_column(values))
            }
        }
        ColumnType::Timestamp { format: TimestampFormat::Iso8601, nullable } => {
            if *nullable {
                (ENC_NULL_TYPED, encode_nullable_column(values, ENC_TIMESTAMP))
            } else {
                (ENC_TIMESTAMP, encode_timestamp_column(values))
            }
        }
        ColumnType::Timestamp { .. } => {
            // Epoch timestamps are already integers; raw passthrough for now.
            (ENC_RAW, encode_raw(values))
        }
        ColumnType::Enum { nullable, .. } => {
            if *nullable {
                (ENC_NULL_TYPED, encode_nullable_column(values, ENC_ENUM))
            } else {
                (ENC_ENUM, encode_enum_column(values))
            }
        }
        ColumnType::Null => {
            // All-null column: store as raw passthrough.
            (ENC_RAW, encode_raw(values))
        }
        // Float, String, Uuid — raw passthrough.
        _ => (ENC_RAW, encode_raw(values)),
    }
}

/// Decode a column based on encoding type, returning text values.
fn decode_column(enc_type: u8, col_data: &[u8]) -> Vec<Vec<u8>> {
    match enc_type {
        ENC_RAW => decode_raw(col_data),
        ENC_DELTA_VARINT => {
            // Need count — stored implicitly. We must count varints.
            // For non-nullable integer columns, count isn't stored explicitly.
            // We decode until data is exhausted.
            decode_integer_column_all(col_data)
        }
        ENC_BITMAP => decode_boolean_column(col_data),
        ENC_NULL_TYPED => decode_nullable_column(col_data),
        ENC_TIMESTAMP => decode_timestamp_column(col_data),
        ENC_ENUM => decode_enum_column(col_data),
        _ => decode_raw(col_data),
    }
}

/// Encode raw: preserve the original \x01-separated format.
fn encode_raw(values: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, val) in values.iter().enumerate() {
        out.extend_from_slice(val);
        if i < values.len() - 1 {
            out.push(VAL_SEP);
        }
    }
    out
}

/// Decode raw: split by \x01.
fn decode_raw(data: &[u8]) -> Vec<Vec<u8>> {
    data.split(|&b| b == VAL_SEP)
        .map(|v| v.to_vec())
        .collect()
}

/// Decode an integer column by consuming all varints in the data.
/// Used for non-nullable integer columns where count is not explicitly stored.
fn decode_integer_column_all(data: &[u8]) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let mut pos = 0;
    let mut prev: i64 = 0;
    let mut i = 0;

    while pos < data.len() {
        let (zz, new_pos) = match leb128_decode(data, pos) {
            Some(v) => v,
            None => break,
        };
        pos = new_pos;
        let delta = zigzag_decode(zz);
        let n = if i == 0 { delta } else { prev + delta };
        prev = n;
        result.push(format_i64(n));
        i += 1;
    }

    result
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse text bytes as i64.
fn parse_i64(val: &[u8]) -> i64 {
    // Safe for validated integer columns.
    match std::str::from_utf8(val) {
        Ok(s) => s.parse::<i64>().unwrap_or(0),
        Err(_) => 0,
    }
}

/// Format i64 as text bytes (decimal representation).
fn format_i64(n: i64) -> Vec<u8> {
    n.to_string().into_bytes()
}

/// Check if a value is "true" (case-sensitive).
fn is_true(val: &[u8]) -> bool {
    val == b"true"
}

/// Parse N ASCII digits from a byte slice starting at `offset`.
fn parse_digits(data: &[u8], offset: usize, count: usize) -> Option<u64> {
    if offset + count > data.len() {
        return None;
    }
    let mut result: u64 = 0;
    for &b in &data[offset..offset + count] {
        if !b.is_ascii_digit() {
            return None;
        }
        result = result * 10 + (b - b'0') as u64;
    }
    Some(result)
}

/// Write a number as zero-padded ASCII digits.
fn write_digits(out: &mut Vec<u8>, val: u64, width: usize) {
    let s = format!("{:0>width$}", val, width = width);
    out.extend_from_slice(s.as_bytes());
}

/// Calculate days since Unix epoch (1970-01-01) for a given date.
/// Returns None if the date is invalid.
fn days_since_epoch(year: i64, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    // civil_from_days / days_from_civil
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;                               // [0, 399]
    let m = month;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;               // [0, 146096]
    let days = era * 146097 + doe as i64 - 719468;                  // days since epoch
    Some(days)
}

/// Convert days since Unix epoch back to (year, month, day).
fn date_from_epoch_days(days: i64) -> (i64, u32, u32) {
    // Reverse of days_since_epoch, from Howard Hinnant's algorithm.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;                             // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);             // [0, 365]
    let mp = (5 * doy + 2) / 153;                                   // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1;                           // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 };                  // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build columnar data from column value slices.
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
    fn test_zigzag_encode() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
        assert_eq!(zigzag_encode(-3), 5);
        assert_eq!(zigzag_encode(3), 6);
        assert_eq!(zigzag_encode(i64::MAX), u64::MAX - 1);
        assert_eq!(zigzag_encode(i64::MIN), u64::MAX);

        // Verify roundtrip for all test values.
        for &n in &[0i64, -1, 1, -2, 2, 100, -100, i64::MAX, i64::MIN] {
            assert_eq!(zigzag_decode(zigzag_encode(n)), n);
        }
    }

    #[test]
    fn test_leb128_roundtrip() {
        let test_values: Vec<u64> = vec![
            0, 1, 127, 128, 255, 256, 16383, 16384, 1_000_000,
            u32::MAX as u64, u64::MAX,
        ];

        for &val in &test_values {
            let mut buf = Vec::new();
            leb128_encode(val, &mut buf);
            let (decoded, _) = leb128_decode(&buf, 0).unwrap();
            assert_eq!(decoded, val, "LEB128 roundtrip failed for {val}");
        }

        // Verify small values use few bytes.
        let mut buf = Vec::new();
        leb128_encode(0, &mut buf);
        assert_eq!(buf.len(), 1, "0 should encode to 1 byte");

        buf.clear();
        leb128_encode(127, &mut buf);
        assert_eq!(buf.len(), 1, "127 should encode to 1 byte");

        buf.clear();
        leb128_encode(128, &mut buf);
        assert_eq!(buf.len(), 2, "128 should encode to 2 bytes");
    }

    #[test]
    fn test_integer_column_roundtrip() {
        let values: &[&[u8]] = &[b"0", b"1", b"-1", b"100", b"-100", b"999999"];
        let encoded = encode_integer_column(values);
        let decoded = decode_integer_column(&encoded, values.len());

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "mismatch at index {i}: expected {:?}, got {:?}",
                String::from_utf8_lossy(val),
                String::from_utf8_lossy(&decoded[i])
            );
        }
    }

    #[test]
    fn test_integer_delta_encoding() {
        // Monotonic sequence: deltas are all 1, so encoding should be tiny.
        let values: Vec<Vec<u8>> = (100..=103).map(|n: i64| n.to_string().into_bytes()).collect();
        let val_refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();

        let encoded = encode_integer_column(&val_refs);

        // First value 100 zigzag = 200, but deltas 1,1,1 zigzag = 2,2,2 (1 byte each).
        // Total: a few bytes for 100 + 3 bytes for deltas.
        assert!(
            encoded.len() <= 6,
            "monotonic [100..103] should encode to <= 6 bytes, got {}",
            encoded.len()
        );

        let decoded = decode_integer_column(&encoded, val_refs.len());
        assert_eq!(decoded.len(), val_refs.len());
        for (i, val) in val_refs.iter().enumerate() {
            assert_eq!(decoded[i], val.to_vec());
        }
    }

    #[test]
    fn test_boolean_column_roundtrip() {
        let values: &[&[u8]] = &[
            b"true", b"false", b"true", b"true", b"false",
            b"false", b"true", b"false", b"true",
        ];
        let encoded = encode_boolean_column(values);
        let decoded = decode_boolean_column(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "boolean mismatch at index {i}"
            );
        }

        // Verify compactness: 9 bools should be 4 (count) + 2 (packed) = 6 bytes.
        assert_eq!(encoded.len(), 6);
    }

    #[test]
    fn test_nullable_integer_roundtrip() {
        let values: &[&[u8]] = &[b"10", b"null", b"30", b"null", b"50"];
        let encoded = encode_nullable_column(values, ENC_DELTA_VARINT);
        let decoded = decode_nullable_column(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "nullable int mismatch at index {i}: expected {:?}, got {:?}",
                String::from_utf8_lossy(val),
                String::from_utf8_lossy(&decoded[i])
            );
        }
    }

    #[test]
    fn test_timestamp_parse_roundtrip() {
        // Parse ISO 8601 to micros and back.
        let ts = b"\"2026-03-15T10:30:00.081Z\"";
        let (micros, fmt, tz, frac) = parse_iso8601_to_micros(ts).unwrap();

        assert_eq!(fmt, TS_FMT_Z);
        assert_eq!(tz, 0);
        assert_eq!(frac, 3);
        assert!(micros > 0);

        let restored = micros_to_iso8601(micros, fmt, tz, frac);
        assert_eq!(
            restored, ts.to_vec(),
            "timestamp parse roundtrip failed: expected {:?}, got {:?}",
            String::from_utf8_lossy(ts),
            String::from_utf8_lossy(&restored)
        );
    }

    #[test]
    fn test_timestamp_column_roundtrip() {
        let values: &[&[u8]] = &[
            b"\"2026-03-15T10:30:00.081Z\"",
            b"\"2026-03-15T10:30:00.234Z\"",
            b"\"2026-03-15T10:30:00.509Z\"",
            b"\"2026-03-15T10:30:00.707Z\"",
            b"\"2026-03-15T10:30:01.013Z\"",
        ];
        let encoded = encode_timestamp_column(values);
        let decoded = decode_timestamp_column(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "timestamp mismatch at index {i}: expected {:?}, got {:?}",
                String::from_utf8_lossy(val),
                String::from_utf8_lossy(&decoded[i])
            );
        }
    }

    #[test]
    fn test_timestamp_delta_encoding_size() {
        // Monotonic timestamps with ~200ms intervals should produce tiny deltas.
        let values: &[&[u8]] = &[
            b"\"2026-03-15T10:30:00.000Z\"",
            b"\"2026-03-15T10:30:00.200Z\"",
            b"\"2026-03-15T10:30:00.400Z\"",
            b"\"2026-03-15T10:30:00.600Z\"",
            b"\"2026-03-15T10:30:00.800Z\"",
            b"\"2026-03-15T10:30:01.000Z\"",
            b"\"2026-03-15T10:30:01.200Z\"",
            b"\"2026-03-15T10:30:01.400Z\"",
            b"\"2026-03-15T10:30:01.600Z\"",
            b"\"2026-03-15T10:30:01.800Z\"",
        ];
        let encoded = encode_timestamp_column(values);

        // Header is 16 bytes. Each delta (200000 micros) zigzag = 400000,
        // which is ~3 bytes LEB128. First delta is 0 (1 byte).
        // 16 + 1 + 9*3 = 44 bytes max.
        // Raw would be 10 * 28 + 9 = 289 bytes.
        assert!(
            encoded.len() < 50,
            "monotonic timestamps should encode compactly: {} bytes",
            encoded.len()
        );

        // Verify roundtrip.
        let decoded = decode_timestamp_column(&encoded);
        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(decoded[i], val.to_vec());
        }
    }

    #[test]
    fn test_enum_column_roundtrip() {
        let values: &[&[u8]] = &[
            b"\"page_view\"",
            b"\"api_call\"",
            b"\"page_view\"",
            b"\"scroll\"",
            b"\"page_view\"",
            b"\"api_call\"",
            b"\"scroll\"",
            b"\"page_view\"",
        ];
        let encoded = encode_enum_column(values);
        let decoded = decode_enum_column(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "enum mismatch at index {i}: expected {:?}, got {:?}",
                String::from_utf8_lossy(val),
                String::from_utf8_lossy(&decoded[i])
            );
        }
    }

    #[test]
    fn test_enum_dictionary_ordering() {
        // Most frequent value should get index 0.
        let values: &[&[u8]] = &[
            b"\"rare\"",
            b"\"common\"",
            b"\"common\"",
            b"\"common\"",
            b"\"rare\"",
            b"\"medium\"",
            b"\"medium\"",
        ];
        let encoded = encode_enum_column(values);

        // The dictionary should be: "common" (3), "rare" (2), "medium" (2).
        // After the dict, count(4 bytes), then ordinals.
        // First byte is dict_count = 3.
        assert_eq!(encoded[0], 3);

        // Verify roundtrip.
        let decoded = decode_enum_column(&encoded);
        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(decoded[i], val.to_vec());
        }
    }

    #[test]
    fn test_nullable_timestamp_roundtrip() {
        let values: &[&[u8]] = &[
            b"\"2026-03-15T10:30:00.081Z\"",
            b"null",
            b"\"2026-03-15T10:30:00.509Z\"",
            b"null",
            b"\"2026-03-15T10:30:01.013Z\"",
        ];
        let encoded = encode_nullable_column(values, ENC_TIMESTAMP);
        let decoded = decode_nullable_column(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (i, val) in values.iter().enumerate() {
            assert_eq!(
                decoded[i],
                val.to_vec(),
                "nullable timestamp mismatch at index {i}: expected {:?}, got {:?}",
                String::from_utf8_lossy(val),
                String::from_utf8_lossy(&decoded[i])
            );
        }
    }

    #[test]
    fn test_full_pipeline_roundtrip() {
        // Multi-column data with enough rows to amortize header overhead.
        // 50 rows of: integers (multi-digit), booleans (5-char "false"), strings.
        let n = 50;
        let ints: Vec<Vec<u8>> = (1000..1000 + n as i64)
            .map(|i| i.to_string().into_bytes())
            .collect();
        let bools: Vec<Vec<u8>> = (0..n)
            .map(|i| if i % 2 == 0 { b"true".to_vec() } else { b"false".to_vec() })
            .collect();
        let strs: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("\"some_string_value_{}\"", i).into_bytes())
            .collect();

        let int_refs: Vec<&[u8]> = ints.iter().map(|v| v.as_slice()).collect();
        let bool_refs: Vec<&[u8]> = bools.iter().map(|v| v.as_slice()).collect();
        let str_refs: Vec<&[u8]> = strs.iter().map(|v| v.as_slice()).collect();

        let data = build_columnar(&[&int_refs, &bool_refs, &str_refs]);

        let result = preprocess(&data);
        assert!(result.is_some(), "preprocess should succeed for typed data with {} rows", n);
        let result = result.unwrap();

        assert!(
            result.data.len() < data.len(),
            "encoded should be smaller: {} < {}",
            result.data.len(),
            data.len()
        );

        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            restored,
            data,
            "full pipeline roundtrip failed"
        );
    }

    #[test]
    fn test_size_check() {
        // Very small data where encoding overhead exceeds savings.
        let data = build_columnar(&[&[b"1", b"2"]]);

        let result = preprocess(&data);
        assert!(
            result.is_none(),
            "preprocess should return None when encoded >= original"
        );
    }

    #[test]
    fn test_real_corpus_roundtrip() {
        // Read test corpus, apply NDJSON columnar, apply typed encoding,
        // reverse both, verify byte-exact match.
        let corpus = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/test-ndjson.ndjson"
        ))
        .expect("failed to read test-ndjson.ndjson");

        // Step 1: NDJSON columnar transform.
        let ndjson_result = crate::format::ndjson::preprocess(&corpus)
            .expect("ndjson::preprocess should succeed");
        let columnar_data = &ndjson_result.data;

        // Step 2: Typed encoding.
        let typed_result = preprocess(columnar_data)
            .expect("typed encoding should succeed on real corpus");

        // Verify it saved space.
        assert!(
            typed_result.data.len() < columnar_data.len(),
            "typed encoding should be smaller: {} < {}",
            typed_result.data.len(),
            columnar_data.len()
        );

        // Step 3: Reverse typed encoding.
        let restored_columnar = reverse(&typed_result.data, &typed_result.metadata);

        // Step 4: Verify byte-exact match with original columnar data.
        assert_eq!(
            restored_columnar.len(),
            columnar_data.len(),
            "restored columnar length mismatch: {} vs {}",
            restored_columnar.len(),
            columnar_data.len()
        );
        assert_eq!(
            restored_columnar,
            columnar_data.to_vec(),
            "restored columnar data does not match original"
        );

        // Step 5: Reverse NDJSON columnar to get original NDJSON.
        let restored_ndjson =
            crate::format::ndjson::reverse(&restored_columnar, &ndjson_result.metadata);
        assert_eq!(
            restored_ndjson, corpus,
            "full roundtrip (NDJSON -> columnar -> typed -> reverse) failed"
        );
    }
}
