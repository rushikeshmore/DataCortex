//! Validation A/B test: does quote stripping help CM compression?
//!
//! Reads corpus/test-ndjson.ndjson, applies NDJSON columnar transform,
//! then compresses two variants with the CM engine (Balanced mode):
//!   A = columnar data as-is (current behavior)
//!   B = columnar data with JSON string quotes stripped
//!
//! If B is smaller, typed encoding will likely help CM => PROCEED.
//! If B is larger, typed encoding should only be used with zstd => DUAL-PIPELINE.
//!
//! Usage: cargo run --release --example validation_ab_test

use datacortex_core::codec::compress_to_vec;
use datacortex_core::dcx::{FormatHint, Mode};
use std::fs;

fn main() {
    // Read test file.
    let data =
        fs::read("corpus/test-ndjson.ndjson").expect("failed to read corpus/test-ndjson.ndjson");
    println!("Original NDJSON: {} bytes", data.len());

    // Apply NDJSON columnar transform (low-level, no value dict).
    let result = datacortex_core::format::ndjson::preprocess(&data)
        .expect("ndjson preprocess failed — is the file valid uniform NDJSON?");
    let columnar = result.data;
    println!("Columnar data:   {} bytes", columnar.len());

    // --- Variant A: compress original columnar data with CM ---
    // Use Generic format hint so compress_to_vec does NOT re-apply transforms.
    eprintln!("[A] Compressing columnar data (this takes a while)...");
    let compressed_a = compress_to_vec(&columnar, Mode::Balanced, Some(FormatHint::Generic))
        .expect("compress variant A");
    let bpb_a = compressed_a.len() as f64 * 8.0 / data.len() as f64;

    // --- Variant B: strip quotes then compress ---
    let stripped = strip_quotes(&columnar);
    let quotes_removed = columnar.len() - stripped.len();
    println!(
        "Quote-stripped:   {} bytes ({:.1}% of columnar, {} quote bytes removed)",
        stripped.len(),
        stripped.len() as f64 / columnar.len() as f64 * 100.0,
        quotes_removed,
    );

    eprintln!("[B] Compressing quote-stripped data (this takes a while)...");
    let compressed_b = compress_to_vec(&stripped, Mode::Balanced, Some(FormatHint::Generic))
        .expect("compress variant B");
    let bpb_b = compressed_b.len() as f64 * 8.0 / data.len() as f64;

    // --- Results ---
    println!();
    println!("=== RESULTS ===");
    println!(
        "Variant A (columnar, no strip): {} bytes, {:.3} bpb",
        compressed_a.len(),
        bpb_a
    );
    println!(
        "Variant B (quote-stripped):      {} bytes, {:.3} bpb",
        compressed_b.len(),
        bpb_b
    );

    let improvement =
        (compressed_a.len() as f64 - compressed_b.len() as f64) / compressed_a.len() as f64 * 100.0;
    println!("Improvement: {:.1}%", improvement);

    if compressed_b.len() < compressed_a.len() {
        println!();
        println!(
            "VERDICT: PROCEED — quote stripping HELPS CM. Typed encoding will likely help too."
        );
    } else {
        println!();
        println!(
            "VERDICT: DUAL-PIPELINE — quote stripping HURTS CM. Use typed encoding + zstd (Fast mode only)."
        );
    }
}

/// Strip leading and trailing `"` from each value in columnar data.
///
/// Columnar layout uses \x00 as column separator and \x01 as value separator.
/// Values that start and end with `"` are JSON strings — we strip those quotes.
/// Non-string values (numbers, booleans, null) are left as-is.
fn strip_quotes(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut val_start = 0;

    for i in 0..=data.len() {
        // At a separator or end of data, process the value.
        let is_sep = if i < data.len() {
            data[i] == 0x00 || data[i] == 0x01
        } else {
            true
        };

        if is_sep {
            let val = &data[val_start..i];
            if val.len() >= 2 && val[0] == b'"' && val[val.len() - 1] == b'"' {
                // Strip the surrounding quotes.
                out.extend_from_slice(&val[1..val.len() - 1]);
            } else {
                // Keep as-is (number, bool, null, or too short).
                out.extend_from_slice(val);
            }
            // Append the separator itself.
            if i < data.len() {
                out.push(data[i]);
            }
            val_start = i + 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_quotes_basic() {
        // "page_view"\x01"api_call"\x01"page_view"
        let input = b"\"page_view\"\x01\"api_call\"\x01\"page_view\"";
        let expected = b"page_view\x01api_call\x01page_view";
        assert_eq!(strip_quotes(input), expected.to_vec());
    }

    #[test]
    fn strip_quotes_mixed() {
        // "hello"\x0142\x01true\x01null\x00"world"\x01false
        let input = b"\"hello\"\x0142\x01true\x01null\x00\"world\"\x01false";
        let expected = b"hello\x0142\x01true\x01null\x00world\x01false";
        assert_eq!(strip_quotes(input), expected.to_vec());
    }

    #[test]
    fn strip_quotes_empty_string() {
        // ""\x01"x"
        let input = b"\"\"\x01\"x\"";
        let expected = b"\x01x";
        assert_eq!(strip_quotes(input), expected.to_vec());
    }

    #[test]
    fn strip_quotes_no_strings() {
        let input = b"42\x01true\x00null\x01false";
        assert_eq!(strip_quotes(input), input.to_vec());
    }
}
