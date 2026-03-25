//! Dump post-transform data for external compression comparison.
//!
//! Usage: cargo run --release --example dump_transform -- <input> <output>
//!
//! Writes the transformed byte stream (what the CM engine sees) to <output>.
//! Reports format, original size, transformed size, and metadata overhead.

use datacortex_core::dcx::Mode;
use datacortex_core::format::{detect_format, preprocess};
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let data = fs::read(input_path).expect("Failed to read input file");

    // Detect format (try extension first, then content).
    let format = datacortex_core::format::detect_from_extension(input_path)
        .unwrap_or_else(|| detect_format(&data));

    // Run preprocessing.
    let (transformed, chain) = preprocess(&data, format, Mode::Balanced);

    // Calculate metadata size.
    let metadata_bytes = chain.serialize();
    let metadata_size = metadata_bytes.len();

    // Write transformed data to output.
    fs::write(output_path, &transformed).expect("Failed to write output file");

    // Report stats.
    let name = std::path::Path::new(input_path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();

    eprintln!("=== {} ===", name);
    eprintln!("Format:      {:?}", format);
    eprintln!("Original:    {} bytes", data.len());
    eprintln!("Transformed: {} bytes", transformed.len());
    eprintln!("Metadata:    {} bytes", metadata_size);
    eprintln!(
        "Total (transform + meta): {} bytes",
        transformed.len() + metadata_size
    );
    eprintln!(
        "Transform ratio: {:.1}%",
        (transformed.len() + metadata_size) as f64 / data.len() as f64 * 100.0
    );

    if transformed.len() == data.len() {
        eprintln!("NOTE: No transform applied (output == input)");
    }
}
