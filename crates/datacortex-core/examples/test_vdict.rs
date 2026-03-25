use datacortex_core::format::value_dict;

fn main() {
    let col_data = std::fs::read("/tmp/transform_dump/test-ndjson.ndjson.transformed").unwrap();
    eprintln!("Column data: {} bytes", col_data.len());

    match value_dict::preprocess(&col_data) {
        Some(result) => {
            eprintln!("Value dict applied!");
            eprintln!("  Encoded data: {} bytes", result.data.len());
            eprintln!("  Metadata: {} bytes", result.metadata.len());
            eprintln!(
                "  Total: {} bytes",
                result.data.len() + result.metadata.len()
            );
            eprintln!(
                "  Savings: {} bytes ({}%)",
                col_data.len() - result.data.len() - result.metadata.len(),
                (col_data.len() - result.data.len() - result.metadata.len()) * 100 / col_data.len()
            );

            // Verify roundtrip.
            let recovered = value_dict::reverse(&result.data, &result.metadata);
            if recovered == col_data {
                eprintln!("  Roundtrip: OK");
            } else {
                eprintln!(
                    "  Roundtrip: FAILED (recovered {} bytes vs {} original)",
                    recovered.len(),
                    col_data.len()
                );
            }
        }
        None => {
            eprintln!("Value dict returned None — not applied.");
        }
    }
}
