use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

use datacortex_vector::{DataCortexDecoder, DataCortexEncoder};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data = if args.len() > 1 {
        let path = Path::new(&args[1]);
        std::fs::read(path).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {e}", path.display());
            std::process::exit(1);
        })
    } else {
        // Generate sample NDJSON
        let mut data = String::new();
        for i in 0..1000 {
            writeln!(
                data,
                "{{\"ts\":\"2026-04-02T12:{:02}:{:02}Z\",\"level\":\"{}\",\"svc\":\"api\",\"msg\":\"request\",\"ms\":{},\"status\":{}}}",
                (i / 60) % 60,
                i % 60,
                ["info", "warn", "error", "debug"][i % 4],
                10 + (i * 3 % 2000),
                [200, 200, 201, 400, 500][i % 5],
            )
            .unwrap();
        }
        println!("Using generated NDJSON ({} rows)", 1000);
        data.into_bytes()
    };

    println!("Original:   {} bytes", data.len());

    // Compress with DataCortex
    let mut compressed = Vec::new();
    let mut encoder = DataCortexEncoder::new(&mut compressed);
    encoder.write_all(&data).unwrap();
    encoder.finish().unwrap();

    #[expect(clippy::cast_precision_loss, reason = "display-only ratio math")]
    let ratio = data.len() as f64 / compressed.len() as f64;
    #[expect(clippy::cast_precision_loss, reason = "display-only bpb math")]
    let bpb = compressed.len() as f64 * 8.0 / data.len() as f64;
    println!(
        "DataCortex: {} bytes ({ratio:.1}x, {bpb:.2} bpb)",
        compressed.len()
    );

    // Verify roundtrip
    let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
    println!("Roundtrip:  OK");
}
