use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

use datacortex_vector::DataCortexEncoder;

fn read_corpus_file(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("corpus")
        .join(name);
    std::fs::read(&path).ok()
}

fn generate_ndjson(rows: usize) -> Vec<u8> {
    let mut data = String::new();
    for i in 0..rows {
        writeln!(
            data,
            "{{\"timestamp\":\"2026-04-02T{:02}:{:02}:{:02}Z\",\"level\":\"{}\",\"service\":\"api-gateway\",\"method\":\"GET\",\"path\":\"/api/v1/users/{}\",\"status\":{},\"duration_ms\":{},\"request_id\":\"req-{:08x}\"}}",
            (i / 3600) % 24,
            (i / 60) % 60,
            i % 60,
            ["info", "warn", "error", "debug"][i % 4],
            i % 1000,
            [200, 200, 200, 201, 400, 404, 500][i % 7],
            10 + (i * 7 % 5000),
            i,
        )
        .unwrap();
    }
    data.into_bytes()
}

fn compress_datacortex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = DataCortexEncoder::new(&mut out);
    enc.write_all(data).unwrap();
    enc.finish().unwrap();
    out
}

fn compress_zstd(data: &[u8], level: i32) -> Vec<u8> {
    zstd::encode_all(std::io::Cursor::new(data), level).unwrap()
}

fn compress_gzip(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// Prints a ratio comparison table. Run with `cargo test -- --nocapture ratio_comparison`.
#[test]
#[expect(clippy::cast_precision_loss, reason = "display-only ratio math")]
fn ratio_comparison() {
    println!("\n=== Compression Ratio Comparison ===\n");
    println!(
        "{:<35} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "File", "Original", "DataCortex", "zstd-19", "gzip-6", "DCX vs zstd"
    );
    println!("{}", "-".repeat(95));

    let mut test_data: Vec<(String, Vec<u8>)> = vec![
        ("generated 100 rows".into(), generate_ndjson(100)),
        ("generated 1K rows".into(), generate_ndjson(1000)),
        ("generated 10K rows".into(), generate_ndjson(10000)),
    ];

    for name in ["test-ndjson.ndjson", "test-api.json"] {
        if let Some(data) = read_corpus_file(name) {
            test_data.push((name.to_string(), data));
        }
    }

    for name in [
        "json-bench/uniform-10k.ndjson",
        "json-bench/twitter.json",
        "json-bench/citm_catalog.json",
    ] {
        if let Some(data) = read_corpus_file(name) {
            test_data.push((name.to_string(), data));
        }
    }

    for (name, data) in &test_data {
        let dcx = compress_datacortex(data);
        let zstd = compress_zstd(data, 19);
        let gzip = compress_gzip(data, 6);

        let dcx_ratio = data.len() as f64 / dcx.len() as f64;
        let zstd_ratio = data.len() as f64 / zstd.len() as f64;
        let gzip_ratio = data.len() as f64 / gzip.len() as f64;
        let advantage = ((dcx_ratio / zstd_ratio) - 1.0) * 100.0;

        println!(
            "{:<35} {:>10} {:>10.1}x {:>10.1}x {:>10.1}x {:>+10.0}%",
            name,
            data.len(),
            dcx_ratio,
            zstd_ratio,
            gzip_ratio,
            advantage,
        );

        // DataCortex should beat zstd-19 on uniform NDJSON
        if name.contains("ndjson") && !name.contains("gharchive") {
            assert!(
                dcx_ratio > zstd_ratio,
                "{name}: DataCortex ({dcx_ratio:.1}x) should beat zstd-19 ({zstd_ratio:.1}x)"
            );
        }
    }
    println!();
}
