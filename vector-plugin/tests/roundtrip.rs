use std::io::Write;
use std::path::Path;

use datacortex_vector::{DataCortexDecoder, DataCortexEncoder};

fn read_corpus_file(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("corpus")
        .join(name);
    std::fs::read(&path).ok()
}

#[expect(clippy::cast_precision_loss, reason = "display-only ratio math")]
fn roundtrip_file(name: &str) {
    let data = read_corpus_file(name).unwrap_or_else(|| panic!("corpus file not found: {name}"));

    let mut compressed = Vec::new();
    let mut encoder = DataCortexEncoder::new(&mut compressed);
    encoder.write_all(&data).unwrap();
    encoder.finish().unwrap();

    let ratio = data.len() as f64 / compressed.len() as f64;
    let bpb = (compressed.len() as f64 * 8.0) / data.len() as f64;
    println!(
        "{name}: {orig} -> {comp} bytes ({ratio:.1}x, {bpb:.2} bpb)",
        orig = data.len(),
        comp = compressed.len(),
    );

    let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
    assert_eq!(decompressed.len(), data.len(), "length mismatch for {name}");
    assert_eq!(decompressed, data, "data mismatch for {name}");
}

#[test]
fn roundtrip_test_ndjson() {
    roundtrip_file("test-ndjson.ndjson");
}

#[test]
fn roundtrip_test_api_json() {
    roundtrip_file("test-api.json");
}

#[test]
fn roundtrip_test_config_json() {
    roundtrip_file("test-config.json");
}

#[test]
fn roundtrip_uniform_10k() {
    if read_corpus_file("json-bench/uniform-10k.ndjson").is_some() {
        roundtrip_file("json-bench/uniform-10k.ndjson");
    }
}

#[test]
fn roundtrip_twitter_json() {
    if read_corpus_file("json-bench/twitter.json").is_some() {
        roundtrip_file("json-bench/twitter.json");
    }
}

#[test]
fn roundtrip_citm_catalog() {
    if read_corpus_file("json-bench/citm_catalog.json").is_some() {
        roundtrip_file("json-bench/citm_catalog.json");
    }
}

#[test]
fn roundtrip_gharchive() {
    if read_corpus_file("json-bench/gharchive-10mb.ndjson").is_some() {
        roundtrip_file("json-bench/gharchive-10mb.ndjson");
    }
}

/// Simulates how Vector would use the encoder: encode events one at a time.
#[test]
fn simulate_vector_event_encoding() {
    let events = vec![
        r#"{"timestamp":"2026-04-02T12:00:00Z","level":"info","service":"api","method":"GET","path":"/health","status":200,"duration_ms":3}"#,
        r#"{"timestamp":"2026-04-02T12:00:01Z","level":"warn","service":"api","method":"POST","path":"/users","status":422,"duration_ms":145}"#,
        r#"{"timestamp":"2026-04-02T12:00:02Z","level":"error","service":"api","method":"POST","path":"/orders","status":500,"duration_ms":2031}"#,
        r#"{"timestamp":"2026-04-02T12:00:03Z","level":"info","service":"worker","method":"PROCESS","path":"job.email","status":200,"duration_ms":892}"#,
    ];

    let mut original = Vec::new();
    let mut compressed = Vec::new();
    let mut encoder = DataCortexEncoder::new(&mut compressed);

    // Vector encodes each event with newline framing, then writes to compressor
    for event in &events {
        let line = format!("{event}\n");
        original.extend_from_slice(line.as_bytes());
        encoder.write_all(line.as_bytes()).unwrap();
    }

    encoder.finish().unwrap();

    let decompressed = DataCortexDecoder::decompress(&compressed).unwrap();
    assert_eq!(decompressed, original);
}
