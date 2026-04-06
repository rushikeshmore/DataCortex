use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

fn bench_datacortex(data: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    let mut encoder = DataCortexEncoder::new(&mut compressed);
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap();
    compressed
}

fn bench_zstd(data: &[u8], level: i32) -> Vec<u8> {
    zstd::encode_all(std::io::Cursor::new(data), level).unwrap()
}

fn bench_gzip(data: &[u8], level: u32) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn compression_benchmarks(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    let mut group = c.benchmark_group("generated_ndjson");
    for &rows in &sizes {
        let data = generate_ndjson(rows);

        group.bench_with_input(BenchmarkId::new("datacortex", rows), &data, |b, data| {
            b.iter(|| bench_datacortex(data));
        });
        group.bench_with_input(BenchmarkId::new("zstd-3", rows), &data, |b, data| {
            b.iter(|| bench_zstd(data, 3));
        });
        group.bench_with_input(BenchmarkId::new("zstd-19", rows), &data, |b, data| {
            b.iter(|| bench_zstd(data, 19));
        });
        group.bench_with_input(BenchmarkId::new("gzip-6", rows), &data, |b, data| {
            b.iter(|| bench_gzip(data, 6));
        });
    }
    group.finish();

    // Corpus files (if available)
    let corpus_files = ["test-ndjson.ndjson", "test-api.json"];

    let mut group = c.benchmark_group("corpus");
    for name in &corpus_files {
        if let Some(data) = read_corpus_file(name) {
            group.bench_with_input(BenchmarkId::new("datacortex", name), &data, |b, data| {
                b.iter(|| bench_datacortex(data));
            });
            group.bench_with_input(BenchmarkId::new("zstd-3", name), &data, |b, data| {
                b.iter(|| bench_zstd(data, 3));
            });
            group.bench_with_input(BenchmarkId::new("zstd-19", name), &data, |b, data| {
                b.iter(|| bench_zstd(data, 19));
            });
            group.bench_with_input(BenchmarkId::new("gzip-6", name), &data, |b, data| {
                b.iter(|| bench_gzip(data, 6));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, compression_benchmarks);
criterion_main!(benches);
