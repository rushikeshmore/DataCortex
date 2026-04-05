# datacortex-cli

Command-line tool for [DataCortex](https://github.com/rushikeshmore/DataCortex), a lossless JSON/NDJSON compressor that beats zstd-19 and brotli-11 on every file tested.

## Install

```bash
cargo install datacortex-cli
```

## Quick Start

```bash
datacortex compress logs.ndjson                    # compress
datacortex decompress logs.ndjson.dcx output.json  # decompress
datacortex bench corpus/ -m fast --compare         # benchmark
```

## Commands

| Command | Description |
|---------|-------------|
| `compress` | Compress a JSON/NDJSON file to .dcx format |
| `decompress` | Decompress a .dcx file back to the original |
| `bench` | Benchmark compression ratio and throughput |
| `info` | Inspect a .dcx file (mode, format, sizes) |
| `train-dict` | Train a custom zstd dictionary from a corpus |

## Usage

```bash
# Streaming (pipe-friendly)
cat logs.ndjson | datacortex compress - -o compressed.dcx
datacortex decompress compressed.dcx -o -

# Chunked compression for large files
datacortex compress logs.ndjson -o out.dcx --chunk-rows 10000

# Custom dictionary for repeated schemas
datacortex train-dict corpus/*.ndjson --output my.dict
datacortex compress logs.ndjson --dict my.dict

# Higher compression (slower encode)
datacortex compress data.ndjson -m fast --level 19
```

## Benchmarks

| File | Size | DataCortex | zstd -19 | vs zstd |
|------|------|-----------|----------|---------|
| k8s structured logs | 9.9 MB | ~40x | 18.9x | +113% |
| NDJSON 10K rows | 3.3 MB | 27.2x | 16.0x | +70% |
| nginx access logs | 9.5 MB | ~28x | 17.3x | +62% |
| Twitter API | 617 KB | 19.7x | 16.7x | +18% |

## Links

- [Core library](https://crates.io/crates/datacortex-core)
- [Python bindings](https://pypi.org/project/datacortex/)
- [GitHub](https://github.com/rushikeshmore/DataCortex)

## License

MIT
