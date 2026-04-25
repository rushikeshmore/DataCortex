# DataCortex

[![Crates.io](https://img.shields.io/crates/v/datacortex-cli)](https://crates.io/crates/datacortex-cli)
[![PyPI](https://img.shields.io/pypi/v/datacortex)](https://pypi.org/project/datacortex/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Lossless compression for JSON and NDJSON. Beats zstd-19 and brotli-11 on every file tested.

```bash
cargo install datacortex-cli        # Rust
pip install datacortex               # Python
```

## Quick Start

```bash
datacortex compress logs.ndjson                   # compress (auto-detects format)
datacortex compress logs.ndjson --turbo            # turbo: 30-55x faster encode
datacortex decompress logs.ndjson.dcx output.json  # decompress (byte-exact)
datacortex bench corpus/ -m fast --compare         # benchmark against zstd
```

## Benchmarks

Compression ratio (smaller output = better):

| File | Size | DataCortex | zstd -19 | brotli -11 | vs best |
|------|------|-----------|----------|------------|---------|
| k8s structured logs | 9.9 MB | **~40x** | 18.9x | - | **+113%** |
| NDJSON (10K rows) | 3.3 MB | **27.9x** | 9.0x | 16.4x | **+70%** |
| nginx access logs | 9.5 MB | **~28x** | 17.3x | - | **+62%** |
| NDJSON (analytics) | 107 KB | **21.7x** | 15.6x | 16.6x | **+31%** |
| Event tickets | 1.7 MB | **221.6x** | 176.0x | 190.0x | **+16%** |
| GH Archive (10 MB) | 10 MB | **8.0x** | 7.5x | 7.7x | **+4%** |
| Twitter API (nested) | 617 KB | **19.7x** | 16.7x | 18.9x | **+4%** |

Throughput (Apple M-series, release build):

| File | Turbo Encode | Normal Encode | Decode |
|------|-------------|---------------|--------|
| GH Archive (10 MB) | **169 MB/s** | 2.7 MB/s | 396 MB/s |
| NDJSON 10K rows (3.3 MB) | **68 MB/s** | 2.3 MB/s | 174 MB/s |
| Event tickets (1.7 MB) | **36 MB/s** | 8.1 MB/s | 1124 MB/s |
| Twitter API (617 KB) | **87 MB/s** | 2.2 MB/s | 377 MB/s |

All results are byte-exact lossless roundtrips.

## When to Use DataCortex

- Batch compression of JSON/NDJSON log files for cold storage
- Reducing S3/GCS storage costs for structured log data
- Real-time pipelines with `--turbo` (99 MB/s average encode)
- Any workload where compression ratio matters more than encode speed
- NDJSON pipelines with uniform or semi-uniform schemas

## When NOT to Use DataCortex

- Binary data, images, or arbitrary text (use zstd directly)
- Files under 1 KB (overhead exceeds benefit)

## Installation

**Rust CLI:**
```bash
cargo install datacortex-cli
```

**Python:**
```bash
pip install datacortex
```

**From source:**
```bash
git clone https://github.com/rushikeshmore/DataCortex
cd DataCortex
cargo build --release
```

Requires Rust 1.85+.

## Usage

```bash
# Compress (auto-detects JSON/NDJSON)
datacortex compress data.ndjson
datacortex compress api-response.json

# Turbo mode: 30-55x faster encode, ~2% ratio tradeoff
datacortex compress data.ndjson --turbo

# Decompress
datacortex decompress data.dcx output.ndjson

# Streaming (pipe-friendly)
cat logs.ndjson | datacortex compress - -o compressed.dcx
datacortex decompress compressed.dcx -o -

# Chunked compression for large NDJSON
datacortex compress logs.ndjson -o out.dcx --chunk-rows 10000

# Custom dictionary for known schemas
datacortex train-dict corpus/*.ndjson --output my.dict
datacortex compress logs.ndjson --dict my.dict

# Benchmark with turbo comparison
datacortex bench corpus/ -m fast --compare --turbo

# Inspect compressed files
datacortex info data.dcx
```

## Python API

```python
import datacortex

# Compress/decompress bytes
compressed = datacortex.compress(json_bytes, mode="fast")
original = datacortex.decompress(compressed)

# File-based
datacortex.compress_file("logs.ndjson", "logs.dcx", mode="fast")
datacortex.decompress_file("logs.dcx", "logs.json")

# Inspect and detect
info = datacortex.detect_format(data)  # "ndjson", "json", "generic"
```

See [PyPI](https://pypi.org/project/datacortex/) for full Python documentation.

## How It Works

1. **Format detection** -- identifies JSON, NDJSON, or generic data
2. **Schema inference** -- discovers column types (integer, boolean, timestamp, enum, string, float, UUID)
3. **Columnar reorg** -- transposes row-oriented NDJSON into column-oriented layout
4. **Type-specific encoding** -- delta+varint for integers, bitmaps for booleans, epoch deltas for timestamps
5. **Auto-fallback** -- tries 6+ compression paths (zstd, brotli, with/without preprocessing) and picks the smallest output. Turbo mode runs 2 optimized paths for maximum speed.

No schema files. No configuration. Fully automatic.

## Compression Modes

| Mode | Engine | Encode Speed | Best for |
|------|--------|-------------|----------|
| **fast** (default) | Columnar + typed encoding + zstd/brotli | 2-8 MB/s | Best ratio on JSON/NDJSON |
| **fast --turbo** | Columnar + typed encoding + zstd-3 | **99 MB/s** | Speed-sensitive pipelines |
| **balanced** | Context mixing (CM) engine | ~0.5 MB/s | General text, small files |
| **max** | CM with larger context maps | ~0.3 MB/s | Maximum compression |

Turbo mode produces the same `.dcx` format -- decompression is identical regardless of which mode was used to compress.

## Technical Background

DataCortex builds on ideas from columnar storage (Parquet, BtrBlocks), context mixing (PAQ/cmix), and format-aware preprocessing. Key techniques:

- Columnar transform with schema-based grouping for diverse NDJSON
- Selective columnar encoding (high-cardinality columns stay row-major)
- Per-column typed encoding: delta-varint integers, bitmap booleans, frequency-sorted enum dictionaries
- Adaptive zstd compression levels tuned for structured JSON
- FSST string compression support (decode-ready, reserved for future use)

Research references: ALP (SIGMOD 2024), FSST (VLDB 2020), BtrBlocks (SIGMOD 2023), CLP (OSDI 2021).

## Development

```bash
cargo test                                      # 393 tests
cargo clippy --all-targets -- -D warnings       # lint
cargo fmt --check                               # formatting
```

## Links
- [crates.io](https://crates.io/crates/datacortex-cli)
- [PyPI](https://pypi.org/project/datacortex/)
- [GitHub](https://github.com/rushikeshmore/DataCortex)

## License

MIT
