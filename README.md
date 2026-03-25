# DataCortex

The best standalone JSON/NDJSON compressor. Beats zstd-19 and brotli-11 on every file tested.

DataCortex auto-infers your JSON schema, applies columnar reorg + type-specific encoding, then picks the optimal entropy coder (zstd or brotli). No schema files, no database, no configuration — just `datacortex compress data.json`.

## Benchmarks

**Fast mode** vs the best general-purpose compressors:

| File | Size | DataCortex | zstd -19 | brotli -11 | vs best |
|------|------|-----------|----------|------------|---------|
| NDJSON (analytics) | 107 KB | **22.4x** | 15.6x | 16.6x | **+35%** |
| NDJSON (10K rows) | 3.3 MB | **27.8x** | 16.0x | 16.4x | **+70%** |
| JSON API response | 36 KB | **16.0x** | 13.2x | 15.0x | **+7%** |
| Twitter API (nested) | 617 KB | **19.7x** | 16.7x | 18.9x | **+4%** |
| Event tickets (repetitive) | 1.7 MB | **222.3x** | 176.0x | 190.0x | **+17%** |

On larger structured logs:

| Data | Size | DataCortex | zstd -19 | Advantage |
|------|------|-----------|----------|-----------|
| k8s structured logs (100K rows) | 9.9 MB | **~40x** | 18.9x | **+113%** |
| nginx access logs (100K rows) | 9.5 MB | **~28x** | 17.3x | **+62%** |

> Higher is better. DataCortex wins on every file. Lossless — byte-exact decompression guaranteed.

## Installation

```bash
cargo install datacortex-cli
```

Or from source:

```bash
git clone https://github.com/rushikeshmore/DataCortex
cd DataCortex
cargo build --release
```

Requires Rust 1.85+.

## Usage

```bash
# Compress (auto-detects format, picks best compression)
datacortex compress data.ndjson
datacortex compress api-response.json
datacortex compress logs.ndjson -m fast          # explicit fast mode

# Decompress
datacortex decompress data.dcx output.ndjson

# Benchmark against zstd
datacortex bench corpus/ -m fast --compare

# Higher compression (slower)
datacortex compress data.ndjson -m fast --level 19

# Inspect a .dcx file
datacortex info data.dcx
```

## Compression modes

| Mode | Engine | Best for |
|------|--------|----------|
| **fast** (default) | Columnar + typed encoding + zstd/brotli | JSON/NDJSON (best ratio at high speed) |
| **balanced** | Context mixing (CM) engine | General text, small files |
| **max** | CM with larger context maps | Maximum compression |

**Fast mode** is recommended for JSON/NDJSON. It runs the full preprocessing pipeline (schema inference, columnar reorg, typed encoding) then picks the best entropy coder automatically.

**Balanced/Max modes** use a bit-level context mixing engine with 13 specialized models. Better for general text but slower.

## Development

```bash
cargo test                                      # 354 tests
cargo clippy --all-targets -- -D warnings       # lint (0 warnings)
cargo fmt --check                               # formatting
cargo build --release                           # optimized build
```

## License

MIT
