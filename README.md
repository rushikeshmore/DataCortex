# DataCortex

The best standalone JSON/NDJSON compressor. Beats zstd-19 and brotli-11 on every file tested.

[Site](https://datacortex-dcx.vercel.app) | [crates.io](https://crates.io/crates/datacortex-cli) | [PyPI](https://pypi.org/project/datacortex/) | [Docs](https://github.com/rushikeshmore/DataCortex)

DataCortex auto-infers your JSON schema, applies columnar reorg + type-specific encoding, then picks the optimal entropy coder (zstd or brotli). No schema files, no database, no configuration. Just `datacortex compress data.json`.

## Benchmarks

**Fast mode** vs the best general-purpose compressors:

| File | Size | DataCortex | zstd -19 | brotli -11 | vs best |
|------|------|-----------|----------|------------|---------|
| NDJSON (analytics) | 107 KB | **22.0x** | 15.6x | 16.6x | **+32%** |
| NDJSON (10K rows) | 3.3 MB | **27.8x** | 16.0x | 16.4x | **+70%** |
| JSON API response | 36 KB | **16.0x** | 13.2x | 15.0x | **+7%** |
| Twitter API (nested) | 617 KB | **19.7x** | 16.7x | 18.9x | **+4%** |
| Event tickets (repetitive) | 1.7 MB | **221.7x** | 176.0x | 190.0x | **+17%** |

On larger structured logs:

| Data | Size | DataCortex | zstd -19 | Advantage |
|------|------|-----------|----------|-----------|
| k8s structured logs (100K rows) | 9.9 MB | **~40x** | 18.9x | **+113%** |
| nginx access logs (100K rows) | 9.5 MB | **~28x** | 17.3x | **+62%** |

> Higher is better. DataCortex wins on every file. Lossless, byte-exact decompression guaranteed.

## Installation

**Rust:**
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
# Compress (auto-detects format, picks best compression)
datacortex compress data.ndjson
datacortex compress api-response.json
datacortex compress logs.ndjson -m fast          # explicit fast mode

# Decompress
datacortex decompress data.dcx output.ndjson

# Streaming (pipe-friendly)
cat logs.ndjson | datacortex compress - -o compressed.dcx
datacortex decompress compressed.dcx -o -        # stdout

# Chunked compression (for large NDJSON)
datacortex compress logs.ndjson -o out.dcx --chunk-rows 10000

# Custom dictionary (for known schemas)
datacortex train-dict corpus/*.ndjson --output my.dict
datacortex compress logs.ndjson --dict my.dict

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

## Python

```python
import datacortex

compressed = datacortex.compress(json_bytes, mode="fast")
original = datacortex.decompress(compressed)

# File-based
datacortex.compress_file("logs.ndjson", "logs.dcx", mode="fast")
datacortex.decompress_file("logs.dcx", "logs.json")

# Format detection
fmt = datacortex.detect_format(data)  # "ndjson", "json", "generic"
```

## How it works

1. **Format detection** - auto-identifies JSON, NDJSON, or generic data
2. **Schema inference** - discovers column types (integer, boolean, timestamp, enum, string, etc.)
3. **Columnar reorg** - transposes row-oriented NDJSON into column-oriented layout
4. **Type-specific encoding** - delta+varint for integers, bitmaps for booleans, epoch deltas for timestamps
5. **Auto-fallback** - tries 6+ compression paths (zstd, brotli, with/without preprocessing) and picks the smallest

No schema files. No configuration. Fully automatic.

## Development

```bash
cargo test                                      # 381 tests
cargo clippy --all-targets -- -D warnings       # lint (0 warnings)
cargo fmt --check                               # formatting
cargo build --release                           # optimized build
```

## License

MIT
