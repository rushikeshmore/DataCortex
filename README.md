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

## How it works

```
Input JSON/NDJSON
  → Format detection (JSON vs NDJSON vs generic)
  → Schema inference (auto-detect column types)
  → Columnar reorg (group values by field)
  → Nested decomposition (flatten objects into sub-columns)
  → Type-specific encoding:
      Integers → delta + ZigZag + LEB128 varint
      Booleans → bitmap (8 per byte)
      Timestamps → epoch micros + delta varint
      Enums → frequency-sorted ordinal dictionary
      Strings → quote strip + length prefix
      UUIDs → 16-byte binary
  → Auto-select best entropy coder (zstd or brotli)
  → .dcx output
```

The auto-fallback tries 6+ compression paths and picks the absolute smallest output. You always get the best result.

## Installation

```bash
# From source
git clone https://github.com/rushikeshmore/DataCortex
cd DataCortex
cargo build --release
# Binary at target/release/datacortex
```

Requires Rust 1.85+ (edition 2024).

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

## Why DataCortex beats zstd on JSON

General-purpose compressors (zstd, brotli, gzip) treat JSON as opaque bytes. They find repeated patterns via LZ77 sliding window matching but don't understand the structure.

DataCortex understands JSON:

1. **Schema inference** — auto-detects that `timestamp` is a timestamp, `status` is a low-cardinality enum, `user_id` is a string
2. **Columnar reorg** — groups all timestamps together, all status codes together (like Parquet, but automatic)
3. **Type-specific encoding** — timestamps become tiny delta-encoded varints, booleans become bitmaps, enums become 1-byte ordinals
4. **The preprocessed data compresses dramatically better** — zstd/brotli on columnar+typed data achieves 2-3x better ratios than on raw JSON

## Architecture

```
datacortex/
  crates/
    datacortex-core/          Core compression library
      src/
        format/               Schema inference, columnar transforms, typed encoding
        model/                CM engine (13 context models)
        mixer/                Triple logistic mixer + 7-stage APM
        entropy/              Binary arithmetic coder
        codec.rs              Pipeline orchestrator + auto-fallback
        dcx.rs                .dcx file format (v3)
    datacortex-cli/           CLI binary
  corpus/                     Test corpus (JSON, NDJSON, text)
  benchmarks/                 Baseline measurements
```

## Development

```bash
cargo test                                      # 354 tests
cargo clippy --all-targets -- -D warnings       # lint (0 warnings)
cargo fmt --check                               # formatting
cargo build --release                           # optimized build
```

## License

MIT
