# datacortex-core

Core compression library for [DataCortex](https://github.com/rushikeshmore/DataCortex). Lossless, format-aware compression for JSON and NDJSON data.

Beats zstd-19 and brotli-11 on every JSON file tested (+4% to +113%).

## Usage

```rust
use datacortex_core::{compress, decompress, Mode, FormatHint};

// Compress JSON/NDJSON bytes
let compressed = compress(json_bytes, Mode::Fast, FormatHint::Auto)?;

// Decompress (byte-exact lossless)
let original = decompress(&compressed)?;

assert_eq!(json_bytes, &original);
```

## Features

- Auto-detection of JSON, NDJSON, and generic text formats
- Schema inference with 8 column types (integer, boolean, timestamp, enum, string, float, UUID, null)
- Columnar reorg for NDJSON with schema-based grouping
- Type-specific encoding (delta-varint, bitmaps, epoch deltas, frequency-sorted dictionaries)
- Auto-fallback across 6+ compression paths (zstd, brotli, with/without preprocessing)
- Custom dictionary training for known schemas
- Parallel compression via rayon (Fast mode)

## Compression Modes

- **Fast** (default): columnar + typed encoding + zstd/brotli auto-fallback. Best for JSON/NDJSON.
- **Balanced**: context mixing engine with 13 specialized models. Better on general text.
- **Max**: same as Balanced with larger context maps.

## Links

- [CLI tool](https://crates.io/crates/datacortex-cli)
- [Python bindings](https://pypi.org/project/datacortex/)
- [GitHub](https://github.com/rushikeshmore/DataCortex)

## License

MIT
