# datacortex-vector

DataCortex compression codec for [Vector.dev](https://vector.dev/) log pipelines.

Compresses NDJSON log data **2-8x smaller than zstd-19** by understanding JSON structure: auto-infers schema, does columnar reorg, applies typed encoding, and picks the best compression path per file.

## Compression Ratios

| File | Size | DataCortex | zstd-19 | vs zstd |
|------|------|-----------|---------|---------|
| NDJSON analytics (200 rows) | 107 KB | **22.0x** | 15.6x | **+41%** |
| NDJSON uniform (10K rows) | 3.3 MB | **27.9x** | 9.0x | **+70%** |
| Twitter API JSON | 617 KB | **19.7x** | 16.7x | **+18%** |
| Event tickets (repetitive) | 1.7 MB | **221.7x** | 176.0x | **+26%** |
| GH Archive (diverse) | 10 MB | **8.0x** | 7.5x | **+7%** |

DataCortex wins on every file tested, with the biggest gains on uniform/structured NDJSON (the typical log pipeline use case).

## Throughput

| Mode | Encode Speed | Ratio | Best for |
|------|-------------|-------|----------|
| **Turbo** (default for pipelines) | **99 MB/s** | ~12% | Real-time log pipelines |
| **Normal Fast** | 2.7 MB/s | ~10% | Maximum ratio, batch jobs |

Turbo mode produces the same `.dcx` format -- decompression is identical (327+ MB/s).

## How It Works

Vector's compression pipeline: `Events -> Encoding (NDJSON) -> Compression -> Sink (S3/file)`.

This codec plugs into the compression step:

1. Buffer all encoded NDJSON bytes
2. On `finish()`, run DataCortex pipeline:
   - Auto-detect NDJSON format
   - Columnar reorg (group values by column)
   - Typed encoding (timestamps -> deltas, booleans -> bitmaps, etc.)
   - Auto-fallback across compression paths, pick smallest output
3. Write compressed `.dcx` bytes to the underlying writer

## Usage

```rust
use std::io::Write;
use datacortex_vector::{DataCortexEncoder, DataCortexDecoder, DataCortexConfig};

// Default: Fast mode, best ratio
let mut output = Vec::new();
let mut encoder = DataCortexEncoder::new(&mut output);
encoder.write_all(ndjson_bytes)?;
let writer = encoder.finish()?;

// Turbo: 33x faster encode for real-time pipelines
let mut output = Vec::new();
let mut encoder = DataCortexEncoder::with_config(&mut output, DataCortexConfig::turbo());
encoder.write_all(ndjson_bytes)?;
let writer = encoder.finish()?;

// Decompress
let original = DataCortexDecoder::decompress(&compressed)?;
```

## Vector Integration (Proposed)

```toml
# vector.toml
[sinks.s3_logs]
type = "aws_s3"
compression = "datacortex"    # <-- proposed new option
encoding.codec = "ndjson"
bucket = "my-logs"
```

Vector does not currently have a plugin system for compression codecs. Integration requires a PR to the [Vector repository](https://github.com/vectordotdev/vector). See the [GitHub Discussion](https://github.com/vectordotdev/vector/discussions/25063) for the proposal.

## Running

```bash
# Build
cargo build --release

# Test (roundtrips all corpus files)
cargo test

# Benchmarks (datacortex vs zstd vs gzip)
cargo bench

# Example
cargo run --example compress_ndjson
cargo run --example compress_ndjson -- path/to/logs.ndjson
```

## Architecture

- `DataCortexEncoder<W: io::Write>` -- matches Vector's SnappyEncoder pattern
- `DataCortexDecoder` -- decompresses `.dcx` data
- `DataCortexConfig` -- mode (Fast/Balanced/Max), format hint, turbo toggle
- Default: Fast mode + NDJSON format (optimal for log pipelines)
- Turbo: `DataCortexConfig::turbo()` for 99 MB/s encode

## File Extension & Content-Encoding

- File extension: `.log.dcx`
- HTTP Content-Encoding: `datacortex`
