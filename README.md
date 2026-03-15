# DataCortex

Lossless text compression engine that understands file structure. Format-aware preprocessing + bit-level context mixing + adaptive entropy coding, written in Rust.

Most compressors treat JSON the same as Shakespeare. DataCortex detects the file format, applies structure-aware transforms, then feeds the result through a 13-model context mixing engine with adaptive prediction.

## Results

**Balanced mode** (13-model context mixing) vs common compressors on our test corpus:

| File | Size | DataCortex | zstd -19 | bzip2 -9 |
|------|------|-----------|----------|----------|
| alice29.txt (prose) | 149 KB | **2.17 bpb** | 2.59 bpb | 2.27 bpb |
| test-api.json | 5.5 KB | 2.07 bpb | 1.80 bpb | 2.07 bpb |
| test-code.rs | 15 KB | **2.11 bpb** | 2.35 bpb | 2.29 bpb |
| test-config.json | 2.0 KB | 3.37 bpb | 3.25 bpb | 3.42 bpb |
| test-doc.md | 12 KB | **3.07 bpb** | 3.47 bpb | 3.47 bpb |
| test-log.log | 8.3 KB | **1.97 bpb** | 2.15 bpb | 2.24 bpb |
| test-ndjson.ndjson | 8.3 KB | 1.51 bpb | 1.37 bpb | 1.77 bpb |
| **Corpus total** | **200 KB** | **2.20 bpb** | **2.54 bpb** | **2.35 bpb** |

DataCortex Balanced beats zstd -19 by **13.5%** overall and bzip2 -9 by **6.4%** on this corpus. Strongest wins are on prose, markdown, logs, and code where the context mixing engine's higher-order models dominate.

> bpb = bits per byte. Lower is better. 8.0 = no compression. Raw text is typically 2-4 bpb with good compressors.

**Fast mode** wraps zstd with format-aware preprocessing. On structured data (JSON, NDJSON), the preprocessing step can improve compression, though on generic text the overhead is marginal.

## Installation

### From source

```bash
git clone https://github.com/RushikeshMore/datacortex
cd datacortex
cargo build --release
# Binary at target/release/datacortex
```

### Cargo install

```bash
cargo install datacortex-cli
```

Requires Rust 1.85+ (edition 2024).

## Usage

```bash
# Compress a file (auto-detects format)
datacortex compress data.json -m balanced
datacortex compress logs.log -m fast

# Decompress
datacortex decompress data.dcx output.json

# Benchmark a directory
datacortex bench corpus/ -m balanced
datacortex bench corpus/ -m balanced --compare   # show zstd comparison

# Inspect a .dcx file
datacortex info data.dcx

# Options
datacortex --help
datacortex compress --help
datacortex -q compress data.json         # quiet mode
datacortex -v bench corpus/ -m balanced  # verbose mode
```

### Compression modes

| Mode | Engine | Speed | Compression | Memory |
|------|--------|-------|-------------|--------|
| **fast** | zstd (level 3) + preprocessing | ~100 MB/s | ~3.0 bpb | ~10 MB |
| **balanced** | 13-model context mixing | ~0.5 MB/s | ~2.2 bpb | ~256 MB |
| **max** | Reserved (neural, not yet available) | - | - | - |

**Balanced** is the default and recommended mode. It runs a full bit-level context mixing engine with 13 specialized models and a triple logistic mixer. Slower than zstd, but significantly better compression on text.

**Fast** mode uses zstd as the backend with format-aware preprocessing on top. Good for when you need speed but want the format detection.

**Max** mode is reserved for a future neural model (RWKV). Currently not functional.

## Format detection

DataCortex auto-detects file formats and applies specialized preprocessing:

| Format | Detection | Preprocessing |
|--------|-----------|---------------|
| **JSON** | `{` / `[` start, valid structure | Key interning (dedup repeated keys) |
| **NDJSON** | Line-delimited JSON | Per-line key interning |
| **Markdown** | `#` headers, `[links]()` | Passthrough (CM handles structure) |
| **Code** | Language keywords, braces | Passthrough |
| **Logs** | Timestamp patterns, log levels | Passthrough |
| **CSV** | Comma/tab delimited, consistent columns | Passthrough |
| **Generic** | Fallback | None |

Format can be overridden with `-f json` / `-f markdown` / etc.

## Architecture

```
Input --> Format Detection --> Preprocessing --> CM Engine --> Entropy Coding --> .dcx
                                                   |
                                            13 Context Models
                                            Triple Logistic Mixer
                                            3-Stage APM Cascade
                                            Binary Arithmetic Coder
```

### Context models (Balanced mode)

The CM engine runs 13 models in parallel, each predicting the next bit:

- **Order 0-7**: Byte-level context models with increasing history (256 direct to 32MB associative)
- **Match model**: Ring buffer (16MB) + hash table (8M entries) for long-range pattern matching
- **Word model**: Word boundary context (16MB)
- **Sparse model**: Skip-byte context for periodic patterns (8MB)
- **Run model**: Run-length encoding context (2MB)
- **JSON model**: Structure-aware context for JSON key/value/array state (4MB)

All predictions are combined through a **triple logistic mixer** (fine 64K + medium 16K + coarse 4K weights) and refined by a **3-stage APM cascade**. The final probability drives a 12-bit binary arithmetic coder.

Total memory: ~256 MB.

### .dcx file format

32-byte header with magic bytes, mode, format hint, original size, compressed size, CRC-32, and optional transform metadata. All multi-byte fields are little-endian.

```
[DCX\x03] [v3] [mode] [format] [flags] [orig_size:8] [comp_size:8] [crc32:4] [meta_len:4] [meta...] [data...]
```

## Project structure

```
datacortex/
  crates/
    datacortex-core/       Core library
      src/
        format/            Format detection + JSON key interning + transforms
        model/             Order-0 through Order-7, match, word, sparse, run, JSON models
        state/             StateTable (256-state), StateMap, ContextMap (lossy/checksum/associative)
        mixer/             Triple logistic mixer + 3-stage APM
        entropy/           Binary arithmetic coder (12-bit, carry-free)
        codec.rs           Pipeline orchestrator
        dcx.rs             .dcx v3 file format
    datacortex-cli/        CLI binary
    datacortex-neural/     Neural models (stub, not yet implemented)
  corpus/                  Tier 1 test corpus (7 files)
  benchmarks/              Benchmark baselines
```

## Development

```bash
cargo test                                      # 146 tests
cargo clippy --all-targets -- -D warnings       # lint
cargo fmt --check                               # formatting
cargo run --release -- bench corpus/ -m balanced # benchmark
```

## License

MIT
