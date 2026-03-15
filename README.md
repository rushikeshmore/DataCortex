# DataCortex

Next-generation lossless text compression engine. Format-aware preprocessing + context mixing + Ramanujan periodic features + entropy coding.

**Target:** Sub-1.0 bpb on enwik8. Sub-0.5 bpb on JSON. Self-contained Rust binary.

## Status

🚧 **Under construction** — Architecture complete, building Phase 0 (scaffold).

## Modes

| Mode | enwik8 Target | JSON Target | Speed |
|------|--------------|-------------|-------|
| **Max** | sub-1.0 bpb | sub-0.5 bpb | ≥1 KB/s |
| **Balanced** | 1.2-1.4 bpb | sub-0.6 bpb | 1-10 MB/s |
| **Fast** | 1.8-2.0 bpb | sub-0.8 bpb | 50-200 MB/s |

## What Makes It Different

Every existing compressor treats JSON the same as Shakespeare. DataCortex understands file structure:

- **JSON/NDJSON** — key extraction, schema detection, delta encoding
- **Markdown** — header hierarchy, link pooling, code block separation
- **Logs** — timestamp factoring, prefix deduplication
- **Code** — indentation normalization, keyword dictionaries

Format-aware preprocessing feeds a bit-level context mixing engine with Ramanujan periodic features for pattern detection.

## Stack

- **Language:** Rust 2024 edition
- **Architecture:** Format Detection → Preprocessing → Context Mixing → Entropy Coding → `.dcx`
- **License:** MIT

## Building

```bash
cargo build --release
```

## Usage

```bash
# Compress
datacortex compress -m balanced input.json output.dcx

# Decompress
datacortex decompress output.dcx roundtrip.json

# Benchmark
datacortex bench corpus/
```

## Benchmarks

*Coming soon — Phase 0 in progress.*

## License

MIT
