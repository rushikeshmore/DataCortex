# datacortex

Python bindings for [DataCortex](https://github.com/rushikeshmore/DataCortex), a lossless JSON/NDJSON compressor that beats zstd-19 and brotli-11 on every file tested.

Built with Rust via PyO3. Native performance, Python convenience.

## Install

```bash
pip install datacortex
```

Requires Python 3.8+. Pre-built wheels available for macOS (ARM).

## Quick Start

```python
import datacortex

# Compress JSON bytes
with open("logs.ndjson", "rb") as f:
    data = f.read()

compressed = datacortex.compress(data, mode="fast")
print(f"Ratio: {len(data) / len(compressed):.1f}x")

# Decompress (byte-exact)
original = datacortex.decompress(compressed)
assert original == data
```

## API Reference

### compress(data, mode="fast")

Compress bytes. Returns compressed bytes in .dcx format.

**Args:**
- `data` (bytes): Input data (JSON, NDJSON, or generic text)
- `mode` (str): `"fast"` (default), `"balanced"`, or `"max"`

**Returns:** bytes

### decompress(data)

Decompress .dcx bytes. Returns the original data, byte-exact.

**Args:**
- `data` (bytes): Compressed .dcx data

**Returns:** bytes

### compress_file(input_path, output_path, mode="fast")

Compress a file to .dcx format.

**Args:**
- `input_path` (str): Path to the input file
- `output_path` (str): Path for the compressed output
- `mode` (str): `"fast"`, `"balanced"`, or `"max"`

### decompress_file(input_path, output_path)

Decompress a .dcx file back to the original.

**Args:**
- `input_path` (str): Path to the .dcx file
- `output_path` (str): Path for the decompressed output

### detect_format(data)

Detect the format of input data.

**Args:**
- `data` (bytes): Input data to analyze

**Returns:** str -- `"ndjson"`, `"json"`, `"json_array"`, or `"generic"`

### info(data)

Inspect compressed .dcx data.

**Args:**
- `data` (bytes): Compressed .dcx data

**Returns:** dict with keys: `mode`, `format`, `original_size`, `compressed_size`, `ratio`

## Compression Modes

| Mode | Engine | Best for |
|------|--------|----------|
| `"fast"` | Columnar + typed encoding + zstd/brotli | JSON/NDJSON (recommended) |
| `"balanced"` | Context mixing engine | General text |
| `"max"` | CM with larger context maps | Maximum compression |

## Benchmarks

| File | Size | DataCortex | zstd -19 | vs zstd |
|------|------|-----------|----------|---------|
| k8s structured logs | 9.9 MB | ~40x | 18.9x | +113% |
| NDJSON 10K rows | 3.3 MB | 27.2x | 16.0x | +70% |
| nginx access logs | 9.5 MB | ~28x | 17.3x | +62% |
| Twitter API | 617 KB | 19.7x | 16.7x | +18% |

## CLI

For command-line usage, install the Rust CLI:

```bash
cargo install datacortex-cli
```

## Links

- [GitHub](https://github.com/rushikeshmore/DataCortex)
- [CLI (crates.io)](https://crates.io/crates/datacortex-cli)
- [Site](https://datacortex-dcx.vercel.app)

## License

MIT
