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

compressed = datacortex.compress(data)
print(f"Ratio: {len(data) / len(compressed):.1f}x")

# Turbo mode: ~30x faster encode, ~2% ratio tradeoff
fast = datacortex.compress(data, turbo=True)

# Decompress (byte-exact)
original = datacortex.decompress(compressed)
assert original == data
```

## API Reference

### compress(data, mode="fast", format="auto", level=None, turbo=False)

Compress bytes. Returns compressed bytes in .dcx format.

**Args:**
- `data` (bytes): Input data (JSON, NDJSON, or generic text)
- `mode` (str): `"fast"` (default), `"balanced"`, or `"max"`
- `format` (str): `"auto"` (default), `"json"`, `"ndjson"`, `"generic"`
- `level` (int, optional): zstd level override (fast mode only)
- `turbo` (bool): Use turbo mode for ~30x faster encode (fast mode only)

**Returns:** bytes

### decompress(data)

Decompress .dcx bytes. Returns the original data, byte-exact.

**Args:**
- `data` (bytes): Compressed .dcx data

**Returns:** bytes

### compress_file(input_path, output_path, mode="fast", level=None, turbo=False)

Compress a file to .dcx format.

**Args:**
- `input_path` (str): Path to the input file
- `output_path` (str): Path for the compressed output
- `mode` (str): `"fast"`, `"balanced"`, or `"max"`
- `level` (int, optional): zstd level override (fast mode only)
- `turbo` (bool): Use turbo mode for ~30x faster encode (fast mode only)

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

**Returns:** dict with keys: `mode`, `format`, `original_size`, `compressed_size`, `crc32`, `entropy_coder`

## Compression Modes

| Mode | Engine | Speed | Best for |
|------|--------|-------|----------|
| `"fast"` | Columnar + typed encoding + zstd/brotli | 2.7 MB/s encode | Best ratio on JSON/NDJSON |
| `"fast"` + `turbo=True` | Columnar + typed encoding + zstd-3 | **99 MB/s encode** | Speed-sensitive pipelines |
| `"balanced"` | Context mixing engine | <1 MB/s | General text |
| `"max"` | CM with larger context maps | <1 MB/s | Maximum compression |

## Benchmarks (v0.6.0)

| File | Size | DataCortex | zstd -19 | vs zstd | Turbo Encode |
|------|------|-----------|----------|---------|-------------|
| k8s structured logs | 9.9 MB | ~40x | 18.9x | **+113%** | -- |
| NDJSON 10K rows | 3.3 MB | 27.9x | 16.0x | **+68%** | 68 MB/s |
| GH Archive | 10 MB | 8.0x | 6.3x | **+26%** | 169 MB/s |
| Twitter API | 617 KB | 19.7x | 14.7x | **+34%** | 87 MB/s |
| Event tickets | 1.7 MB | 221.7x | 189.8x | **+17%** | 36 MB/s |

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
