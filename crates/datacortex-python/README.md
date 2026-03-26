# datacortex

Python bindings for [DataCortex](https://github.com/rushikeshmore/DataCortex), the best standalone JSON/NDJSON compressor.

Beats zstd-19 and brotli-11 on every JSON file tested.

## Install

```
pip install datacortex
```

## Usage

```python
import datacortex

# Compress bytes
compressed = datacortex.compress(json_bytes, mode="fast")
original = datacortex.decompress(compressed)

# Compress files
datacortex.compress_file("data.json", "data.dcx", mode="fast")
datacortex.decompress_file("data.dcx", "output.json")

# Inspect compressed data
info = datacortex.info(compressed)
# {'mode': 'fast', 'format': 'json', 'original_size': 435889, ...}

# Detect format
fmt = datacortex.detect_format(json_bytes)
# 'json', 'ndjson', 'generic', etc.
```

## Modes

- `"fast"` (default) - Best for JSON/NDJSON. Uses format-aware preprocessing + zstd/brotli.
- `"balanced"` - Context mixing engine. Better on general text, slower.
- `"max"` - Maximum compression. Slowest.

## License

MIT
