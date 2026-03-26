"""DataCortex - The best standalone JSON/NDJSON compressor.

Beats zstd-19 and brotli-11 on every JSON file tested.

Usage:
    import datacortex

    compressed = datacortex.compress(json_bytes)
    original = datacortex.decompress(compressed)

    datacortex.compress_file("data.json", "data.dcx")
    datacortex.decompress_file("data.dcx", "data.json")

    info = datacortex.info(compressed)
"""

from datacortex._datacortex import (
    compress,
    decompress,
    compress_file,
    decompress_file,
    info,
    detect_format,
    __version__,
)

__all__ = [
    "compress",
    "decompress",
    "compress_file",
    "decompress_file",
    "info",
    "detect_format",
    "__version__",
]
