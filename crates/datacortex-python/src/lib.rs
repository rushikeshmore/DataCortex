use std::io::Cursor;

use datacortex_core::dcx::{DcxHeader, FormatHint, Mode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_mode(mode: &str) -> PyResult<Mode> {
    match mode.to_lowercase().as_str() {
        "fast" => Ok(Mode::Fast),
        "balanced" => Ok(Mode::Balanced),
        "max" => Ok(Mode::Max),
        _ => Err(PyValueError::new_err(format!(
            "unknown mode: '{}' (expected: fast, balanced, max)",
            mode
        ))),
    }
}

fn parse_format(format: &str) -> PyResult<FormatHint> {
    match format.to_lowercase().as_str() {
        "auto" | "" => Ok(FormatHint::Generic),
        "json" => Ok(FormatHint::Json),
        "ndjson" | "jsonl" => Ok(FormatHint::Ndjson),
        "generic" => Ok(FormatHint::Generic),
        _ => Err(PyValueError::new_err(format!(
            "unknown format: '{}' (expected: auto, json, ndjson, generic)",
            format
        ))),
    }
}

/// Compress JSON/NDJSON data.
///
/// Args:
///     data: Input bytes to compress.
///     mode: Compression mode - "fast" (default), "balanced", or "max".
///     format: Format hint - "auto" (default), "json", "ndjson", "generic".
///     level: Optional zstd level override (fast mode only).
///
/// Returns:
///     Compressed bytes in .dcx format.
#[pyfunction]
#[pyo3(signature = (data, mode="fast", format="auto", level=None))]
fn compress(data: &[u8], mode: &str, format: &str, level: Option<i32>) -> PyResult<Vec<u8>> {
    let m = parse_mode(mode)?;
    let fmt = parse_format(format)?;
    let fmt_override = if fmt == FormatHint::Generic {
        None
    } else {
        Some(fmt)
    };

    datacortex_core::codec::compress_to_vec_with_options(data, m, fmt_override, None, level)
        .map_err(|e| PyValueError::new_err(format!("compression failed: {}", e)))
}

/// Decompress .dcx data back to the original bytes.
///
/// Args:
///     data: Compressed .dcx bytes.
///
/// Returns:
///     Original decompressed bytes.
#[pyfunction]
fn decompress(data: &[u8]) -> PyResult<Vec<u8>> {
    datacortex_core::codec::decompress_from_slice(data)
        .map_err(|e| PyValueError::new_err(format!("decompression failed: {}", e)))
}

/// Compress a file to .dcx format.
///
/// Args:
///     input_path: Path to the input file.
///     output_path: Path for the compressed output file.
///     mode: Compression mode - "fast" (default), "balanced", or "max".
///     level: Optional zstd level override (fast mode only).
#[pyfunction]
#[pyo3(signature = (input_path, output_path, mode="fast", level=None))]
fn compress_file(
    input_path: &str,
    output_path: &str,
    mode: &str,
    level: Option<i32>,
) -> PyResult<()> {
    let data = std::fs::read(input_path)
        .map_err(|e| PyValueError::new_err(format!("cannot read '{}': {}", input_path, e)))?;

    let m = parse_mode(mode)?;
    let fmt = datacortex_core::detect_format(&data);

    let mut out =
        std::io::BufWriter::new(std::fs::File::create(output_path).map_err(|e| {
            PyValueError::new_err(format!("cannot create '{}': {}", output_path, e))
        })?);

    datacortex_core::compress_with_options(&data, m, Some(fmt), None, level, &mut out)
        .map_err(|e| PyValueError::new_err(format!("compression failed: {}", e)))?;

    use std::io::Write;
    out.flush()
        .map_err(|e| PyValueError::new_err(format!("flush failed: {}", e)))?;

    Ok(())
}

/// Decompress a .dcx file back to the original.
///
/// Args:
///     input_path: Path to the .dcx file.
///     output_path: Path for the decompressed output file.
#[pyfunction]
fn decompress_file(input_path: &str, output_path: &str) -> PyResult<()> {
    let file = std::fs::File::open(input_path)
        .map_err(|e| PyValueError::new_err(format!("cannot open '{}': {}", input_path, e)))?;
    let mut reader = std::io::BufReader::new(file);

    let data = datacortex_core::decompress(&mut reader)
        .map_err(|e| PyValueError::new_err(format!("decompression failed: {}", e)))?;

    std::fs::write(output_path, &data)
        .map_err(|e| PyValueError::new_err(format!("cannot write '{}': {}", output_path, e)))?;

    Ok(())
}

/// Get metadata from compressed .dcx bytes.
///
/// Args:
///     data: Compressed .dcx bytes.
///
/// Returns:
///     Dict with mode, format, original_size, compressed_size, crc32, entropy_coder.
#[pyfunction]
fn info(py: Python, data: &[u8]) -> PyResult<Py<PyDict>> {
    let mut cursor = Cursor::new(data);
    let header = DcxHeader::read_from(&mut cursor)
        .map_err(|e| PyValueError::new_err(format!("invalid .dcx data: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("mode", header.mode.name())?;
    dict.set_item("format", header.format_hint.name())?;
    dict.set_item("original_size", header.original_size)?;
    dict.set_item("compressed_size", header.compressed_size)?;
    dict.set_item("crc32", header.crc32)?;
    dict.set_item(
        "entropy_coder",
        if header.use_brotli { "brotli" } else { "zstd" },
    )?;

    Ok(dict.into())
}

/// Detect the format of input data.
///
/// Args:
///     data: Input bytes.
///
/// Returns:
///     Detected format string: "json", "ndjson", "generic", etc.
#[pyfunction]
fn detect_format(data: &[u8]) -> String {
    datacortex_core::detect_format(data).name().to_string()
}

#[pymodule]
fn _datacortex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(compress_file, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_file, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
