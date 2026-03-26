use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use datacortex_core::{
    codec::{compress_to_vec, compress_with_full_options, train_dict},
    dcx::{FormatHint, Mode},
    decompress_with_model, detect_format,
    format::detect_from_extension,
    raw_zstd_compress, read_header,
};

#[derive(Parser)]
#[command(
    name = "datacortex",
    about = "Lossless text compression engine with format-aware preprocessing and context mixing.",
    long_about = "DataCortex — lossless text compression engine.\n\n\
        Understands file structure (JSON, Markdown, NDJSON, logs, code) and applies\n\
        format-aware preprocessing before bit-level context mixing with 13 models.\n\n\
        Modes:\n  \
          fast       — zstd backend with preprocessing (~3 bpb, fast)\n  \
          balanced   — CM engine, 13 models, ~256MB (~2.2 bpb, slow)\n  \
          max        — CM engine, 13 models, ~512MB (best compression, slower)",
    version,
    after_help = "Examples:\n  \
        datacortex compress data.json -m fast\n  \
        datacortex compress data.json -o output.dcx -m fast\n  \
        cat logs.ndjson | datacortex compress - -o logs.dcx -m fast\n  \
        datacortex compress large.ndjson --chunk-rows 10000 -m fast\n  \
        datacortex decompress data.dcx output.json\n  \
        datacortex bench corpus/ -m fast --compare\n  \
        datacortex info data.dcx"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Show detailed output (per-model predictions in bench)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Path to GGUF model for neural Max mode (or set DATACORTEX_MODEL env var)
    #[arg(long, global = true)]
    model_path: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a file to .dcx format (use - for stdin)
    Compress {
        /// Input file path (use - for stdin)
        input: PathBuf,
        /// Output file path (default: input.dcx, required when reading stdin)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Compression mode: max, balanced, fast
        #[arg(short, long, default_value = "balanced")]
        mode: String,
        /// Force format hint (auto-detected if omitted)
        #[arg(short, long)]
        format: Option<String>,
        /// Override zstd compression level (Fast mode only; default: mode-based)
        #[arg(long)]
        level: Option<i32>,
        /// Split NDJSON into chunks of N rows, each compressed as an independent frame.
        /// Enables bounded-memory compression of large files.
        #[arg(long)]
        chunk_rows: Option<usize>,
        /// Use a pre-trained dictionary for compression (from train-dict command).
        #[arg(long)]
        dict: Option<PathBuf>,
    },
    /// Decompress a .dcx file (supports multi-frame files)
    Decompress {
        /// Input .dcx file (use - for stdin)
        input: PathBuf,
        /// Output file path (use - for stdout)
        output: PathBuf,
    },
    /// Train a compression dictionary from sample files
    TrainDict {
        /// Sample files to train on (JSON/NDJSON)
        samples: Vec<PathBuf>,
        /// Output dictionary file
        #[arg(short, long, default_value = "datacortex.dict")]
        output: PathBuf,
        /// Maximum dictionary size in bytes
        #[arg(long, default_value = "65536")]
        max_size: usize,
    },
    /// Benchmark compression on a corpus directory
    Bench {
        /// Corpus directory (default: corpus/)
        #[arg(default_value = "corpus")]
        dir: PathBuf,
        /// Compression mode
        #[arg(short, long, default_value = "balanced")]
        mode: String,
        /// Show zstd comparison column
        #[arg(long)]
        compare: bool,
        /// Save results to benchmarks/baseline.json
        #[arg(long)]
        save: bool,
    },
    /// Show .dcx file info
    Info {
        /// .dcx file to inspect
        file: PathBuf,
    },
}

fn parse_mode(s: &str) -> io::Result<Mode> {
    match s.to_lowercase().as_str() {
        "max" => Ok(Mode::Max),
        "balanced" => Ok(Mode::Balanced),
        "fast" => Ok(Mode::Fast),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown mode: '{s}' (expected: max, balanced, fast)"),
        )),
    }
}

fn parse_format(s: &str) -> io::Result<FormatHint> {
    match s.to_lowercase().as_str() {
        "generic" => Ok(FormatHint::Generic),
        "json" => Ok(FormatHint::Json),
        "markdown" | "md" => Ok(FormatHint::Markdown),
        "ndjson" | "jsonl" => Ok(FormatHint::Ndjson),
        "csv" => Ok(FormatHint::Csv),
        "code" => Ok(FormatHint::Code),
        "log" => Ok(FormatHint::Log),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unknown format: '{s}' (expected: generic, json, markdown, ndjson, csv, code, log)"
            ),
        )),
    }
}

fn cmd_train_dict(
    samples: &[PathBuf],
    output: &Path,
    max_size: usize,
    quiet: bool,
) -> io::Result<()> {
    if samples.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no sample files provided",
        ));
    }

    let mut sample_data: Vec<Vec<u8>> = Vec::new();
    let mut total_bytes: u64 = 0;

    for path in samples {
        if !path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("sample file not found: {}", path.display()),
            ));
        }
        let data = fs::read(path)?;
        total_bytes += data.len() as u64;
        sample_data.push(data);
    }

    if !quiet {
        eprintln!(
            "Training dictionary from {} files ({} bytes)...",
            samples.len(),
            total_bytes
        );
    }

    let refs: Vec<&[u8]> = sample_data.iter().map(|d| d.as_slice()).collect();
    let start = Instant::now();
    let dict = train_dict(&refs, max_size)?;
    let elapsed = start.elapsed();

    fs::write(output, &dict)?;

    if !quiet {
        println!(
            "Dictionary trained: {} bytes, saved to {} ({:.1}ms)",
            dict.len(),
            output.display(),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}

fn is_stdin(p: &Path) -> bool {
    p.as_os_str() == "-"
}

fn is_stdout(p: &Path) -> bool {
    p.as_os_str() == "-"
}

fn read_input(input: &Path) -> io::Result<Vec<u8>> {
    if is_stdin(input) {
        let mut data = Vec::new();
        io::stdin().read_to_end(&mut data)?;
        Ok(data)
    } else {
        fs::read(input)
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_compress(
    input: &Path,
    output: Option<&Path>,
    mode: Mode,
    format_override: Option<FormatHint>,
    model_path: Option<&str>,
    zstd_level_override: Option<i32>,
    chunk_rows: Option<usize>,
    dict_path: Option<&Path>,
    quiet: bool,
    verbose: bool,
) -> io::Result<()> {
    if !is_stdin(input) {
        if !input.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("file not found: {}", input.display()),
            ));
        }
        if !input.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("not a file: {}", input.display()),
            ));
        }
    }

    let data = read_input(input)?;
    if data.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input is empty",
        ));
    }

    // Determine output path.
    let output_path = if let Some(p) = output {
        p.to_path_buf()
    } else if is_stdin(input) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "output path required when reading from stdin (use -o)",
        ));
    } else {
        input.with_extension("dcx")
    };

    let writing_stdout = is_stdout(&output_path);

    // Load external dictionary if provided.
    let dict_data = if let Some(dp) = dict_path {
        Some(fs::read(dp).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("cannot read dictionary '{}': {}", dp.display(), e),
            )
        })?)
    } else {
        None
    };
    let ext_dict = dict_data.as_deref();

    // Detect format.
    let format = format_override.unwrap_or_else(|| {
        let detected = detect_format(&data);
        if detected == FormatHint::Generic && !is_stdin(input) {
            detect_from_extension(input.to_str().unwrap_or("")).unwrap_or(FormatHint::Generic)
        } else {
            detected
        }
    });

    if verbose && !quiet {
        let input_name = if is_stdin(input) {
            "stdin".to_string()
        } else {
            input.display().to_string()
        };
        eprintln!("  input:  {} ({} bytes)", input_name, data.len());
        eprintln!("  mode:   {mode}");
        eprintln!("  format: {format}");
        if let Some(cr) = chunk_rows {
            eprintln!("  chunk:  {cr} rows per frame");
        }
    }

    let start = Instant::now();

    // Chunked mode: split NDJSON into N-row chunks, each compressed as independent frame.
    if let Some(chunk_size) = chunk_rows {
        if chunk_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--chunk-rows must be > 0",
            ));
        }

        let lines: Vec<&[u8]> = data
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .collect();
        let num_chunks = lines.len().div_ceil(chunk_size);

        let mut out: Box<dyn Write> = if writing_stdout {
            Box::new(BufWriter::new(io::stdout().lock()))
        } else {
            Box::new(BufWriter::new(fs::File::create(&output_path)?))
        };

        let mut total_compressed: u64 = 0;
        for (ci, chunk_lines) in lines.chunks(chunk_size).enumerate() {
            let mut chunk_data = Vec::new();
            for (i, line) in chunk_lines.iter().enumerate() {
                if i > 0 {
                    chunk_data.push(b'\n');
                }
                chunk_data.extend_from_slice(line);
            }
            chunk_data.push(b'\n');

            compress_with_full_options(
                &chunk_data,
                mode,
                Some(format),
                model_path,
                zstd_level_override,
                ext_dict,
                &mut out,
            )?;

            if verbose && !quiet {
                eprintln!(
                    "  chunk {}/{}: {} rows, {} bytes",
                    ci + 1,
                    num_chunks,
                    chunk_lines.len(),
                    chunk_data.len()
                );
            }
        }
        out.flush()?;
        drop(out);

        let elapsed = start.elapsed();

        if !writing_stdout {
            total_compressed = fs::metadata(&output_path)?.len();
        }

        if !quiet && !writing_stdout {
            let input_name = if is_stdin(input) {
                "stdin".to_string()
            } else {
                input.display().to_string()
            };
            println!(
                "{} -> {} ({} -> {} bytes, {:.2} bpb, {:.1}% ratio, mode={}, format={}, {} chunks, {:.1}ms)",
                input_name,
                output_path.display(),
                data.len(),
                total_compressed,
                (total_compressed as f64 * 8.0) / data.len() as f64,
                total_compressed as f64 / data.len() as f64 * 100.0,
                mode,
                format,
                num_chunks,
                elapsed.as_secs_f64() * 1000.0,
            );
        }

        return Ok(());
    }

    // Single-frame mode (default).
    let mut out: Box<dyn Write> = if writing_stdout {
        Box::new(BufWriter::new(io::stdout().lock()))
    } else {
        Box::new(BufWriter::new(fs::File::create(&output_path)?))
    };

    compress_with_full_options(
        &data,
        mode,
        Some(format),
        model_path,
        zstd_level_override,
        ext_dict,
        &mut out,
    )?;
    out.flush()?;
    drop(out);
    let elapsed = start.elapsed();

    if !quiet && !writing_stdout {
        let output_size = fs::metadata(&output_path)?.len();
        let bpb = (output_size as f64 * 8.0) / data.len() as f64;
        let ratio = output_size as f64 / data.len() as f64;
        let input_name = if is_stdin(input) {
            "stdin".to_string()
        } else {
            input.display().to_string()
        };
        println!(
            "{} -> {} ({} -> {} bytes, {:.2} bpb, {:.1}% ratio, mode={}, format={}, {:.1}ms)",
            input_name,
            output_path.display(),
            data.len(),
            output_size,
            bpb,
            ratio * 100.0,
            mode,
            format,
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}

fn cmd_decompress(
    input: &Path,
    output: &Path,
    model_path: Option<&str>,
    quiet: bool,
) -> io::Result<()> {
    let writing_stdout = is_stdout(output);

    // Read input (file or stdin).
    let file_data = if is_stdin(input) {
        let mut data = Vec::new();
        io::stdin().read_to_end(&mut data)?;
        data
    } else {
        if !input.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("file not found: {}", input.display()),
            ));
        }
        if !input.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("not a file: {}", input.display()),
            ));
        }
        fs::read(input)?
    };

    let start = Instant::now();

    // Check for multi-frame: read first header, see if data extends beyond first frame.
    let mut cursor = io::Cursor::new(&file_data);
    let first_header = read_header(&mut cursor)?;
    let first_frame_end = cursor.position() as usize + first_header.compressed_size as usize;

    let data = if first_frame_end < file_data.len() {
        // Multi-frame: decompress all frames sequentially.
        let mut output = Vec::new();
        let mut pos = 0;
        while pos < file_data.len() {
            let mut frame_cursor = io::Cursor::new(&file_data[pos..]);
            let frame_data = decompress_with_model(&mut frame_cursor, model_path)?;
            pos += frame_cursor.position() as usize;
            output.extend_from_slice(&frame_data);
        }
        output
    } else {
        // Single frame.
        let mut reader = io::Cursor::new(&file_data);
        decompress_with_model(&mut reader, model_path)?
    };

    let elapsed = start.elapsed();

    if writing_stdout {
        io::stdout().write_all(&data)?;
        io::stdout().flush()?;
    } else {
        fs::write(output, &data)?;
    }

    if !quiet && !writing_stdout {
        let input_name = if is_stdin(input) {
            "stdin".to_string()
        } else {
            input.display().to_string()
        };
        println!(
            "{} -> {} ({} bytes, {:.1}ms)",
            input_name,
            output.display(),
            data.len(),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}

fn cmd_bench(
    dir: &Path,
    mode: Mode,
    compare: bool,
    save: bool,
    quiet: bool,
    verbose: bool,
) -> io::Result<()> {
    if !dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("directory not found: {}", dir.display()),
        ));
    }
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("not a directory: {}", dir.display()),
        ));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();
    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        if !quiet {
            println!("No files found in {}", dir.display());
        }
        return Ok(());
    }

    let show_zstd = compare || mode == Mode::Fast;

    if !quiet {
        println!();
        println!("  DataCortex Benchmark");
        println!(
            "  Mode: {mode} | Files: {} | Dir: {}",
            entries.len(),
            dir.display()
        );
        println!();

        if show_zstd {
            println!(
                "  {:<24} {:>8} {:>8} {:>7} {:>8} {:>7} {:>8} {:>8}",
                "File", "Original", "DCX", "bpb", "zstd", "bpb", "vs zstd", "Format"
            );
            println!("  {}", "-".repeat(91));
        } else {
            println!(
                "  {:<24} {:>8} {:>8} {:>7} {:>9} {:>8}",
                "File", "Original", "DCX", "bpb", "Time", "Format"
            );
            println!("  {}", "-".repeat(68));
        }
    }

    let mut total_original: u64 = 0;
    let mut total_compressed: u64 = 0;
    let mut total_raw_zstd: u64 = 0;
    let mut total_time_ms: f64 = 0.0;
    let mut results = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let data = fs::read(&path)?;

        let start = Instant::now();
        let compressed = compress_to_vec(&data, mode, None)?;
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0;

        let orig_size = data.len() as u64;
        let comp_size = compressed.len() as u64;
        let bpb = if orig_size == 0 {
            0.0
        } else {
            (comp_size as f64 * 8.0) / orig_size as f64
        };

        let detected = detect_format(&data);

        total_original += orig_size;
        total_compressed += comp_size;
        total_time_ms += time_ms;

        if show_zstd {
            let zstd_level = match mode {
                Mode::Fast => 3,
                Mode::Balanced => 19,
                Mode::Max => 22,
            };
            let raw = raw_zstd_compress(&data, zstd_level)?;
            let raw_size = raw.len() as u64;
            let raw_bpb = if orig_size == 0 {
                0.0
            } else {
                (raw_size as f64 * 8.0) / orig_size as f64
            };
            let delta_pct = if raw_size == 0 {
                0.0
            } else {
                (1.0 - comp_size as f64 / raw_size as f64) * 100.0
            };

            total_raw_zstd += raw_size;

            if !quiet {
                println!(
                    "  {:<24} {:>8} {:>8} {:>6.2}  {:>8} {:>6.2}  {:>+6.1}% {:>8}",
                    name,
                    format_size(orig_size),
                    format_size(comp_size),
                    bpb,
                    format_size(raw_size),
                    raw_bpb,
                    delta_pct,
                    detected,
                );
            }

            if verbose && !quiet {
                eprintln!("    time: {time_ms:.1}ms");
            }

            results.push(serde_json::json!({
                "file": name,
                "original_bytes": orig_size,
                "compressed_bytes": comp_size,
                "bpb": (bpb * 1000.0).round() / 1000.0,
                "raw_zstd_bytes": raw_size,
                "raw_zstd_bpb": (raw_bpb * 1000.0).round() / 1000.0,
                "improvement_pct": (delta_pct * 10.0).round() / 10.0,
                "time_ms": (time_ms * 100.0).round() / 100.0,
                "format": detected.to_string(),
            }));
        } else {
            if !quiet {
                println!(
                    "  {:<24} {:>8} {:>8} {:>6.3}  {:>7.1}ms {:>8}",
                    name,
                    format_size(orig_size),
                    format_size(comp_size),
                    bpb,
                    time_ms,
                    detected,
                );
            }

            if verbose && !quiet {
                let ratio = if orig_size == 0 {
                    0.0
                } else {
                    comp_size as f64 / orig_size as f64
                };
                eprintln!(
                    "    ratio: {:.1}% | speed: {:.1} KB/s",
                    ratio * 100.0,
                    if time_ms > 0.0 {
                        orig_size as f64 / 1024.0 / (time_ms / 1000.0)
                    } else {
                        0.0
                    },
                );
            }

            results.push(serde_json::json!({
                "file": name,
                "original_bytes": orig_size,
                "compressed_bytes": comp_size,
                "bpb": (bpb * 1000.0).round() / 1000.0,
                "time_ms": (time_ms * 100.0).round() / 100.0,
                "format": detected.to_string(),
            }));
        }
    }

    let total_bpb = if total_original == 0 {
        0.0
    } else {
        (total_compressed as f64 * 8.0) / total_original as f64
    };

    if !quiet {
        if show_zstd {
            println!("  {}", "-".repeat(91));
            let total_raw_bpb = if total_original == 0 {
                0.0
            } else {
                (total_raw_zstd as f64 * 8.0) / total_original as f64
            };
            let total_delta = if total_raw_zstd == 0 {
                0.0
            } else {
                (1.0 - total_compressed as f64 / total_raw_zstd as f64) * 100.0
            };
            println!(
                "  {:<24} {:>8} {:>8} {:>6.2}  {:>8} {:>6.2}  {:>+6.1}%",
                "TOTAL",
                format_size(total_original),
                format_size(total_compressed),
                total_bpb,
                format_size(total_raw_zstd),
                total_raw_bpb,
                total_delta,
            );
        } else {
            println!("  {}", "-".repeat(68));
            println!(
                "  {:<24} {:>8} {:>8} {:>6.3}  {:>7.1}ms",
                "TOTAL",
                format_size(total_original),
                format_size(total_compressed),
                total_bpb,
                total_time_ms,
            );
        }

        let overall_ratio = if total_original == 0 {
            0.0
        } else {
            total_compressed as f64 / total_original as f64
        };
        println!();
        println!(
            "  Overall: {:.1}% compression ratio | {:.1}ms total",
            overall_ratio * 100.0,
            total_time_ms,
        );
        println!();
    }

    if save {
        let baseline = serde_json::json!({
            "mode": mode.to_string(),
            "version": env!("CARGO_PKG_VERSION"),
            "files": results,
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "total_bpb": (total_bpb * 1000.0).round() / 1000.0,
        });

        let baseline_path = Path::new("benchmarks/baseline.json");
        if let Some(parent) = baseline_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(baseline_path, serde_json::to_string_pretty(&baseline)?)?;
        if !quiet {
            println!("  Saved to {}", baseline_path.display());
        }
    }

    Ok(())
}

fn cmd_info(file: &Path, quiet: bool) -> io::Result<()> {
    if !file.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("file not found: {}", file.display()),
        ));
    }
    if !file.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not a file: {}", file.display()),
        ));
    }

    let f = fs::File::open(file)?;
    let mut reader = BufReader::new(f);
    let header = read_header(&mut reader)?;

    let overhead = header.total_size() as f64;
    let data_bpb = if header.original_size == 0 {
        0.0
    } else {
        ((header.compressed_size as f64 + overhead) * 8.0) / header.original_size as f64
    };
    let ratio = if header.original_size == 0 {
        0.0
    } else {
        (header.compressed_size as f64 + overhead) / header.original_size as f64
    };

    if !quiet {
        println!();
        println!("  DataCortex .dcx File Info");
        println!("  {}", "-".repeat(40));
        println!("  File:            {}", file.display());
        println!("  Mode:            {}", header.mode);
        println!("  Format:          {}", header.format_hint);
        println!(
            "  Original size:   {} ({})",
            format_size(header.original_size),
            header.original_size
        );
        println!(
            "  Compressed size: {} ({})",
            format_size(header.compressed_size),
            header.compressed_size
        );
        println!("  Header size:     {} bytes", header.total_size());
        println!("  CRC-32:          {:#010X}", header.crc32);
        println!(
            "  Entropy coder:   {}",
            if header.use_brotli { "brotli" } else { "zstd" }
        );
        if !header.transform_metadata.is_empty() {
            println!(
                "  Transform meta:  {} bytes",
                header.transform_metadata.len()
            );
        }
        println!("  bpb:             {:.3}", data_bpb);
        println!("  Ratio:           {:.1}%", ratio * 100.0);
        println!();
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.quiet && cli.verbose {
        eprintln!("Error: --quiet and --verbose cannot be used together");
        std::process::exit(1);
    }

    let result = match &cli.command {
        Command::Compress {
            input,
            output,
            mode,
            format,
            level,
            chunk_rows,
            dict,
        } => {
            let mode = parse_mode(mode).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let format = format.as_deref().map(|f| {
                parse_format(f).unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                })
            });
            cmd_compress(
                input,
                output.as_deref(),
                mode,
                format,
                cli.model_path.as_deref(),
                *level,
                *chunk_rows,
                dict.as_deref(),
                cli.quiet,
                cli.verbose,
            )
        }
        Command::Decompress { input, output } => {
            cmd_decompress(input, output, cli.model_path.as_deref(), cli.quiet)
        }
        Command::TrainDict {
            samples,
            output,
            max_size,
        } => cmd_train_dict(samples, output, *max_size, cli.quiet),
        Command::Bench {
            dir,
            mode,
            compare,
            save,
        } => {
            let mode = parse_mode(mode).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            cmd_bench(dir, mode, *compare, *save, cli.quiet, cli.verbose)
        }
        Command::Info { file } => cmd_info(file, cli.quiet),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
