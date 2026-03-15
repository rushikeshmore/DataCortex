use std::fs;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use datacortex_core::{
    codec::compress_to_vec,
    compress,
    dcx::{FormatHint, Mode},
    decompress, detect_format,
    format::detect_from_extension,
    raw_zstd_compress, read_header,
};

#[derive(Parser)]
#[command(name = "datacortex", about = "Lossless text compression engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a file to .dcx format
    Compress {
        /// Input file path
        input: PathBuf,
        /// Output file path (default: input.dcx)
        output: Option<PathBuf>,
        /// Compression mode: max, balanced, fast
        #[arg(short, long, default_value = "balanced")]
        mode: String,
        /// Force format hint (auto-detected if omitted)
        #[arg(short, long)]
        format: Option<String>,
    },
    /// Decompress a .dcx file
    Decompress {
        /// Input .dcx file
        input: PathBuf,
        /// Output file path
        output: PathBuf,
    },
    /// Benchmark compression on a corpus directory
    Bench {
        /// Corpus directory (default: corpus/)
        #[arg(default_value = "corpus")]
        dir: PathBuf,
        /// Compression mode
        #[arg(short, long, default_value = "balanced")]
        mode: String,
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
            format!("unknown mode: {s} (expected: max, balanced, fast)"),
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
            format!("unknown format: {s}"),
        )),
    }
}

fn cmd_compress(
    input: &Path,
    output: Option<&Path>,
    mode: Mode,
    format_override: Option<FormatHint>,
) -> io::Result<()> {
    let data = fs::read(input)?;
    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("dcx"));

    // Use extension hint if content detection returns Generic.
    let format = format_override.unwrap_or_else(|| {
        let detected = detect_format(&data);
        if detected == FormatHint::Generic {
            detect_from_extension(input.to_str().unwrap_or("")).unwrap_or(FormatHint::Generic)
        } else {
            detected
        }
    });

    let mut out = BufWriter::new(fs::File::create(&output_path)?);
    compress(&data, mode, Some(format), &mut out)?;

    let output_size = fs::metadata(&output_path)?.len();
    let bpb = if data.is_empty() {
        0.0
    } else {
        (output_size as f64 * 8.0) / data.len() as f64
    };

    println!(
        "Compressed {} → {} ({} → {} bytes, {:.2} bpb, mode={}, format={})",
        input.display(),
        output_path.display(),
        data.len(),
        output_size,
        bpb,
        mode,
        format,
    );

    Ok(())
}

fn cmd_decompress(input: &Path, output: &Path) -> io::Result<()> {
    let file = fs::File::open(input)?;
    let mut reader = BufReader::new(file);
    let data = decompress(&mut reader)?;

    fs::write(output, &data)?;

    println!(
        "Decompressed {} → {} ({} bytes)",
        input.display(),
        output.display(),
        data.len(),
    );

    Ok(())
}

fn cmd_bench(dir: &Path, mode: Mode, save: bool) -> io::Result<()> {
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("corpus directory not found: {}", dir.display()),
        ));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();
    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        println!("No files found in {}", dir.display());
        return Ok(());
    }

    let compare_zstd = mode == Mode::Fast;

    println!("DataCortex Benchmark — mode: {mode}");
    println!("{}", "─".repeat(110));
    if compare_zstd {
        println!(
            "{:<25} {:>7} {:>7} {:>6} {:>7} {:>6} {:>7} {:>8}",
            "File", "Orig", "DCX", "bpb", "zstd", "bpb", "Δ%", "Format"
        );
    } else {
        println!(
            "{:<25} {:>7} {:>7} {:>7} {:>7} {:>8}",
            "File", "Orig", "Comp", "bpb", "Time", "Format"
        );
    }
    println!("{}", "─".repeat(110));

    let mut total_original: u64 = 0;
    let mut total_compressed: u64 = 0;
    let mut total_raw_zstd: u64 = 0;
    let mut results = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let data = fs::read(&path)?;

        let start = Instant::now();
        let compressed = compress_to_vec(&data, mode, None)?;
        let elapsed = start.elapsed();

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

        if compare_zstd {
            // Raw zstd at same level DCX uses for fair comparison.
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

            println!(
                "{:<25} {:>7} {:>7} {:>6.2} {:>7} {:>6.2} {:>+6.1}% {:>8}",
                name,
                format_size(orig_size),
                format_size(comp_size),
                bpb,
                format_size(raw_size),
                raw_bpb,
                delta_pct,
                detected,
            );

            results.push(serde_json::json!({
                "file": name,
                "original_bytes": orig_size,
                "compressed_bytes": comp_size,
                "bpb": (bpb * 1000.0).round() / 1000.0,
                "raw_zstd_bytes": raw_size,
                "raw_zstd_bpb": (raw_bpb * 1000.0).round() / 1000.0,
                "improvement_pct": (delta_pct * 10.0).round() / 10.0,
                "time_ms": (elapsed.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "format": detected.to_string(),
            }));
        } else {
            println!(
                "{:<25} {:>7} {:>7} {:>7.3} {:>6.1}ms {:>8}",
                name,
                format_size(orig_size),
                format_size(comp_size),
                bpb,
                elapsed.as_secs_f64() * 1000.0,
                detected,
            );

            results.push(serde_json::json!({
                "file": name,
                "original_bytes": orig_size,
                "compressed_bytes": comp_size,
                "bpb": (bpb * 1000.0).round() / 1000.0,
                "time_ms": (elapsed.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "format": detected.to_string(),
            }));
        }
    }

    let total_bpb = if total_original == 0 {
        0.0
    } else {
        (total_compressed as f64 * 8.0) / total_original as f64
    };

    println!("{}", "─".repeat(110));
    if compare_zstd {
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
            "{:<25} {:>7} {:>7} {:>6.2} {:>7} {:>6.2} {:>+6.1}%",
            "TOTAL",
            format_size(total_original),
            format_size(total_compressed),
            total_bpb,
            format_size(total_raw_zstd),
            total_raw_bpb,
            total_delta,
        );
    } else {
        println!(
            "{:<25} {:>7} {:>7} {:>7.3}",
            "TOTAL",
            format_size(total_original),
            format_size(total_compressed),
            total_bpb,
        );
    }

    if save {
        let baseline = serde_json::json!({
            "mode": mode.to_string(),
            "phase": "1-fast-preprocess",
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
        println!("\nBaseline saved to {}", baseline_path.display());
    }

    Ok(())
}

fn cmd_info(file: &Path) -> io::Result<()> {
    let f = fs::File::open(file)?;
    let mut reader = BufReader::new(f);
    let header = read_header(&mut reader)?;

    let overhead = header.total_size() as f64;
    let data_bpb = if header.original_size == 0 {
        0.0
    } else {
        ((header.compressed_size as f64 + overhead) * 8.0) / header.original_size as f64
    };

    println!("DataCortex .dcx File Info");
    println!("{}", "─".repeat(40));
    println!("  Mode:            {}", header.mode);
    println!("  Format:          {}", header.format_hint);
    println!("  Original size:   {} bytes", header.original_size);
    println!("  Compressed size: {} bytes", header.compressed_size);
    println!("  Header size:     {} bytes", header.total_size());
    println!("  CRC-32:          {:#010X}", header.crc32);
    println!(
        "  Transform meta:  {} bytes",
        header.transform_metadata.len()
    );
    println!("  bpb:             {:.3}", data_bpb);

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1}MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Command::Compress {
            input,
            output,
            mode,
            format,
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
            cmd_compress(input, output.as_deref(), mode, format)
        }
        Command::Decompress { input, output } => cmd_decompress(input, output),
        Command::Bench { dir, mode, save } => {
            let mode = parse_mode(mode).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            cmd_bench(dir, mode, *save)
        }
        Command::Info { file } => cmd_info(file),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
