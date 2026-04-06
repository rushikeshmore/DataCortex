# DataCortex - Claude Code Instructions

## What This Is
JSON/NDJSON-focused lossless compression engine in Rust. Dual-pipeline: Fast mode (columnar + typed encoding + zstd/brotli auto-fallback) and Balanced mode (columnar + CM engine). Beats zstd-19 (+4% to +113%) and brotli-11 on every JSON file tested.

## Architecture
```
Input → Format Detection → NDJSON Columnar Reorg → [Nested Decomposition] → [Typed Encoding (Fast only)] → [Value Dict] → [zstd Dict Training (Fast)] / [CM Engine (Balanced)] → .dcx output
```

Key modules:
- `format/ndjson.rs` -Columnar transform (uniform Strategy 1 + grouped Strategy 2)
- `format/schema.rs` -Auto schema inference (8 types: Integer, Float, Boolean, Timestamp, UUID, Enum, String, Null)
- `format/typed_encoding.rs` -Type-specific binary encoding (delta varint, bitmap, enum dict, timestamp delta, etc.)
- `format/value_dict.rs` -Per-column dictionary encoding
- `format/json.rs` -Key interning (Balanced/Max only)
- `format/json_array.rs` -JSON array columnar
- `model/` -CM engine: Order0-9, match, word, sparse, run, JSON, indirect, PPM, DMC models
- `mixer/` -Triple logistic mixer + 7-stage APM + GRU MetaMixer
- `entropy/` -Binary arithmetic coder (12-bit)
- `codec.rs` -Pipeline orchestrator with zstd dict training
- `dcx.rs` -.dcx v3 file format

## Current Status
**v0.6.0.** JSON/NDJSON focused. 393 tests. Published: crates.io (core + CLI), PyPI (datacortex). Site: datacortex-dcx.vercel.app.

**Benchmark results:**
| File | Size | DataCortex | zstd-19 | brotli-11 | vs best |
|------|------|-----------|---------|-----------|---------|
| NDJSON analytics | 107 KB | **22.0x** | 15.6x | 16.6x | **+32%** |
| NDJSON 10K rows | 3.3 MB | **27.9x** | 9.0x | 16.4x | **+70%** |
| k8s logs 100K rows | 9.9 MB | **~40x** | 18.9x | - | **+113%** |
| GH Archive | 10 MB | **8.0x** | 7.5x | 7.7x | **+4%** |
| Twitter API | 617 KB | **19.7x** | 16.7x | 18.9x | **+4%** |
| Event tickets | 1.7 MB | **221.7x** | 176.0x | 190.0x | **+17%** |

**Throughput (v0.6.0):**
- Turbo mode: **99 MB/s** encode average (169 MB/s on GH Archive 10MB)
- Normal Fast: 2.7 MB/s encode (best ratio)
- Decode: 327-430 MB/s

## Dual Pipeline (Gotcha #35)
- **Fast mode:** columnar → typed encoding → zstd dict → auto-fallback (6+ paths: zstd/brotli × raw/preprocessed/embedded). Parallel via rayon.
- **Fast mode (turbo):** columnar → typed encoding → 2 paths (preprocessed+zstd-3, raw+zstd-3). 30-55x faster.
- **Balanced mode:** columnar → CM engine (no typed encoding). Typed encoding HURTS CM.
- **Max mode:** same as Balanced with larger context maps.

## Features (v0.6.0)
- **Turbo mode** (`--turbo`): 99 MB/s encode, ~2% ratio tradeoff. Same .dcx format.
- Streaming stdin/stdout (`compress - -o -`)
- Chunked compression (`--chunk-rows N`)
- Custom dictionary training (`train-dict` command + `--dict` flag)
- Parallel Fast mode (rayon, 247% CPU utilization)
- Python bindings via PyO3 (`pip install datacortex`)
- Auto-fallback picks smallest output from 6+ compression paths

## Build & Test
```bash
cargo build --release
cargo test                    # 393 tests (<5s for lib, minutes for integration)
cargo clippy --all-targets -- -D warnings
```

## Rules
1. **Roundtrip is sacred.** Compress → decompress must produce identical output. Always.
2. **Typed encoding is Fast-mode-only.** Never apply to Balanced/Max (gotcha #35).
3. **Benchmark after every change.** Use corpus/test-ndjson.ndjson + corpus/json-bench/uniform-10k.ndjson.
4. **A/B test new encoders.** Test with both Fast (zstd) and Balanced (CM) backends.
5. **Solo model test before mixing.** New CM models get solo bpb test first.
6. **No external deps for parsing.** Manual ISO 8601, UUID, etc. parsing (no chrono, no regex).

## Key Gotchas (51 total)
- **#35:** Typed encoding HURTS CM, HELPS zstd. Fast-mode-only.
- **#33:** Columnar transform + strong CM = worse than raw + strong CM (confirmed with cmix).
- **#34:** Value dict saves 55% raw but only 3% compressed (CM already predicts repetition).
- **#38:** Non-JSON transforms removed, archived at datacortex-general repo.
- **#39:** Hex-to-binary removes Huffman-exploitable alphabet structure.
- **#41:** Auto-fallback with 6+ paths is the key architecture.
- **#44:** Mixed-type columns corrupt data if typed-encoded as String. Check `has_mixed_quoting()`.
- **#48:** zstd levels 9-15 are a ratio plateau on structured JSON. Skip to 19 (≤16MB) or 16 (16-64MB).
- **#49:** zstd-19 is the absolute bottleneck for encode speed. Turbo mode uses zstd-3.
- **#50:** Turbo needs only 2 paths (preprocessed+zstd-3, raw+zstd-3). No brotli, no dict.
- **#51:** Preprocessing advantage persists at low zstd levels. Turbo still beats raw zstd-3 by 14%.
- Match model: rolling hash must be non-cumulative.
- Multi-set mixer FAILS with <100 inputs.
- η=2 for fine mixer (64K weights), η=4 for coarse (4K).

## Corpus
- `corpus/test-ndjson.ndjson` -200 rows, 14 columns, uniform schema (primary test)
- `corpus/test-api.json` -JSON API response
- `corpus/test-config.json` -Small JSON config
- `corpus/alice29.txt` -English prose (general text reference)
- `corpus/json-bench/uniform-10k.ndjson` -10K rows (scaling test)
- `corpus/json-bench/gharchive-10mb.ndjson` -Real GH Archive (diverse schemas)
- `corpus/json-bench/twitter.json` -simdjson benchmark
- `corpus/json-bench/citm_catalog.json` -Highly repetitive JSON
- `corpus/json-bench/canada.json` -GeoJSON (numeric-heavy)

## Typed Encoding (format/typed_encoding.rs)
- Integer: delta + ZigZag + LEB128 varint
- Boolean: bitmap (8 per byte)
- Timestamp: ISO 8601 → epoch micros → delta varint
- Enum: frequency-sorted ordinal dictionary (1 byte per value)
- String: quote strip + length prefix
- UUID: 38 bytes → 16 bytes binary
- Float: raw passthrough (roundtrip risk)

## Commits
Include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` in commit messages.

<!-- codecortex:start -->
## CodeCortex — Project Knowledge (auto-updated)

### Architecture
**datacortex** — rust, python — 53 files, 1160 symbols
- **Modules (4):** datacortex-core (22291loc), datacortex-cli (999loc), datacortex-neural (773loc), datacortex-python (224loc)

### Risk Map
**High-risk files:**
- `CLAUDE.md` — 47 changes, 5 bug-fixes, volatile
- `.codecortex/cortex.yaml` — 38 changes, 4 bug-fixes, volatile
- `.codecortex/constitution.md` — 37 changes, 4 bug-fixes, volatile
- `.codecortex/graph.json` — 37 changes, 4 bug-fixes, volatile
- `.codecortex/hotspots.md` — 37 changes, 4 bug-fixes, volatile

**Hidden couplings (co-change, no import):**
- `crates/datacortex-core/src/format/mod.rs` ↔ `crates/datacortex-core/src/format/transform.rs` (55% co-change)
- `crates/datacortex-core/src/mixer/dual_mixer.rs` ↔ `crates/datacortex-core/src/model/engine.rs` (71% co-change)
- `crates/datacortex-cli/src/main.rs` ↔ `crates/datacortex-core/src/lib.rs` (60% co-change)

**Bug-prone files:**
- `crates/datacortex-core/src/format/ndjson.rs` — 5 bug-fix commits
- `.../datacortex-core/src/format/typed_encoding.rs` — 5 bug-fix commits
- `.codecortex/symbols.json` — 4 bug-fix commits

### Before Editing
Check `.codecortex/hotspots.md` for risk-ranked files before editing.
If CodeCortex MCP tools are available, call `get_edit_briefing` for coupling + risk details.
If not, read `.codecortex/modules/<module>.md` for the relevant module's dependencies and bug history.

### Project Knowledge
Read these files directly (always available, no tool call needed):
- `.codecortex/hotspots.md` — risk-ranked files with coupling + bug data
- `.codecortex/modules/*.md` — module docs, dependencies, temporal signals
- `.codecortex/constitution.md` — full architecture overview
- `.codecortex/patterns.md` — coding conventions
- `.codecortex/decisions/*.md` — architectural decisions

### MCP Tools (if available)
If a CodeCortex MCP server is connected, these tools provide live analysis:
- `get_edit_briefing` — risk + coupling + bugs for files you plan to edit.
- `get_change_coupling` — files that co-change (hidden dependencies).
- `get_project_overview` — architecture + dependency graph summary.
- `get_dependency_graph` — scoped import/call graph for file or module.
- `lookup_symbol` — precise symbol search (name, kind, file filters).
<!-- codecortex:end -->
