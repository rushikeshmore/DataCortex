# DataCortex — Claude Code Instructions

## What This Is
JSON/NDJSON-focused lossless compression engine in Rust. Dual-pipeline: Fast mode (columnar + typed encoding + zstd dict) and Balanced mode (columnar + CM engine). Beats zstd-19 by 14-49% on NDJSON.

## Architecture
```
Input → Format Detection → NDJSON Columnar Reorg → [Nested Decomposition] → [Typed Encoding (Fast only)] → [Value Dict] → [zstd Dict Training (Fast)] / [CM Engine (Balanced)] → .dcx output
```

Key modules:
- `format/ndjson.rs` — Columnar transform (uniform Strategy 1 + grouped Strategy 2)
- `format/schema.rs` — Auto schema inference (8 types: Integer, Float, Boolean, Timestamp, UUID, Enum, String, Null)
- `format/typed_encoding.rs` — Type-specific binary encoding (delta varint, bitmap, enum dict, timestamp delta, etc.)
- `format/value_dict.rs` — Per-column dictionary encoding
- `format/json.rs` — Key interning (Balanced/Max only)
- `format/json_array.rs` — JSON array columnar
- `model/` — CM engine: Order0-9, match, word, sparse, run, JSON, indirect, PPM, DMC models
- `mixer/` — Triple logistic mixer + 7-stage APM + GRU MetaMixer
- `entropy/` — Binary arithmetic coder (12-bit)
- `codec.rs` — Pipeline orchestrator with zstd dict training
- `dcx.rs` — .dcx v3 file format

## Current Status
**v0.2.0-pivot.** JSON/NDJSON focused. 311 tests. ~40 commits.

**Benchmark results:**
| File | Size | zstd-19 | DCX Fast | Advantage |
|------|------|---------|----------|-----------|
| test-ndjson (200 rows) | 110KB | 15.5x | 17.7x | +14% |
| uniform-10k (10K rows) | 3.3MB | 15.9x | 23.6x | +49% |
| GH Archive (diverse) | 10MB | 7.5x | 6.7x | -11% |

## Dual Pipeline (Gotcha #35)
- **Fast mode:** columnar → typed encoding → zstd dict → zstd-9. Best ratio on NDJSON.
- **Balanced mode:** columnar → CM engine (no typed encoding). Typed encoding HURTS CM.
- **Max mode:** same as Balanced with larger context maps.

## Build & Test
```bash
cargo build --release
cargo test                    # 311 tests (<5s for lib, minutes for integration)
cargo clippy --all-targets -- -D warnings
```

## Rules
1. **Roundtrip is sacred.** Compress → decompress must produce identical output. Always.
2. **Typed encoding is Fast-mode-only.** Never apply to Balanced/Max (gotcha #35).
3. **Benchmark after every change.** Use corpus/test-ndjson.ndjson + corpus/json-bench/uniform-10k.ndjson.
4. **A/B test new encoders.** Test with both Fast (zstd) and Balanced (CM) backends.
5. **Solo model test before mixing.** New CM models get solo bpb test first.
6. **No external deps for parsing.** Manual ISO 8601, UUID, etc. parsing (no chrono, no regex).

## Key Gotchas
- **#35:** Typed encoding HURTS CM, HELPS zstd. Fast-mode-only.
- **#33:** Columnar transform + strong CM = worse than raw + strong CM (confirmed with cmix).
- **#34:** Value dict saves 55% raw but only 3% compressed (CM already predicts repetition).
- **#38:** Non-JSON transforms removed, archived at datacortex-general repo.
- Match model: rolling hash must be non-cumulative.
- Multi-set mixer FAILS with <100 inputs.
- η=2 for fine mixer (64K weights), η=4 for coarse (4K).

## Corpus
- `corpus/test-ndjson.ndjson` — 200 rows, 14 columns, uniform schema (primary test)
- `corpus/test-api.json` — JSON API response
- `corpus/test-config.json` — Small JSON config
- `corpus/alice29.txt` — English prose (general text reference)
- `corpus/json-bench/uniform-10k.ndjson` — 10K rows (scaling test)
- `corpus/json-bench/gharchive-10mb.ndjson` — Real GH Archive (diverse schemas)
- `corpus/json-bench/twitter.json` — simdjson benchmark
- `corpus/json-bench/citm_catalog.json` — Highly repetitive JSON
- `corpus/json-bench/canada.json` — GeoJSON (numeric-heavy)

## Typed Encoding (format/typed_encoding.rs)
- Integer: delta + ZigZag + LEB128 varint
- Boolean: bitmap (8 per byte)
- Timestamp: ISO 8601 → epoch micros → delta varint
- Enum: frequency-sorted ordinal dictionary (1 byte per value)
- String: quote strip + length prefix
- UUID: 38 bytes → 16 bytes binary
- Float: raw passthrough (roundtrip risk)

## Full Documentation (Obsidian Vault)
All detailed docs in `Rushikesh OS/2. Projects/06. DataCortex/`:
- `Build SOP's/gotchas.md` — 38 institutional lessons
- `Build SOP's/testing.md`, `deployment.md`, `coding.md` — SOPs
- Session logs in `00. Logs/`

## Commits
Include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in commit messages.

<!-- codecortex:start -->
## CodeCortex — Project Knowledge (auto-updated)

### Architecture
**datacortex** — rust — 49 files, 1061 symbols
- **Modules (3):** datacortex-core (20077loc), datacortex-neural (773loc), datacortex-cli (653loc)

### Risk Map
**High-risk files:**
- `CLAUDE.md` — 25 changes, volatile
- `.codecortex/constitution.md` — 20 changes, volatile
- `.codecortex/cortex.yaml` — 20 changes, volatile
- `.codecortex/graph.json` — 20 changes, volatile
- `.codecortex/hotspots.md` — 20 changes, volatile

**Hidden couplings (co-change, no import):**
- `crates/datacortex-core/src/format/mod.rs` ↔ `crates/datacortex-core/src/format/transform.rs` (71% co-change)
- `crates/datacortex-core/src/mixer/dual_mixer.rs` ↔ `crates/datacortex-core/src/model/engine.rs` (86% co-change)
- `crates/datacortex-core/src/model/engine.rs` ↔ `crates/datacortex-core/src/model/mod.rs` (57% co-change)

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
