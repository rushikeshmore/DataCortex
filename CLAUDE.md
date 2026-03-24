# DataCortex — Claude Code Instructions

## What This Is
Lossless text compression engine in Rust. Three modes: Max (sub-1.0 bpb), Balanced (1.2-1.4 bpb), Fast (1.8-2.0 bpb). Format-aware preprocessing for JSON/MD/NDJSON/CSV/code/logs + bit-level context mixing + Ramanujan periodic features + entropy coding.

## Architecture
```
Input → Format Detection → Preprocessing → CM Engine (or Fast zstd) → .dcx output
```
- `crates/datacortex-core/` — Core library
  - `format/` — Detection (7 types) + JSON key interning + transform pipeline
  - `model/` — Order0-9, match model, word model, sparse, run, JSON, indirect, XML tracker; CMEngine orchestrator
  - `state/` — StateTable (256-state), StateMap (adaptive), ContextMap (lossy hash, checksum, 2-way assoc)
  - `mixer/` — Triple logistic mixer (fine 64K + med 16K + coarse 4K), squash/stretch, 3-stage APM
  - `entropy/` — Binary arithmetic coder (12-bit, carry-free)
  - `codec.rs` — Pipeline orchestrator (Fast=zstd, Balanced=CM, Max=CM)
  - `dcx.rs` — .dcx v3 file format (32-byte header + CRC-32)
- `crates/datacortex-cli/` — CLI binary (compress, decompress, bench, info)
- `crates/datacortex-neural/` — Optional RWKV model for Max mode (stub)
- `corpus/` — 7 Tier 1 test files (checked in, never change)
- `benchmarks/` — baseline.json

## Current Status
**Phase 26 (v0.2.0-dev).** 2.15 bpb alice29, **1.84 bpb enwik8** (confirmed full 100MB). 393 tests (389 unit + 4 integration). ~400MB memory. 29 commits.

**Engine:** 16 CM models + PPM + DMC + GRU byte mixer (128 cells, BPTT-10, 12% blend) + 7-stage APM + triple logistic mixer. 9 format transforms, 11 detected formats.

**BPTT-10 (uncommitted):** Truncated backprop on GRU. cargo test passed. alice29 2.15 bpb (-0.01). enwik8 benchmark running.

## Build & Test
```bash
cargo build --release
cargo test                    # Tier 1: roundtrip + unit tests (<5s)
cargo clippy --all-targets -- -D warnings
cargo bench --bench compress_bench  # Tier 2: full benchmark (~1 min)
```

## Rules
1. **Roundtrip is sacred.** Compress → decompress must produce identical output. Always.
2. **Benchmark after every change.** Tier 1 corpus. Log bpb. Regression >0.5% = stop and fix.
3. **Solo model test before mixing.** New models get solo bpb on alice29.txt before entering mixer.
4. **No speculative optimization.** Correct first, fast second.
5. **V2 dead ends stay dead.** Don't retry: logistic byte-level mixing, geometric mixing, hybrid byte+bit, η>5, 4+ APM stages on small files. See gotchas below.
6. **Format-aware advantage >10% on JSON** vs raw zstd. If not meeting this, focus on preprocessing.

## Key Gotchas (Don't Repeat — see vault gotchas.md for all 31)
- Match model: rolling hash must be non-cumulative (V2 bug cost 0.44 bpb)
- Logistic mixing at byte-level = +69% regression. Bit-level ONLY.
- Adding weak models dilutes mixer weights. Solo test first.
- η=2 for fine mixer (64K weights), η=4 for coarse (4K). Don't increase without A/B test.
- WRT regresses on XML-heavy content. Only enable per-format after A/B test.
- **Multi-set mixer FAILS with <100 inputs** (+0.44 regression at 28 inputs). Need multi-output ContextMap2 FIRST.
- **ISSE negligible** (-0.001 bpb, redundant with APM).
- **Hierarchical mixer** causes information bottleneck (-0.02 regression).
- **Memory scaling negligible** (450MB→1GB = -0.003 bpb). Architecture is bottleneck, not memory.
- **Transforms beat models** for structured data. For general text, paradigm diversity (PPM/DMC/neural) is the path.
- **LLM byte mapping:** unmapped bytes get mean logit, not -100. Keep blend weight low until comprehensive.

## V3 Engine (Phase 5+ — current)
- 16 models: Order-0 (256 direct), Order-1 (32MB), Order-2 (16MB), Order-3 (32MB checksum), Order-4 (32MB checksum), Order-5 (32MB assoc), Order-6 (16MB assoc), Order-7 (32MB assoc), Order-8 (32MB assoc), Order-9 (16MB assoc), Match (16MB ring + 8M hash, multi-candidate), Word (16MB), Sparse (16MB), Run (4MB), JSON (8MB), Indirect (8MB + 2MB pred table)
- XML state tracker: 8-state FSM provides context bits for markup-heavy content (tag/content/attr/comment/entity)
- Multi-output ContextMaps: each order model (O1-O9) produces 2 predictions (state + run-count). 28 total mixer inputs (was 19).
- Triple logistic mixer: fine (64K, η=2), medium (16K, η=3), coarse (4K, η=4). XML state + run-length context in mixer hash.
- Multi-set mixer available but not used (regresses on 1MB; needs larger data).
- 7-stage APM cascade: 2K/16K/4K/4K/4K/2K/4K contexts. XML state injected into APM2 and APM5.
- byte_class: 12 classes (was 8) — high bytes split into 4 WRT groups + escape for future WRT support.
- StateTable (256-state PAQ8), StateMap (adaptive 1/n), ContextMap (lossy/checksum/2-way assoc)
- Binary AC: 12-bit precision, carry-free
- JSON key interning: Balanced/Max only (hurts Fast due to zstd redundancy)
- WRT: implemented but disabled (regresses on XML-heavy content; keep for future per-format enable)
- Total memory: ~264MB

## Corpus (Tier 1 — run on every cargo test)
- `corpus/alice29.txt` — English prose (152 KB)
- `corpus/test-api.json` — JSON API responses (~100 KB)
- `corpus/test-config.json` — JSON configs (~10 KB)
- `corpus/test-doc.md` — Markdown docs (~50 KB)
- `corpus/test-ndjson.ndjson` — NDJSON stream (~100 KB)
- `corpus/test-log.log` — Server logs (~100 KB)
- `corpus/test-code.rs` — Rust source (~20 KB)

## Phase Gates
| Phase | Gate | bpb |
|-------|------|-----|
| 0 | cargo test passes, benchmark harness runs | — |
| 1 | Format-aware >10% on JSON vs zstd | Fast ships |
| 2 | Roundtrip all corpus files | ~7.0 (O0) |
| 3 | ≤2.10 bpb alice29 (match V2) | 2.10 |
| 4 | <1.80 alice29, <1.00 JSON-API | 1.80/1.00 |
| 5 | 1.2-1.4 enwik8 (Balanced) | 1.40 |
| 6 | sub-1.0 enwik8 (Max+RWKV) | 1.00 |
| 7 | All targets met, published | All |

## Full Documentation (Obsidian Vault)
All detailed docs are in `Rushikesh OS/2. Projects/06. DataCortex/`:
- `CLAUDE-Instructions.md` — Autopilot bootstrap (session start/end protocol)
- `Architecture-Strategy.md` — Full architecture, project structure, phases
- `Benchmark-Targets.md` — bpb targets per mode per file type
- `DataCortex-Retrospective.md` — V1/V2 learnings
- `Landscape-Analysis.md` — SOTA survey (21 compressors)
- `Ramanujan-Structures.md` — RPT, filter banks, Stern-Brocot research
- `Build SOP's/` — testing, deployment, gotchas, surfaces, coding

## Commits
Include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` in commit messages.

<!-- codecortex:start -->
## CodeCortex — Project Knowledge (auto-updated)

### Architecture
**datacortex** — rust — 50 files, 968 symbols
- **Modules (3):** datacortex-core (16457loc), datacortex-neural (773loc), datacortex-cli (635loc)

### Risk Map
**High-risk files:**
- `crates/datacortex-core/src/format/mod.rs` — 14 changes, volatile, coupled to: transform.rs ⚠, dcx.rs ⚠
- `crates/datacortex-core/src/model/engine.rs` — 13 changes, volatile, coupled to: dual_mixer.rs ⚠, mod.rs ⚠
- `CLAUDE.md` — 12 changes, volatile
- `crates/datacortex-core/src/codec.rs` — 11 changes, volatile, coupled to: mod.rs ⚠, mod.rs ⚠
- `crates/datacortex-core/src/mixer/dual_mixer.rs` — 11 changes, volatile, coupled to: engine.rs ⚠, mod.rs ⚠

**Hidden couplings (co-change, no import):**
- `crates/datacortex-core/src/mixer/dual_mixer.rs` ↔ `crates/datacortex-core/src/model/engine.rs` (85% co-change)
- `crates/datacortex-core/src/format/mod.rs` ↔ `crates/datacortex-core/src/format/transform.rs` (71% co-change)
- `crates/datacortex-core/src/model/engine.rs` ↔ `crates/datacortex-core/src/model/mod.rs` (62% co-change)

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
