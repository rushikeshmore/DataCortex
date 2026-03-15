# DataCortex — Claude Code Instructions

## What This Is
Lossless text compression engine in Rust. Three modes: Max (sub-1.0 bpb), Balanced (1.2-1.4 bpb), Fast (1.8-2.0 bpb). Format-aware preprocessing for JSON/MD/NDJSON/CSV/code/logs + bit-level context mixing + Ramanujan periodic features + entropy coding.

## Architecture
```
Input → Format Detection → Preprocessing → CM Engine (or Fast zstd) → .dcx output
```
- `crates/datacortex-core/` — Core library
  - `format/` — Detection (7 types) + JSON key interning + transform pipeline
  - `model/` — Order0-4, match model, word model, CMEngine orchestrator
  - `state/` — StateTable (256-state), StateMap (adaptive), ContextMap (lossy hash)
  - `mixer/` — Triple logistic mixer (fine 64K + med 16K + coarse 4K), squash/stretch, 3-stage APM
  - `entropy/` — Binary arithmetic coder (12-bit, carry-free)
  - `codec.rs` — Pipeline orchestrator (Fast=zstd, Balanced=CM, Max=CM)
  - `dcx.rs` — .dcx v3 file format (32-byte header + CRC-32)
- `crates/datacortex-cli/` — CLI binary (compress, decompress, bench, info)
- `crates/datacortex-neural/` — Optional RWKV model for Max mode (stub)
- `corpus/` — 7 Tier 1 test files (checked in, never change)
- `benchmarks/` — baseline.json

## Current Status
**Phase 3 complete.** 2.34 bpb alice29. 123 tests. Next: Phase 4 (format context + RPT).

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

## Key V2 Gotchas (Don't Repeat)
- Match model: rolling hash must be non-cumulative (V2 bug cost 0.44 bpb)
- Match model: linear confidence ramp, not step function
- Match model: length tracking must reset on mismatch
- Logistic mixing at byte-level = +69% regression. Bit-level ONLY.
- Adding weak models dilutes mixer weights. Solo test first.
- η=2 for fine mixer (64K weights), η=4 for coarse (4K). Don't increase without A/B test.
- 2-stage APM for Balanced, 3-stage only for Max mode on large files.

## V3 Engine (Phase 3 — implemented)
- 7 models: Order-0 (256 direct), Order-1 (8MB), Order-2 (2MB), Order-3 (4MB), Order-4 (4MB), Match (8MB ring + 2M hash), Word (2MB)
- Triple logistic mixer: fine (64K, η=2), medium (16K, η=3), coarse (4K, η=4)
- 3-stage APM cascade: 2K/16K/4K contexts, 50/25/25% blend
- StateTable (256-state), StateMap (adaptive 1/n), ContextMap (lossy hash)
- Binary AC: 12-bit precision, carry-free
- JSON key interning: Balanced/Max only (hurts Fast due to zstd redundancy)

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
