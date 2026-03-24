# Module: datacortex-neural

## Purpose
3 files, 773 lines (rust). Auto-generated from code structure. Updated on each commit via git hooks.

## Data Flow
implementation: 3 files (lib.rs, llm.rs, meta_mixer.rs)

## Public API
- `LlmError (enum, crates/datacortex-neural/src/llm.rs:29-35)`
- `LlmPredictor (class, crates/datacortex-neural/src/llm.rs:64-91)`
- `new (function, crates/datacortex-neural/src/llm.rs:95-97)`
- `with_ctx_size (function, crates/datacortex-neural/src/llm.rs:100-161)`
- `predict_byte_probs (function, crates/datacortex-neural/src/llm.rs:167-200)`
- `predict_bit (function, crates/datacortex-neural/src/llm.rs:319-372)`
- `mapped_bytes (function, crates/datacortex-neural/src/llm.rs:375-377)`
- `vocab_size (function, crates/datacortex-neural/src/llm.rs:380-382)`
- `MetaMixer (class, crates/datacortex-neural/src/meta_mixer.rs:17-30)`
- `new (function, crates/datacortex-neural/src/meta_mixer.rs:35-58)`
- `blend (function, crates/datacortex-neural/src/meta_mixer.rs:67-84)`
- `update (function, crates/datacortex-neural/src/meta_mixer.rs:88-93)`
- `last_prediction (function, crates/datacortex-neural/src/meta_mixer.rs:96-98)`

## Dependencies

## Temporal Signals
- **Churn:** 2 changes (stable)
- **Coupled with:** none
- **Stability:** stable
- **Last changed:** 2026-03-16T00:04:59+05:30
