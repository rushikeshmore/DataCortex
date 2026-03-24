# Module: datacortex-core

## Purpose
52 files, 17949 lines (rust). Auto-generated from code structure. Updated on each commit via git hooks.

## Data Flow
implementation: 52 files (dump_transform.rs, test_pipeline.rs, test_vdict.rs, ...)

## Public API
- `preprocess (function, crates/datacortex-core/src/format/value_dict.rs:29-112)`
- `reverse (function, crates/datacortex-core/src/format/value_dict.rs:115-201)`
- `compress (function, crates/datacortex-core/src/codec.rs:411-418)`
- `compress_with_model (function, crates/datacortex-core/src/codec.rs:421-525)`
- `decompress (function, crates/datacortex-core/src/codec.rs:528-530)`
- `decompress_with_model (function, crates/datacortex-core/src/codec.rs:533-659)`
- `compress_to_vec (function, crates/datacortex-core/src/codec.rs:662-670)`
- `compress_to_vec_with_model (function, crates/datacortex-core/src/codec.rs:673-682)`
- `decompress_from_slice (function, crates/datacortex-core/src/codec.rs:685-688)`
- `read_header (function, crates/datacortex-core/src/codec.rs:691-693)`
- `raw_zstd_compress (function, crates/datacortex-core/src/codec.rs:696-698)`
- `Mode (enum, crates/datacortex-core/src/dcx.rs:25-29)`
- `from_u8 (function, crates/datacortex-core/src/dcx.rs:32-42)`
- `name (function, crates/datacortex-core/src/dcx.rs:44-50)`
- `FormatHint (enum, crates/datacortex-core/src/dcx.rs:62-74)`
- `from_u8 (function, crates/datacortex-core/src/dcx.rs:77-95)`
- `name (function, crates/datacortex-core/src/dcx.rs:97-111)`
- `DcxHeader (class, crates/datacortex-core/src/dcx.rs:122-129)`
- `write_to (function, crates/datacortex-core/src/dcx.rs:133-152)`
- `read_from (function, crates/datacortex-core/src/dcx.rs:155-196)`
- `total_size (function, crates/datacortex-core/src/dcx.rs:199-201)`
- `ArithmeticEncoder (class, crates/datacortex-core/src/entropy/arithmetic.rs:17-21)`
- `new (function, crates/datacortex-core/src/entropy/arithmetic.rs:25-31)`
- `encode (function, crates/datacortex-core/src/entropy/arithmetic.rs:35-59)`
- `finish (function, crates/datacortex-core/src/entropy/arithmetic.rs:63-70)`
- `ArithmeticDecoder (class, crates/datacortex-core/src/entropy/arithmetic.rs:82-88)`
- `new (function, crates/datacortex-core/src/entropy/arithmetic.rs:92-105)`
- `decode (function, crates/datacortex-core/src/entropy/arithmetic.rs:109-136)`
- `detect_csv (function, crates/datacortex-core/src/format/csv.rs:165-167)`
- `preprocess (function, crates/datacortex-core/src/format/csv.rs:172-303)`
- `reverse (function, crates/datacortex-core/src/format/csv.rs:306-394)`
- `preprocess (function, crates/datacortex-core/src/format/json.rs:94-149)`
- `reverse (function, crates/datacortex-core/src/format/json.rs:152-201)`
- `preprocess (function, crates/datacortex-core/src/format/json_array.rs:381-494)`
- `reverse (function, crates/datacortex-core/src/format/json_array.rs:497-642)`
- `preprocess (function, crates/datacortex-core/src/format/log_transform.rs:237-343)`
- `reverse (function, crates/datacortex-core/src/format/log_transform.rs:346-419)`
- `detect_logfmt (function, crates/datacortex-core/src/format/logfmt.rs:123-142)`
- `preprocess (function, crates/datacortex-core/src/format/logfmt.rs:147-297)`
- `reverse (function, crates/datacortex-core/src/format/logfmt.rs:300-398)`
- `...and 190 more`

## Dependencies

## Temporal Signals
- **Churn:** 12 changes (stabilizing)
- **Coupled with:** crates/datacortex-core/src/model/engine.rs (10 co-changes, 83%), crates/datacortex-core/src/format/transform.rs (8 co-changes, 80%), crates/datacortex-core/src/model/mod.rs (7 co-changes, 58%), crates/datacortex-core/src/mixer/mod.rs (5 co-changes, 50%), crates/datacortex-core/src/model/mod.rs (5 co-changes, 50%), crates/datacortex-core/src/model/mod.rs (5 co-changes, 50%), crates/datacortex-core/src/model/word_model.rs (5 co-changes, 42%), crates/datacortex-core/src/lib.rs (5 co-changes, 50%), crates/datacortex-core/src/model/mod.rs (4 co-changes, 40%), crates/datacortex-core/src/mixer/mod.rs (4 co-changes, 40%), crates/datacortex-core/src/model/cm_model.rs (4 co-changes, 40%), crates/datacortex-core/src/model/engine.rs (4 co-changes, 33%), crates/datacortex-core/src/model/engine.rs (4 co-changes, 33%), crates/datacortex-core/src/model/word_model.rs (4 co-changes, 40%), crates/datacortex-cli/src/main.rs (4 co-changes, 40%), crates/datacortex-cli/src/main.rs (4 co-changes, 80%), crates/datacortex-core/src/model/match_model.rs (4 co-changes, 33%), crates/datacortex-core/src/model/word_model.rs (4 co-changes, 80%), crates/datacortex-core/src/state/mod.rs (4 co-changes, 40%), crates/datacortex-core/src/format/mod.rs (3 co-changes, 30%), crates/datacortex-core/src/model/engine.rs (3 co-changes, 25%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 30%), Cargo.toml (3 co-changes, 30%), Cargo.toml (3 co-changes, 60%), crates/datacortex-core/Cargo.toml (3 co-changes, 30%), crates/datacortex-core/Cargo.toml (3 co-changes, 60%), crates/datacortex-core/src/model/mod.rs (3 co-changes, 30%), crates/datacortex-core/src/model/json_model.rs (3 co-changes, 25%), crates/datacortex-core/src/model/run_model.rs (3 co-changes, 25%), crates/datacortex-core/src/model/sparse_model.rs (3 co-changes, 25%), crates/datacortex-core/src/model/match_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/run_model.rs (3 co-changes, 100%), crates/datacortex-core/src/model/sparse_model.rs (3 co-changes, 100%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 60%), crates/datacortex-core/src/model/mod.rs (3 co-changes, 30%), crates/datacortex-core/src/model/run_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/sparse_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 30%), crates/datacortex-core/src/model/sparse_model.rs (3 co-changes, 100%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 60%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 60%), crates/datacortex-core/src/mixer/dual_mixer.rs (3 co-changes, 30%), crates/datacortex-core/src/model/cm_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/engine.rs (3 co-changes, 25%), crates/datacortex-core/src/model/match_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 60%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 75%), crates/datacortex-core/src/model/match_model.rs (3 co-changes, 30%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 30%), crates/datacortex-core/src/model/match_model.rs (3 co-changes, 75%), crates/datacortex-core/src/model/word_model.rs (3 co-changes, 60%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 75%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 25%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 75%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 60%), crates/datacortex-core/src/model/mod.rs (3 co-changes, 30%), crates/datacortex-core/src/state/mod.rs (3 co-changes, 75%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 75%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 30%), crates/datacortex-core/src/state/state_map.rs (3 co-changes, 75%), crates/datacortex-core/src/state/mod.rs (3 co-changes, 30%), crates/datacortex-core/src/state/mod.rs (3 co-changes, 43%)
- **Stability:** stabilizing
- **Last changed:** 2026-03-17T02:09:05+05:30
