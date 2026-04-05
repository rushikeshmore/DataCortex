# DataCortex Rust Style Guide

Definitive coding conventions for the DataCortex codebase. Derived from the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/), production
patterns from ripgrep/serde/tokio/zstd-rs, and DataCortex-specific requirements.

---

## Lint Policy

Every crate's `lib.rs` (or `main.rs` for CLI) must declare lint levels:

```rust
// Enforce documentation on all public items.
#![warn(missing_docs)]
// Catch common mistakes beyond default clippy.
#![warn(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::checked_conversions,
    clippy::ptr_arg,
)]
// Selective allows for pedantic lints that conflict with our codebase style.
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::missing_errors_doc,     // enable once all pub fns have # Errors
    clippy::missing_panics_doc,     // enable once all pub fns have # Panics
)]
```

`cargo clippy --all-targets -- -D warnings` must pass. Zero warnings.
`cargo fmt` is law. No manual formatting exceptions.

---

## Naming (RFC 430)

| Kind | Convention | Example |
|------|-----------|---------|
| Types, Traits, Enums | `UpperCamelCase` | `ContextMap`, `FormatHint` |
| Functions, Methods, Modules | `snake_case` | `compress`, `detect_format` |
| Constants, Statics | `SCREAMING_SNAKE_CASE` | `DCX_MAGIC`, `MAX_ORDER` |
| Type parameters | Single uppercase | `T`, `E` |
| Lifetimes | Short lowercase | `'a`, `'input` |
| Acronyms | Treat as one word | `Uuid` not `UUID`, `Ndjson` not `NDJSON` |

**Conversions** follow Rust convention:
- `as_*` -- free, borrow-to-borrow (e.g., `as_bytes()`)
- `to_*` -- expensive, produces new value (e.g., `to_string()`)
- `into_*` -- consumes self (e.g., `into_bytes()`)

**Getters** have no `get_` prefix: use `fn mode(&self)` not `fn get_mode()`.

**Predicates** start with `is_` or a question word: `is_empty()`, `has_schema()`.

---

## Documentation

### Public items -- always document

Every `pub` item gets a `///` doc comment. First line is a single sentence in
third-person singular, ending with a period:

```rust
/// Compresses input bytes using the specified mode.
pub fn compress(input: &[u8], mode: Mode) -> Result<Vec<u8>>
```

### Required doc sections

| Section | When |
|---------|------|
| `# Errors` | Function returns `Result` |
| `# Panics` | Function can panic |
| `# Safety` | Function is `unsafe` |
| `# Examples` | Public API entry points |

```rust
/// Compresses input bytes using the specified mode.
///
/// # Errors
///
/// Returns an error if the input is empty or if compression fails
/// due to an invalid format hint.
///
/// # Examples
///
/// ```
/// use datacortex_core::{compress, Mode, FormatHint};
///
/// let data = b"{\"key\": \"value\"}";
/// let compressed = compress(data, Mode::Fast, FormatHint::Auto)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compress(input: &[u8], mode: Mode, hint: FormatHint) -> Result<Vec<u8>>
```

### Module-level docs

Use `//!` at the top of each module file:

```rust
//! Schema inference for JSON/NDJSON data.
//!
//! Analyzes column values to determine types (integer, boolean,
//! timestamp, enum, string, float, UUID) for type-specific encoding.
```

### Private items -- document when non-obvious

Skip docs on trivial private functions (`new()`, simple getters). Add brief
comments on private functions with non-obvious logic or magic numbers.

### Comments style

- Use `//` line comments, never `/* */` block comments.
- Comments explain *why*, not *what*. If code needs a *what* comment, refactor.
- Sentences start with a capital letter and end with punctuation.
- No em dashes. Use `--` if needed.

---

## Error Handling

- **Library code (`datacortex-core`):** Return `Result<T, E>`. No `.unwrap()`.
  Use `?` propagation. Error types implement `std::error::Error + Send + Sync`.
- **CLI code (`datacortex-cli`):** `.unwrap()` is acceptable. Use `anyhow` or
  simple `eprintln!` + `process::exit`.
- **Tests:** `.unwrap()` is fine.
- **Error messages:** Lowercase, no trailing punctuation:
  `"invalid .dcx header"` not `"Invalid .dcx header."`.

---

## Imports

Three groups separated by blank lines, each group alphabetized:

```rust
use std::io::{self, Write};

use brotli::CompressorWriter;
use zstd::stream::encode_all;

use crate::dcx::{FormatHint, Mode};
use crate::format::schema::InferredSchema;
```

- Import types and traits directly: `use std::io::Read;`
- Qualify function calls with module: `mem::replace(...)` not `use std::mem::replace;`
- Never use glob `use *` except `use super::*` in `#[cfg(test)]` modules.

---

## Attributes

### Performance

| Attribute | When to use |
|-----------|-------------|
| `#[inline]` | Small functions called across crate boundaries |
| `#[inline(always)]` | Per-bit/per-byte hot path functions (squash, stretch, predict). Always measure. |
| `#[cold]` | Error constructors and panic helper functions |
| `#[must_use]` | Functions where ignoring the return value is always a bug |

```rust
#[cold]
fn compression_failed(reason: &str) -> DataCortexError {
    DataCortexError::CompressionFailed(reason.to_string())
}

#[must_use]
pub fn compress(input: &[u8], mode: Mode) -> Result<Vec<u8>>
```

### Safety

```rust
// SAFETY: We verified that `idx < self.table.len()` on the line above.
unsafe { *self.table.get_unchecked(idx) }
```

Every `unsafe` block requires a `// SAFETY:` comment explaining why the
invariant holds. Every `unsafe` block gets a corresponding unit test.

---

## Module Organization

- One primary concept per file. `state_table.rs` contains `StateTable`.
- `mod.rs` (or parent module) contains only `mod` declarations and re-exports.
  No implementation logic.
- Modules over 100 lines go in separate files.
- Re-export key types at the crate root for ergonomic public API.
- Keep internal implementation details in private submodules.

### File layout order

```rust
//! Module-level documentation.

// Imports (three groups)

// Constants

// Type definitions (structs, enums)

// Trait implementations

// Inherent implementations

// Free functions

// #[cfg(test)] mod tests { ... }
```

---

## Testing

- Unit tests live in `#[cfg(test)] mod tests` at the bottom of each module.
- Integration tests live in `tests/` at the crate root.
- Test names describe the behavior: `compress_roundtrip_preserves_exact_bytes`.
- Use `assert_eq!` with descriptive messages for non-obvious assertions.
- `use super::*` is acceptable inside test modules.

---

## Performance-Critical Code (Hot Paths)

DataCortex has two distinct code temperatures:

**Hot path (per-bit, per-byte -- CM engine, mixer, arithmetic coder):**
- Static dispatch only. Zero trait objects.
- Fixed-size arrays, not `Vec`, for known-size data.
- `i32` for probability math (12-bit range). No `f32`/`f64`.
- Lookup tables over computation.
- No allocation in loops. No `HashMap` (use flat arrays indexed by hash).
- No format-type branching in per-bit code.

**Cold path (per-file, per-block -- format detection, schema inference):**
- `Vec`, `HashMap`, `String`, dynamic dispatch all fine.
- Readability over micro-optimization.

---

## Type Design

- Implement `Debug` on all public types (derive when possible).
- Implement `Clone` unless the type manages a unique resource.
- Use `#[non_exhaustive]` on public enums that may gain variants.
- Keep struct fields private. Expose via methods.
- Put trait bounds on `impl` blocks, not struct definitions.
- Implement `Default` alongside `new()` when a sensible default exists.

---

## Build Configuration

```toml
# Cargo.toml [profile.release]
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

For benchmarks, add `RUSTFLAGS="-C target-cpu=native"`.

---

## Commit Style

```
type: short description (imperative mood)

[optional body]

Benchmarks:
- file: X.Xx (was Y.Yx)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `release`.
Always include benchmark numbers in commits that change compression output.
