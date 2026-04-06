//! Compression configuration for the Vector plugin.
//!
//! [`DataCortexConfig`] controls which compression mode and format hint
//! are passed to `datacortex-core`. Defaults to Fast mode with NDJSON
//! format, which is optimal for Vector log pipelines.

use datacortex_core::dcx::{FormatHint, Mode};

/// Configuration for `DataCortex` compression in Vector pipelines.
#[derive(Debug, Clone, Copy)]
pub struct DataCortexConfig {
    mode: Mode,
    format_hint: FormatHint,
    turbo: bool,
}

impl DataCortexConfig {
    /// Create a new config with explicit mode and format hint.
    #[must_use]
    pub const fn new(mode: Mode, format_hint: FormatHint) -> Self {
        Self {
            mode,
            format_hint,
            turbo: false,
        }
    }

    /// Create a turbo config (Fast mode, NDJSON, 30-55x faster encode).
    ///
    /// Turbo mode uses zstd-3 with only 2 compression paths instead of 6+.
    /// ~2% ratio tradeoff for ~33x faster encode. Same `.dcx` format.
    #[must_use]
    pub const fn turbo() -> Self {
        Self {
            mode: Mode::Fast,
            format_hint: FormatHint::Ndjson,
            turbo: true,
        }
    }

    /// Enable or disable turbo mode (Fast mode only).
    #[must_use]
    pub const fn with_turbo(mut self, turbo: bool) -> Self {
        self.turbo = turbo;
        self
    }

    /// Compression mode.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// Format hint for the compression pipeline.
    #[must_use]
    pub const fn format_hint(&self) -> FormatHint {
        self.format_hint
    }

    /// Whether turbo mode is enabled.
    #[must_use]
    pub const fn is_turbo(&self) -> bool {
        self.turbo
    }
}

impl Default for DataCortexConfig {
    /// Default: Fast mode + NDJSON format (optimal for log pipelines).
    fn default() -> Self {
        Self {
            mode: Mode::Fast,
            format_hint: FormatHint::Ndjson,
            turbo: false,
        }
    }
}
