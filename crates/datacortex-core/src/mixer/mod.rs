//! Mixer — logistic transforms, dual logistic mixer, hierarchical mixer, and APM cascade.
//!
//! Phase 2: squash/stretch logistic transforms.
//! Phase 3: dual logistic mixer (fine + coarse) + two-stage APM.
//! Sprint 2: hierarchical mixer (experimental, not used by default).

pub mod apm;
pub mod dual_mixer;
pub mod hierarchical_mixer;
pub mod logistic;

pub use apm::APMStage;
pub use dual_mixer::{DualMixer, NUM_MODELS, byte_class};
pub use logistic::{squash, stretch};
