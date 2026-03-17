//! Mixer — logistic transforms, dual logistic mixer, hierarchical mixer, APM cascade, and ISSE chain.
//!
//! Phase 2: squash/stretch logistic transforms.
//! Phase 3: dual logistic mixer (fine + coarse) + two-stage APM.
//! Sprint 2: hierarchical mixer (experimental, not used by default).
//! ISSE: Indirect Secondary Symbol Estimation chain (ZPAQ-style).

pub mod apm;
pub mod dual_mixer;
pub mod hierarchical_mixer;
pub mod isse;
pub mod logistic;
pub mod meta_mixer;
pub mod multi_set_mixer;

pub use apm::APMStage;
pub use dual_mixer::{DualMixer, NUM_MODELS, byte_class};
pub use isse::IsseChain;
pub use logistic::{squash, stretch};
pub use meta_mixer::MetaMixer;
pub use multi_set_mixer::MultiSetMixer;
