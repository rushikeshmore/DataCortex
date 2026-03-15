//! Mixer — logistic transforms and model prediction combining.
//!
//! Phase 2: squash/stretch logistic transforms.
//! Phase 3+: logistic mixer in log-odds space, dual/triple hierarchy, APM.

pub mod logistic;

pub use logistic::{squash, stretch};
