//! Context models — predict next bit probability from context.
//!
//! Phase 2: Order-0 model (256-context partial byte predictor).
//! Phase 3+: Order-1 through Order-5, match model, word model, sparse models.

pub mod order0;

pub use order0::Order0Model;
