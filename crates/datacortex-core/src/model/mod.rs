//! Context models — predict next bit probability from context.
//!
//! Phase 2: Order-0 model (256-context partial byte predictor).
//! Phase 3: Order-1/2/3 context models, match model, CM engine.

pub mod cm_model;
pub mod engine;
pub mod json_model;
pub mod match_model;
pub mod order0;
pub mod run_model;
pub mod sparse_model;
pub mod word_model;

pub use cm_model::{AssociativeContextModel, ChecksumContextModel, ContextModel};
pub use engine::{CMConfig, CMEngine};
pub use match_model::MatchModel;
pub use order0::Order0Model;
pub use run_model::RunModel;
pub use sparse_model::SparseModel;
pub use word_model::WordModel;
