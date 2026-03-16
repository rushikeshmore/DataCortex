//! Context models — predict next bit probability from context.
//!
//! Phase 2: Order-0 model (256-context partial byte predictor).
//! Phase 3: Order-1/2/3 context models, match model, CM engine.

pub mod cm_model;
pub mod dmc_model;
pub mod engine;
pub mod indirect_model;
pub mod json_model;
pub mod match_model;
pub mod neural_model;
pub mod order0;
pub mod ppm_model;
pub mod run_model;
pub mod sparse_model;
pub mod word_model;
pub mod xml_model;

pub use cm_model::{AssociativeContextModel, ChecksumContextModel, ContextModel};
pub use dmc_model::DmcModel;
pub use engine::{CMConfig, CMEngine};
pub use match_model::MatchModel;
pub use neural_model::NeuralModel;
pub use order0::Order0Model;
pub use ppm_model::{PpmConfig, PpmModel};
pub use run_model::RunModel;
pub use sparse_model::SparseModel;
pub use word_model::WordModel;
