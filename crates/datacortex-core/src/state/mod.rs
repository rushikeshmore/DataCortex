//! State primitives — StateTable, StateMap, ContextMap for bit-level state tracking.
//!
//! Phase 2: 256-state bit history machine + adaptive state→probability mapping.
//! Phase 3: ContextMap — lossy hash table for higher-order context models.

pub mod context_map;
pub mod state_map;
pub mod state_table;

pub use context_map::{AssociativeContextMap, ChecksumContextMap, ContextMap};
pub use state_map::StateMap;
pub use state_table::StateTable;
