//! State primitives — StateTable, StateMap for bit-level state tracking.
//!
//! Phase 2: 256-state bit history machine + adaptive state→probability mapping.

pub mod state_map;
pub mod state_table;

pub use state_map::StateMap;
pub use state_table::StateTable;
