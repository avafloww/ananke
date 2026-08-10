//! Feasibility checks, eviction planning, and the balloon resolver.
//!
//! Re-exported from `ananke-allocator` so `crate::allocator::…` paths
//! inside the daemon are unchanged by the split.

pub use ananke_allocator::{
    AllocationTable, NoFit, balloon, can_fit, can_fit_after_eviction, eviction,
};
pub use ananke_placement as placement;
