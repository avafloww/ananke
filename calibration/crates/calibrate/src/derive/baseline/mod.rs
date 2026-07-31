//! What the process holds beyond its buffers.
//!
//! Split by what the term is *for*: `offset` holds the corrections charged into
//! `host_overhead_bytes`, and `headroom` holds the two allowances that are reserved
//! beside it but deliberately not charged into it.

pub mod headroom;
pub mod offset;

pub use headroom::{checkpoint_headroom, per_slot_bytes};
pub use offset::{baseline_offset, per_device_bytes, tensor_split_baseline};
