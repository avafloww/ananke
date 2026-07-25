//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

mod command_gpu;
mod cpu_capacity;
mod entry;
mod experts_ncmoe;
mod experts_nonexpert;
mod finish;
mod layer_walk;
mod mtp;
mod packer;
mod reserve;
mod sharded;
mod types;

#[cfg(test)]
mod test_support;

pub use command_gpu::{check_command_placement_override, pick_command_gpu};
pub use entry::{pack, pack_optimistic};
pub use types::{CommandArgs, PackError, Packed};
