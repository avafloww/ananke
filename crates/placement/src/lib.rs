//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

use ananke_config::placement::DeviceSlot;

pub mod devices;
pub mod registry;

pub use command_gpu::{check_command_placement_override, command_gpu_shortfalls, pick_command_gpu};
pub use entry::{PackMode, pack, pack_corrected, pack_demand, pack_optimistic};
pub use registry::{DrainReason, KillHandle, ServiceRegistry, slot_to_key};
pub use types::{CommandArgs, DeviceShortfall, PackError, Packed, RollingInputs};

/// The reservation table: what each service has pledged on each device.
pub type AllocationTable =
    std::collections::BTreeMap<smol_str::SmolStr, std::collections::BTreeMap<DeviceSlot, u64>>;

/// The gated correction factors a placement runs with, one per pool.
///
/// The packer scales every byte it charges to a device by that device's factor, so
/// the reservation it produces is what we predict the service will *actually* use
/// — and the fit decisions (which layers land on which card, how many experts
/// spill to the host) are made against the corrected numbers rather than the raw
/// estimate.
#[derive(Debug, Clone, Copy)]
pub struct Corrections {
    pub vram: f64,
    pub host: f64,
}

impl Corrections {
    /// No correction in either pool: what an untrained service, a preview of bare
    /// hardware, or a test runs with.
    pub const NEUTRAL: Self = Self {
        vram: 1.0,
        host: 1.0,
    };

    /// The factor for a destination slot.
    pub fn for_slot(&self, slot: &DeviceSlot) -> f64 {
        match slot {
            DeviceSlot::Cpu => self.host,
            DeviceSlot::Gpu(_) => self.vram,
        }
    }

    /// Scale `bytes` by the factor for `slot`.
    ///
    /// Rounds *up*: truncating a scaled byte count is an under-reservation,
    /// which is the failure the correction exists to prevent. Short-circuits an
    /// exactly-neutral factor so an untrained service's reservation is bit-for-bit
    /// the estimate.
    pub fn scale(&self, slot: &DeviceSlot, bytes: u64) -> u64 {
        let factor = self.for_slot(slot);
        if factor == 1.0 {
            return bytes;
        }
        (bytes as f64 * factor).ceil() as u64
    }
}

impl Default for Corrections {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

mod charge;
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
