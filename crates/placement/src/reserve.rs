//! Turning the live pledge book and configured reserves into "bytes still
//! available on this device", shared by the packer and the command-template
//! GPU picker.

use ananke_config::placement::{DeviceSlot, PlacementInputs, PlacementPolicy};
use smol_str::SmolStr;

use crate::{AllocationTable, devices::DeviceSnapshot};

/// The GPUs this service may be placed on: every GPU in the snapshot, narrowed
/// by `gpu_allow` when set, and empty for a `CpuOnly` service.
pub(super) fn allowed_gpu_list(placement: &PlacementInputs, snapshot: &DeviceSnapshot) -> Vec<u32> {
    if placement.policy == PlacementPolicy::CpuOnly {
        return Vec::new();
    }
    let all: Vec<u32> = snapshot.gpus.iter().map(|g| g.id).collect();
    if placement.gpu_allow.is_empty() {
        all
    } else {
        placement
            .gpu_allow
            .iter()
            .copied()
            .filter(|id| all.contains(id))
            .collect()
    }
}

pub(crate) fn sum_reserved(table: &AllocationTable, slot: &DeviceSlot, exclude: &SmolStr) -> u64 {
    table
        .iter()
        .filter(|(k, _)| k.as_str() != exclude.as_str())
        .filter_map(|(_, alloc)| alloc.get(slot))
        .sum::<u64>()
        * 1024
        * 1024
}

/// VRAM (bytes) this service keeps free on `gpu`: the global `[devices]`
/// reserve for that GPU (per-GPU override, else default) plus the service's
/// own `gpu_headroom_mb`.
pub(crate) fn gpu_reserve_bytes(placement: &PlacementInputs, gpu: u32) -> u64 {
    let r = &placement.reserves;
    let mb = r
        .per_gpu_mb
        .get(&gpu)
        .copied()
        .unwrap_or(r.default_gpu_mb)
        .saturating_add(placement.gpu_headroom_mb);
    mb.saturating_mul(1024 * 1024)
}
