//! Per-device placement previews that report a [`FitVerdict`] instead of
//! rendering argv: whether a service fits now, would need eviction, or
//! cannot fit at all, for each of the estimator, override, and
//! command-template placement paths.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use ananke_api::internal::fit_verdict::{DeviceShortfall, FitVerdict};

use crate::{
    allocator::{
        AllocationTable,
        placement::{self, PackError},
    },
    config::{AllocationMode, DeviceSlot, PlacementPolicy, ServiceConfig},
    devices::{DeviceId, DeviceSnapshot},
    estimator::Estimate,
    supervise::preview::PlacementOutcome,
    tracking::rolling::Corrections,
};

/// Compute where a llama service's memory would land per device and whether it
/// fits without eviction, by running the packer against the live snapshot and
/// pledge book. `corrections` are the service's learned per-pool estimator
/// corrections, applied by the packer to every byte it charges. This
/// is the estimator path only — the caller must not pass a service with a
/// manual `placement_override` (its placement is the override, not a pack).
///
/// `running` short-circuits the verdict to [`FitVerdict::Fits`]: a live
/// service is by definition placed, and the strict nvml-free check would
/// otherwise be confounded by the service's own resident VRAM (which lowers
/// reported free without being a true obstacle to its own placement).
pub fn preview_placement(
    svc: &ServiceConfig,
    est: &Estimate,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
    corrections: Corrections,
) -> PlacementOutcome {
    // Strict honours currently-free memory (what the daemon checks before
    // deciding to evict); optimistic trusts the pledge book; on-empty models
    // the bare hardware capacity (could it ever fit on the allowed devices).
    let strict = placement::pack_corrected(est, svc, snapshot, table, corrections, false).ok();
    let optimistic = placement::pack_corrected(est, svc, snapshot, table, corrections, true).ok();
    // The on-empty pack is the last word on "can this ever be placed", so its
    // error is the one that explains a `DoesNotFit` to the operator.
    let on_empty = placement::pack_corrected(
        est,
        svc,
        snapshot,
        &AllocationTable::new(),
        corrections,
        true,
    );

    let verdict = if running || strict.is_some() {
        FitVerdict::Fits
    } else {
        match &on_empty {
            Ok(_) => FitVerdict::NeedsEviction,
            Err(e) => does_not_fit(e),
        }
    };
    let on_empty = on_empty.ok();

    // Prefer the strict allocation (what it would actually get now), then the
    // pledge-book shape, then the bare-hardware shape — so a service that
    // needs eviction still shows where it would land once room is freed.
    let (devices, expert_offload_bytes, expert_offload_layers) =
        match strict.or(optimistic).or(on_empty) {
            Some(p) => (
                p.allocation.bytes,
                p.expert_offload_bytes,
                p.expert_offload_layers,
            ),
            None => (BTreeMap::new(), 0, 0),
        };

    PlacementOutcome {
        devices,
        verdict,
        expert_offload_bytes,
        expert_offload_layers,
    }
}

/// Placement for a service that declares a manual `placement_override`. The
/// per-device split is the override itself (the daemon honours it verbatim
/// rather than packing); the verdict checks each pledged GPU slot against the
/// live snapshot the same way [`preview_placement`] does — strict (currently
/// free) for `Fits`, bare-hardware capacity for `NeedsEviction`. Works for
/// both llama-cpp and command (e.g. multi-GPU vLLM) override services.
pub fn preview_override_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
) -> PlacementOutcome {
    let devices = svc
        .placement_override
        .iter()
        .map(|(slot, mib)| {
            let id = match slot {
                DeviceSlot::Cpu => DeviceId::Cpu,
                DeviceSlot::Gpu(n) => DeviceId::Gpu(*n),
            };
            (id, mib.saturating_mul(1024 * 1024))
        })
        .collect();

    let fits_now = placement::check_command_placement_override(svc, snapshot, table, false).is_ok();
    let on_empty =
        placement::check_command_placement_override(svc, snapshot, &AllocationTable::new(), true);
    let verdict = if running || fits_now {
        FitVerdict::Fits
    } else {
        match &on_empty {
            Ok(()) => FitVerdict::NeedsEviction,
            Err(e) => does_not_fit(e),
        }
    };

    PlacementOutcome {
        devices,
        verdict,
        expert_offload_bytes: 0,
        expert_offload_layers: 0,
    }
}

/// Placement for a command-template service that picks a GPU dynamically (no
/// `placement_override`): it reserves `min_mb` on the GPU with the most
/// headroom, or pins to the CPU for a cpu-only service. Mirrors
/// [`crate::supervise::RunLoop::compute_command_reservation`]. Returns `None` when the
/// service reserves nothing (`AllocationMode::None`), so the caller renders no
/// placement at all.
pub fn preview_command_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
) -> Option<PlacementOutcome> {
    let (min_mb, prefer_mb) = match svc.allocation_mode {
        AllocationMode::Static { reserve_mb } => (reserve_mb, Some(reserve_mb)),
        AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
        AllocationMode::None => return None,
    };
    if min_mb == 0 {
        return None;
    }
    let bytes = min_mb.saturating_mul(1024 * 1024);

    if matches!(svc.placement_policy, PlacementPolicy::CpuOnly) {
        let mut devices = BTreeMap::new();
        devices.insert(DeviceId::Cpu, bytes);
        return Some(PlacementOutcome {
            devices,
            verdict: FitVerdict::Fits,
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
        });
    }

    let strict = placement::pick_command_gpu(svc, snapshot, table, min_mb, prefer_mb, false);
    let on_empty = placement::pick_command_gpu(
        svc,
        snapshot,
        &AllocationTable::new(),
        min_mb,
        prefer_mb,
        true,
    );
    let verdict = if running || strict.is_some() {
        FitVerdict::Fits
    } else if on_empty.is_some() {
        FitVerdict::NeedsEviction
    } else {
        // No allowed GPU could host `min_mb` even on bare hardware, so report
        // each one's headroom against the ask.
        does_not_fit(&PackError::WeightsDoNotFit {
            shortfalls: placement::command_gpu_shortfalls(
                svc,
                snapshot,
                &AllocationTable::new(),
                min_mb,
                true,
            ),
        })
    };

    // Show the strict pick (where it would land now), else the bare-hardware
    // pick so a needs-eviction service still shows its target GPU.
    let mut devices = BTreeMap::new();
    if let Some(gpu) = strict.or(on_empty) {
        devices.insert(DeviceId::Gpu(gpu), bytes);
    }
    Some(PlacementOutcome {
        devices,
        verdict,
        expert_offload_bytes: 0,
        expert_offload_layers: 0,
    })
}

/// Render a packer failure as the wire verdict, carrying its per-device
/// shortfalls so the caller can say *which* device came up short and by how
/// much. The binding constraint is frequently host RAM — an expert-offloaded
/// MoE spills most of its weight to the CPU — so a bare "does not fit" would
/// point the operator at the GPUs by implication.
fn does_not_fit(err: &PackError) -> FitVerdict {
    FitVerdict::DoesNotFit {
        shortfalls: err
            .shortfalls()
            .into_iter()
            .map(|s| DeviceShortfall {
                device: s.device.as_display(),
                requested_bytes: s.requested_bytes,
                available_bytes: s.available_bytes,
            })
            .collect(),
    }
}
