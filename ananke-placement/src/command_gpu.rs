//! GPU picking and capacity checks for command-template services: services
//! that don't go through the layer packer but still need a VRAM-aware GPU
//! pick or a placement-override fit check.

use ananke_config::placement::{DeviceSlot, PlacementInputs};

use crate::{
    AllocationTable,
    devices::{DeviceId, DeviceSnapshot},
    reserve::{allowed_gpu_list, gpu_reserve_bytes, sum_reserved},
    types::{DeviceShortfall, PackError},
};

/// VRAM-aware GPU pick for a command-template service.
///
/// Walks `svc`'s allowed GPU list (filtered by `gpu_allow` if set) and
/// returns the GPU with the most available capacity that still satisfies
/// `min_mb`. When `prefer_mb` is `Some`, GPUs that meet that headroom
/// target take priority; only if no candidate hits `prefer_mb` do we fall
/// back to "best of those that meet `min_mb`". This lets dynamic services
/// (`min_mb` floor, `max_mb` growth ceiling) favour a GPU with room to grow.
///
/// `optimistic_remaining` mirrors [`crate::pack_optimistic`]:
/// when `false` the availability view is `min(nvml_free, total - pledged)`;
/// when `true` we trust the pledge book exclusively (`total - pledged`).
/// Eviction-retry passes `true` because nvml hasn't yet caught up to the
/// in-flight drains.
///
/// Returns `None` when no allowed GPU can host `min_mb` — the caller should
/// surface this as a [`PackError::WeightsDoNotFit`] so the supervisor can
/// run its eviction-retry loop.
pub fn pick_command_gpu(
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    min_mb: u64,
    prefer_mb: Option<u64>,
    optimistic_remaining: bool,
) -> Option<u32> {
    let allowed = allowed_gpu_list(placement, snapshot);
    if allowed.is_empty() {
        return None;
    }
    let need_min_bytes = min_mb.saturating_mul(1024 * 1024);
    let prefer_bytes = prefer_mb.map(|m| m.saturating_mul(1024 * 1024));

    let mut candidates: Vec<(u32, u64)> = allowed
        .into_iter()
        .map(|gpu| {
            (
                gpu,
                command_gpu_available(placement, snapshot, reserved, gpu, optimistic_remaining),
            )
        })
        .filter(|(_, available)| *available >= need_min_bytes)
        .collect();

    if candidates.is_empty() {
        return None;
    }

    if let Some(target) = prefer_bytes
        && candidates.iter().any(|(_, a)| *a >= target)
    {
        candidates.retain(|(_, a)| *a >= target);
    }
    // Sort: most-available first, ties broken by ascending GPU id for determinism.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(candidates[0].0)
}

/// Per-allowed-GPU breakdown of why [`pick_command_gpu`] found no home for
/// `min_mb`, in ascending GPU-id order. Callers pair this with
/// [`PackError::WeightsDoNotFit`] so the failure names the cards it was
/// measured against rather than reporting a bare "does not fit".
pub fn command_gpu_shortfalls(
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    min_mb: u64,
    optimistic_remaining: bool,
) -> Vec<DeviceShortfall> {
    let requested_bytes = min_mb.saturating_mul(1024 * 1024);
    let mut allowed = allowed_gpu_list(placement, snapshot);
    allowed.sort_unstable();
    allowed
        .into_iter()
        .map(|gpu| DeviceShortfall {
            device: DeviceId::Gpu(gpu),
            requested_bytes,
            available_bytes: command_gpu_available(
                placement,
                snapshot,
                reserved,
                gpu,
                optimistic_remaining,
            ),
        })
        .collect()
}

/// Capacity check for a command-template service that pinned the
/// reservation across multiple devices via `placement_override`.
///
/// Each `(slot, mib)` entry in the override is checked against the
/// device's available bytes using the same `min(nvml_free, total
/// pledged)` view as [`pick_command_gpu`]. CPU entries are skipped
/// (we don't model CPU capacity). Returns `Ok` when every entry fits;
/// otherwise the first slot that overflows is reported as
/// [`PackError::WeightsDoNotFit`] so the supervisor's eviction-retry
/// loop can engage.
pub fn check_command_placement_override(
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    optimistic_remaining: bool,
) -> Result<(), PackError> {
    for (slot, mib) in &placement.placement_override {
        let DeviceSlot::Gpu(gid) = slot else { continue };
        let need_bytes = mib.saturating_mul(1024 * 1024);
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let total = snapshot.total_bytes(slot).unwrap_or(free);
        let pledged = sum_reserved(reserved, slot, &placement.name);
        let via_pledge = total.saturating_sub(pledged);
        let available = if optimistic_remaining {
            via_pledge
        } else {
            free.min(via_pledge)
        };
        let available = available.saturating_sub(gpu_reserve_bytes(placement, *gid));
        if available < need_bytes {
            return Err(PackError::WeightsDoNotFit {
                shortfalls: vec![DeviceShortfall {
                    device: DeviceId::Gpu(*gid),
                    requested_bytes: need_bytes,
                    available_bytes: available,
                }],
            });
        }
    }
    Ok(())
}

/// Bytes `gpu` can offer this service: `min(nvml_free, total - pledged)`
/// normally, or `total - pledged` when `optimistic_remaining` trusts the
/// pledge book alone, less the configured per-GPU reserve. Mirrors
/// [`crate::packer::Packer::gpu_available`] for the
/// command-template path, which never builds a `Packer`.
fn command_gpu_available(
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    gpu: u32,
    optimistic_remaining: bool,
) -> u64 {
    let slot = DeviceSlot::Gpu(gpu);
    let free = snapshot.free_bytes(&slot).unwrap_or(0);
    let total = snapshot.total_bytes(&slot).unwrap_or(free);
    let pledged = sum_reserved(reserved, &slot, &placement.name);
    let via_pledge = total.saturating_sub(pledged);
    let available = if optimistic_remaining {
        via_pledge
    } else {
        free.min(via_pledge)
    };
    available.saturating_sub(gpu_reserve_bytes(placement, gpu))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ananke_config::placement::PlacementPolicy;
    use smol_str::SmolStr;

    use super::*;
    use crate::test_support::{MIB, snapshot, svc};

    /// The override check must fail with a shortfall naming the GPU that
    /// overflowed, the bytes asked of it, and what it could actually offer —
    /// the operator's cross-reference into `GET /api/devices`.
    fn assert_shortfall_on(result: &Result<(), PackError>, gpu: u32, requested: u64) {
        let Err(err) = result else {
            panic!("expected a placement failure, got {result:?}");
        };
        let shortfalls = err.shortfalls();
        assert_eq!(
            shortfalls.len(),
            1,
            "the first overflowing slot is reported, got {shortfalls:?}"
        );
        let s = shortfalls[0];
        assert_eq!(s.device, DeviceId::Gpu(gpu));
        assert_eq!(s.requested_bytes, requested);
        assert!(
            s.available_bytes < requested,
            "a shortfall must report less available than requested, got {s:?}"
        );
    }

    /// `pick_command_gpu` should return the GPU with the most available
    /// capacity when several satisfy `min_mb`. Ties broken by ascending id.
    #[test]
    fn pick_command_gpu_prefers_most_free() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // GPU 0 has 6 GB free, GPU 1 has 18 GB, GPU 2 has 12 GB.
        let snap = snapshot(&[6, 18, 12]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, Some(1), "GPU 1 has the most free; should be picked");
    }

    /// When a `prefer_mb` headroom target is set (dynamic services'
    /// `max_mb`), `pick_command_gpu` should reject GPUs whose available
    /// capacity falls below that target if any other GPU does meet it.
    #[test]
    fn pick_command_gpu_honours_prefer_headroom() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // GPU 0 = 4 GB free (above min, below prefer), GPU 1 = 10 GB.
        let snap = snapshot(&[4, 10]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(
            &s,
            &snap,
            &table,
            2 * 1024,       // min_mb: 2 GB
            Some(8 * 1024), // prefer_mb: 8 GB
            false,
        );
        assert_eq!(pick, Some(1), "GPU 1 meets prefer_mb headroom");
    }

    /// When no GPU meets `prefer_mb`, fall back to "best of those that meet
    /// `min_mb`" rather than returning None — the pick is still better than
    /// no pick at all, and the dynamic balloon resolver will fast-kill the
    /// service if it actually overshoots.
    #[test]
    fn pick_command_gpu_falls_back_when_prefer_unmet() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // Both GPUs satisfy min (2 GB) but neither meets prefer (16 GB).
        let snap = snapshot(&[6, 10]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, Some(16 * 1024), false);
        assert_eq!(pick, Some(1), "fall back to most-free when prefer unmet");
    }

    /// A pledge from another service must be subtracted from availability,
    /// so a busy GPU 0 cedes to a free GPU 1 even when nvml currently reports
    /// the same free bytes for both.
    #[test]
    fn pick_command_gpu_subtracts_pledged_reservations() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        let snap = snapshot(&[20, 20]); // both GPUs report 20 GB free
        // Another service has 19 GB pledged on GPU 0.
        let mut table = AllocationTable::new();
        let mut other = BTreeMap::new();
        other.insert(DeviceSlot::Gpu(0), 19 * 1024u64); // MB
        table.insert(SmolStr::new("other"), other);
        let pick = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(pick, Some(1), "GPU 1 has the pledged-aware headroom");
    }

    /// `gpu_allow` is a hard restriction. Even if a non-listed GPU has more
    /// free capacity, the pick must come from the allowed set.
    #[test]
    fn pick_command_gpu_respects_gpu_allow() {
        // GPU 0 = 24 GB, GPU 1 = 4 GB, GPU 2 = 16 GB; allow only [1, 2].
        let s = svc(PlacementPolicy::GpuOnly, Some(vec![1, 2]));
        let snap = snapshot(&[24, 4, 16]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, Some(2), "GPU 2 is the most-free among allowed");
    }

    /// When no GPU has enough capacity to host `min_mb`, return `None` so
    /// the supervisor can run its eviction-retry path.
    #[test]
    fn pick_command_gpu_returns_none_when_nothing_fits() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        let snap = snapshot(&[1, 1]); // 1 GB free each
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(pick, None);
    }

    /// `optimistic_remaining = true` ignores nvml's view of free bytes and
    /// trusts the pledge book alone, matching `pack_optimistic`.
    #[test]
    fn pick_command_gpu_optimistic_ignores_nvml_free() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // nvml reports 0 free on both, but total = 24 GB and the pledge book
        // says nothing is reserved.
        let snap = snapshot(&[0, 0]);
        let table = AllocationTable::new();

        // Conservative: clamps to nvml_free, nothing fits.
        let conservative = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(conservative, None);

        // Optimistic: pledge book has 24 GB headroom, GPU 0 wins on tiebreak.
        let optimistic = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, true);
        assert_eq!(optimistic, Some(0));
    }

    /// `placement_policy = CpuOnly` collapses the allowed-GPU list to empty;
    /// the helper must return `None` so the caller routes the reservation
    /// onto Cpu instead.
    #[test]
    fn pick_command_gpu_cpu_only_returns_none() {
        let s = svc(PlacementPolicy::CpuOnly, None);
        let snap = snapshot(&[24, 24]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, None);
    }

    /// Build a service with an explicit per-GPU placement_override.
    /// Mirrors the multi-GPU vLLM use case: TP=2 across two devices,
    /// each pledged separately.
    fn svc_with_override(pairs: &[(u32, u64)]) -> PlacementInputs {
        let mut overrides = BTreeMap::new();
        for (id, mb) in pairs {
            overrides.insert(DeviceSlot::Gpu(*id), *mb);
        }
        PlacementInputs {
            policy: PlacementPolicy::GpuOnly,
            placement_override: overrides,
            ..PlacementInputs::named("vllm-demo")
        }
    }

    /// Two-GPU pledge that fits on both devices: every per-slot pledge
    /// has room, so `check_command_placement_override` should accept.
    #[test]
    fn check_placement_override_accepts_multi_gpu_pledge_that_fits() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        let snap = snapshot(&[24, 24]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_eq!(r, Ok(()));
    }

    /// One slot in the override exceeds that GPU's free capacity. The
    /// helper must surface `WeightsDoNotFit` so the supervisor's
    /// eviction-retry loop can engage; partial fits never silently land.
    #[test]
    fn check_placement_override_rejects_when_one_slot_overflows() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        // GPU 1 only has 16 GB free; pledge of 22 GB doesn't fit.
        let snap = snapshot(&[24, 16]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_shortfall_on(&r, 1, 22 * 1024 * MIB);
    }

    /// Existing peer reservations on a slot have to be subtracted from
    /// available capacity. A second 22 GiB pledge on a GPU that's
    /// already pledged 10 GiB to a peer should fail.
    #[test]
    fn check_placement_override_subtracts_existing_pledges() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        let snap = snapshot(&[24, 24]);
        let mut table = AllocationTable::new();
        let mut peer = BTreeMap::new();
        peer.insert(DeviceSlot::Gpu(0), 10 * 1024);
        table.insert(SmolStr::new("peer"), peer);
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_shortfall_on(&r, 0, 22 * 1024 * MIB);
    }

    /// Optimistic mode (eviction retry) ignores nvml-free and trusts the
    /// pledge book. A pledged-but-not-yet-drained peer should NOT block
    /// our pledge once the supervisor has removed it from the table.
    #[test]
    fn check_placement_override_optimistic_ignores_nvml_free() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        // nvml shows GPU 0 nearly full (peer is still draining), but the
        // pledge book is empty — optimistic mode should accept.
        let snap = snapshot(&[2, 24]);
        let table = AllocationTable::new();
        let conservative = check_command_placement_override(&s, &snap, &table, false);
        assert_shortfall_on(&conservative, 0, 22 * 1024 * MIB);
        let optimistic = check_command_placement_override(&s, &snap, &table, true);
        assert_eq!(optimistic, Ok(()));
    }

    /// CPU entries in the override are ignored (we don't model CPU
    /// capacity here). A pledge that's only on CPU should accept.
    #[test]
    fn check_placement_override_ignores_cpu_slots() {
        let mut overrides = BTreeMap::new();
        overrides.insert(DeviceSlot::Cpu, 8 * 1024);
        let placement = PlacementInputs {
            policy: PlacementPolicy::CpuOnly,
            placement_override: overrides,
            ..PlacementInputs::named("demo")
        };
        let snap = snapshot(&[]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&placement, &snap, &table, false);
        assert_eq!(r, Ok(()));
    }
}
