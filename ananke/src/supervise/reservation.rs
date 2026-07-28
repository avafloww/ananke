//! [`RunLoop`]'s reservation-map computation: running the estimator + packer
//! (or the command-template GPU pick) to decide what a spawn would reserve,
//! without touching the allocation table.

use crate::supervise::{
    RunLoop,
    ensure::{MisconfiguredKind, ReservationFailure, pack_err_to_reservation_failure},
};

impl RunLoop {
    /// Determine the reservation map for an Ensure. On the llama-cpp path this
    /// runs the estimator + packer and caches `Packed` on `self` for the
    /// eventual `render_argv` call. Typed `Err` so the caller can branch on
    /// pack failures (retry with eviction) vs config / estimator failures
    /// (surface verbatim).
    pub(crate) fn compute_reservation_map(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.compute_reservation_map_inner(snap, table, false)
    }

    /// Variant of [`compute_reservation_map`] that uses the optimistic
    /// planner (`pack_optimistic`). Used by the retry-after-eviction path,
    /// where `table` has had victims filtered out and nvml still shows their
    /// realized usage.
    pub(crate) fn compute_reservation_map_optimistic(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.compute_reservation_map_inner(snap, table, true)
    }

    fn compute_reservation_map_inner(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
        optimistic: bool,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        let current = self.current_svc();
        let svc = &current;
        if matches!(svc.template(), crate::config::Template::Command) {
            return self.compute_command_reservation(svc, snap, table, optimistic);
        }
        if !svc.placement_override.is_empty() {
            self.packed_for_spawn = None;
            return Ok(svc.placement_override.clone());
        }

        // Estimator + placement path.
        let inputs = crate::estimator::EstimatorInputs::from_service(svc)
            .map(|i| i.with_visible_devices(snap.gpus.len() as u32))
            .ok_or(ReservationFailure::Misconfigured(
                MisconfiguredKind::NoModelPath,
            ))?;
        let fingerprint = inputs.config_fingerprint();
        let (summary, est) =
            crate::estimator::estimate_with_summary(self.deps.system.fs.as_ref(), &inputs)
                .map_err(ReservationFailure::EstimatorError)?;
        // Warm the daemon-wide estimate cache with the *base* estimate
        // (pre-rolling-correction) and the GGUF summary we just
        // parsed. The management `ServiceDetail` handler reads this
        // cache instead of re-parsing the file on every detail poll;
        // populating it here turns the first detail-page view of a
        // running service into a cache hit. The cache stores the base
        // numbers because that's what the wire `EstimateSummary`
        // documents — the rolling correction applied below is a
        // supervisor-internal placement tweak, not a user-facing
        // estimate.
        let lc = svc.llama_cpp();
        if let Some(lc) = lc {
            self.deps.estimate_cache.insert(
                svc.name.clone(),
                crate::daemon::estimate_cache::CacheEntry::build(
                    &summary,
                    &est,
                    lc.model.clone(),
                    lc.mmproj.clone(),
                    fingerprint,
                ),
            );
        }
        // Hand the service's learned per-pool corrections to the packer, which
        // scales every byte it charges to a device by that pool's factor.
        // `RollingCorrection::corrections` gates each factor to a neutral 1.0
        // until enough samples accumulate, so a single noisy early observation
        // can't over-pledge a shard past a GPU's capacity.
        let corrections = self.deps.rolling.get(&svc.name).corrections();
        let packed = crate::allocator::placement::pack_corrected(
            &est,
            svc,
            snap,
            table,
            corrections,
            optimistic,
        )
        .map_err(pack_err_to_reservation_failure)?;
        // Convert Allocation bytes (per-DeviceId, in bytes) to the
        // BTreeMap<DeviceSlot, u64> in MB that can_fit + insert expects.
        let want_mb: std::collections::BTreeMap<crate::config::DeviceSlot, u64> = packed
            .allocation
            .bytes
            .iter()
            .map(|(id, bytes)| {
                let slot = match id {
                    crate::devices::DeviceId::Cpu => crate::config::DeviceSlot::Cpu,
                    crate::devices::DeviceId::Gpu(n) => crate::config::DeviceSlot::Gpu(*n),
                };
                // Convert bytes → MB, rounding up so we never under-reserve.
                let mb = bytes.div_ceil(1024 * 1024);
                (slot, mb)
            })
            .collect();
        self.packed_for_spawn = Some(packed);
        Ok(want_mb)
    }

    /// Reservation-map computation for command-template services.
    ///
    /// Picks the GPU with the most available headroom (subject to
    /// `gpu_allow`) via `placement::pick_command_gpu`, falling back to
    /// `PackError::WeightsDoNotFit` when nothing fits — that variant is the
    /// trigger for the supervisor's eviction-retry path. The chosen
    /// allocation is stashed on `packed_for_spawn` so `handle_active_lifecycle`
    /// renders `CUDA_VISIBLE_DEVICES` against the actual pick instead of the
    /// always-empty `init.allocation` (which is built from
    /// `placement_override` only).
    pub(crate) fn compute_command_reservation(
        &mut self,
        svc: &crate::config::ServiceConfig,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
        optimistic: bool,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.packed_for_spawn = None;
        let (min_mb, prefer_mb) = match svc.allocation_mode {
            crate::config::AllocationMode::Static { reserve_mb } => (reserve_mb, Some(reserve_mb)),
            crate::config::AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
            crate::config::AllocationMode::None => (0, None),
        };
        let mut map = std::collections::BTreeMap::new();
        if min_mb == 0 {
            // No reservation requested. Still publish an empty Packed so the
            // spawn path renders a deterministic (empty) CUDA_VISIBLE_DEVICES.
            let alloc = crate::devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(crate::allocator::placement::Packed {
                allocation: alloc,
                args: crate::allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
                // A command service's reservation is the operator's declared
                // `min_mb`, not an estimate, so there is nothing for the
                // rolling correction to learn about. Zero bases skip it.
                rolling: crate::allocator::placement::RollingInputs::default(),
            });
            return Ok(map);
        }
        // Operator pinned the layout explicitly (e.g. multi-GPU vLLM with
        // TP=2). Honour the per-device pledge verbatim instead of trying to
        // land the whole `min_mb` on a single GPU.
        if !svc.placement_override.is_empty() {
            crate::allocator::placement::check_command_placement_override(
                svc, snap, table, optimistic,
            )
            .map_err(ReservationFailure::PackFailed)?;
            map = svc.placement_override.clone();
            let alloc = crate::devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(crate::allocator::placement::Packed {
                allocation: alloc,
                args: crate::allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
                // A command service's reservation is the operator's declared
                // `min_mb`, not an estimate, so there is nothing for the
                // rolling correction to learn about. Zero bases skip it.
                rolling: crate::allocator::placement::RollingInputs::default(),
            });
            return Ok(map);
        }
        let slot = if matches!(
            svc.placement_policy,
            crate::config::PlacementPolicy::CpuOnly
        ) {
            crate::config::DeviceSlot::Cpu
        } else {
            match crate::allocator::placement::pick_command_gpu(
                svc, snap, table, min_mb, prefer_mb, optimistic,
            ) {
                Some(id) => crate::config::DeviceSlot::Gpu(id),
                None if snap.gpus.is_empty() => {
                    // No GPUs visible at all (typical in tests with a CPU-only
                    // snapshot). Fall back to CPU so the reservation lands
                    // somewhere — matches the pre-fix behaviour and keeps
                    // the test harness working.
                    crate::config::DeviceSlot::Cpu
                }
                None => {
                    return Err(ReservationFailure::PackFailed(
                        crate::allocator::placement::PackError::WeightsDoNotFit {
                            shortfalls: crate::allocator::placement::command_gpu_shortfalls(
                                svc, snap, table, min_mb, optimistic,
                            ),
                        },
                    ));
                }
            }
        };
        map.insert(slot.clone(), min_mb);
        let alloc = crate::devices::Allocation::from_override(&map);
        self.packed_for_spawn = Some(crate::allocator::placement::Packed {
            allocation: alloc,
            args: crate::allocator::placement::CommandArgs::default(),
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
            // See the `min_mb == 0` arm above: an operator-declared reservation
            // is not an estimate.
            rolling: crate::allocator::placement::RollingInputs::default(),
        });
        Ok(map)
    }
}
