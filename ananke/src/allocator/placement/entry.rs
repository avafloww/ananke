//! Public entry points: `pack` and `pack_optimistic` drive the packer
//! through its steps in order and return the finished `Packed` result.

use crate::{
    allocator::{
        AllocationTable,
        placement::{Packed, packer::Packer, types::PackError},
    },
    config::ServiceConfig,
    devices::DeviceSnapshot,
    estimator::Estimate,
};

/// Number of per-layer-equivalents added to every active backend as slop
/// tolerance for tensor-split rounding. Bumped if empirical overruns
/// show tensor_split's remainder exceeds one layer's worth.
pub(crate) const ONE_LAYER_FUDGE_MULTIPLIER: u64 = 1;

/// `-ngl` value meaning "offload every layer to the GPU". Used when we
/// reserved whole-model space on a GPU without per-layer detail.
pub(crate) const NGL_OFFLOAD_ALL: u32 = 999;

/// `-ngl` value meaning "run entirely on CPU".
pub(crate) const NGL_CPU_ONLY: u32 = 0;

/// Pack `estimate` onto allowed devices, respecting `policy`,
/// `override_tensor`, and live device capacity (`snapshot` minus any
/// already-reserved bytes from `reserved`).
pub fn pack(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    pack_inner(estimate, svc, snapshot, reserved, false)
}

/// Pack variant that trusts the pledge book (`total - reserved`) exclusively
/// rather than taking `min(nvml_free, total - reserved)`. Intended for the
/// retry-after-eviction path, where victims have been removed from `reserved`
/// to model "if they were gone" — nvml still shows their realized usage
/// until drains actually land.
pub fn pack_optimistic(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    pack_inner(estimate, svc, snapshot, reserved, true)
}

fn pack_inner(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    optimistic_remaining: bool,
) -> Result<Packed, PackError> {
    let mut packer = Packer::new(estimate, svc, snapshot, reserved, optimistic_remaining);
    // Sharded (tensor/row) split distributes every layer across all spanned
    // GPUs in parallel — a fundamentally different shape from the first-fit
    // layer walk. Taken only when the service opts in, at least two GPUs are
    // available to span, and the estimator gave a per-layer breakdown to
    // halve. Otherwise fall through to the layer path: a single-GPU "tensor
    // split" is just an ordinary placement, and a fallback-arch model (no
    // per-layer detail) can't be evenly sharded.
    if packer.svc.split_mode.is_sharded()
        && packer.allowed_gpus.len() >= 2
        && !packer.per_layer.is_empty()
    {
        packer.distribute_sharded()?;
        return Ok(packer.finish());
    }
    packer.seed_non_layer();
    packer.seed_mtp_overhead();
    if packer.expert_aware {
        // Two-phase MoE placement: pin every layer's attention + KV on a GPU,
        // then offload the trailing surplus expert *layers* to CPU as whole
        // units via `--n-cpu-moe`. Whole-layer offload (rather than per-tensor
        // `-ot`) keeps the runtime's fused multi-threaded CPU MoE kernel
        // engaged and stays under llama.cpp's graph-split limit.
        packer.place_nonexpert_layers()?;
        packer.distribute_experts_ncmoe()?;
    } else {
        packer.walk_layers()?;
        packer.place_fallback_weights()?;
    }
    packer.add_kv_bytes();
    packer.add_compute_buffer();
    packer.add_one_layer_fudge();
    packer.check_cpu_capacity()?;
    Ok(packer.finish())
}
