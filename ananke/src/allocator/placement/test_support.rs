//! Shared fixtures for the placement test modules: a minimal service/snapshot
//! builder pair, plus trivial and MoE `Estimate` builders.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    allocator::placement::Packed,
    config::{
        OffloadMode, PlacementPolicy, ServiceConfig,
        validate::{
            DeviceSlot,
            test_fixtures::{expect_llama_cpp, minimal_service},
        },
    },
    devices::{CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot},
    estimator::{Estimate, ExpertKind, ExpertTensor, NonLayer},
};

pub(crate) const MIB: u64 = 1024 * 1024;
pub(crate) const GIB: u64 = 1024 * 1024 * 1024;

pub(crate) fn svc(policy: PlacementPolicy, gpu_allow: Option<Vec<u32>>) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 1000);
    let mut svc = minimal_service("demo");
    svc.placement_override = placement;
    svc.placement_policy = policy;
    if let Some(a) = gpu_allow {
        svc.gpu_allow = a;
    }
    svc
}

pub(crate) fn snapshot(free_gpu_gb: &[u64]) -> DeviceSnapshot {
    let gpus = free_gpu_gb
        .iter()
        .enumerate()
        .map(|(i, gb)| GpuSnapshot {
            id: i as u32,
            name: format!("GPU {i}"),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: gb * 1024 * 1024 * 1024,
        })
        .collect();
    DeviceSnapshot {
        gpus,
        cpu: Some(CpuSnapshot {
            total_bytes: 128 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    }
}

pub(crate) fn trivial_estimate(n_layers: u32, per_layer_mb: u64) -> Estimate {
    Estimate {
        weights_bytes: per_layer_mb * 1024 * 1024 * n_layers as u64,
        kv_per_token: 0,
        compute_buffer_mb: 400,
        output_buffer_bytes: 0,
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        tensor_split_replicated_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        per_layer_bytes: Some(vec![per_layer_mb * 1024 * 1024; n_layers as usize]),
        attention_layers: None,
        non_layer: NonLayer::default(),
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_tensors: None,
        context: 4096,
        architecture: SmolStr::new("qwen3"),
    }
}

/// A Hybrid llama-cpp service with `expert_offload` set, for the
/// expert-aware packer path. `placement_override` is cleared so `pack`
/// takes the estimator path.
pub(crate) fn moe_svc(offload: OffloadMode) -> ServiceConfig {
    let mut svc = minimal_service("moe");
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::Hybrid;
    expect_llama_cpp(&mut svc).expert_offload = offload;
    svc
}

/// A MoE estimate: every layer carries `nonexp_mb` of non-expert weight
/// plus three fused expert tensors of `exp_mb` each. `per_layer_bytes`
/// holds the full cost; `expert_tensors` itemises the experts (already
/// counted in the per-layer total).
pub(crate) fn moe_estimate(n_layers: u32, nonexp_mb: u64, exp_mb: u64) -> Estimate {
    let layer_total = (nonexp_mb + 3 * exp_mb) * MIB;
    let mut per_layer = Vec::new();
    let mut experts = Vec::new();
    for layer in 0..n_layers {
        per_layer.push(layer_total);
        for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
            experts.push(ExpertTensor {
                layer,
                kind,
                bytes: exp_mb * MIB,
            });
        }
    }
    Estimate {
        weights_bytes: layer_total * n_layers as u64,
        kv_per_token: 0,
        compute_buffer_mb: 400,
        output_buffer_bytes: 0,
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        tensor_split_replicated_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer: NonLayer::default(),
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: (0..n_layers).collect(),
        expert_tensors: Some(experts),
        context: 4096,
        architecture: SmolStr::new("qwen3moe"),
    }
}

pub(crate) fn cpu_bytes(p: &Packed) -> u64 {
    p.allocation.bytes.get(&DeviceId::Cpu).copied().unwrap_or(0)
}
