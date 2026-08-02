//! Shared fixtures for the placement test modules: a minimal service/snapshot
//! builder pair, plus trivial and MoE `Estimate` builders.

use std::collections::BTreeMap;

use ananke_config::placement::{DeviceSlot, OffloadMode, PlacementInputs, PlacementPolicy};
pub(crate) use ananke_config::units::{GIB, MIB};
use ananke_estimate::{Buffers, Estimate, ExpertKind, ExpertTensor, Layout};
use ananke_gguf::Architecture;

use crate::{
    Packed,
    devices::{CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot},
};

/// A placement with a pinned override, built through the same distiller the
/// daemon uses so the tests exercise that conversion rather than bypassing it.
pub(crate) fn svc(policy: PlacementPolicy, gpu_allow: Option<Vec<u32>>) -> PlacementInputs {
    let mut overrides = BTreeMap::new();
    overrides.insert(DeviceSlot::Gpu(0), 1000);
    PlacementInputs {
        policy,
        placement_override: overrides,
        gpu_allow: gpu_allow.unwrap_or_default(),
        ..PlacementInputs::named("demo")
    }
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
        layout: Layout {
            per_layer_bytes: Some(vec![per_layer_mb * 1024 * 1024; n_layers as usize]),
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 400,
            ..Buffers::default()
        },
        ..Estimate::empty(Architecture::Qwen3, 4096)
    }
}

/// A Hybrid llama-cpp service with `expert_offload` set, for the
/// expert-aware packer path. `placement_override` is cleared so `pack`
/// takes the estimator path.
pub(crate) fn moe_svc(offload: OffloadMode) -> PlacementInputs {
    PlacementInputs {
        policy: PlacementPolicy::Hybrid,
        expert_offload: offload,
        ..PlacementInputs::named("moe")
    }
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
        layout: Layout {
            per_layer_bytes: Some(per_layer),
            expert_layers: (0..n_layers).collect(),
            expert_tensors: Some(experts),
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 400,
            ..Buffers::default()
        },
        ..Estimate::empty(Architecture::Qwen3Moe, 4096)
    }
}

pub(crate) fn cpu_bytes(p: &Packed) -> u64 {
    p.allocation.bytes.get(&DeviceId::Cpu).copied().unwrap_or(0)
}
