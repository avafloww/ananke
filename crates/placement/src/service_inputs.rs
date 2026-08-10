//! Distil the packer-relevant fields out of a `ServiceConfig`.
//!
//! Same reasoning as the estimator's service_inputs: the packer is a pure
//! function over a placement, an estimate, and a device snapshot, and should
//! not have to know what a `ServiceConfig` is.

use ananke_config::{ServiceConfig, placement::PlacementInputs};

/// Distil the packer-relevant fields out of a `ServiceConfig`.
///
/// `reserves` is cloned rather than borrowed because it is three small
/// fields and the `Arc` on the config exists for a different sharing
/// pattern.
pub fn placement_inputs(svc: &ServiceConfig) -> PlacementInputs {
    PlacementInputs {
        name: svc.name.clone(),
        policy: svc.placement_policy,
        placement_override: svc.placement_override.clone(),
        split_mode: svc.split_mode,
        gpu_allow: svc.gpu_allow.clone(),
        gpu_headroom_mb: svc.gpu_headroom_mb,
        reserves: (*svc.reserves).clone(),
        ik_llama: svc.llama_cpp().is_some_and(|lc| lc.runtime.ik().is_some()),
        expert_offload: svc
            .llama_cpp()
            .map(|lc| lc.expert_offload)
            .unwrap_or_default(),
        tensor_split_weights: svc.tensor_split_weights.clone(),
        override_tensor: svc
            .llama_cpp()
            .map(|lc| lc.override_tensor.clone())
            .unwrap_or_default(),
    }
}
