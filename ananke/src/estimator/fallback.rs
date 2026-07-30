//! Fallback estimator for unknown architectures.

use std::collections::BTreeMap;

use smol_str::SmolStr;
use tracing::warn;

use crate::{
    estimator::{
        compute_buffer,
        types::{Estimate, EstimatorInputs, NonLayer},
    },
    gguf::GgufSummary,
};

/// Multiplier applied to the GGUF's on-disk tensor bytes as a rough
/// headroom factor for the unmodelled non-tensor overhead (KV, compute
/// buffer, context scratch).
const FALLBACK_WEIGHTS_SCALE: f64 = 1.15;

/// Flat headroom added on top of the scaled weights.
const FALLBACK_WEIGHTS_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// Produce a coarse estimate for any GGUF: scaled tensor bytes plus a flat
/// headroom landing in `weights_bytes`; no KV modelling; no per-layer
/// split. Emits a warning so the operator knows rolling correction is
/// the only tuning they'll get.
pub fn estimate_fallback(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    warn!(
        architecture = %summary.architecture,
        "unknown architecture — using fallback estimator"
    );
    let weights = ((summary.total_tensor_bytes as f64) * FALLBACK_WEIGHTS_SCALE) as u64
        + FALLBACK_WEIGHTS_HEADROOM_BYTES;
    Estimate {
        weights_bytes: weights,
        kv_per_token: 0,
        // The compute model's pooled default entry, which is fitted across every
        // measured architecture rather than borrowed from whichever one happens to
        // be listed first. An unknown architecture has no head count to size an
        // unfused score matrix from, so that term stays zero regardless.
        compute_buffer_mb: compute_buffer::per_device_for(summary, inputs),
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        tensor_split_replicated_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        output_buffer_bytes: 0,
        per_layer_bytes: None,
        attention_layers: None,
        non_layer: NonLayer {
            output_head_bytes: 0,
            token_embd_bytes: 0,
            tied_head_bytes: 0,
            other_bytes: 0,
        },
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_tensors: None,
        context: inputs.context,
        architecture: SmolStr::new(summary.architecture.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn summary_with(total_bytes: u64, arch: &str) -> GgufSummary {
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: total_bytes,
            tensors: Default::default(),
            metadata: Default::default(),
            block_count: None,
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn fallback_applies_declared_scale_and_headroom() {
        let s = summary_with(1_000_000_000, "nonsense-arch");
        const EMPTY: &[String] = &[];
        let inputs = EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            ubatch: None,
            visible_devices: 1,
            host_resident_experts: false,
            split_mode: crate::config::validate::SplitMode::Layer,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: EMPTY,
            compute_buffer_mb: None,
            allow_fallback: true,
            mtp: false,
            draft_model: None,
            ik_llama: false,
            ik_dsa: false,
            parallel: None,
            flash_attn: None,
            kv_unified: None,
            cache_ram_mb: None,
        };
        let e = estimate_fallback(&s, &inputs);
        // Assert against the named constants so the test tracks any
        // future re-tuning without silently drifting.
        assert_eq!(
            e.weights_bytes,
            (1_000_000_000f64 * FALLBACK_WEIGHTS_SCALE) as u64 + FALLBACK_WEIGHTS_HEADROOM_BYTES
        );
        assert_eq!(e.kv_per_token, 0);
    }
}
