//! Weights accounting for llama-family models: per-layer tensor collection,
//! non-layer tensor bucketing, and the top-level `estimate` entry point.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    estimator::{
        llama::kv_per_token::compute_kv_per_token,
        types::{Estimate, EstimatorInputs, NonLayer},
    },
    gguf::GgufSummary,
};

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer_bytes = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer_bytes.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    let kv_per_token = compute_kv_per_token(summary, arch, n_layers, inputs);

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: crate::estimator::compute_buffer::per_device_for(summary, inputs),
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        output_buffer_bytes: 0,
        per_layer_bytes: Some(per_layer_bytes),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_tensors: None,
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

pub(crate) fn collect_per_layer(summary: &GgufSummary, n_layers: u32) -> Vec<u64> {
    let mut out = vec![0u64; n_layers as usize];
    for tensor in summary.tensors.values() {
        if let Some(idx) = layer_index(&tensor.name)
            && (idx as usize) < out.len()
        {
            out[idx as usize] += tensor.byte_size;
        }
    }
    out
}

pub(crate) fn collect_non_layer(summary: &GgufSummary) -> NonLayer {
    let mut nl = NonLayer::default();
    for (name, tensor) in &summary.tensors {
        if layer_index(name).is_some() {
            continue;
        }
        match name.as_str() {
            "output.weight" => nl.output_head_bytes += tensor.byte_size,
            // Gemma 4's `per_layer_token_embd.weight` is a 42-slot embedding
            // stack (one per transformer block) that llama.cpp keeps on CPU
            // alongside `token_embd.weight`. For the E4B quant it's ~2.8 GiB
            // — bucketing it as GPU-resident caused the packer to over-
            // reserve a small single-GPU fit by ~3 GiB.
            "token_embd.weight" | "per_layer_token_embd.weight" => {
                nl.token_embd_bytes += tensor.byte_size
            }
            _ => nl.other_bytes += tensor.byte_size,
        }
    }
    // Only meaningful once every tensor has been seen: the table is the output
    // head exactly when the model ships no head of its own.
    if nl.output_head_bytes == 0 {
        nl.tied_head_bytes = summary
            .tensors
            .get("token_embd.weight")
            .map(|t| t.byte_size)
            .unwrap_or(0);
    }
    nl
}

/// Extract the N in a tensor name like `blk.N.attn_q.weight`.
pub(crate) fn layer_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use super::*;
    use crate::{
        estimator::llama::test_support::{fake_summary, inputs, tensor},
        gguf::types::GgufSummary,
    };

    #[test]
    fn sums_per_layer_and_non_layer() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // per-layer: 2 layers × 3 tensors × 1 MiB = 6 MiB weights from layers.
        // non-layer: 2 MiB output + 4 MiB token_embd = 6 MiB.
        assert_eq!(e.weights_bytes, 12 * 1024 * 1024);
        assert_eq!(e.per_layer_bytes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn layer_index_extracts() {
        assert_eq!(layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(layer_index("blk.42.ffn_down.weight"), Some(42));
        assert_eq!(layer_index("output.weight"), None);
        assert_eq!(layer_index("token_embd.weight"), None);
    }

    #[test]
    fn gemma3n_ple_tensor_is_cpu_resident() {
        // The E4B quant carries a ~2.3 GiB `per_layer_token_embd.weight`
        // PLE table plus MatFormer `altup_*` / `per_layer_*_proj.weight`
        // tensors. The PLE must land on CPU (same rule as gemma4 above);
        // altup/proj are tiny and fall through to `other_bytes` which is
        // fine — they genuinely do live on GPU at runtime.
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            tensor("token_embd.weight", 352 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            tensor("per_layer_token_embd.weight", 2380 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("altup_proj.weight"),
            tensor("altup_proj.weight", 24 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("altup_unembd_proj.weight"),
            tensor("altup_unembd_proj.weight", 24 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_model_proj.weight"),
            tensor("per_layer_model_proj.weight", 35 * 1024 * 1024),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata: std::collections::BTreeMap::new(),
            block_count: Some(0),
            architecture: SmolStr::new("gemma3n"),
            shards: vec!["/fake".into()],
        };
        let nl = collect_non_layer(&summary);
        // PLE + token_embd both route to CPU.
        assert_eq!(nl.token_embd_bytes, (352 + 2380) * 1024 * 1024);
        // altup + per_layer_model_proj are GPU-resident (together < 100 MiB).
        assert_eq!(nl.other_bytes, (24 + 24 + 35) * 1024 * 1024);
        // No explicit output head — gemma3n uses weight-tied output via
        // token_embd, so `output_head_bytes` stays zero.
        assert_eq!(nl.output_head_bytes, 0);
    }

    #[test]
    fn per_layer_token_embd_is_cpu_resident() {
        // Gemma 4 E-variants carry a large `per_layer_token_embd.weight`
        // tensor (2.8 GiB for E4B) that llama.cpp keeps on CPU alongside
        // `token_embd.weight`. Bucketing it as GPU-resident caused the
        // packer to over-reserve a single-GPU fit by ~3 GiB.
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            tensor("token_embd.weight", 100 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            tensor("per_layer_token_embd.weight", 300 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("output_norm.weight"),
            tensor("output_norm.weight", 1024),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata: std::collections::BTreeMap::new(),
            block_count: Some(0),
            architecture: SmolStr::new("gemma4"),
            shards: vec!["/fake".into()],
        };
        let nl = collect_non_layer(&summary);
        assert_eq!(nl.token_embd_bytes, 400 * 1024 * 1024);
        assert_eq!(nl.other_bytes, 1024);
    }
}
