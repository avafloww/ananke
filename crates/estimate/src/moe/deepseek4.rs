//! KV cache sizing for DeepSeek-V4-Flash (deepseek4), whose compressed
//! sparse attention (CSA) layers are the only ones with a context-scaling
//! cache.

use ananke_gguf::{GgufSummary, GgufType, keys};

use crate::{kv, tuning::DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16, types::EstimatorInputs};

/// The `attention.compress_ratios` value that marks a CSA (compressed
/// sparse attention) layer — the only layers whose KV cache scales with
/// context. Layers with the ratio-128 HCA value or the leading `0` full-
/// attention value do not carry a context-scaling cache worth modelling.
const DEEPSEEK4_CSA_RATIO: u32 = 4;

/// KV bytes per context token for deepseek4 (DeepSeek-V4-Flash).
///
/// Only the CSA layers keep a context-scaling cache; it is key-only (the
/// value projection is absorbed into the compressed latent, so llama.cpp
/// reports `V (f16): 0.00 MiB`), so the per-token cost tracks the
/// `cache_type_k` element width alone. The sibling HCA cache (ratio 128 →
/// `n_ctx / 128` cells) and the fixed sliding-window cache are small and
/// context-flat, so they fall into the compute-buffer headroom rather than
/// this per-token term. Returns `kv_per_token` so the packer multiplies by
/// context exactly as it does for every other family.
pub(crate) fn deepseek4_kv_per_token(
    summary: &GgufSummary,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    let arch = &summary.architecture;
    if inputs.context == 0 || n_layers == 0 {
        return 0;
    }
    let bytes_k = kv::kv_bytes_per_element(inputs.cache_type_k.unwrap_or(GgufType::F16));

    // Count the CSA layers from `attention.compress_ratios` when present;
    // fall back to the observed "roughly half the layers are CSA" ratio so
    // a quant that drops the array still gets a sane, non-zero estimate.
    let csa_layers = summary
        .metadata
        .get(&keys::attention_compress_ratios(arch))
        .and_then(|v| v.as_u32_array())
        .map(|ratios| ratios.iter().filter(|&&r| r == DEEPSEEK4_CSA_RATIO).count() as u64)
        .filter(|&n| n > 0)
        .unwrap_or((n_layers / 2) as u64);

    // f16 baseline scaled to the actual K-cache element width (f16 = 2.0).
    let per_layer = DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16 * (bytes_k / 2.0);
    (csa_layers as f64 * per_layer) as u64
}

#[cfg(test)]
mod tests {
    use ananke_gguf::{
        Architecture, keys,
        types::{GgufSummary, GgufTensor, GgufType, GgufValue},
    };
    use smol_str::SmolStr;

    use crate::{moe::estimate::estimate, types::EstimatorInputs};

    #[test]
    fn deepseek4_kv_uses_csa_layer_count_not_naive_mla() {
        use std::path::Path;

        // 43-layer deepseek4 shape: layers 0-1 are ratio 0 (full attn),
        // then the rest alternate 4 (CSA) / 128 (HCA) → 21 CSA layers.
        let n_layers = 43u32;
        let compress_ratios: Vec<GgufValue> = (0..n_layers)
            .map(|i| {
                GgufValue::U32(match i {
                    0 | 1 => 0,
                    i if i % 2 == 0 => 4,
                    _ => 128,
                })
            })
            .collect();
        let csa_layers = compress_ratios
            .iter()
            .filter(|v| matches!(v, GgufValue::U32(4)))
            .count();
        assert_eq!(csa_layers, 21, "sanity: fixture has 21 CSA layers");

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..n_layers {
            for kind in ["attn_kv", "ffn_gate_exps", "ffn_gate_shexp"] {
                let name = format!("blk.{layer}.{kind}.weight");
                tensors.insert(
                    SmolStr::new(&name),
                    GgufTensor {
                        name: SmolStr::new(&name),
                        dtype: GgufType::F16,
                        shape: vec![512 * 1024],
                        byte_size: 1024 * 1024,
                        shard_idx: 0,
                        offset: 0,
                    },
                );
            }
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("deepseek4".into()),
        );
        metadata.insert(
            SmolStr::new("deepseek4.block_count"),
            GgufValue::U32(n_layers),
        );
        // Naive-MLA metadata that `kv_for_hybrid` would otherwise consume.
        metadata.insert(
            SmolStr::new("deepseek4.attention.head_count_kv"),
            GgufValue::U32(1),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.key_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.compress_ratios"),
            GgufValue::Array(compress_ratios),
        );

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: Architecture::DeepSeek4,
            shards: vec!["/fake".into()],
        };
        let inputs = EstimatorInputs {
            name: "demo",
            context: 131072,
            cache_type_k: Some(GgufType::F16),
            cache_type_v: Some(GgufType::F16),
            ..EstimatorInputs::empty(Path::new("/fake"))
        };

        let e = estimate(&summary, &inputs);
        // 21 CSA layers × 317 B/token (f16) = 6657 B/token — matches the
        // measured ~6.65 KiB/token, an order of magnitude below the naive
        // MLA formula (1 kv-head × (512+512) × 2 × 43 = 88_064 B/token).
        assert_eq!(e.kv_per_token, 6657);
        assert!(
            e.kv_per_token < 88_064 / 10,
            "deepseek4 KV must be far below the naive MLA estimate; got {}",
            e.kv_per_token
        );
        // Expert itemisation still works: 43 fused gate tensors, shared
        // experts (`_shexp`) excluded.
        assert_eq!(e.expert_tensors.as_ref().unwrap().len(), 43);
    }

    #[test]
    fn deepseek4_kv_scales_with_cache_type() {
        use std::path::Path;

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("deepseek4".into()),
        );
        metadata.insert(SmolStr::new("deepseek4.block_count"), GgufValue::U32(43));
        // No compress_ratios → falls back to n_layers / 2 = 21 CSA layers.
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(43),
            architecture: Architecture::DeepSeek4,
            shards: vec!["/fake".into()],
        };
        let mk = |ctk: GgufType| EstimatorInputs {
            name: "demo",
            context: 131072,
            cache_type_k: Some(ctk),
            cache_type_v: Some(ctk),
            ..EstimatorInputs::empty(Path::new("/fake"))
        };
        // Fallback layer count (21) at f16 reproduces the 6657 figure.
        assert_eq!(estimate(&summary, &mk(GgufType::F16)).kv_per_token, 6657);
        // q8_0 K-cache (1.0625 B/elem vs 2.0) shrinks the per-token cost.
        assert!(estimate(&summary, &mk(GgufType::Q8_0)).kv_per_token < 6657);
    }
}
