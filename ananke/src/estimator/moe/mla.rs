//! KV cache sizing for MLA (multi-head latent attention) architectures
//! (glm-dsa), which carry no V cache at all.

use crate::{
    estimator::{kv, types::EstimatorInputs},
    gguf::GgufSummary,
};

/// Fallback `attention.key_length` for MLA archs whose quant dropped the
/// key: kv_lora_rank (512) + rope dims (64), the GLM-5 / DeepSeek-V3
/// compressed-cache width.
const MLA_DEFAULT_KEY_LENGTH: u64 = 576;

/// KV bytes per context token for MLA architectures (glm-dsa).
///
/// llama.cpp allocates no V cache for MLA (`has_v = !is_mla` in
/// `llama-kv-cache.cpp`) — the value states are recovered from the
/// compressed latent via `attn_v_b` at compute time — so the per-token
/// cost is a single K tensor of `attention.key_length` elements per
/// layer, priced off `cache_type_k` alone. The trailing
/// `nextn_predict_layers` MTP block is excluded: llama.cpp's
/// `hparams.n_layer()` (which sizes the main-context cache) subtracts it.
pub(crate) fn mla_kv_per_token(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if inputs.context == 0 || n_layers == 0 {
        return 0;
    }
    let bytes_k = kv::kv_bytes_per_element(inputs.cache_type_k.unwrap_or("f16"));

    let key_length = summary
        .metadata
        .get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .map(u64::from)
        .unwrap_or(MLA_DEFAULT_KEY_LENGTH);
    let nextn_layers = summary
        .metadata
        .get(&*format!("{arch}.nextn_predict_layers"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0);
    let kv_layers = n_layers.saturating_sub(nextn_layers) as u64;

    (kv_layers as f64 * key_length as f64 * bytes_k) as u64
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use crate::{
        estimator::{moe::estimate::estimate, types::EstimatorInputs},
        gguf::types::{GgufSummary, GgufValue},
    };

    #[test]
    fn glm_dsa_kv_is_key_only_and_excludes_nextn_layers() {
        use std::path::Path;

        // GLM-5.2 shape: 79 blocks, 1 NextN layer, MLA cache of 576
        // K elements per token per layer, phantom value_length of 512
        // that must NOT be priced.
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("glm-dsa".into()),
        );
        metadata.insert(SmolStr::new("glm-dsa.block_count"), GgufValue::U32(79));
        metadata.insert(
            SmolStr::new("glm-dsa.attention.head_count_kv"),
            GgufValue::U32(1),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.attention.key_length"),
            GgufValue::U32(576),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.nextn_predict_layers"),
            GgufValue::U32(1),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(79),
            architecture: SmolStr::new("glm-dsa"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let mk = |ctk: Option<&'static str>| EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 32768,
            ubatch: None,
            cache_type_k: ctk,
            cache_type_v: None,
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
            ik_llama: false,
            ik_dsa: false,
            parallel: None,
            flash_attn: None,
            kv_unified: None,
            cache_ram_mb: None,
        };
        // 78 KV layers × 576 elems × 2 bytes (f16) = 89856 bytes/token.
        // The naive K+V formula would give 79 × (576 + 512) × 2 = 171904.
        assert_eq!(estimate(&summary, &mk(None)).kv_per_token, 89_856);
        // q8_0 K-cache shrinks it by the element-width ratio; the V type
        // is irrelevant because no V cache exists.
        assert_eq!(
            estimate(&summary, &mk(Some("q8_0"))).kv_per_token,
            (78.0f64 * 576.0 * 1.0625) as u64
        );
    }
}
