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
        + indexer_cache_bytes_per_token(summary, arch, n_layers.saturating_sub(nextn_layers))
}

/// Bytes per context token of the sparse-attention indexer's own cache.
///
/// A DSA (DeepSeek sparse attention) layer scores every cache token against the
/// batch's queries, and to do that it keeps a second, much narrower cache: one
/// `attention.indexer.key_length` key per token per indexing layer, always f16
/// — the `--cache-type-k` flag does not reach it. llama.cpp reports it apart
/// from the main cache, as `KV self indexer size`.
///
/// Which layers index is not in the metadata; it is in the tensor table, since
/// an indexing layer carries `blk.N.indexer.*` weights and a dense one does
/// not. GLM-5.2 carries them on 22 layers — 0, 1, 2, then every fourth — of
/// which the runtime uses 21: the twenty-second is the MTP head's block, which
/// sits outside the main context's layer span, and the loader says so
/// (`layer 78 is 'full' per indexer_types but has no indexer weights`).
///
/// 21 layers x 128 elements x 2 bytes is 5376 bytes per token, which reproduces
/// every measured figure exactly: 42, 168, 336, and 672 MiB at contexts 8192,
/// 32768, 65536, and 131072.
fn indexer_cache_bytes_per_token(summary: &GgufSummary, arch: &str, span: u32) -> u64 {
    let Some(key_length) = summary
        .metadata
        .get(&*format!("{arch}.attention.indexer.key_length"))
        .and_then(|v| v.as_u32())
        .map(u64::from)
    else {
        return 0;
    };
    let indexing_layers = summary
        .tensors
        .keys()
        .filter_map(|name| {
            let rest = name.strip_prefix("blk.")?;
            let (index, tail) = rest.split_once('.')?;
            tail.starts_with("indexer.")
                .then(|| index.parse::<u32>().ok())
                .flatten()
        })
        .filter(|&index| index < span)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u64;
    indexing_layers * key_length * INDEXER_CACHE_BYTES_PER_ELEMENT
}

/// The indexer cache is f16 whatever `--cache-type-k` says: it holds scores'
/// keys rather than the attention cache llama.cpp lets an operator quantise.
const INDEXER_CACHE_BYTES_PER_ELEMENT: u64 = 2;

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
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
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
        // No indexer tensors here, so no indexer cache.
        assert_eq!(estimate(&summary, &mk(None)).kv_per_token, 89_856);
        // q8_0 K-cache shrinks it by the element-width ratio; the V type
        // is irrelevant because no V cache exists.
        assert_eq!(
            estimate(&summary, &mk(Some("q8_0"))).kv_per_token,
            (78.0f64 * 576.0 * 1.0625) as u64
        );
    }

    /// GLM-5.2's indexer cache, against the four contexts it was measured at.
    #[test]
    fn glm_dsa_prices_the_sparse_attention_indexer_cache() {
        use std::path::Path;

        use crate::gguf::types::{GgufTensor, GgufType};

        let mut metadata = std::collections::BTreeMap::new();
        for (key, value) in [
            ("glm-dsa.block_count", 79u32),
            ("glm-dsa.attention.key_length", 576),
            ("glm-dsa.nextn_predict_layers", 1),
            ("glm-dsa.attention.indexer.key_length", 128),
            ("glm-dsa.attention.indexer.head_count", 32),
        ] {
            metadata.insert(SmolStr::new(key), GgufValue::U32(value));
        }
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("glm-dsa".into()),
        );
        // Indexer weights on layers 0, 1, 2 and every fourth from 6 to 78 —
        // 22 layers, of which the last is the MTP head's and unused.
        let mut tensors = std::collections::BTreeMap::new();
        for layer in [0u32, 1, 2].into_iter().chain((6..79).step_by(4)) {
            let name = format!("blk.{layer}.indexer.attn_k.weight");
            tensors.insert(
                SmolStr::new(&name),
                GgufTensor {
                    name: SmolStr::new(&name),
                    dtype: GgufType::F16,
                    shape: vec![128],
                    byte_size: 256,
                    shard_idx: 0,
                    offset: 0,
                },
            );
        }
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(79),
            architecture: SmolStr::new("glm-dsa"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let at = |context: u32| EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            ubatch: None,
            cache_type_k: Some("f16"),
            cache_type_v: None,
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
            ik_llama: true,
            ik_dsa: true,
            parallel: None,
            flash_attn: None,
            kv_unified: None,
            cache_ram_mb: None,
        };
        // 21 indexing layers × 128 × 2 bytes = 5376 bytes/token on top of the
        // 89856 the main cache costs.
        let per_token = estimate(&summary, &at(32768)).kv_per_token;
        assert_eq!(per_token, 89_856 + 5_376);
        // `KV self indexer size` measured 42, 168, 336, and 672 MiB.
        for (context, indexer_mib) in [(8192u32, 42u64), (32768, 168), (65536, 336), (131072, 672)]
        {
            let total = per_token * u64::from(context);
            let main = 89_856 * u64::from(context);
            assert_eq!(
                (total - main) / (1024 * 1024),
                indexer_mib,
                "at ctx {context}"
            );
        }
    }
}
