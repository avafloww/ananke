//! Hybrid-architecture estimator (SSM + attention, no MoE).
//!
//! Applies to: jamba, qwen35.
//!
//! These models mix standard attention layers with SSM (State Space Model)
//! layers. Only every `full_attention_interval`-th layer runs full attention
//! with a KV cache; the rest run a recurrent SSM that carries constant
//! per-layer state instead of context-dependent KV.
//!
//! Weight accounting uses the same `blk.N.*` layout as llama-family.
//! KV cache is scaled down by the attention interval.

use ananke_gguf::{Architecture, GgufSummary, GgufType, keys};

use crate::{
    compute_buffer, kv,
    llama::{collect_non_layer, collect_per_layer},
    recurrent,
    types::{Buffers, Estimate, EstimatorInputs, Layout},
};

/// Architectures that mix attention with recurrent SSM layers (no MoE).
pub const HYBRID_FAMILY: &[Architecture] = &[Architecture::Jamba, Architecture::Qwen35];

pub fn is_hybrid(arch: &Architecture) -> bool {
    HYBRID_FAMILY.contains(arch)
}

/// Compute the KV cost per token for a hybrid model, scaling by
/// `full_attention_interval`. Only every N-th layer has a KV cache;
/// the rest carry recurrent state instead.
///
/// Returns `kv_per_token` so the caller can multiply by context to recover
/// total context bytes. The recurrent layers' context-independent state is
/// folded in as a per-token equivalent (`state / context`) so the downstream
/// `kv_per_token × context` recovers `attention_kv + recurrent_state`, which
/// is what llama.cpp allocates and reports as one "context" bucket in its
/// memory breakdown — and is distributed across devices the same way, since
/// both scale with how many of the model's layers a device holds.
///
/// `n_layers` is the model's full block count; the layer counting below uses
/// [`recurrent::context_layer_span`] to drop an MTP head's trailing block,
/// which belongs to the separate draft context rather than to this one.
pub fn kv_for_hybrid(summary: &GgufSummary, n_layers: u32, inputs: &EstimatorInputs<'_>) -> u64 {
    let arch = &summary.architecture;
    let cache_k = inputs.cache_type_k.unwrap_or(GgufType::F16);
    let cache_v = inputs.cache_type_v.unwrap_or(GgufType::F16);
    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    let n_kv_heads = summary
        .meta_u32(&keys::attention_head_count_kv(arch))
        .unwrap_or(0) as u64;
    let key_length = summary
        .meta_u32(&keys::attention_key_length(arch))
        .unwrap_or(128) as u64;
    let value_length = summary
        .meta_u32(&keys::attention_value_length(arch))
        .unwrap_or(128) as u64;

    // `full_attention_interval = N`: only every N-th layer runs full
    // attention; the rest are SSM with no KV cache. Absent / 1 = every
    // layer has KV (the jamba case).
    let full_attention_interval = summary
        .meta_u32(&keys::full_attention_interval(arch))
        .unwrap_or(1)
        .max(1);
    let span = recurrent::context_layer_span(summary).min(n_layers);
    let kv_layer_count = (span / full_attention_interval) as u64;

    let attention_kv = if kv_layer_count > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv =
            n_kv_heads * ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
        kv_layer_count * per_layer_bytes_kv
    } else {
        0
    };

    // The remaining layers carry recurrent state, which llama.cpp allocates
    // alongside the KV cache and reports in the same "context" bucket. Its
    // size follows from the model's `ssm.*` metadata — see
    // [`crate::recurrent`].
    let recurrent_layers = span as u64 - kv_layer_count;
    let state = recurrent::state_bytes(summary, recurrent_layers, inputs);

    // Fold the state into the per-token figure so `kv_per_token × context`
    // recovers `attention_kv + state`.
    let context = inputs.context as u64;
    attention_kv + state.checked_div(context).unwrap_or(0)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = &summary.architecture;
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);
    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    let kv_per_token = kv_for_hybrid(summary, n_layers, inputs);

    Estimate {
        weights_bytes,
        kv_per_token,
        layout: Layout {
            per_layer_bytes: Some(per_layer),
            non_layer,
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: compute_buffer::per_device_for(summary, inputs),
            ..Buffers::default()
        },
        ..Estimate::empty(arch.clone(), inputs.context)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ananke_gguf::{
        keys::suffix,
        types::{GgufSummary, GgufTensor, GgufType, GgufValue},
    };
    use smol_str::SmolStr;

    use super::*;

    fn fake_hybrid_summary(
        arch: &Architecture,
        n_layers: u32,
        interval: Option<u32>,
    ) -> GgufSummary {
        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..n_layers {
            let name = format!("blk.{layer}.attn_q.weight");
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
        tensors.insert(
            SmolStr::new("output.weight"),
            GgufTensor {
                name: SmolStr::new("output.weight"),
                dtype: GgufType::F16,
                shape: vec![1024 * 1024],
                byte_size: 2 * 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            GgufTensor {
                name: SmolStr::new("token_embd.weight"),
                dtype: GgufType::F16,
                shape: vec![2 * 1024 * 1024],
                byte_size: 4 * 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String(arch.as_str().into()),
        );
        metadata.insert(keys::block_count(arch), GgufValue::U32(n_layers));
        metadata.insert(keys::attention_head_count_kv(arch), GgufValue::U32(4));
        metadata.insert(keys::attention_key_length(arch), GgufValue::U32(128));
        metadata.insert(keys::attention_value_length(arch), GgufValue::U32(128));
        if let Some(interval) = interval {
            metadata.insert(
                keys::full_attention_interval(arch),
                GgufValue::U32(interval),
            );
        }
        // Qwen3.6-27B's recurrent block, as its GGUF declares it.
        for (key, value) in [
            (suffix::SSM_CONV_KERNEL, 4u32),
            (suffix::SSM_INNER_SIZE, 6144),
            (suffix::SSM_STATE_SIZE, 128),
            (suffix::SSM_GROUP_COUNT, 16),
        ] {
            metadata.insert(keys::scoped(arch, key), GgufValue::U32(value));
        }

        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: arch.clone(),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs<'a>(context: u32) -> EstimatorInputs<'a> {
        EstimatorInputs {
            context,
            name: "demo",
            cache_type_k: Some(GgufType::F16),
            cache_type_v: Some(GgufType::F16),
            ..EstimatorInputs::empty(Path::new("/fake"))
        }
    }

    #[test]
    fn qwen35_kv_scales_with_full_attention_interval() {
        // 64 layers, interval=4 → 16 attention layers, 48 recurrent.
        // Attention KV: 16 × 4 heads × (128+128) × 2 bytes (f16) = 32768
        //   bytes/token.
        // Recurrent state, one slot, no speculation:
        //   R = (4-1) × (6144 + 2×16×128) = 30720 elements
        //   S = 128 × 6144 = 786432 elements
        //   48 layers × 817152 × 4 bytes = 156_893_184 bytes, which is the
        //   5.62 + 144.00 MiB llama.cpp reports for this model.
        //   Folded into per-token at ctx 4096: 156_893_184 / 4096 = 38304.
        // kv_per_token = 32768 + 38304 = 71072.
        let s = fake_hybrid_summary(&Architecture::Qwen35, 64, Some(4));
        let e = estimate(&s, &inputs(4096));
        assert_eq!(e.kv_per_token, 71072);
    }

    #[test]
    fn jamba_kv_no_interval_scales_all_layers() {
        // No full_attention_interval key → defaults to 1 (all layers).
        let s = fake_hybrid_summary(&Architecture::Jamba, 80, None);
        let e = estimate(&s, &inputs(4096));
        // 80 layers × 2048 bytes = 163840.
        assert_eq!(e.kv_per_token, 163840);
    }

    #[test]
    fn hybrid_is_recognised() {
        assert!(is_hybrid(&Architecture::Jamba));
        assert!(is_hybrid(&Architecture::Qwen35));
    }
}
