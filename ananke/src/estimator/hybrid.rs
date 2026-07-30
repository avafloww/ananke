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

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    estimator::{
        compute_buffer, kv,
        llama::{collect_non_layer, collect_per_layer},
        recurrent,
        types::{Estimate, EstimatorInputs},
    },
    gguf::GgufSummary,
};

/// Architectures that mix attention with recurrent SSM layers (no MoE).
pub const HYBRID_FAMILY: &[&str] = &["jamba", "qwen35"];

pub fn is_hybrid(arch: &str) -> bool {
    HYBRID_FAMILY.contains(&arch)
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
pub fn kv_for_hybrid(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");
    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    let n_kv_heads = summary
        .metadata
        .get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;
    let key_length = summary
        .metadata
        .get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;
    let value_length = summary
        .metadata
        .get(&*format!("{arch}.attention.value_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;

    // `full_attention_interval = N`: only every N-th layer runs full
    // attention; the rest are SSM with no KV cache. Absent / 1 = every
    // layer has KV (the jamba case).
    let full_attention_interval = summary
        .metadata
        .get(&*format!("{arch}.full_attention_interval"))
        .and_then(|v| v.as_u32())
        .unwrap_or(1)
        .max(1);
    let span = recurrent::context_layer_span(summary, arch).min(n_layers);
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
    // [`crate::estimator::recurrent`].
    let recurrent_layers = span as u64 - kv_layer_count;
    let state = recurrent::state_bytes(summary, arch, recurrent_layers, inputs);

    // Fold the state into the per-token figure so `kv_per_token × context`
    // recovers `attention_kv + state`.
    let context = inputs.context as u64;
    attention_kv + state.checked_div(context).unwrap_or(0)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);
    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    let kv_per_token = kv_for_hybrid(summary, arch, n_layers, inputs);

    Estimate {
        weights_bytes,
        kv_per_token,
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
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_tensors: None,
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

    fn fake_hybrid_summary(arch: &str, n_layers: u32, interval: Option<u32>) -> GgufSummary {
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
            SmolStr::new("general.architecture"),
            GgufValue::String(arch.into()),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.block_count")),
            GgufValue::U32(n_layers),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.head_count_kv")),
            GgufValue::U32(4),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.key_length")),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.value_length")),
            GgufValue::U32(128),
        );
        if let Some(interval) = interval {
            metadata.insert(
                SmolStr::new(format!("{arch}.full_attention_interval")),
                GgufValue::U32(interval),
            );
        }
        // Qwen3.6-27B's recurrent block, as its GGUF declares it.
        for (key, value) in [
            ("ssm.conv_kernel", 4u32),
            ("ssm.inner_size", 6144),
            ("ssm.state_size", 128),
            ("ssm.group_count", 16),
        ] {
            metadata.insert(SmolStr::new(format!("{arch}.{key}")), GgufValue::U32(value));
        }

        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs<'a>(context: u32, empty: &'a [String]) -> EstimatorInputs<'a> {
        EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: crate::config::validate::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            ubatch: None,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
            override_tensor: empty,
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
        let s = fake_hybrid_summary("qwen35", 64, Some(4));
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs(4096, &empty));
        assert_eq!(e.kv_per_token, 71072);
    }

    #[test]
    fn jamba_kv_no_interval_scales_all_layers() {
        // No full_attention_interval key → defaults to 1 (all layers).
        let s = fake_hybrid_summary("jamba", 80, None);
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs(4096, &empty));
        // 80 layers × 2048 bytes = 163840.
        assert_eq!(e.kv_per_token, 163840);
    }

    #[test]
    fn hybrid_is_recognised() {
        assert!(is_hybrid("jamba"));
        assert!(is_hybrid("qwen35"));
    }
}
