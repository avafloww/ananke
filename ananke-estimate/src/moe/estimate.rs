//! Weights accounting and expert-tensor itemisation for MoE models, plus the
//! top-level `estimate` entry point that dispatches KV sizing to the
//! deepseek4 / MLA specialisations or the generic hybrid path.

use std::collections::BTreeMap;

use ananke_gguf::GgufSummary;
use smol_str::SmolStr;

use crate::{
    llama::{collect_non_layer, layer_index},
    moe::{deepseek4::deepseek4_kv_per_token, mla::mla_kv_per_token},
    types::{Estimate, EstimatorInputs, ExpertKind, ExpertTensor},
};

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    // Per-layer split into {non-expert, expert} bytes, and itemise every
    // offloadable fused expert tensor for the packer.
    let mut per_layer_nonexp = vec![0u64; n_layers as usize];
    let mut per_layer_exp = vec![0u64; n_layers as usize];
    let mut expert_tensors: Vec<ExpertTensor> = Vec::new();

    for (name, t) in &summary.tensors {
        let Some(idx) = layer_index(name) else {
            continue;
        };
        if (idx as usize) >= per_layer_nonexp.len() {
            continue;
        }
        if let Some(kind) = expert_kind(name) {
            per_layer_exp[idx as usize] += t.byte_size;
            expert_tensors.push(ExpertTensor {
                layer: idx,
                kind,
                bytes: t.byte_size,
            });
        } else {
            per_layer_nonexp[idx as usize] += t.byte_size;
        }
    }

    let non_layer = collect_non_layer(summary);

    // Full per-layer cost (non-expert + experts). Experts are itemised in
    // `expert_tensors` but stay counted here; the packer subtracts them only
    // for the experts it actually moves off the GPU.
    let per_layer_total: Vec<u64> = per_layer_nonexp
        .iter()
        .zip(per_layer_exp.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let weights_bytes = per_layer_total.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // KV cost per token. deepseek4's compressed caches need bespoke
    // handling (see `deepseek4_kv_per_token`); glm-dsa uses MLA. Architectures
    // with `full_attention_interval` (qwen35moe's SSM hybrid) use the shared
    // hybrid logic. Everything else — including MoE architectures with
    // sliding-window attention like laguna — uses the llama-family KV
    // function, which handles SWA capping, per-layer head_count_kv arrays,
    // and shared-KV layers.
    let has_full_attention_interval = summary
        .metadata
        .contains_key(&*format!("{arch}.full_attention_interval"));
    let kv_per_token = if arch == "deepseek4" {
        deepseek4_kv_per_token(summary, arch, n_layers, inputs)
    } else if arch == "glm-dsa" {
        mla_kv_per_token(summary, arch, n_layers, inputs)
    } else if has_full_attention_interval {
        crate::hybrid::kv_for_hybrid(summary, arch, n_layers, inputs)
    } else {
        crate::llama::compute_kv_per_token(summary, arch, n_layers, inputs)
    };

    let expert_layers: Vec<u32> = per_layer_exp
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if *b > 0 { Some(i as u32) } else { None })
        .collect();

    // Stable order so the packer's offload selection and the synthesised `-ot`
    // rules are deterministic across runs.
    expert_tensors.sort_by_key(|e| (e.layer, e.kind));

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: crate::compute_buffer::per_device_for(summary, inputs),
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        mtp_head_expert_layers: 0,
        tensor_split_replicated_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        output_buffer_bytes: 0,
        per_layer_bytes: Some(per_layer_total),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers,
        expert_tensors: Some(expert_tensors),
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

/// Classify an expert weight tensor by its projection.
/// Pattern: `blk.N.ffn_{gate,up,down}_exps.weight`. The `_shexp` (shared
/// expert) counterparts are *not* offloadable experts and return `None`.
pub(crate) fn expert_kind(name: &str) -> Option<ExpertKind> {
    let rest = name.strip_prefix("blk.")?;
    let (_, kind) = rest.split_once('.')?;
    if kind.contains("shexp") {
        return None;
    }
    if kind.starts_with("ffn_gate_exps") {
        Some(ExpertKind::Gate)
    } else if kind.starts_with("ffn_up_exps") {
        Some(ExpertKind::Up)
    } else if kind.starts_with("ffn_down_exps") {
        Some(ExpertKind::Down)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35moe_kv_scales_with_full_attention_interval() {
        use std::path::Path;

        use ananke_gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

        use crate::types::EstimatorInputs;

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..8u32 {
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
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen35moe".into()),
        );
        metadata.insert(SmolStr::new("qwen35moe.block_count"), GgufValue::U32(8));
        metadata.insert(
            SmolStr::new("qwen35moe.attention.head_count_kv"),
            GgufValue::U32(4),
        );
        metadata.insert(
            SmolStr::new("qwen35moe.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("qwen35moe.attention.value_length"),
            GgufValue::U32(128),
        );
        // Every 4th layer is full-attention → 2 layers × KV; the other 6
        // are recurrent and contribute no KV.
        metadata.insert(
            SmolStr::new("qwen35moe.full_attention_interval"),
            GgufValue::U32(4),
        );
        // Qwen3.6-35B-A3B's recurrent block, as its GGUF declares it.
        for (key, value) in [
            ("ssm.conv_kernel", 4u32),
            ("ssm.inner_size", 4096),
            ("ssm.state_size", 128),
            ("ssm.group_count", 16),
        ] {
            metadata.insert(
                SmolStr::new(format!("qwen35moe.{key}")),
                GgufValue::U32(value),
            );
        }

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(8),
            architecture: SmolStr::new("qwen35moe"),
            shards: vec!["/fake".into()],
        };

        let empty: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            ubatch: None,
            cache_type_k: None,
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

        let e = estimate(&summary, &inputs);
        // Attention KV: 4 heads × (128+128) × 2 bytes (f16) = 2048 bytes/token.
        // With interval=4, only 2 of 8 layers are attention → 2 × 2048 = 4096.
        // Recurrent state over the other 6 layers, one slot, no speculation:
        //   R = (4-1) × (4096 + 2×16×128) = 24576 elements
        //   S = 128 × 4096 = 524288 elements
        //   6 × 548864 × 4 bytes = 13_172_736, folded into per-token at
        //   ctx 4096 → 3216.
        // kv_per_token = 4096 + 3216 = 7312.
        assert_eq!(e.kv_per_token, 7312);
    }

    #[test]
    fn expert_pattern_matches() {
        assert_eq!(
            expert_kind("blk.0.ffn_gate_exps.weight"),
            Some(ExpertKind::Gate)
        );
        assert_eq!(
            expert_kind("blk.1.ffn_up_exps.weight"),
            Some(ExpertKind::Up)
        );
        assert_eq!(
            expert_kind("blk.5.ffn_down_exps.weight"),
            Some(ExpertKind::Down)
        );
        assert_eq!(expert_kind("blk.0.ffn_gate.weight"), None);
        assert_eq!(expert_kind("blk.0.ffn_gate_shexp.weight"), None);
        assert_eq!(expert_kind("output.weight"), None);
    }

    #[test]
    fn itemises_expert_tensors_with_full_per_layer() {
        use std::path::Path;

        use ananke_gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};
        use smol_str::SmolStr;

        use crate::types::EstimatorInputs;

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..3u32 {
            // Base layer tensors: 1 MiB.
            let attn = format!("blk.{layer}.attn_q.weight");
            tensors.insert(
                SmolStr::new(&attn),
                GgufTensor {
                    name: SmolStr::new(&attn),
                    dtype: GgufType::F16,
                    shape: vec![512 * 1024],
                    byte_size: 1024 * 1024,
                    shard_idx: 0,
                    offset: 0,
                },
            );
            // Expert tensors: different size by layer.
            let size = match layer {
                0 => 4,
                1 => 10,
                2 => 2,
                _ => unreachable!(),
            };
            let exp = format!("blk.{layer}.ffn_gate_exps.weight");
            tensors.insert(
                SmolStr::new(&exp),
                GgufTensor {
                    name: SmolStr::new(&exp),
                    dtype: GgufType::F16,
                    shape: vec![size * 512 * 1024],
                    byte_size: size * 1024 * 1024,
                    shard_idx: 0,
                    offset: 0,
                },
            );
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen3moe".into()),
        );
        metadata.insert(SmolStr::new("qwen3moe.block_count"), GgufValue::U32(3));

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(3),
            architecture: SmolStr::new("qwen3moe"),
            shards: vec!["/fake".into()],
        };

        let empty_override: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            ubatch: None,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &empty_override,
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

        let e = estimate(&summary, &inputs);
        // Every layer's experts are itemised; nothing is pre-offloaded.
        let experts = e.expert_tensors.expect("MoE arch must itemise experts");
        assert_eq!(experts.len(), 3, "one fused gate tensor per layer");
        // Sorted by (layer, kind).
        assert_eq!(experts[0].layer, 0);
        assert_eq!(experts[0].bytes, 4 * 1024 * 1024);
        assert_eq!(experts[1].layer, 1);
        assert_eq!(experts[1].bytes, 10 * 1024 * 1024);
        assert_eq!(experts[2].layer, 2);
        assert_eq!(experts[2].bytes, 2 * 1024 * 1024);
        // per_layer_bytes keeps the full cost (1 MiB attn + experts).
        let per_layer = e.per_layer_bytes.expect("per-layer breakdown");
        assert_eq!(per_layer[1], (1 + 10) * 1024 * 1024);
    }

    #[test]
    fn laguna_kv_uses_scalar_head_count_kv_not_variable_head_count() {
        // Laguna's Q-projection head count is a per-layer array, but KV uses
        // a scalar `head_count_kv` — the array must not leak into the KV term.
        use std::path::Path;

        use ananke_gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

        use crate::types::EstimatorInputs;

        let n_layers = 48u32;
        let mut tensors = std::collections::BTreeMap::new();
        // Layer 0 is dense (no experts); layers 1..47 are MoE.
        for layer in 0..n_layers {
            for kind in ["attn_q", "attn_k", "attn_v", "attn_output"] {
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
            if layer > 0 {
                // Fused routed experts + shared expert per MoE layer.
                for kind in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
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
                for kind in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
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
        }

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("laguna".into()),
        );
        metadata.insert(SmolStr::new("laguna.block_count"), GgufValue::U32(n_layers));
        // The variable Q head count — must not affect KV.
        let head_count: Vec<GgufValue> = (0..n_layers)
            .map(|i| GgufValue::U32(if i % 4 == 0 { 48 } else { 72 }))
            .collect();
        metadata.insert(
            SmolStr::new("laguna.attention.head_count"),
            GgufValue::Array(head_count),
        );
        // Scalar KV head count — this is what kv_for_hybrid reads.
        metadata.insert(
            SmolStr::new("laguna.attention.head_count_kv"),
            GgufValue::U32(8),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.value_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.sliding_window"),
            GgufValue::U32(512),
        );

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new("laguna"),
            shards: vec!["/fake".into()],
        };

        let empty: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 32768,
            ubatch: None,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
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

        let e = estimate(&summary, &inputs);

        // KV must use the scalar head_count_kv (8), not the variable
        // head_count array (48/72). Laguna uses a 1:3 global:SWA pattern
        // (every 4th layer is global, the rest cap at sliding_window=512).
        // 12 global layers × 8 × (128+128) × 2 × 32768 +
        // 36 SWA layers × 8 × (128+128) × 2 × 512 = 1_686_110_208 bytes.
        // kv_per_token = 1_686_110_208 / 32768 = 51_456 bytes/token.
        assert_eq!(e.kv_per_token, 51_456);

        // 47 MoE layers × 3 fused expert projections (gate/up/down) =
        // 141 itemised expert tensors. The dense layer 0 has none, and
        // the `_shexp` shared experts are excluded (always-on, not
        // offloadable).
        let experts = e.expert_tensors.expect("MoE arch must itemise experts");
        assert_eq!(experts.len(), 141);
        assert_eq!(e.expert_layers.len(), 47);
        // Layer 0 (dense) must not appear in the expert layer list.
        assert!(!e.expert_layers.contains(&0u32));

        // q8_0 KV shrinks by the element-width ratio (1.0625 / 2.0).
        let inputs_q8 = EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            ..inputs
        };
        let e_q8 = estimate(&summary, &inputs_q8);
        // q8_0: 1.0625 bytes/element (block overhead). 12 global + 36 SWA
        // layers × 8 heads × 272 bytes/head. SWA layers cap at window 512.
        assert_eq!(e_q8.kv_per_token, 27_336);
        assert!(e_q8.kv_per_token < e.kv_per_token);
    }
}
