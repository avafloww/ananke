//! Shared GGUF fixture builders for the llama-family estimator's unit tests.
//!
//! Centralised so `estimate` and `kv_per_token` don't each carry their own
//! copy of the same fake tensor/summary/inputs constructors.

use std::path::Path;

use smol_str::SmolStr;

use crate::{
    estimator::types::EstimatorInputs,
    gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue},
};

pub fn tensor(name: &str, bytes: u64) -> GgufTensor {
    GgufTensor {
        name: SmolStr::new(name),
        dtype: GgufType::F16,
        shape: vec![bytes / 2],
        byte_size: bytes,
        shard_idx: 0,
        offset: 0,
    }
}

pub fn fake_summary() -> GgufSummary {
    let mut tensors = std::collections::BTreeMap::new();
    // 2 layers × 3 tensors per layer.
    for layer in 0..2u32 {
        for kind in ["attn_q", "attn_k", "ffn_down"] {
            let name = format!("blk.{layer}.{kind}.weight");
            tensors.insert(SmolStr::new(&name), tensor(&name, 1024 * 1024));
        }
    }
    tensors.insert(
        SmolStr::new("output.weight"),
        tensor("output.weight", 2 * 1024 * 1024),
    );
    tensors.insert(
        SmolStr::new("token_embd.weight"),
        tensor("token_embd.weight", 4 * 1024 * 1024),
    );

    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        SmolStr::new("general.architecture"),
        GgufValue::String("qwen3".into()),
    );
    metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(2));
    metadata.insert(
        SmolStr::new("qwen3.attention.head_count_kv"),
        GgufValue::U32(4),
    );
    metadata.insert(
        SmolStr::new("qwen3.attention.key_length"),
        GgufValue::U32(128),
    );
    metadata.insert(
        SmolStr::new("qwen3.attention.value_length"),
        GgufValue::U32(128),
    );

    GgufSummary {
        path: "/fake".into(),
        total_tensor_bytes: 6 * 1024 * 1024 + 6 * 1024 * 1024,
        tensors,
        metadata,
        block_count: Some(2),
        architecture: SmolStr::new("qwen3"),
        shards: vec!["/fake".into()],
    }
}

pub fn inputs<'a>(
    cache_k: &'a str,
    cache_v: &'a str,
    context: u32,
    empty: &'a [String],
) -> EstimatorInputs<'a> {
    EstimatorInputs {
        name: "demo",
        model: Path::new("/fake"),
        mmproj: None,
        context,
        ubatch: None,
        cache_type_k: Some(cache_k),
        cache_type_v: Some(cache_v),
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
