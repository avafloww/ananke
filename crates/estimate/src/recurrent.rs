//! Recurrent-state sizing for architectures whose layers are not all attention.
//!
//! Beside the attention KV cache, llama.cpp keeps a second memory module —
//! `llama_memory_recurrent` — for every layer that carries recurrent state
//! instead of a cache. It holds two f32 tensors per such layer, both
//! independent of the context length:
//!
//! - **R**, the rolling convolution state, `n_embd_r` elements. A Mamba-style
//!   `ssm.*` block sizes it `(d_conv - 1) × (d_inner + 2 × n_group × d_state)`;
//!   a short-convolution block (LFM2) sizes it `n_embd × (l_cache - 1)`.
//! - **S**, the SSM state proper, `n_embd_s = d_state × d_inner` elements. A
//!   short-convolution block has none.
//!
//! The whole module is replicated `parallel × (rollback_depth + 1)` times. Two
//! things about that multiplier are easy to get wrong:
//!
//! - It scales with `--parallel`, **not** with the stream count. `--kv-unified`
//!   collapses the attention cache across slots and leaves this untouched:
//!   the `mtpslot-*-none-np{1,2,4}` cells all set `kv_unified` and their
//!   recurrent module still grows 1×, 2×, 4×.
//! - Speculative decoding replicates it again, once per rollback slot plus the
//!   live one, so a `--spec-type draft-mtp` service pays four times what its
//!   slot count alone suggests.
//!
//! Every term is GGUF metadata or a service flag, so nothing here is fitted.
//! `ananke-calibrate`'s deriver holds this formula to all 13 distinct
//! (architecture, slots, rollback) combinations in the measurement set and
//! fails `emit` if any of them moves.

use ananke_gguf::{GgufSummary, keys};

use crate::{tuning::SPEC_RECURRENT_ROLLBACK_DEPTH, types::EstimatorInputs};

/// Bytes of recurrent state llama.cpp allocates for `recurrent_layers` layers
/// of this model under `inputs`, or 0 when the architecture carries none.
pub(crate) fn state_bytes(
    summary: &GgufSummary,
    recurrent_layers: u64,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if recurrent_layers == 0 {
        return 0;
    }
    let per_layer = per_layer_elements(summary);
    if per_layer == 0 {
        return 0;
    }
    // A rollback depth is only allocated when something can roll back.
    let rollback = if inputs.mtp {
        SPEC_RECURRENT_ROLLBACK_DEPTH
    } else {
        0
    };
    let copies = u64::from(inputs.parallel.unwrap_or(1).max(1)) * (rollback + 1);
    recurrent_layers * per_layer * F32_BYTES * copies
}

/// The number of a model's blocks the main context spans.
///
/// An MTP head's trailing blocks are excluded: llama.cpp reports the span as
/// `n_layer` and the full block count as `n_layer_all`, and its contexts cover
/// only the former — the head's own attention layer belongs to the separate
/// draft context. Without this the recurrent layer count runs one high on
/// every Qwen 3.6 model.
pub(crate) fn context_layer_span(summary: &GgufSummary) -> u32 {
    let arch = &summary.architecture;
    let blocks = summary.block_count.unwrap_or(0);
    let nextn = summary
        .meta_u32(&keys::nextn_predict_layers(arch))
        .unwrap_or(0);
    blocks.saturating_sub(nextn)
}

/// `n_embd_r + n_embd_s`: the elements one recurrent layer holds per copy.
fn per_layer_elements(summary: &GgufSummary) -> u64 {
    let arch = &summary.architecture;
    let meta = |key| summary.meta_u64(&key).unwrap_or(0);
    let d_conv = meta(keys::ssm_conv_kernel(arch));
    if d_conv > 0 {
        let d_inner = meta(keys::ssm_inner_size(arch));
        let d_state = meta(keys::ssm_state_size(arch));
        let n_group = meta(keys::ssm_group_count(arch));
        let n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state);
        let n_embd_s = d_state * d_inner;
        return n_embd_r + n_embd_s;
    }
    // A short-convolution block keeps `l_cache - 1` past activations of the
    // full hidden width and no SSM state at all.
    let l_cache = meta(keys::shortconv_l_cache(arch));
    if l_cache > 1 {
        return meta(keys::embedding_length(arch)) * (l_cache - 1);
    }
    0
}

/// llama.cpp allocates both recurrent tensors as f32 regardless of the
/// model's quantisation or the `--cache-type-*` flags, which apply only to the
/// attention cache.
const F32_BYTES: u64 = 4;

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use ananke_gguf::{
        Architecture,
        keys::suffix,
        types::{GgufSummary, GgufValue},
    };
    use smol_str::SmolStr;

    use super::*;

    /// Qwen3.6-35B-A3B's recurrent shape, as its GGUF declares it.
    fn qwen35moe() -> GgufSummary {
        summary_with(
            &Architecture::Qwen35Moe,
            41,
            1,
            &[
                (suffix::SSM_CONV_KERNEL, 4),
                (suffix::SSM_INNER_SIZE, 4096),
                (suffix::SSM_STATE_SIZE, 128),
                (suffix::SSM_GROUP_COUNT, 16),
            ],
        )
    }

    fn summary_with(
        arch: &Architecture,
        blocks: u32,
        nextn: u32,
        entries: &[(&str, u32)],
    ) -> GgufSummary {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String(arch.as_str().into()),
        );
        metadata.insert(keys::nextn_predict_layers(arch), GgufValue::U32(nextn));
        for (key, value) in entries {
            metadata.insert(keys::scoped(arch, key), GgufValue::U32(*value));
        }
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata,
            block_count: Some(blocks),
            architecture: arch.clone(),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs<'a>(parallel: Option<u32>, mtp: bool, empty: &'a [String]) -> EstimatorInputs<'a> {
        EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 32768,
            ubatch: None,
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            override_tensor: empty,
            compute_buffer_mb: None,
            mtp,
            draft_model: None,
            ik_llama: false,
            ik_dsa: false,
            parallel,
            flash_attn: None,
            kv_unified: None,
            cache_ram_mb: None,
        }
    }

    #[test]
    fn qwen35moe_reproduces_the_measured_single_slot_state() {
        // `llama_memory_recurrent: size = 62.81 MiB (1 cells, 40 layers,
        // 1 seqs 0 rs_seq), R (f32): 2.81 MiB, S (f32): 60.00 MiB`
        // — 30 recurrent layers of the 40 the context spans.
        let s = qwen35moe();
        assert_eq!(context_layer_span(&s), 40);
        let empty: Vec<String> = Vec::new();
        let bytes = state_bytes(&s, 30, &inputs(Some(1), false, &empty));
        // R: (4-1) × (4096 + 2×16×128) = 24576 elements → 2.8125 MiB over 30.
        // S: 128 × 4096 = 524288 elements → 60 MiB over 30.
        assert_eq!(
            bytes,
            (2.8125_f64 * 1024.0 * 1024.0) as u64 + 60 * 1024 * 1024
        );
    }

    #[test]
    fn scales_with_parallel_even_under_unified_kv() {
        // The `mtpslot-*-none` cells all set `--kv-unified` and still show the
        // recurrent module growing with the slot count: 62.81, 125.62, 251.25
        // MiB at np 1, 2, 4. A stream count would have collapsed all three.
        let s = qwen35moe();
        let empty: Vec<String> = Vec::new();
        let at = |np: u32| {
            let mut i = inputs(Some(np), false, &empty);
            i.kv_unified = Some(true);
            state_bytes(&s, 30, &i)
        };
        assert_eq!(at(2), at(1) * 2);
        assert_eq!(at(4), at(1) * 4);
    }

    #[test]
    fn speculative_decoding_replicates_the_whole_module() {
        // Production Qwen3.6-35B-A3B: 2 slots with `--spec-type draft-mtp`
        // measures 502.50 MiB against 62.81 at one slot without it, exactly
        // `2 × (3 + 1)`.
        let s = qwen35moe();
        let empty: Vec<String> = Vec::new();
        let plain = state_bytes(&s, 30, &inputs(Some(1), false, &empty));
        let spec = state_bytes(&s, 30, &inputs(Some(2), true, &empty));
        assert_eq!(spec, plain * 2 * (SPEC_RECURRENT_ROLLBACK_DEPTH + 1));
    }

    #[test]
    fn shortconv_has_a_rolling_state_and_no_ssm_state() {
        // LFM2.5-Embedding-350M: `l_cache = 3`, `n_embd = 1024`, 10 of its 16
        // layers recurrent. Measured `R (f32): 0.08 MiB, S (f32): 0.00 MiB`.
        let s = summary_with(
            &Architecture::Lfm2,
            16,
            0,
            &[
                (suffix::SHORTCONV_L_CACHE, 3),
                (suffix::EMBEDDING_LENGTH, 1024),
            ],
        );
        let empty: Vec<String> = Vec::new();
        let bytes = state_bytes(&s, 10, &inputs(Some(1), false, &empty));
        assert_eq!(bytes, 10 * 1024 * 2 * 4);
    }

    #[test]
    fn a_pure_attention_model_has_no_recurrent_state() {
        let s = summary_with(
            &Architecture::Llama,
            32,
            0,
            &[(suffix::EMBEDDING_LENGTH, 4096)],
        );
        let empty: Vec<String> = Vec::new();
        assert_eq!(state_bytes(&s, 0, &inputs(Some(1), false, &empty)), 0);
        // Even if a caller miscounted the layers, an architecture with neither
        // an `ssm.*` block nor a short convolution allocates nothing.
        assert_eq!(state_bytes(&s, 32, &inputs(Some(1), false, &empty)), 0);
    }
}
