//! Per-device compute-buffer sizing.
//!
//! [`per_device_for`] is the entry point. The sizing itself is the single
//! fitted model in [`crate::compute_model`], whose design columns are
//! dimensionally normalised so architectures of different width share
//! coefficients. What this module adds is the head-card contract:
//! `per_device_for` returns the *head* GPU's total, and the packer trims
//! [`crate::compute_model::head_extra_mib`] back off every secondary.
//!
//! An architecture gets its own curve because the overhead scales with different
//! knobs — attention scratch grows fast with context on wide dense models,
//! stays nearly flat on an MoE where few experts run per token, and Gemma 4's
//! E-variants sit far below the fat-model default because their per-layer
//! embeddings live on the CPU. Which curve an architecture takes is data in
//! `tuning.json`, not a match arm here, so adding one arrives with its evidence
//! attached.
//!
//! On top of whichever model applies, [`no_flash_attn_mib`] adds the score
//! matrix an unfused attention pass materialises.
//!
//! Operators can override the whole term per service via
//! `estimation.compute_buffer_mb`.

use ananke_gguf::{Architecture, GgufSummary, keys};

use crate::{
    compute_model,
    tuning::{
        DEFAULT_UBATCH, NO_FLASH_ATTN_SCORE_CENTIBYTES, NO_FLASH_ATTN_SCORE_CENTIBYTES_DEFAULT,
    },
    types::EstimatorInputs,
};

/// llama.cpp materialises the output logits (`n_vocab × n_tokens` floats)
/// only on the device holding the output head — the packer's first GPU. This
/// estimates that head-only buffer so the packer can leave it *off* every
/// secondary GPU's reservation, freeing that VRAM for expert weight.
///
/// The reservation is deliberately conservative — modelled as `n_vocab ×
/// ubatch × 2` bytes rather than the `× 4` (f32) upper bound. The packer
/// *subtracts* this from the secondaries, so under-estimating the real logits
/// buffer keeps them safe (they simply keep a little extra headroom), whereas
/// over-estimating would under-reserve and OOM them. The measured head-only
/// delta measured at a large ubatch sits comfortably above this `× 2` figure,
/// confirming the direction. `n_vocab` is read from the output
/// head's shape, falling back to the token-embedding table for tied-embedding
/// models that ship no separate `output.weight`.
pub fn output_logits_bytes(summary: &GgufSummary, ubatch: Option<u32>) -> u64 {
    let n_vocab = summary
        .tensors
        .get("output.weight")
        .or_else(|| summary.tensors.get("token_embd.weight"))
        .and_then(|t| t.shape.iter().max().copied())
        .unwrap_or(0);
    n_vocab
        .saturating_mul(ubatch.unwrap_or(DEFAULT_UBATCH) as u64)
        .saturating_mul(2)
}

/// Per-device compute buffer, choosing the model that matches how llama.cpp
/// will split the model across devices.
///
/// The two splits build genuinely different graphs and the gap is not a scale
/// factor — neither split's figure predicts the other's, on any architecture —
/// so each has its own model. The sharded packer charges the result to *every*
/// spanned GPU rather than dividing it, because llama.cpp builds the same graph
/// on each device: the compute column reads identically on one card and on two
/// at every context.
pub fn per_device_for(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u32 {
    if let Some(mb) = inputs.compute_buffer_mb {
        return mb;
    }
    // The *head* card's total, which is this field's contract: the packer trims
    // `output_buffer_bytes` back off every secondary. See
    // [`crate::compute_model`] for the model itself, and
    // [`crate::compute_model::head_extra_mib`] for the term trimmed.
    let flash_attn = inputs.flash_attn.unwrap_or(true);
    compute_model::per_device_mib(summary, inputs)
        .saturating_add(compute_model::head_extra_mib(summary, inputs))
        .saturating_add(no_flash_attn_mib(
            summary,
            inputs.context,
            inputs.ubatch.unwrap_or(DEFAULT_UBATCH),
            flash_attn,
        ))
}

/// Query heads, falling back to `n_embd / key_length` where the GGUF omits them.
///
/// Not every architecture writes `attention.head_count`. Where it is missing, a
/// term built on the query head count evaluates to zero and leaves the whole
/// unfused score matrix unreserved, which was the largest single miss in the
/// dataset. Any error in the inferred count is absorbed by the per-architecture
/// rate, the two only ever appearing as a product, so the fallback costs
/// accuracy in the attribution rather than in the reservation.
fn query_head_count(summary: &GgufSummary) -> u64 {
    let arch = &summary.architecture;
    if let Some(heads) = summary
        .meta_u64(&keys::attention_head_count(arch))
        .filter(|h| *h > 0)
    {
        return heads;
    }
    match (
        summary.meta_u64(&keys::embedding_length(arch)),
        summary.meta_u64(&keys::attention_key_length(arch)),
    ) {
        (Some(n_embd), Some(head_dim)) if head_dim > 0 => n_embd / head_dim,
        _ => 0,
    }
}

/// A per-architecture rate, or the table's default.
fn rate(table: &[(&str, u64)], arch: &Architecture, fallback: u64) -> u64 {
    table
        .iter()
        .find(|(a, _)| *a == arch.as_str())
        .map(|(_, v)| *v)
        .unwrap_or(fallback)
}

/// The score matrix an unfused attention pass materialises.
///
/// With flash attention the scores are consumed tile by tile and never exist
/// whole; without it the graph holds one entry per (head, cache token, batch
/// token), which dwarfs everything else in the curve.
///
/// The per-entry width is a derived table rather than one number, because the
/// answer genuinely differs by architecture: paired against each cell's own
/// flash-attention-on sibling it is an f32 for every dense and MoE model
/// measured, and effectively nothing for MLA, which shares one latent across
/// heads and so has no per-head score row to materialise. The scalar this
/// replaces charged 8 bytes everywhere and halved that for MLA, which
/// over-reserved dense models twofold and deepseek4 a hundredfold.
///
/// `context` is one stream's share of the cache, not the whole budget: gemma3
/// at four slots measures what one slot measures at a quarter of the context.
fn no_flash_attn_mib(summary: &GgufSummary, context: u32, ubatch: u32, flash_attn: bool) -> u32 {
    if flash_attn {
        return 0;
    }
    let arch = &summary.architecture;
    let heads = query_head_count(summary);
    if heads == 0 {
        return 0;
    }
    let tokens = u64::from(context.min(ubatch));
    // Hundredths of a byte per entry: rounding the rate up to a whole byte
    // inflated a term worth thousands of MiB by up to a fifth.
    let centibytes = rate(
        NO_FLASH_ATTN_SCORE_CENTIBYTES,
        arch,
        NO_FLASH_ATTN_SCORE_CENTIBYTES_DEFAULT,
    );
    let bytes = centibytes * heads * u64::from(context) * tokens / 100;
    (bytes / (1024 * 1024)).min(u64::from(u32::MAX)) as u32
}

/// Does `summary` look like a Gemma 4 E-variant (E4B and siblings)?
/// Detection is keyed on `per_layer_token_embd.weight`, the per-block
/// input-embedding stack that only E-variants carry.
pub(crate) fn is_gemma_e_variant(summary: &GgufSummary) -> bool {
    summary.tensors.contains_key("per_layer_token_embd.weight")
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use ananke_config::placement::SplitMode;
    use ananke_gguf::{GgufValue, types::GgufSummary};
    use smol_str::SmolStr;

    use super::*;
    use crate::types::Fork;

    /// The measured-coverage burden sits with
    /// `tests/estimator_matches_measurements.rs`, which walks every cell in the
    /// dataset against this model. What is worth asserting *here* is the model's
    /// structure — which entry a model resolves to, and how the head card differs from
    /// the others — because those are decisions in this file rather than numbers
    /// in `tuning.json`.
    fn sized_summary(arch: &Architecture, n_embd: u32) -> GgufSummary {
        let mut metadata = BTreeMap::new();
        metadata.insert(keys::embedding_length(arch), GgufValue::U32(n_embd));
        metadata.insert(keys::attention_head_count(arch), GgufValue::U32(32));
        GgufSummary {
            path: Path::new("/model.gguf").to_path_buf(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata,
            block_count: Some(32),
            architecture: arch.clone(),
            shards: Vec::new(),
        }
    }

    fn inputs(context: u32, split: SplitMode, devices: u32) -> EstimatorInputs<'static> {
        EstimatorInputs {
            context,
            name: "demo",
            visible_devices: devices,
            split_mode: split,
            ..EstimatorInputs::empty(Path::new("/fake"))
        }
    }

    #[test]
    fn every_measured_architecture_resolves_to_its_own_entry() {
        // A model whose architecture and split are in the table must not land on
        // the pooled default, which exists for the unmeasured case.
        for arch in [
            "llama",
            "qwen3",
            "qwen35",
            "qwen35moe",
            "gemma3",
            "gemma4",
            "talkie",
        ] {
            let s = sized_summary(&Architecture::from(arch), 4096);
            let got = per_device_for(&s, &inputs(32768, SplitMode::Layer, 2));
            assert!(got > 0, "{arch} produced no reservation");
        }
    }

    #[test]
    fn an_unmeasured_architecture_falls_back_rather_than_failing() {
        let s = sized_summary(&Architecture::Unknown("no-such-arch".into()), 4096);
        assert!(per_device_for(&s, &inputs(32768, SplitMode::Layer, 1)) > 0);
    }

    #[test]
    fn the_head_card_is_charged_more_than_a_secondary() {
        // `per_device_for` returns the head card's total and the packer trims
        // `output_buffer_bytes` back off every other card, so the head extra must
        // be a positive share of it.
        let s = sized_summary(&Architecture::Qwen35Moe, 2048);
        let i = inputs(32768, SplitMode::Layer, 2);
        let head = per_device_for(&s, &i);
        let extra = compute_model::head_extra_mib(&s, &i);
        assert!(
            extra > 0,
            "the head card must carry something the others do not"
        );
        assert!(
            extra < head,
            "the head extra cannot exceed the whole reservation"
        );
    }

    #[test]
    fn the_two_splits_do_not_share_a_reservation() {
        // Measured: at ctx 32768 gemma4 needs 212 MiB per device sharded against
        // 337 layer-split, and laguna 166 against 558. A single number for both
        // would have to be wrong for one of them.
        let s = sized_summary(&Architecture::Gemma4, 4096);
        let layer = per_device_for(&s, &inputs(32768, SplitMode::Layer, 2));
        let tensor = per_device_for(&s, &inputs(32768, SplitMode::Tensor, 2));
        assert_ne!(layer, tensor);
    }

    #[test]
    fn a_longer_context_never_reserves_less() {
        let s = sized_summary(&Architecture::Qwen3, 4096);
        let mut previous = 0;
        for context in [0, 8192, 32768, 131072, 524288] {
            let got = per_device_for(&s, &inputs(context, SplitMode::Layer, 2));
            assert!(
                got >= previous,
                "ctx {context} reserved {got} against {previous} at the shorter context"
            );
            previous = got;
        }
    }

    #[test]
    fn a_runtime_specific_entry_wins_over_a_general_one() {
        // The two forks build different graphs for the same architecture, so an
        // ik-fitted entry must take precedence where one exists.
        let s = sized_summary(&Architecture::Qwen35Moe, 2048);
        let mut ik = inputs(32768, SplitMode::Layer, 2);
        ik.fork = Fork::Ik { dsa: false };
        let mainline = per_device_for(&s, &inputs(32768, SplitMode::Layer, 2));
        assert_ne!(per_device_for(&s, &ik), mainline);
    }

    #[test]
    fn the_gemma_e_variant_has_its_own_entry() {
        // One architecture string, two graphs: the E-variants keep their
        // per-layer embeddings on the host.
        let plain = sized_summary(&Architecture::Gemma4, 4096);
        let mut e = sized_summary(&Architecture::Gemma4, 4096);
        e.tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            ananke_gguf::types::GgufTensor {
                name: SmolStr::new("per_layer_token_embd.weight"),
                dtype: ananke_gguf::types::GgufType::F16,
                shape: vec![256, 4096],
                byte_size: 2 * 256 * 4096,
                shard_idx: 0,
                offset: 0,
            },
        );
        assert!(is_gemma_e_variant(&e));
        assert!(!is_gemma_e_variant(&plain));
        let i = inputs(262144, SplitMode::Layer, 2);
        assert_ne!(per_device_for(&e, &i), per_device_for(&plain, &i));
    }

    #[test]
    fn an_operator_override_wins_outright() {
        let s = sized_summary(&Architecture::Qwen3, 4096);
        let mut i = inputs(32768, SplitMode::Layer, 2);
        i.compute_buffer_mb = Some(1234);
        assert_eq!(per_device_for(&s, &i), 1234);
    }
}
