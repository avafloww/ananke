//! Per-architecture compute-buffer sizing.
//!
//! The packer multiplies this number by the count of active devices
//! (GPUs + CPU when token embeddings land there) and adds it to the
//! per-device reservation. It's the only term in the estimate that
//! reflects *unmodelled* overhead — CUDA context + cuBLAS workspace +
//! attention scratch + KV ring buffers, all lumped together.
//!
//! Per-architecture tuning exists because the overhead scales with
//! different knobs on different architectures:
//!
//! - Dense large-hidden (gemma3/gemma4 with hidden ≥ 4k): attention
//!   scratch grows quickly with context.
//! - Dense standard-hidden (llama, qwen3 < 5k): slower context scaling.
//! - MoE (gpt-oss, mixtral, qwen3moe): compute buffer is almost flat
//!   because only a few experts run per token.
//! - Hybrid MoE + SSM (qwen35moe): SSM state scales with `ssm_d_*`
//!   constants, not context, so the slope is small.
//! - Gemma 4 E-variants (detected by `per_layer_token_embd.weight`):
//!   small hidden + per-layer embeddings on CPU; fits under a much
//!   lower curve than the fat-model gemma4 default.
//!
//! Operators can still override per service via
//! `estimation.compute_buffer_mb`, which short-circuits this table.

use crate::{
    estimator::tuning::{
        CURVES, DEFAULT_CURVE, DEFAULT_UBATCH, NO_FLASH_ATTN_COMPUTE_HEAD_FACTOR,
        QUANTISED_KV_COMPUTE_BYTES_PER_CTX_TOKEN,
    },
    gguf::GgufSummary,
};

/// Per-architecture knobs: `base + slope × (ctx / 1024)` MiB per device.
#[derive(Debug, Clone, Copy)]
struct Tuning {
    base: u32,
    base_batch: u32,
    slope: u32,
}

/// The curve for this model, from the generated table.
///
/// The architecture-to-curve mapping lives in `tuning.json` rather than here:
/// which curve an architecture gets is data, so adding one is a JSON edit and
/// arrives with its evidence attached, and an architecture nobody has
/// measured says so in its own comment instead of looking like every other
/// row.
fn tuning_for(summary: &GgufSummary, ubatch: u32) -> Tuning {
    let arch = summary.architecture.as_str();
    let variant = variant_of(summary);
    let curve = CURVES
        .iter()
        .find(|c| c.archs.contains(&arch) && c.variant == variant)
        .unwrap_or(&DEFAULT_CURVE);

    // A slope that scales with ubatch is calibrated at 512 and taken linear
    // off that point, floored at 1 so a tiny batch still reserves a non-zero
    // context term.
    let slope = if curve.slope_scales_with_ubatch {
        let scaled = (curve.slope_mib_per_1k as u64 * ubatch.max(1) as u64) / DEFAULT_UBATCH as u64;
        scaled.max(1) as u32
    } else {
        curve.slope_mib_per_1k
    };
    Tuning {
        base: curve.base_mib,
        base_batch: curve.base_batch_mib,
        slope,
    }
}

/// Which variant of an architecture this model is, where one architecture
/// string covers models whose graphs differ enough to need separate curves.
///
/// `None` matches an entry with no variant, so a general entry still applies
/// to a model that is not any special variant.
fn variant_of(summary: &GgufSummary) -> Option<&'static str> {
    is_gemma_e_variant(summary).then_some("gemma_e")
}

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
/// delta on Laguna (2×3090, ub2048) was ~660 MiB, comfortably above this
/// `× 2` figure, confirming the direction. `n_vocab` is read from the output
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

/// Default per-device compute-buffer reservation for `summary` at
/// `context` tokens and the service's physical batch size. `ubatch = None`
/// (or an unset config) means llama.cpp's [`DEFAULT_UBATCH`]. Operators can
/// override the whole term per service via `estimation.compute_buffer_mb`.
/// `ubatch` only affects the deepseek4 curve.
pub fn default_for(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
) -> u32 {
    default_for_streams(summary, context, ubatch, flash_attn, 1)
}

/// As [`default_for`], for a service whose KV cache is split across slots.
///
/// `streams` is the number of separate caches — the slot count unless they
/// share a unified one. It affects only the unfused-attention term, whose
/// score matrix is built against one sequence's share of the context:
/// gemma-3-27B and Qwen3.6-27B both measure 3.4x less of it at four slots
/// than at one, on two cards, with the card count and everything else held
/// constant.
pub fn default_for_streams(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
    streams: u32,
) -> u32 {
    default_for_inputs(summary, context, ubatch, flash_attn, streams, false)
}

/// As [`default_for_streams`], plus the allowance a quantised KV cache needs.
///
/// A quantised cache costs the *compute* buffer as well as the pinned host
/// one: paired cells differing in nothing else measure up to 394 MiB more per
/// device at ctx 65536, 1.86x on gemma3 and 1.95x on qwen3. It used to be
/// absorbed into the curves, which are fitted from the worst cell at each
/// context and did not key on cache type — so one quantised cell set the
/// slope and every f16 service paid it.
pub fn default_for_inputs(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
    streams: u32,
    quantised_kv: bool,
) -> u32 {
    let batch = ubatch.unwrap_or(DEFAULT_UBATCH);
    let t = tuning_for(summary, batch);
    // The batch-scaling constant, off the 512-token calibration point. It is
    // separate from `base` because it is not flat — 357 MiB on gemma3 at
    // ubatch 512 and four times that at 2048 — and separate from `slope`
    // because it does not grow with context.
    let batch_term =
        (u64::from(t.base_batch) * u64::from(batch.max(1)) / u64::from(DEFAULT_UBATCH)) as u32;
    let quantised = if quantised_kv {
        let bytes = QUANTISED_KV_COMPUTE_BYTES_PER_CTX_TOKEN
            * u64::from(context)
            * u64::from(context.min(batch));
        (bytes / (1024 * 1024)).min(u64::from(u32::MAX)) as u32
    } else {
        0
    };
    t.base
        .saturating_add(batch_term)
        .saturating_add(t.slope.saturating_mul(context / 1024))
        .saturating_add(no_flash_attn_mib(
            summary,
            context / streams.max(1),
            batch,
            flash_attn,
        ))
        .saturating_add(quantised)
}

/// The score matrix an unfused attention pass materialises.
///
/// With flash attention the scores are consumed tile by tile and never exist
/// whole; without it the graph holds `n_head x n_kv x n_tokens` f32 entries,
/// which dwarfs everything else in the curve — measured at ten times the
/// reserved figure at ub 2048, in the direction that OOMs a load.
fn no_flash_attn_mib(summary: &GgufSummary, context: u32, ubatch: u32, flash_attn: bool) -> u32 {
    if flash_attn {
        return 0;
    }
    let arch = summary.architecture.as_str();
    let heads = summary
        .metadata
        .get(&smol_str::SmolStr::new(format!(
            "{arch}.attention.head_count"
        )))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;
    if heads == 0 {
        return 0;
    }
    let tokens = u64::from(context.min(ubatch));
    // Halved for MLA, which shares one latent across heads rather than
    // materialising a score row per head: normalised by `n_head x ctx x
    // n_tokens x 4`, every dense and MoE architecture measures 1.07-1.88 while
    // deepseek4 measures 0.49. Charging it the same factor as the rest is what
    // made its flash-attention-off cell reserve 12066 MiB against the 2435 the
    // runtime took — the largest over-reservation left in the table.
    let per_head = if crate::estimator::host_buffer::is_mla(arch) {
        u64::from(NO_FLASH_ATTN_COMPUTE_HEAD_FACTOR) / 2
    } else {
        u64::from(NO_FLASH_ATTN_COMPUTE_HEAD_FACTOR)
    };
    let bytes = per_head * heads * u64::from(context) * tokens * 4;
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
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType};

    fn summary_for(arch: &str) -> GgufSummary {
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata: BTreeMap::new(),
            block_count: None,
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn gemma4_e_variant_summary() -> GgufSummary {
        let mut s = summary_for("gemma4");
        s.tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            GgufTensor {
                name: SmolStr::new("per_layer_token_embd.weight"),
                dtype: GgufType::F32,
                shape: vec![1, 1],
                byte_size: 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );
        s
    }

    /// The default curve is `base + slope * (ctx / 1024)`, whatever those are.
    ///
    /// Asserted against the generated table rather than against literals: the
    /// values come from the measurement dataset now, so a test that spelled
    /// them out would have to be edited every time the data was extended, and
    /// would be asserting the generator rather than the arithmetic.
    #[test]
    fn llama_family_default_tuning() {
        // An architecture with no entry of its own, so it lands on the default
        // curve. qwen3 used to serve here and no longer does: it has its own.
        let s = summary_for("brand-new-arch");
        // `base` here is the flat term plus the batch term, which is charged
        // in full at the default 512-token batch; only the slope varies with
        // the context.
        let base = DEFAULT_CURVE.base_mib + DEFAULT_CURVE.base_batch_mib;
        let slope = DEFAULT_CURVE.slope_mib_per_1k;
        assert_eq!(default_for(&s, 2048, None, true), base + slope * 2);
        assert_eq!(default_for(&s, 32768, None, true), base + slope * 32);
    }

    /// gemma's curve is steeper than the llama default, though it no longer
    /// starts higher.
    ///
    /// It used to, and the belief that it must was inherited: gemma's
    /// full-attention layers were thought to need a large scratch even at
    /// small context. Fitting both to the dataset says otherwise — gemma
    /// starts *below* the default and overtakes it, because what it actually
    /// has is a steeper context term, not a bigger floor. The claim is kept,
    /// corrected, rather than deleted, since the crossover is the real
    /// behaviour and worth guarding.
    #[test]
    fn gemma_family_no_longer_needs_a_steeper_curve() {
        // It used to, and the belief was an artifact. The curves are fitted
        // from the worst cell at each context and did not key on cache type,
        // and a quantised KV cache costs the compute buffer up to 394 MiB more
        // at ctx 65536 — so the gemma family's apparent steepness was one
        // quantised cell setting the slope. Fitted on f16 cells alone, gemma
        // and the llama default have the *same* slope, and gemma's base is
        // lower.
        //
        // Kept as a guard rather than deleted: if a future refit makes gemma
        // steeper again, that is worth knowing rather than absorbing.
        let gemma = summary_for("gemma4");
        // The default curve, via an architecture with no entry of its own:
        // qwen3 has had one since it stopped sharing the default.
        let llama = summary_for("brand-new-arch");
        let slope_of =
            |s: &GgufSummary| default_for(s, 65536, None, true) - default_for(s, 0, None, true);
        // Within a unit of each other, not identical. They were identical at
        // one point and are 7 against 6 after later refits, and pinning the
        // equality made a test of the artifact's *absence* fail whenever the
        // fit moved at all. What is worth guarding is that gemma no longer
        // needs a curve several times steeper, which is what the quantised
        // cells made it look like.
        let (gemma_slope, llama_slope) = (slope_of(&gemma), slope_of(&llama));
        let ratio = f64::from(gemma_slope.max(llama_slope))
            / f64::from(gemma_slope.min(llama_slope).max(1));
        assert!(
            ratio < 1.5,
            "gemma's slope should be within half again of the default's, not \
             several times it (gemma {gemma_slope}, default {llama_slope})"
        );
    }

    #[test]
    fn gemma4_e_variant_uses_smaller_curve() {
        // E-variants ship a `per_layer_token_embd.weight` tensor and have a
        // small hidden size. This entry was the one curve nobody derived — a
        // hand-set 1100/7 that stayed put while every other curve moved, which
        // briefly put it *above* the general gemma curve once that one was
        // corrected. It is now fitted from the E-variant's own cells.
        let regular = default_for(&summary_for("gemma4"), 262144, None, true);
        let e_variant = default_for(&gemma4_e_variant_summary(), 262144, None, true);
        assert!(
            e_variant < regular,
            "E-variant cb should be strictly lower than regular gemma4 \
             (e={e_variant} regular={regular})"
        );
    }

    #[test]
    fn moe_tuning_is_flatter_than_dense() {
        let dense_32k = default_for(&summary_for("qwen3"), 32768, None, true);
        let moe_32k = default_for(&summary_for("gpt-oss"), 32768, None, true);
        assert!(
            moe_32k < dense_32k,
            "MoE compute buffer should be flatter than dense at long context; \
             dense={dense_32k} moe={moe_32k}"
        );
    }

    #[test]
    fn qwen35moe_sits_between_moe_only_and_dense() {
        // Hybrid SSM+MoE: full-attention layers are a minority but they
        // do cost more per 1k context than pure-MoE's near-flat curve.
        let moe_only_262k = default_for(&summary_for("gpt-oss"), 262144, None, true);
        let qwen35moe_262k = default_for(&summary_for("qwen35moe"), 262144, None, true);
        let dense_262k = default_for(&summary_for("qwen3"), 262144, None, true);
        assert!(
            moe_only_262k <= qwen35moe_262k && qwen35moe_262k <= dense_262k,
            "qwen35moe should land between MoE-only and dense at 262k \
             (moe={moe_only_262k} qwen35moe={qwen35moe_262k} dense={dense_262k})"
        );
    }

    #[test]
    fn talkie_is_tighter_than_llama_default_and_covers_measured() {
        // The talkie curve was calibrated against a single-GPU sweep whose
        // residual compute buffer stayed ~414-428 MiB across 2048..16384.
        // It must (a) stay strictly below the conservative dense default it
        // would otherwise inherit, and (b) still cover the measured peak
        // (~428 MiB warmed) at the model's native 2048 context.
        let talkie_2k = default_for(&summary_for("talkie"), 2048, None, true);
        // The default curve, via an architecture with no entry of its own.
        let llama_2k = default_for(&summary_for("brand-new-arch"), 2048, None, true);
        assert!(
            talkie_2k < llama_2k,
            "talkie cb should be tighter than the dense default \
             (talkie={talkie_2k} llama={llama_2k})"
        );
        assert!(
            talkie_2k >= 440,
            "talkie cb at 2048 must cover the measured ~428 MiB peak with headroom \
             (got {talkie_2k})"
        );
    }

    #[test]
    fn talkie_floors_to_base() {
        let curve = CURVES
            .iter()
            .find(|c| c.archs.contains(&"talkie"))
            .expect("talkie has a curve");
        assert_eq!(
            default_for(&summary_for("talkie"), 0, None, true),
            curve.base_mib + curve.base_batch_mib
        );
    }

    #[test]
    fn glm_dsa_covers_measured_dsa_compute() {
        // Recalibrated for the ik `-dsa` path (2026-07-23): the head-GPU
        // compute buffer measured 4578 MiB at 131072, ub2048. The DSA indexer
        // scratch scales with context, so the curve must (a) cover the 4578
        // measurement at 131072 with headroom, and (b) still stay below
        // deepseek4. That ordering no longer holds and the assertion is gone:
        // the campaign measured deepseek4's per-device compute flat at 1976 MiB
        // from ctx 8192 to 131072, so glm-dsa's DSA scratch is the steeper of
        // the two.
        let glm = summary_for("glm-dsa");
        let glm_base = CURVES
            .iter()
            .find(|c| c.archs.contains(&"glm-dsa"))
            .expect("glm-dsa has a curve")
            .base_mib;
        assert!(
            default_for(&glm, 131072, None, true) >= 4578,
            "must cover the measured 4578 MiB -dsa compute at 131072 (got {})",
            default_for(&glm, 131072, None, true)
        );
        // The context term scales with batch here as everywhere else. It did
        // not until the campaign measured the compute buffer proportional to
        // ubatch on every architecture it could compare — Magidonia at 388 MiB
        // and 1552 for ub 512 and 2048, Qwen3.6-27B at 290 and 1160 — which a
        // flat slope under-reserves fourfold.
        assert_eq!(
            default_for(&glm, 131072, Some(2048), true) - glm_base,
            (default_for(&glm, 131072, Some(512), true) - glm_base) * 4
        );
    }

    #[test]
    fn deepseek4_covers_measured_indexer_buffer() {
        // The curve this replaces charged 66 MiB per 1k of context on the
        // premise that the NSA indexer scales with ubatch × context. The
        // campaign falsified it: `indexer.top_k` is 512, a fixed working set,
        // and the primary device's compute buffer measures 1976 MiB at ctx
        // 8192, 32768, 65536, and 131072 alike — identical to the megabyte
        // across a sixteenfold range, and 1976/1984/2001 across ubatch
        // 512/1024/2048. What does scale with context lives on the *secondary*
        // device, at roughly 1 MiB per 1k, which is the slope now fitted.
        //
        // The worst per-device need over the whole sweep is 2401 MiB, compute
        // plus its unaccounted remainder.
        let ds4 = summary_for("deepseek4");
        for context in [8192, 32768, 65536, 131072] {
            assert!(
                default_for(&ds4, context, None, true) >= 2401,
                "must cover the measured 2401 MiB worst per-device need at \
                 ctx {context} (got {})",
                default_for(&ds4, context, None, true)
            );
        }
        // And it must not run away with context the way its predecessor did,
        // which reserved 10348 MiB at 131072 for a buffer that had not moved.
        assert!(default_for(&ds4, 131072, None, true) < 2 * default_for(&ds4, 8192, None, true));
    }

    #[test]
    fn deepseek4_compute_buffer_scales_with_ubatch() {
        let ds4 = summary_for("deepseek4");
        // Unset ubatch (None) resolves to llama.cpp's default of 512.
        assert_eq!(
            default_for(&ds4, 131072, None, true),
            default_for(&ds4, 131072, Some(512), true)
        );
        let base = CURVES
            .iter()
            .find(|c| c.archs.contains(&"deepseek4"))
            .expect("deepseek4 has a curve entry")
            .base_mib;
        let slope_at = |ub| default_for(&ds4, 131072, Some(ub), true) - base;
        // The context-scaling term is linear in ubatch off the 512 baseline.
        // Checked at 131072 because the fitted slope is 1 MiB per 1k: at a
        // shorter context the halving case lands inside integer rounding.
        assert_eq!(slope_at(1024), slope_at(512) * 2);
        assert_eq!(slope_at(2048), slope_at(512) * 4);
        // And so does every other architecture, for the same reason.
        let qwen = summary_for("qwen3");
        let qwen_base = CURVES
            .iter()
            .find(|c| c.archs.contains(&"qwen3"))
            .map_or(DEFAULT_CURVE.base_mib, |c| c.base_mib);
        assert_eq!(
            default_for(&qwen, 131072, Some(2048), true) - qwen_base,
            (default_for(&qwen, 131072, None, true) - qwen_base) * 4
        );
    }

    #[test]
    fn unknown_arch_falls_back_to_llama_default() {
        // Matches the conservative dense-family curve so unknown archs
        // that slip through the fallback still over-reserve safely.
        assert_eq!(
            default_for(&summary_for("brand-new-arch"), 8192, None, true),
            DEFAULT_CURVE.base_mib
                + DEFAULT_CURVE.base_batch_mib
                + DEFAULT_CURVE.slope_mib_per_1k * 8
        );
    }

    #[test]
    fn absent_context_floors_to_base() {
        // The floor at zero context is the flat base *plus* the batch term,
        // which is charged in full at the 512-token calibration batch. Only
        // the context-scaling part disappears.
        let base_of = |arch: &str| {
            CURVES
                .iter()
                .find(|c| c.archs.contains(&arch) && c.variant.is_none())
                .map_or(DEFAULT_CURVE.base_mib + DEFAULT_CURVE.base_batch_mib, |c| {
                    c.base_mib + c.base_batch_mib
                })
        };
        assert_eq!(
            default_for(&summary_for("qwen3"), 0, None, true),
            base_of("qwen3")
        );
        assert_eq!(
            default_for(&summary_for("gpt-oss"), 512, None, true),
            base_of("gpt-oss")
        );
        assert_eq!(
            default_for(&summary_for("gemma4"), 0, None, true),
            base_of("gemma4")
        );
        assert_eq!(
            default_for(&summary_for("qwen35moe"), 0, None, true),
            base_of("qwen35moe")
        );
        // Against the entry rather than a literal: the E-variant curve is
        // derived now, so a hardcoded figure fails on every recalibration
        // while checking nothing about the floor.
        let e = CURVES
            .iter()
            .find(|c| c.variant == Some("gemma_e"))
            .expect("the E-variant entry");
        assert_eq!(
            default_for(&gemma4_e_variant_summary(), 0, None, true),
            e.base_mib + e.base_batch_mib
        );
    }
}
