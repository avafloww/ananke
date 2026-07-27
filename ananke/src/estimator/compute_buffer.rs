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
    estimator::tuning::{CURVES, DEFAULT_CURVE, DEFAULT_UBATCH},
    gguf::GgufSummary,
};

/// Per-architecture knobs: `base + slope × (ctx / 1024)` MiB per device.
#[derive(Debug, Clone, Copy)]
struct Tuning {
    base: u32,
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
pub fn default_for(summary: &GgufSummary, context: u32, ubatch: Option<u32>) -> u32 {
    let t = tuning_for(summary, ubatch.unwrap_or(DEFAULT_UBATCH));
    t.base
        .saturating_add(t.slope.saturating_mul(context / 1024))
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

    #[test]
    fn llama_family_default_tuning() {
        let s = summary_for("qwen3");
        assert_eq!(default_for(&s, 2048, None), 700 + 8 * 2);
        assert_eq!(default_for(&s, 32768, None), 700 + 8 * 32);
    }

    #[test]
    fn gemma_family_has_higher_base_than_llama_default() {
        // gemma-4-31B's full-attention layers drive a big attention
        // scratch allocation even at small context — the gemma tuning
        // has to start well above the llama default to cover it.
        let gemma_2k = default_for(&summary_for("gemma4"), 2048, None);
        let llama_2k = default_for(&summary_for("qwen3"), 2048, None);
        assert!(
            gemma_2k > llama_2k,
            "gemma base should exceed llama default at 2k (gemma={gemma_2k} llama={llama_2k})"
        );
    }

    #[test]
    fn gemma4_e_variant_uses_smaller_curve() {
        // E-variants ship a `per_layer_token_embd.weight` tensor and
        // have a small hidden size; the fat-model gemma4 tuning over-
        // reserves them by ~2 GiB at 262k otherwise.
        let regular = default_for(&summary_for("gemma4"), 262144, None);
        let e_variant = default_for(&gemma4_e_variant_summary(), 262144, None);
        assert!(
            e_variant < regular,
            "E-variant cb should be strictly lower than regular gemma4 \
             (e={e_variant} regular={regular})"
        );
    }

    #[test]
    fn moe_tuning_is_flatter_than_dense() {
        let dense_32k = default_for(&summary_for("qwen3"), 32768, None);
        let moe_32k = default_for(&summary_for("gpt-oss"), 32768, None);
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
        let moe_only_262k = default_for(&summary_for("gpt-oss"), 262144, None);
        let qwen35moe_262k = default_for(&summary_for("qwen35moe"), 262144, None);
        let dense_262k = default_for(&summary_for("qwen3"), 262144, None);
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
        let talkie_2k = default_for(&summary_for("talkie"), 2048, None);
        let llama_2k = default_for(&summary_for("qwen3"), 2048, None);
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
        assert_eq!(default_for(&summary_for("talkie"), 0, None), 500);
    }

    #[test]
    fn glm_dsa_covers_measured_dsa_compute() {
        // Recalibrated for the ik `-dsa` path (2026-07-23): the head-GPU
        // compute buffer measured 4578 MiB at 131072, ub2048. The DSA indexer
        // scratch scales with context, so the curve must (a) cover the 4578
        // measurement at 131072 with headroom, and (b) still stay below
        // deepseek4, whose NSA indexer scales with ubatch × context and is far
        // steeper.
        let glm = summary_for("glm-dsa");
        assert!(
            default_for(&glm, 131072, None) >= 4578,
            "must cover the measured 4578 MiB -dsa compute at 131072 (got {})",
            default_for(&glm, 131072, None)
        );
        assert!(
            default_for(&glm, 131072, None) < default_for(&summary_for("deepseek4"), 131072, None),
            "glm-dsa's indexer is less steep than deepseek4's NSA curve"
        );
        // ubatch is not a factor for glm-dsa (unlike deepseek4).
        assert_eq!(
            default_for(&glm, 131072, Some(2048)),
            default_for(&glm, 131072, Some(512))
        );
    }

    #[test]
    fn deepseek4_covers_measured_indexer_buffer() {
        // The NSA indexer's prompt scratch is the steepest curve in the
        // table. It must (a) grow faster than every other MoE arch and
        // (b) cover the measured per-card residuals (~9.3 GiB at 131072,
        // ~17.5 GiB at 262144) with a little headroom.
        let ds4 = summary_for("deepseek4");
        let moe = summary_for("gpt-oss");
        assert!(
            default_for(&ds4, 262144, None) > default_for(&moe, 262144, None) * 4,
            "deepseek4 cb must dwarf the flat MoE curve at long context"
        );
        assert!(
            default_for(&ds4, 131072, None) >= 9297,
            "must cover the measured ~9.3 GiB residual at 131072 (got {})",
            default_for(&ds4, 131072, None)
        );
        assert!(
            default_for(&ds4, 262144, None) >= 17519,
            "must cover the measured ~17.5 GiB residual at 262144 (got {})",
            default_for(&ds4, 262144, None)
        );
    }

    #[test]
    fn deepseek4_compute_buffer_scales_with_ubatch() {
        let ds4 = summary_for("deepseek4");
        // Unset ubatch (None) resolves to llama.cpp's default of 512.
        assert_eq!(
            default_for(&ds4, 131072, None),
            default_for(&ds4, 131072, Some(512))
        );
        let base = CURVES
            .iter()
            .find(|c| c.archs.contains(&"deepseek4"))
            .expect("deepseek4 has a curve entry")
            .base_mib;
        let slope_at = |ub| default_for(&ds4, 131072, Some(ub)) - base;
        // The context-scaling term is linear in ubatch off the 512 baseline.
        assert_eq!(slope_at(1024), slope_at(512) * 2);
        assert_eq!(slope_at(2048), slope_at(512) * 4);
        assert_eq!(slope_at(256), slope_at(512) / 2);
        // Every other arch ignores ubatch entirely.
        let qwen = summary_for("qwen3");
        assert_eq!(
            default_for(&qwen, 131072, Some(2048)),
            default_for(&qwen, 131072, None)
        );
    }

    #[test]
    fn unknown_arch_falls_back_to_llama_default() {
        // Matches the conservative dense-family curve so unknown archs
        // that slip through the fallback still over-reserve safely.
        assert_eq!(
            default_for(&summary_for("brand-new-arch"), 8192, None),
            700 + 8 * 8
        );
    }

    #[test]
    fn absent_context_floors_to_base() {
        assert_eq!(default_for(&summary_for("qwen3"), 0, None), 700);
        assert_eq!(default_for(&summary_for("gpt-oss"), 512, None), 600);
        assert_eq!(default_for(&summary_for("gemma4"), 0, None), 2000);
        assert_eq!(default_for(&summary_for("qwen35moe"), 0, None), 900);
        assert_eq!(default_for(&gemma4_e_variant_summary(), 0, None), 1100);
    }
}
