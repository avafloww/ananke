//! Per-device compute-buffer sizing.
//!
//! [`per_device_for`] is the entry point, and it picks between three models,
//! because the graph a device builds depends on how the model was split across
//! devices and on which runtime built it. Each is documented where it lives:
//!
//! - [`tensor_split_per_device`] — `--split-mode tensor`. Built from the model's
//!   hyperparameters: hidden-width intermediates per batch token, an f16 KQ
//!   mask, and a dequantisation term. Charged to *every* spanned GPU, since
//!   llama.cpp builds the same graph on each rather than dividing one.
//! - [`ik_layer_split_per_device`] — a layer split on ik_llama, where the fork
//!   has been measured. Also from hyperparameters, but with a batch term that
//!   steps by a constant per doubling above the fork's attention chunk — a shape
//!   no affine curve can express.
//! - [`layer_split_per_device`] — a layer split otherwise, from the fitted
//!   per-architecture curves in `tuning.json`. This is the one remaining term
//!   that is a curve rather than a model: `base + base_batch × k + slope ×
//!   ctx/1024`, with `k = ubatch / 512`.
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
//! Operators can still override the whole term per service via
//! `estimation.compute_buffer_mb`.

use crate::{
    config::validate::SplitMode,
    estimator::{
        tuning::{
            CURVES, Curve, DEFAULT_CURVE, DEFAULT_UBATCH, IK_COMPUTE_FLAT_BYTES,
            IK_COMPUTE_FLAT_BYTES_DEFAULT, IK_COMPUTE_PER_BATCH_TOKEN_BYTES,
            IK_COMPUTE_PER_BATCH_TOKEN_BYTES_DEFAULT, IK_COMPUTE_PER_DOUBLING_BYTES,
            IK_COMPUTE_PER_DOUBLING_BYTES_DEFAULT, IK_COMPUTE_PER_TOKEN_PAIR_BYTES,
            IK_COMPUTE_PER_TOKEN_PAIR_BYTES_DEFAULT, NO_FLASH_ATTN_SCORE_BYTES,
            NO_FLASH_ATTN_SCORE_BYTES_DEFAULT, TENSOR_COMPUTE_INTERMEDIATES,
            TENSOR_COMPUTE_INTERMEDIATES_DEFAULT, TENSOR_COMPUTE_QUANTISED_RATE_DEFAULT,
            TENSOR_COMPUTE_QUANTISED_RATES, TENSOR_COMPUTE_SHADOW_BYTES,
            TENSOR_COMPUTE_SHADOW_BYTES_DEFAULT, TENSOR_MASK_BYTES_PER_TOKEN_PAIR,
        },
        types::EstimatorInputs,
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
fn tuning_for(summary: &GgufSummary, ik_llama: bool) -> Tuning {
    let arch = summary.architecture.as_str();
    let variant = variant_of(summary);
    // A fork-specific entry wins over a general one for the same architecture:
    // the two runtimes build different graphs, and where the fork has been
    // measured its own fit is the better answer. Falling back to the general
    // entry is deliberate — most architectures have no fork cells at all.
    let matches = |c: &&Curve| c.archs.contains(&arch) && c.variant == variant;
    let runtime = ik_llama.then_some("ik");
    let curve = CURVES
        .iter()
        .find(|c| matches(c) && c.runtime == runtime)
        .or_else(|| CURVES.iter().find(|c| matches(c) && c.runtime.is_none()))
        .unwrap_or(&DEFAULT_CURVE);

    // The slope does not scale with ubatch. The data confirms this: at
    // ub=2048 (k=4), no architecture's compute buffer grows 4× — the
    // highest ratio is 2.58× (llama) and most are under 2×. The
    // batch-dependent growth is carried entirely by `base_batch`, which
    // scales linearly with `ub/512`.
    let slope = curve.slope_mib_per_1k;
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

/// As [`default_for_streams`], for callers that need to distinguish a quantised
/// KV cache. The quantised-KV compute-buffer allowance was removed: the
/// compute-buffer curves are fitted from the worst (quantised) cell at each
/// context, so the allowance was already absorbed into the slope and charging
/// it again double-counted. A quantised cache *reduces* total VRAM (the
/// smaller KV outweighs the small compute-buffer increase), so an additional
/// GPU charge was the wrong direction.
pub fn default_for_inputs(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
    streams: u32,
    _quantised_kv: bool,
) -> u32 {
    layer_split_per_device(summary, context, ubatch, flash_attn, streams, false)
}

/// Per-device compute buffer, choosing the model that matches how llama.cpp
/// will split the model across devices.
///
/// The two splits build genuinely different graphs and the gap is not a scale
/// factor: at ctx 32768 gemma4 needs 212 MiB per device sharded against 337
/// layer-split, while laguna needs 166 against 558. Neither figure predicts the
/// other, so each split has its own model — and the sharded packer charges the
/// result to *every* spanned GPU rather than dividing it, because llama.cpp
/// builds the same graph on each device. The measured compute column reads
/// identically on one card and on two at every context in the dataset.
pub fn per_device_for(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u32 {
    if let Some(mb) = inputs.compute_buffer_mb {
        return mb;
    }
    let flash_attn = inputs.flash_attn.unwrap_or(true);
    match inputs.split_mode {
        // Note that the tensor model is fitted entirely on mainline cells: the
        // dataset holds 160 mainline tensor-split cells and *no* ik ones, so a
        // fork service configured with `--split-mode tensor` is estimated from
        // the other runtime's graph. Nothing in the data says whether that is
        // close; measuring one fork tensor cell would say.
        SplitMode::Tensor | SplitMode::Row => tensor_split_per_device(
            summary,
            inputs.context,
            inputs.ubatch,
            flash_attn,
            inputs.streams(),
            crate::estimator::host_buffer::quantised_kv(inputs),
        ),
        SplitMode::Layer => layer_split_per_device(
            summary,
            inputs.context,
            inputs.ubatch,
            flash_attn,
            inputs.streams(),
            inputs.ik_llama,
        ),
    }
}

/// Per-device compute buffer under `--split-mode tensor`, from the model's own
/// hyperparameters rather than from a fitted curve.
///
/// Three terms, each measured to be exactly what it claims:
///
/// - **Graph intermediates.** `K × n_embd × 4` bytes per batch token, where `K`
///   counts the hidden-width f32 buffers the graph holds live. Flat in context
///   and exactly proportional to the batch — laguna measures 67, 134, 270, and
///   544 MiB of it at ubatch 256, 512, 1024, and 2048. `K` is the one fitted
///   quantity, dimensionless and near-integral (12 on talkie, 15 on qwen3, 21
///   on gemma4, 22 on llama), so it transfers across model sizes within a
///   family instead of being a curve to extrapolate.
/// - **The KQ mask.** Two bytes — one f16 entry — per (batch token, cache
///   token). Read from the graph, not fitted, and it is why every architecture
///   in the set grows by exactly 1.00 MiB per 1024 cache tokens at ubatch 512.
///   The cache in question is one *stream's*, so `--kv-unified` widens it and
///   `--parallel` without unification divides it.
/// - **Quantised-cache dequantisation.** Zero on an f16 cache. It follows the
///   **full** context rather than one stream's share: Qwen3-4B at ctx 32768
///   shows the same absolute increase at one slot as at four, where a term
///   following the mask would have quartered.
///
/// Reproduces all three production tensor-split cells within 37 to 60 MiB per
/// device, over-reserving in every case.
pub fn tensor_split_per_device(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
    streams: u32,
    quantised_kv: bool,
) -> u32 {
    let arch = summary.architecture.as_str();
    let batch = u64::from(ubatch.unwrap_or(DEFAULT_UBATCH).max(1));
    let n_embd = summary
        .metadata
        .get(&smol_str::SmolStr::new(format!("{arch}.embedding_length")))
        .and_then(|v| v.as_u32())
        .map(u64::from)
        .unwrap_or(0);
    let intermediates = rate(
        TENSOR_COMPUTE_INTERMEDIATES,
        arch,
        TENSOR_COMPUTE_INTERMEDIATES_DEFAULT,
    );
    let quantised_rate = if quantised_kv {
        rate(
            TENSOR_COMPUTE_QUANTISED_RATES,
            arch,
            TENSOR_COMPUTE_QUANTISED_RATE_DEFAULT,
        )
    } else {
        0
    };
    let n_kv = u64::from(context / streams.max(1));
    let per_token = intermediates * n_embd * F32_BYTES
        + TENSOR_MASK_BYTES_PER_TOKEN_PAIR * n_kv
        + quantised_rate * u64::from(context);
    let shadow = rate(
        TENSOR_COMPUTE_SHADOW_BYTES,
        arch,
        TENSOR_COMPUTE_SHADOW_BYTES_DEFAULT,
    );
    let bytes = batch * per_token + shadow;
    let mib = (bytes / (1024 * 1024)).min(u64::from(u32::MAX)) as u32;
    // The unfused-attention score matrix is the same quantity under either
    // split and dwarfs everything above when it exists at all.
    mib.saturating_add(no_flash_attn_mib(
        summary,
        context / streams.max(1),
        batch as u32,
        flash_attn,
    ))
}

/// The per-architecture rate for `arch`, or the fallback.
fn rate(table: &[(&str, u64)], arch: &str, fallback: u64) -> u64 {
    table
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, v)| *v)
        .unwrap_or(fallback)
}

const F32_BYTES: u64 = 4;

/// ik's attention chunk: the batch above which its graph stops handling the
/// whole micro-batch in one pass. This is the fork's `-amb` default, and the
/// GLM-5.2 service passes the same value explicitly. A service that overrode
/// `-amb` would need it read rather than assumed — ananke does not model the
/// flag today, so nothing can.
const IK_ATTENTION_CHUNK: u32 = 512;

/// Per-device compute buffer for a layer split on ik_llama, or `None` for an
/// architecture the fork has not been measured on.
///
/// The fork's buffer has a shape the affine curves cannot express:
///
/// ```text
/// flat + per_batch_token × min(ubatch, chunk)
///      + per_doubling × log2(max(ubatch, chunk) / chunk)
///      + per_token_pair × ubatch × n_kv
/// ```
///
/// The third term is the one that matters. Above the attention chunk the buffer
/// grows by a *constant per doubling* of the batch rather than in proportion to
/// it: Qwen3.6-35B-A3B measures 719, 1227, and 1732 MiB at ubatch 512, 1024, and
/// 2048 — two equal steps of ~505 — so a curve fitted at 512 and scaled by
/// `ubatch/512` misses the 2048 cells by 30 to 40%. Below the chunk it is
/// linear, which ubatch 256's 586 MiB pins.
///
/// The last term is the mask, and its width separates the architectures: one
/// byte per (batch token, cache token) for qwen35moe's full attention, three for
/// laguna's sliding window, nineteen for glm-dsa — whose sparse-attention
/// indexer scores every cache token and so carries far more than a mask.
///
/// Fitted across thirteen two-card cells for qwen35moe with a worst residual of
/// 5 MiB. `None` falls back to the general curve rather than borrowing another
/// architecture's rates.
fn ik_layer_split_per_device(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    streams: u32,
) -> Option<u32> {
    let arch = summary.architecture.as_str();
    let flat = rate(IK_COMPUTE_FLAT_BYTES, arch, IK_COMPUTE_FLAT_BYTES_DEFAULT);
    if flat == 0 {
        return None;
    }
    let batch = u64::from(ubatch.unwrap_or(DEFAULT_UBATCH).max(1));
    let chunk = u64::from(IK_ATTENTION_CHUNK);
    let doublings = (batch.max(chunk) / chunk).ilog2();
    let n_kv = u64::from(context / streams.max(1));
    let bytes = flat
        + rate(
            IK_COMPUTE_PER_BATCH_TOKEN_BYTES,
            arch,
            IK_COMPUTE_PER_BATCH_TOKEN_BYTES_DEFAULT,
        ) * batch.min(chunk)
        + rate(
            IK_COMPUTE_PER_DOUBLING_BYTES,
            arch,
            IK_COMPUTE_PER_DOUBLING_BYTES_DEFAULT,
        ) * u64::from(doublings)
        + rate(
            IK_COMPUTE_PER_TOKEN_PAIR_BYTES,
            arch,
            IK_COMPUTE_PER_TOKEN_PAIR_BYTES_DEFAULT,
        ) * batch
            * n_kv;
    Some((bytes / (1024 * 1024)).min(u64::from(u32::MAX)) as u32)
}

/// Per-device compute buffer under a layer split, from the fitted curves.
fn layer_split_per_device(
    summary: &GgufSummary,
    context: u32,
    ubatch: Option<u32>,
    flash_attn: bool,
    streams: u32,
    ik_llama: bool,
) -> u32 {
    // The fork has its own model where it has been measured; the curves
    // describe mainline's graph.
    if ik_llama && let Some(mib) = ik_layer_split_per_device(summary, context, ubatch, streams) {
        return mib.saturating_add(no_flash_attn_mib(
            summary,
            context / streams.max(1),
            ubatch.unwrap_or(DEFAULT_UBATCH),
            flash_attn,
        ));
    }
    let batch = ubatch.unwrap_or(DEFAULT_UBATCH);
    let t = tuning_for(summary, ik_llama);
    // The batch-scaling constant, off the 512-token calibration point. It is
    // separate from `base` because it is not flat — 357 MiB on gemma3 at
    // ubatch 512 and four times that at 2048 — and separate from `slope`
    // because it does not grow with context.
    let batch_term =
        (u64::from(t.base_batch) * u64::from(batch.max(1)) / u64::from(DEFAULT_UBATCH)) as u32;
    t.base
        .saturating_add(batch_term)
        .saturating_add(t.slope.saturating_mul(context / 1024))
        .saturating_add(no_flash_attn_mib(
            summary,
            context / streams.max(1),
            batch,
            flash_attn,
        ))
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
    let per_entry = rate(
        NO_FLASH_ATTN_SCORE_BYTES,
        arch,
        NO_FLASH_ATTN_SCORE_BYTES_DEFAULT,
    );
    let bytes = per_entry * heads * u64::from(context) * tokens;
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
        // Within a factor of three, not several times it. The ratio widened
        // when the curves started including quantised-cache cells (which
        // grow with context on gemma3) rather than f16-only — that is
        // correct, not an artifact: the curve must cover the worst cell.
        let (gemma_slope, llama_slope) = (slope_of(&gemma), slope_of(&llama));
        let ratio = f64::from(gemma_slope.max(llama_slope))
            / f64::from(gemma_slope.min(llama_slope).max(1));
        assert!(
            ratio < 3.0,
            "gemma's slope should be within 3x of the default's, not \
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

    /// A summary carrying just the hidden size, which is all the tensor-split
    /// compute model reads from the file.
    fn sized_summary(arch: &str, n_embd: u32) -> GgufSummary {
        let mut summary = summary_for(arch);
        summary.metadata.insert(
            SmolStr::new(format!("{arch}.embedding_length")),
            crate::gguf::types::GgufValue::U32(n_embd),
        );
        summary
    }

    /// Every production tensor-split cell, against the figure llama.cpp
    /// actually reserved on each card.
    ///
    /// These are the three configurations the model has to get right, and none
    /// of them resembles a calibration cell: two run a quantised cache at a
    /// context an order of magnitude past the sweep, one runs four slots
    /// against a unified cache. Covering them by 37 to 60 MiB is the whole
    /// claim, so it is pinned rather than left to the campaign scripts.
    #[test]
    fn tensor_model_covers_every_production_cell() {
        // (arch, n_embd, context, streams, quantised, measured per-device MiB)
        let cells = [
            // prod-qwen36-35b-a3b: ctx 524288, 2 slots, no unification, q8_0.
            ("qwen35moe", 2048, 524288u32, 2u32, true, 1332u32),
            // prod-qwen36-27b: ctx 360448 after rounding, 2 slots, q8_0.
            ("qwen35", 5120, 360448, 2, true, 1664),
            // prod-gemma4-31b-qat: ctx 240000, 4 slots sharing a unified
            // cache, f16.
            ("gemma4", 5376, 240000, 1, false, 418),
        ];
        for (arch, n_embd, context, streams, quantised, measured) in cells {
            let got = tensor_split_per_device(
                &sized_summary(arch, n_embd),
                context,
                None,
                true,
                streams,
                quantised,
            );
            assert!(
                got >= measured,
                "{arch} at ctx {context} must cover the measured {measured} MiB, got {got}"
            );
            // The shadow term alone is ~360-400 MiB, so a bound much tighter
            // than this would be pinning the shadow rather than the model.
            assert!(
                got <= measured + 512,
                "{arch} at ctx {context} over-reserves: {got} against {measured} MiB"
            );
        }
    }

    /// The context term is the f16 KQ mask and nothing else: two bytes per
    /// (batch token, cache token), which is 1.00 MiB per 1024 cache tokens at
    /// the calibration batch. Every architecture measured grows at exactly
    /// that rate, from talkie's 13B to gemma4's 31B, which is why the term is
    /// read from the graph instead of fitted per architecture.
    #[test]
    fn the_tensor_context_slope_is_one_mib_per_1k_at_the_default_batch() {
        for arch in ["talkie", "qwen3", "qwen35", "gemma4", "llama"] {
            let at = |context: u32| {
                tensor_split_per_device(&sized_summary(arch, 4096), context, None, true, 1, false)
            };
            assert_eq!(
                at(65536) - at(32768),
                32,
                "{arch}: 32768 more cache tokens must cost 32 MiB"
            );
        }
    }

    /// Flat in context and exactly proportional to the batch — laguna measures
    /// 67, 134, 270, and 544 MiB of graph intermediates at ubatch 256, 512,
    /// 1024, and 2048. The mask scales with the batch too, so the whole
    /// per-device figure does, net of the flat shadow.
    #[test]
    fn the_tensor_model_scales_with_the_batch() {
        let s = sized_summary("laguna", 3072);
        let net = |ubatch: u32| {
            let shadow = rate(
                TENSOR_COMPUTE_SHADOW_BYTES,
                "laguna",
                TENSOR_COMPUTE_SHADOW_BYTES_DEFAULT,
            ) / (1024 * 1024);
            u64::from(tensor_split_per_device(
                &s,
                32768,
                Some(ubatch),
                true,
                1,
                false,
            )) - shadow
        };
        assert_eq!(net(1024), net(512) * 2);
        assert_eq!(net(2048), net(512) * 4);
    }

    /// The whole Qwen3.6-35B-A3B grid on ik, in both axes.
    ///
    /// Twelve cells the sweep measured, and the point of them is the batch axis:
    /// 719 → 1227 → 1732 MiB at ubatch 512 → 1024 → 2048 is two *equal* steps,
    /// not growth proportional to the batch. An affine curve scaled by
    /// `ubatch/512` cannot produce that shape at all, whatever its coefficients,
    /// which is why the fork needed its own model rather than its own fit.
    #[test]
    fn the_ik_model_reproduces_the_measured_grid() {
        let s = sized_summary("qwen35moe", 2048);
        // (context, ubatch) → measured per-device MiB, two cards, f16 cache.
        let cells = [
            ((8192u32, 512u32), 719u32),
            ((8192, 1024), 1227),
            ((8192, 2048), 1732),
            ((32768, 256), 586),
            ((32768, 512), 731),
            ((32768, 1024), 1251),
            ((32768, 2048), 1780),
            ((65536, 512), 747),
            ((65536, 1024), 1283),
            ((65536, 2048), 1843),
            ((131072, 512), 779),
            ((131072, 1024), 1337),
            ((131072, 2048), 1971),
        ];
        for ((context, ubatch), measured) in cells {
            let got = ik_layer_split_per_device(&s, context, Some(ubatch), 1)
                .expect("qwen35moe has ik rates");
            let drift = got.abs_diff(measured);
            assert!(
                drift <= 24,
                "ctx {context} ub {ubatch}: modelled {got} against measured \
                 {measured}"
            );
        }
    }

    /// An architecture the fork has not been measured on falls back to the
    /// general curve rather than borrowing another's rates — the mask width
    /// alone spans one byte to nineteen across the three that were measured, so
    /// a borrowed rate would be worse than no rate.
    #[test]
    fn an_unmeasured_architecture_has_no_ik_model() {
        assert!(ik_layer_split_per_device(&sized_summary("llama", 4096), 32768, None, 1).is_none());
        assert!(
            ik_layer_split_per_device(&sized_summary("qwen35moe", 2048), 32768, None, 1).is_some()
        );
    }

    #[test]
    fn moe_tuning_has_lower_slope_than_dense() {
        // MoE-only architectures (gpt-oss, mixtral) have a higher base but
        // lower context slope than dense models: their compute buffer is
        // dominated by the expert routing overhead (flat) rather than the
        // attention score matrix (grows with context). The crossover point
        // depends on the specific architectures; what holds is the slope
        // ordering.
        let moe_curve = CURVES
            .iter()
            .find(|c| c.archs.contains(&"gpt-oss"))
            .expect("gpt-oss has a curve");
        let dense_curve = CURVES
            .iter()
            .find(|c| c.archs.contains(&"qwen3"))
            .expect("qwen3 has a curve");
        assert!(
            moe_curve.slope_mib_per_1k < dense_curve.slope_mib_per_1k,
            "MoE compute buffer slope should be flatter than dense \
             (moe={} dense={})",
            moe_curve.slope_mib_per_1k,
            dense_curve.slope_mib_per_1k,
        );
    }

    #[test]
    fn qwen35moe_has_lower_slope_than_dense() {
        // Hybrid SSM+MoE: full-attention layers are a minority (1/4 of
        // layers), so the context-scaling part of the compute buffer is
        // smaller than a dense model's. The slope is lower than dense.
        let qwen35moe_slope = CURVES
            .iter()
            .find(|c| c.archs.contains(&"qwen35moe"))
            .map(|c| c.slope_mib_per_1k)
            .expect("qwen35moe has a curve");
        let dense_slope = CURVES
            .iter()
            .find(|c| c.archs.contains(&"qwen3"))
            .map(|c| c.slope_mib_per_1k)
            .expect("qwen3 has a curve");
        assert!(
            qwen35moe_slope < dense_slope,
            "qwen35moe slope should be lower than dense \
             (qwen35moe={qwen35moe_slope} dense={dense_slope})"
        );
    }

    #[test]
    fn talkie_covers_measured() {
        // The talkie curve was calibrated against a single-GPU sweep whose
        // residual compute buffer stayed ~414-428 MiB across 2048..16384.
        // It must cover the measured ~428 MiB peak at the model's native
        // 2048 context.
        let talkie_2k = default_for(&summary_for("talkie"), 2048, None, true);
        assert!(
            talkie_2k >= 428,
            "talkie cb at 2048 must cover the measured ~428 MiB peak \
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
        // The context term does not scale with batch — the data shows the
        // compute buffer grows at most 2.58× at ub=2048, not 4×. The
        // batch-dependent growth is carried by `base_batch`.
        assert_eq!(
            default_for(&glm, 131072, Some(2048), true) - glm_base,
            default_for(&glm, 131072, Some(512), true) - glm_base
        );
    }

    #[test]
    fn deepseek4_compute_buffer_is_flat_in_context() {
        // The NSA indexer's compute buffer is flat across context: the
        // primary device measures ~1976 MiB at ctx 8192, 32768, 65536, and
        // 131072 alike. What does scale with context lives on the secondary
        // device, at roughly 1 MiB per 1k. The curve is fitted to the
        // per-run average, so it charges the mean of both devices to each
        // GPU — lower than the primary's peak but matching the total.
        let ds4 = summary_for("deepseek4");
        let cb_8k = default_for(&ds4, 8192, None, true);
        let cb_131k = default_for(&ds4, 131072, None, true);
        // Flat: the ratio should be close to 1.
        assert!(
            cb_131k < 2 * cb_8k,
            "deepseek4 cb must not run away with context \
             (8k={cb_8k} 131k={cb_131k})"
        );
        // The per-run average at ctx 8192 is ~1495 MiB, but the curve base
        // is raised to cover the worst cell at ctx 32768 (~1704 MiB), so
        // the curve over-reserves at short contexts by design. The curve
        // should be within 20% of the measured value — enough to confirm
        // it is in the right neighbourhood without rejecting the coverage
        // requirement that pushes it higher.
        assert!(
            (cb_8k as f64 / 1495.0).abs() - 1.0 < 0.20,
            "deepseek4 cb at 8k should be within 20% of measured 1495 MiB \
             (got {cb_8k})"
        );
    }

    #[test]
    fn deepseek4_compute_buffer_does_not_scale_slope_with_ubatch() {
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
        // The slope does not scale with ubatch — the data shows no
        // architecture's compute buffer grows 4× at ub=2048. The
        // batch-dependent growth is carried by `base_batch`.
        let base_batch = CURVES
            .iter()
            .find(|c| c.archs.contains(&"deepseek4"))
            .expect("deepseek4 has a curve entry")
            .base_batch_mib;
        let slope_at = |ub| {
            let k = ub / 512;
            default_for(&ds4, 131072, Some(ub), true) - base - base_batch * k
        };
        assert_eq!(slope_at(1024), slope_at(512));
        assert_eq!(slope_at(2048), slope_at(512));
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
