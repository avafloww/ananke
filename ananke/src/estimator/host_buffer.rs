//! Host memory llama-server holds that is neither model weights nor KV cache.
//!
//! Three terms, all present whatever the placement — a fully GPU-offloaded
//! model pays every one of them:
//!
//! - **The pinned graph arena.** ggml assigns every `GGML_TENSOR_FLAG_INPUT`
//!   tensor to the last backend, which is the CPU one, and when any GPU is
//!   present that backend's buffer type is swapped for the first device's
//!   *host* buffer type — `cudaMallocHost`, page-locked and unswappable.
//!   llama.cpp logs it as `CUDA_Host compute buffer size`, not `CPU`.
//! - **The process baseline.** The CUDA runtime's host-side allocations, the
//!   tokenizer, per-slot sampler state, graph metadata. Roughly fixed for a
//!   given model, and not derivable from ggml's source — this is the one term
//!   that had to be measured.
//! - **The prompt cache.** `-cram`, serialized evicted prompts. A *cap* rather
//!   than an allocation: it fills with use, so it is reserved but deliberately
//!   kept out of the rolling correction's base (see `Charge::Slop`).
//!
//! Before this module the `Cpu` slot was charged the GPU-calibrated
//! `compute_buffer_mb` — a number derived from `nvidia-smi` VRAM readings and
//! never measured against a host backend — and the other two terms were not
//! modelled at all.
//!
//! # Calibration
//!
//! The arena's shape is read from llama.cpp's graph construction rather than
//! fitted, then checked against hardware; the baseline is a fit. Measured on
//! 2×RTX 3090 (CUDA) over Qwen3-4B (36 layers, n_embd 2560), Qwen3.6-27B (64,
//! 5120), and Gemma-4-31B-QAT (60, 5376, SWA), sweeping context 8k–64k, ubatch
//! 512–4096, `-np` 1–4, and flash attention on and off. The arena model
//! reproduces all 15 sweep points to within 0.05 MiB.
//!
//! A second axis matters as much as those: **how much of the model is on the
//! GPU**. The arena turns out to be independent of it — 18.01 MiB at `-ngl 99`,
//! at `-ngl 18`, and at `-ngl 0` alike — while the host side grows entirely
//! through the CPU's share of the KV cache, which the packer charges
//! separately. Whole-host predictions across that spread land within 1.2% (see
//! `the_host_model_holds_across_the_offload_spread`).
//!
//! Both terms were measured on mainline llama.cpp. ik_llama differs — it does
//! not map host-side weights even without `--no-mmap` — which the *observation*
//! side handles by measuring rather than inferring; see
//! [`crate::supervise::rolling::RollingBase::host_peak`].
//!
//! Validated at production scale on the shape issue #34 was about: a
//! DeepSeek-V4-Flash hybrid (96 GiB, `--n-cpu-moe 40`, 2 GPUs) holds **90.7
//! GiB of expert weights entirely in `RssFile`** and only 589 MiB of owned
//! memory. That is the whole premise of pairing an owned-memory numerator with
//! a weights-excluded denominator, measured rather than assumed.
//!
//! The exception is a service with **no GPU visible at all**, where ggml never
//! swaps the CPU backend's buffer type: nothing is pinned, and the arena
//! instead holds the CPU-executed op intermediates that a GPU run offloads to
//! the device — measured 88 MiB against this model's 18, while the absent CUDA
//! runtime makes the baseline ~130 MiB smaller. The two errors offset to a
//! ~60 MiB over-prediction, which is left to the rolling correction rather
//! than threading placement into the estimator for a term that size.
//!
//! Two known under-predictions, both small and both left to the correction:
//! spanning a second GPU on the layer-split path adds ~18 MiB of baseline and
//! ~24 MiB of arena (tensor-split adds neither), and the estimator runs before
//! the packer has chosen a span, so it cannot see it.
//!
//! See CONTRIBUTING for the procedure.

use crate::{estimator::types::EstimatorInputs, gguf::GgufSummary};

/// llama.cpp's default `-cram` in MiB (`common/common.h`'s
/// `cache_ram_mib = 8192`).
pub const DEFAULT_CACHE_RAM_MB: u32 = 8192;

/// llama.cpp's default physical batch (`-ub`).
const DEFAULT_UBATCH: u32 = 512;

/// Granularity llama.cpp pads a KV cache to.
const KV_CACHE_PAD: u64 = 256;

/// Pinned host bytes that exist alongside the graph arena but are not part of
/// it: the output buffer and the CUDA driver's own pinned allocations.
/// Measured at 12–14 MiB across the calibration models; rounded up.
const PINNED_EXTRA_BYTES: u64 = 16 * 1024 * 1024;

/// Fixed part of the process baseline — CUDA runtime, tokenizer, HTTP stack,
/// and the per-request scratch that only appears once the service has served
/// something. Calibrated against *serving* processes: measuring at idle
/// understates it by ~26 MiB, which is the shape of a term that is allocated
/// on first use.
const PROCESS_BASE_BYTES: u64 = 112 * 1024 * 1024;

/// Per-layer part of the process baseline: graph metadata scales with the
/// tensor count, which scales with layers. Measured 3.4 MiB/layer across
/// dense models of 36, 60, and 65 layers, predicting within 1.3%.
const PROCESS_BASE_BYTES_PER_LAYER: u64 = 3_565_158; // 3.4 MiB

/// Additional baseline for a mixture-of-experts model.
///
/// Layer count badly under-predicts an MoE — a 41-layer MoE was measured
/// holding more than a 65-layer dense model — but nothing available in the
/// GGUF predicts the rest. Three MoEs measured 400, 503, and 546 MiB while all
/// having 256 experts, and their layer counts run the *wrong* way (48 layers
/// for the smallest baseline). So this is a flat allowance, not a fit.
///
/// Its value is chosen by a different criterion than closeness: the estimate
/// has to land within the rolling correction's `[0.8, 1.5]` clamp of reality,
/// because that band is the whole distance the correction can travel. At 200
/// MiB every model measured lands inside it (worst 1.19), so each service can
/// be corrected to its true value. Larger allowances fit the two big MoEs
/// better and push Laguna to 0.78 — outside the band, where no amount of
/// observation could ever bring the reservation back down.
const PROCESS_BASE_BYTES_MOE: u64 = 200 * 1024 * 1024;

/// Host bytes per batch token that ik_llama's arena carries when the MoE
/// expert ops run on the CPU. Measured at exactly 81 KiB/token on
/// Qwen3.6-35B-A3B at two batch sizes, and independent of context and of how
/// many expert layers are CPU-resident.
///
/// It is not constant across models — Laguna-S-2.1 measured 161 KiB/token —
/// and two models are not enough to say what it scales with (expert width and
/// active-expert count both move the right way, neither alone fits). The lower
/// figure is deliberate: under-predicting Laguna by 1.36x leaves the rolling
/// correction able to close the gap, where over-predicting Qwen3.6-35B-A3B by
/// 2x would put it outside the clamp permanently.
///
/// The term only exists *below* ik's op-offload threshold, so production
/// batches (2048) never pay it; a service that lowers `ubatch_size` does.
const IK_MOE_CPU_BYTES_PER_TOKEN: u64 = 81 * 1024;

/// ik_llama ships MoE ops to the GPU once the batch is large enough, at which
/// point the host term above disappears entirely. The threshold is
/// `n_tokens × n_expert_used >= MIN_BATCH × n_expert`.
const IK_OP_OFFLOAD_MIN_BATCH: u64 = 32;

/// Host memory a multi-token-prediction draft context holds.
///
/// `Estimate::mtp_bytes` covers the *VRAM* the draft context takes, which the
/// packer reserves on a GPU. It also costs host memory — a second context
/// brings its own pinned buffers — and that had no term at all until it was
/// measured: gemma-4-31B-QAT at its production settings held 504 MiB more
/// owned host memory with `--spec-type draft-mtp -md` than without, split
/// 374 MiB pinned and 130 MiB anonymous, with the graph arena unchanged.
///
/// One measurement, so a flat allowance rather than a law. Without it the
/// service's host prediction sits at a ratio of 2.15 — beyond the rolling
/// correction's reach — and with it, 1.15.
const MTP_HOST_BYTES_SEPARATE_DRAFT: u64 = 512 * 1024 * 1024;

/// Host memory an *embedded* MTP head costs, which is a different shape and a
/// much smaller number: the trailing nextn layers are resident as part of the
/// target model, so there is no second GGUF, no second set of weights, and no
/// second graph — only the draft context.
///
/// Measured on Qwen3.6-27B (ctx 32768, `-np 2`, tensor split, q8_0 cache):
/// 829 MiB owned without, 1038 with, so 209 MiB. Charging the separate-draft
/// figure here instead would over-predict by ~300 MiB and put the service
/// outside the correction's reach in the *over*-reserving direction.
const MTP_HOST_BYTES_EMBEDDED: u64 = 224 * 1024 * 1024;

/// Extra pinned bytes per batch token when flash attention is off. Measured
/// at exactly 8 KiB/token, consistently across context and batch; its origin
/// is not identified in ggml's source, so it is carried as a measurement.
const NO_FLASH_ATTN_BYTES_PER_TOKEN: u64 = 8 * 1024;

/// Host bytes the runtime is *predicted* to hold, excluding weights and KV.
///
/// This is the figure the rolling correction divides an observation by, so it
/// deliberately excludes the prompt-cache cap — see [`prompt_cache_bytes`].
pub fn host_overhead_bytes(summary: &GgufSummary, arch: &str, inputs: &EstimatorInputs<'_>) -> u64 {
    // The two MTP shapes cost materially different amounts of host memory,
    // and `spec_type` alone does not distinguish them — a separate draft GGUF
    // brings a whole second model, an embedded head brings only a context.
    let mtp = match (inputs.mtp, inputs.draft_model.is_some()) {
        (true, true) => MTP_HOST_BYTES_SEPARATE_DRAFT,
        (true, false) => MTP_HOST_BYTES_EMBEDDED,
        (false, _) => 0,
    };
    pinned_graph_bytes(summary, arch, inputs)
        .saturating_add(PINNED_EXTRA_BYTES)
        .saturating_add(process_base_bytes(summary))
        .saturating_add(mtp)
}

/// The process's fixed host baseline for this model.
fn process_base_bytes(summary: &GgufSummary) -> u64 {
    let layers = summary.block_count.unwrap_or(0) as u64;
    let moe = if has_experts(summary) {
        PROCESS_BASE_BYTES_MOE
    } else {
        0
    };
    PROCESS_BASE_BYTES + layers * PROCESS_BASE_BYTES_PER_LAYER + moe
}

/// Whether the model carries expert tensors, i.e. is a mixture of experts.
fn has_experts(summary: &GgufSummary) -> bool {
    summary.tensors.keys().any(|n| n.contains("_exps"))
}

/// The prompt-cache cap in bytes. Zero when the operator disables it.
///
/// Reserved but not predicted. The cache fills with use rather than at load —
/// measured identical host memory at `-cram 0` and `-cram 4096` on a freshly
/// started server — so counting the cap as a prediction would make every
/// observation read as a large over-reservation and pin the host correction to
/// its clamp floor. The packer charges it as slop for exactly that reason.
pub fn prompt_cache_bytes(inputs: &EstimatorInputs<'_>) -> u64 {
    inputs.cache_ram_mb.unwrap_or(DEFAULT_CACHE_RAM_MB) as u64 * 1024 * 1024
}

/// Bytes of pinned host memory the graph allocator reserves.
///
/// Three measured components, each scaling with the batch:
///
/// - **The KQ mask**, `n_kv × n_tokens` elements — f16 under flash attention,
///   f32 otherwise. `n_kv` is one slot's share of the cache, since a
///   non-unified cache divides the context across slots; `n_tokens` is
///   `min(context, ubatch)`, because graph reservation models a full cache
///   against one batch.
/// - **The hidden-state graph inputs**, two `n_embd × n_tokens` f32 buffers.
///   Larger than the mask at short contexts and absent from any source-level
///   reasoning about "the compute buffer" — the token embeddings stay on the
///   CPU backend, so the embedding lookup and the split-boundary copy both
///   land here.
/// - **A per-token term when flash attention is off**, on top of the mask
///   already doubling.
///
/// An interleaved-SWA model builds a second mask against its sliding-window
/// cache, sized `n_swa + n_tokens` rather than the window alone.
pub fn pinned_graph_bytes(summary: &GgufSummary, arch: &str, inputs: &EstimatorInputs<'_>) -> u64 {
    let context = inputs.context.max(1) as u64;
    let ubatch = inputs.ubatch.unwrap_or(DEFAULT_UBATCH).max(1) as u64;
    let n_tokens = context.min(ubatch);

    // `EstimatorInputs::from_service` resolves `-fa auto` the way the runtime
    // will, so an unset value here only arises in tests. Default to off there,
    // which is the larger mask.
    let flash_attn = inputs.flash_attn.unwrap_or(false);
    let element_bytes = if flash_attn { 2 } else { 4 };

    let streams = if inputs.kv_unified.unwrap_or(false) {
        1
    } else {
        inputs.parallel.unwrap_or(1).max(1) as u64
    };
    let n_embd = embedding_length(summary, arch);
    let has_swa = sliding_window(summary, arch).is_some();

    if inputs.ik_llama {
        // ik sizes every mask against the *whole* cache — it does not divide
        // by the slot count, and an interleaved-SWA model's second mask gets
        // the full context rather than the window — and pins one
        // hidden-state buffer where mainline pins two.
        // One ordinary mask, a second for an interleaved-SWA model — sized
        // against the full cache, not the window — and two more when DSA is
        // on, for the MLA and sparse-indexer masks. Measured: GLM-5.2 at
        // ctx 131072 / ub 2048 came to exactly three 512 MiB masks.
        let masks = 1 + u64::from(has_swa) + if inputs.ik_dsa { 2 } else { 0 };
        let arena = masks * context * n_tokens * element_bytes + n_embd * n_tokens * 4;
        return arena + ik_moe_cpu_bytes(summary, arch, n_tokens);
    }

    let n_kv = pad_to_kv_cache(context / streams);
    // An MLA architecture compresses the KV cache into a single latent
    // tensor, and its mask comes out at half the width the cache size
    // suggests. Measured on DeepSeek-V4-Flash across context and batch: half
    // width fits all three points to 0.3 MiB, full width over-predicts by 30%
    // — far enough that the rolling correction could not pull it back.
    let base_mask = if is_mla(arch) {
        n_kv * n_tokens * element_bytes / 2
    } else {
        n_kv * n_tokens * element_bytes
    };
    // A sparse-attention indexer builds its own mask alongside the ordinary
    // one. The ik path models this from the `-dsa` flag; on mainline there is
    // no flag, so it keys on the architecture. Measured on GLM-5.2: two masks
    // plus hidden comes to 40.00 MiB against 40.07 observed, where one mask
    // gives 32.00.
    let indexer_masks = extra_full_masks(arch) * n_kv * n_tokens * element_bytes;

    let swa_mask = match sliding_window(summary, arch) {
        Some(window) => pad_to_kv_cache(window as u64 + n_tokens) * n_tokens * element_bytes,
        None => 0,
    };

    let hidden_inputs = 2 * n_embd * n_tokens * 4;
    let no_fa_extra = if flash_attn {
        0
    } else {
        NO_FLASH_ATTN_BYTES_PER_TOKEN * n_tokens
    };

    base_mask + indexer_masks + swa_mask + hidden_inputs + no_fa_extra
}

/// Additional full-width masks an architecture's sparse-attention indexer
/// builds, beyond the ordinary one.
fn extra_full_masks(arch: &str) -> u64 {
    match arch {
        "glm-dsa" => 1,
        _ => 0,
    }
}

/// The host-side intermediates ik allocates when a batch is too small for it
/// to ship the CPU-resident expert ops to the GPU.
///
/// Zero above the threshold, which is where production batches sit — but a
/// service that lowers `ubatch_size` crosses back over it and the term
/// reappears, so it cannot simply be ignored.
fn ik_moe_cpu_bytes(summary: &GgufSummary, arch: &str, n_tokens: u64) -> u64 {
    let experts = meta_u32(summary, arch, "expert_count").unwrap_or(0) as u64;
    let used = meta_u32(summary, arch, "expert_used_count").unwrap_or(0) as u64;
    if experts == 0 || used == 0 {
        return 0;
    }
    if n_tokens * used >= IK_OP_OFFLOAD_MIN_BATCH * experts {
        return 0;
    }
    IK_MOE_CPU_BYTES_PER_TOKEN * n_tokens
}

/// Read a `{arch}.{key}` u32 from the GGUF metadata.
fn meta_u32(summary: &GgufSummary, arch: &str, key: &str) -> Option<u32> {
    summary
        .metadata
        .get(&smol_str::SmolStr::new(format!("{arch}.{key}")))
        .and_then(|v| v.as_u32())
}

/// Round a cell count up to the KV cache's padding granularity.
fn pad_to_kv_cache(cells: u64) -> u64 {
    cells.div_ceil(KV_CACHE_PAD).max(1) * KV_CACHE_PAD
}

/// The model's hidden size. Zero when absent, which drops the hidden-input
/// term rather than guessing a width.
fn embedding_length(summary: &GgufSummary, arch: &str) -> u64 {
    meta_u32(summary, arch, "embedding_length").unwrap_or(0) as u64
}

/// Architectures whose attention compresses the KV cache into a latent
/// (multi-head latent attention), which halves the graph mask.
///
/// `glm-dsa` is *not* in this list despite also being MLA: it was measured
/// needing *more* than the plain law, not less. It gets an extra mask instead,
/// via [`extra_full_masks`].
fn is_mla(arch: &str) -> bool {
    // `deepseek4` only. The DeepSeek-V2/V3/R1 family (`deepseek2`) is also
    // MLA, but it has *no measurement here* and a materially different graph —
    // no NSA indexer, no sliding window. Halving its mask on the strength of a
    // different architecture's numbers is the mistake this module keeps
    // making; add it when someone measures it.
    arch == "deepseek4"
}

/// The model's sliding-window size, when it advertises one.
fn sliding_window(summary: &GgufSummary, arch: &str) -> Option<u32> {
    meta_u32(summary, arch, "attention.sliding_window").filter(|w| *w > 0)
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::{GgufSummary, GgufValue};

    const MIB: f64 = (1024 * 1024) as f64;

    fn inputs<'a>(
        context: u32,
        ubatch: u32,
        flash_attn: bool,
        parallel: u32,
    ) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "t",
            model: std::path::Path::new("/m.gguf"),
            mmproj: None,
            context,
            ubatch: Some(ubatch),
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &[],
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
            ik_llama: false,
            ik_dsa: false,
            parallel: Some(parallel),
            flash_attn: Some(flash_attn),
            kv_unified: Some(false),
            cache_ram_mb: None,
        }
    }

    fn summary(arch: &str, n_embd: u32, layers: u32, swa: Option<u32>) -> GgufSummary {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(format!("{arch}.embedding_length")),
            GgufValue::U32(n_embd),
        );
        if let Some(w) = swa {
            metadata.insert(
                SmolStr::new(format!("{arch}.attention.sliding_window")),
                GgufValue::U32(w),
            );
        }
        GgufSummary {
            path: std::path::PathBuf::from("/m.gguf"),
            total_tensor_bytes: 0,
            tensors: Default::default(),
            metadata,
            block_count: Some(layers),
            architecture: SmolStr::new(arch),
            shards: Vec::new(),
        }
    }

    /// Every point from the hardware sweep, reproduced by the model. These are
    /// the measurements the constants come from, so a change that breaks them
    /// is a change that no longer describes the runtime.
    ///
    /// One measured point from the sweep.
    struct Point {
        n_embd: u32,
        layers: u32,
        swa: Option<u32>,
        context: u32,
        ubatch: u32,
        flash_attn: bool,
        parallel: u32,
        /// `CUDA_Host compute buffer size`, as llama.cpp logged it.
        measured_mib: f64,
    }

    #[test]
    fn the_arena_model_reproduces_the_hardware_sweep() {
        #[rustfmt::skip]
        let points = [
            // Qwen3-4B: context sweep at a fixed batch.
            Point { n_embd: 2560, layers: 36, swa: None, context: 8192, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 12.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 16384, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 14.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 18.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 65536, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 26.01 },
            // … batch sweep at a fixed context.
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 1024, flash_attn: true, parallel: 4, measured_mib: 36.02 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 2048, flash_attn: true, parallel: 4, measured_mib: 72.05 },
            // … slot count, which divides the cache and so the mask.
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 1, measured_mib: 42.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 2, measured_mib: 26.01 },
            // … flash attention off: the mask doubles and a per-token term
            // appears.
            Point { n_embd: 2560, layers: 36, swa: None, context: 16384, ubatch: 512, flash_attn: false, parallel: 4, measured_mib: 22.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: false, parallel: 4, measured_mib: 30.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 65536, ubatch: 512, flash_attn: false, parallel: 4, measured_mib: 46.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 1024, flash_attn: false, parallel: 4, measured_mib: 60.02 },
            // Qwen3.6-27B: double the hidden size, which doubles the term the
            // mask alone cannot explain.
            Point { n_embd: 5120, layers: 64, swa: None, context: 8192, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 22.02 },
            Point { n_embd: 5120, layers: 64, swa: None, context: 8192, ubatch: 1024, flash_attn: true, parallel: 4, measured_mib: 44.04 },
            // Gemma-4-31B-QAT: interleaved SWA, so a second window-sized mask.
            Point { n_embd: 5376, layers: 60, swa: Some(1024), context: 8192, ubatch: 512, flash_attn: true, parallel: 4, measured_mib: 24.52 },
        ];
        for p in points {
            let got = pinned_graph_bytes(
                &summary("t", p.n_embd, p.layers, p.swa),
                "t",
                &inputs(p.context, p.ubatch, p.flash_attn, p.parallel),
            ) as f64
                / MIB;
            assert!(
                (got - p.measured_mib).abs() < 0.1,
                "n_embd={} ctx={} ub={} fa={} np={}: modelled {got:.2} MiB \
                 against {:.2} MiB measured",
                p.n_embd,
                p.context,
                p.ubatch,
                p.flash_attn,
                p.parallel,
                p.measured_mib
            );
        }
    }

    /// The mainline law is validated on dense, SWA, and plain-MoE models. It
    /// is *not* right for the MLA/sparse-attention architectures, which build
    /// masks this model has no term for and which differ from each other:
    /// `deepseek4` counts its mask at half width (a compressed KV cache) and
    /// `glm-dsa` carries an extra full one. One to three points per
    /// architecture is not enough to derive either, so what is asserted here
    /// is only that the estimate stays inside the band the rolling correction
    /// can travel — which it does, because these are a few MiB against host
    /// slots of tens of GiB.
    #[test]
    fn the_mainline_model_stays_reachable_on_sparse_attention_models() {
        // (arch, n_embd, swa, ctx, ubatch, parallel, measured MiB)
        #[rustfmt::skip]
        let points = [
            // DeepSeek-V4-Flash: MLA + NSA indexer + a 128-token window.
            ("deepseek4", 4096u32, Some(128u32), 32768u32,  512u32, 4u32, 21.09f64),
            ("deepseek4", 4096, Some(128), 32768, 1024, 4, 43.12),
            ("deepseek4", 4096, Some(128), 65536,  512, 4, 25.09),
            // GLM-5.2: MLA + DSA, no window — the indexer's extra mask.
            ("glm-dsa", 6144, None, 32768, 512, 4, 40.07),
            // Plain MoE, which the law *does* fit exactly.
            ("qwen35moe", 2048, None, 32768, 512, 4, 16.02),
            ("laguna", 3072, Some(512), 32768, 512, 4, 21.02),
        ];
        for (arch, n_embd, swa, ctx, ub, par, measured) in points {
            let got = pinned_graph_bytes(
                &summary(arch, n_embd, 48, swa),
                arch,
                &inputs(ctx, ub, true, par),
            ) as f64
                / MIB;
            let ratio = measured / got;
            assert!(
                (0.8..=1.5).contains(&ratio),
                "{arch} ctx={ctx} ub={ub}: modelled {got:.2} MiB against \
                 {measured:.2} MiB measured is a ratio of {ratio:.2}"
            );
        }
    }

    /// ik_llama's arena follows a different law, measured on the packaged
    /// `ik-llama-server` build against the same models:
    ///
    /// - every mask is sized against the **whole** cache — `-np` does not
    ///   divide it, and an interleaved-SWA model's second mask gets the full
    ///   context rather than the window;
    /// - one hidden-state buffer is pinned, not two;
    /// - GPU count makes no difference (37.01 MiB on one card and on two);
    /// - below ik's op-offload threshold the CPU-resident expert ops allocate
    ///   host intermediates, at 81 KiB per batch token.
    #[test]
    fn the_ik_arena_model_reproduces_its_own_sweep() {
        /// One measured ik point. `experts` is `(count, used)` for a
        /// mixture-of-experts model.
        struct IkPoint {
            n_embd: u32,
            swa: Option<u32>,
            experts: Option<(u32, u32)>,
            dsa: bool,
            context: u32,
            ubatch: u32,
            parallel: u32,
            measured_mib: f64,
        }
        #[rustfmt::skip]
        let points = [
            // Qwen3-4B, dense — the same four points swept on mainline.
            IkPoint { n_embd: 2560, swa: None, experts: None, dsa: false, context: 32768, ubatch: 512, parallel: 4, measured_mib: 37.01 },
            IkPoint { n_embd: 2560, swa: None, experts: None, dsa: false, context: 65536, ubatch: 512, parallel: 4, measured_mib: 69.01 },
            IkPoint { n_embd: 2560, swa: None, experts: None, dsa: false, context: 32768, ubatch: 2048, parallel: 4, measured_mib: 148.02 },
            // `-np` changes nothing here, where it halves the mainline arena.
            IkPoint { n_embd: 2560, swa: None, experts: None, dsa: false, context: 32768, ubatch: 512, parallel: 1, measured_mib: 37.01 },
            // Qwen3.6-35B-A3B, MoE: below the offload threshold at ub 256/512,
            // above it at 1024/2048 where the host term vanishes.
            IkPoint { n_embd: 2048, swa: None, experts: Some((256, 8)), dsa: false, context: 32768, ubatch: 256, parallel: 4, measured_mib: 38.26 },
            IkPoint { n_embd: 2048, swa: None, experts: Some((256, 8)), dsa: false, context: 32768, ubatch: 512, parallel: 4, measured_mib: 76.51 },
            IkPoint { n_embd: 2048, swa: None, experts: Some((256, 8)), dsa: false, context: 32768, ubatch: 1024, parallel: 4, measured_mib: 72.03 },
            IkPoint { n_embd: 2048, swa: None, experts: Some((256, 8)), dsa: false, context: 32768, ubatch: 2048, parallel: 4, measured_mib: 144.05 },
            IkPoint { n_embd: 2048, swa: None, experts: Some((256, 8)), dsa: false, context: 65536, ubatch: 512, parallel: 4, measured_mib: 108.51 },
            // Laguna-S-2.1 at its production settings: SWA, so two full-cache
            // masks, and above the threshold so no host MoE term.
            IkPoint { n_embd: 3072, swa: Some(512), experts: Some((256, 10)), dsa: false, context: 131072, ubatch: 2048, parallel: 1, measured_mib: 1048.02 },
            // …and below the threshold, where the host MoE term reappears.
            IkPoint { n_embd: 3072, swa: Some(512), experts: Some((256, 10)), dsa: false, context: 32768, ubatch: 512, parallel: 4, measured_mib: 150.50 },
            // gemma-3-27b: dense + SWA, confirming the full-cache second mask
            // is not a mixture-of-experts artefact.
            IkPoint { n_embd: 5376, swa: Some(1024), experts: None, dsa: false, context: 8192, ubatch: 512, parallel: 4, measured_mib: 26.51 },
            // GLM-5.2 at its production settings: MLA + DSA, so three
            // full-cache masks, and above the MoE threshold.
            IkPoint { n_embd: 6144, swa: None, experts: Some((256, 8)), dsa: true, context: 131072, ubatch: 2048, parallel: 1, measured_mib: 1584.02 },
        ];
        for p in points {
            let mut sum = summary("t", p.n_embd, 36, p.swa);
            if let Some((count, used)) = p.experts {
                sum.metadata
                    .insert(SmolStr::new("t.expert_count"), GgufValue::U32(count));
                sum.metadata
                    .insert(SmolStr::new("t.expert_used_count"), GgufValue::U32(used));
            }
            let mut i = inputs(p.context, p.ubatch, true, p.parallel);
            i.ik_llama = true;
            i.ik_dsa = p.dsa;
            let got = pinned_graph_bytes(&sum, "t", &i) as f64 / MIB;
            // The mask and hidden-buffer terms are exact. The host MoE term is
            // calibrated on one model, so where it applies the guarantee is
            // only that the estimate stays inside the band the rolling
            // correction can travel.
            let nt = u64::from(p.context.min(p.ubatch));
            let moe_term_applies = p
                .experts
                .is_some_and(|(count, used)| nt * u64::from(used) < 32 * u64::from(count));
            if moe_term_applies {
                let ratio = p.measured_mib / got;
                assert!(
                    (0.8..=1.5).contains(&ratio),
                    "ik n_embd={} ctx={} ub={}: modelled {got:.2} MiB against \
                     {:.2} MiB measured is a ratio of {ratio:.2}, outside the \
                     band the correction can travel",
                    p.n_embd,
                    p.context,
                    p.ubatch,
                    p.measured_mib
                );
            } else {
                assert!(
                    (got - p.measured_mib).abs() < 0.1,
                    "ik n_embd={} ctx={} ub={} np={}: modelled {got:.2} MiB \
                     against {:.2} MiB measured",
                    p.n_embd,
                    p.context,
                    p.ubatch,
                    p.parallel,
                    p.measured_mib
                );
            }
        }
    }

    /// The two forks disagree on the same inputs, so the runtime has to be
    /// part of the estimate rather than a detail of how it is served.
    #[test]
    fn the_forks_disagree_on_the_same_inputs() {
        let s = summary("t", 2560, 36, None);
        let mainline = pinned_graph_bytes(&s, "t", &inputs(32768, 512, true, 4));
        let mut ik = inputs(32768, 512, true, 4);
        ik.ik_llama = true;
        let ik_bytes = pinned_graph_bytes(&s, "t", &ik);
        assert_eq!(mainline as f64 / MIB, 18.0);
        assert_eq!(ik_bytes as f64 / MIB, 37.0);
    }

    /// The hidden-state inputs dominate at short contexts, which is why a
    /// mask-only model (or a per-device GPU curve) cannot stand in for this.
    #[test]
    fn the_hidden_state_inputs_dominate_a_short_context() {
        let s = summary("t", 5120, 64, None);
        let total = pinned_graph_bytes(&s, "t", &inputs(8192, 512, true, 4));
        let mask = 2048 * 512 * 2;
        assert!(
            total - mask > 9 * mask,
            "the mask is a small share of the arena here ({mask} of {total})"
        );
    }

    /// Give a summary an expert tensor, so it reads as a mixture of experts.
    fn with_experts(mut s: GgufSummary) -> GgufSummary {
        s.tensors.insert(
            SmolStr::new("blk.0.ffn_gate_exps.weight"),
            crate::gguf::GgufTensor {
                name: SmolStr::new("blk.0.ffn_gate_exps.weight"),
                dtype: crate::gguf::GgufType::F16,
                shape: vec![1],
                byte_size: 2,
                shard_idx: 0,
                offset: 0,
            },
        );
        s
    }

    /// Layer count under-predicts a mixture of experts badly — a 41-layer MoE
    /// was measured holding more host memory than a 65-layer dense model — so
    /// the baseline carries a flat allowance for one.
    ///
    /// Measured with the weights excluded, so no CPU KV or mapped weight
    /// confounds it: Qwen3.6-35B-A3B (41 layers) 503 MiB,
    /// DeepSeek-V4-Flash (43) 546 MiB, Laguna-S-2.1 (48) 400 MiB.
    ///
    /// The requirement is not closeness but *reachability*: every prediction
    /// must sit within the rolling correction's `[0.8, 1.5]` clamp of the
    /// measurement, since that is as far as the correction can move it.
    #[test]
    fn a_mixture_of_experts_gets_its_own_allowance() {
        for (layers, measured) in [(41u32, 503.0), (43, 546.0), (48, 400.0)] {
            let s = with_experts(summary("t", 2048, layers, None));
            let predicted = process_base_bytes(&s) as f64 / MIB;
            let ratio = measured / predicted;
            assert!(
                (0.8..=1.5).contains(&ratio),
                "{layers}-layer MoE: predicted {predicted:.0} MiB against \
                 {measured:.0} MiB measured is a ratio of {ratio:.2}, outside \
                 the band the correction can travel"
            );
        }
        // A dense model of the same size gets no allowance.
        let dense = process_base_bytes(&summary("t", 2048, 41, None));
        let moe = process_base_bytes(&with_experts(summary("t", 2048, 41, None)));
        assert!(moe > dense);
    }

    /// The baseline is charged per model, not per placement: a fully
    /// GPU-offloaded service still pays it.
    ///
    /// Pinned against the one point measured in the *serving* state — a
    /// 36-layer model, whose owned footprint less the arena and its pinned
    /// companions came to 233 MiB. The idle figures the layer slope was fitted
    /// from run ~26 MiB lower and are deliberately not asserted here, since a
    /// service that has never served is not the state worth predicting.
    #[test]
    fn the_process_baseline_matches_a_serving_process() {
        let small = process_base_bytes(&summary("t", 2560, 36, None)) as f64 / MIB;
        assert!(
            (small - 233.0).abs() < 8.0,
            "36-layer baseline modelled at {small:.0} MiB against 233 MiB measured"
        );
        let large = process_base_bytes(&summary("t", 5120, 64, None)) as f64 / MIB;
        assert!(large > small, "more layers means more graph metadata");
    }

    /// The whole host prediction, checked against a sweep from fully
    /// GPU-offloaded to fully CPU-resident. The arena is offload-independent,
    /// so what moves is the CPU's KV share — supplied here from the same runs'
    /// logged `CPU KV buffer size`, since the packer rather than this module
    /// charges it.
    ///
    /// Measurements are `RssAnon + RssShmem` on a process that has served a
    /// request. Qwen3-4B, 36 layers, ctx 32768, ubatch 512, `-np 4`, FA on.
    #[test]
    fn the_host_model_holds_across_the_offload_spread() {
        // (label, CPU KV MiB, measured owned MiB)
        let regimes = [
            ("-ngl 99, 37/37 on GPU", 0.0, 267.0),
            ("-ngl 18, 18/37 on GPU", 2432.0, 2689.0),
            ("-ngl 6,   6/37 on GPU", 3968.0, 4223.0),
            ("-ngl 0,   0/37 on GPU", 4608.0, 4819.0),
        ];
        let s = summary("t", 2560, 36, None);
        let overhead = host_overhead_bytes(&s, "t", &inputs(32768, 512, true, 4)) as f64 / MIB;
        for (label, cpu_kv, measured) in regimes {
            let predicted = overhead + cpu_kv;
            let ratio = measured / predicted;
            assert!(
                (0.95..=1.02).contains(&ratio),
                "{label}: predicted {predicted:.0} MiB against {measured:.0} MiB \
                 measured (ratio {ratio:.3})"
            );
            assert!(
                predicted >= measured * 0.99,
                "{label}: the prediction must not fall short of the measurement"
            );
        }
    }

    /// A draft context costs host memory as well as the VRAM the packer
    /// reserves for it, and at production settings that is enough to put the
    /// prediction outside the correction's reach if unmodelled.
    ///
    /// The two shapes cost different amounts and `spec_type` alone cannot
    /// tell them apart, so the term keys on whether a draft GGUF is loaded.
    ///
    /// Separate draft, gemma-4-31B-QAT (ctx 240000, `-np 4 -kvu`, tensor
    /// split): 770 MiB owned without MTP, 1274 with — a 504 MiB delta.
    /// Embedded head, Qwen3.6-27B (ctx 32768, `-np 2`, tensor split, q8_0):
    /// 829 without, 1038 with — 209 MiB. Charging the separate-draft figure
    /// for an embedded head over-predicts by ~300 MiB.
    #[test]
    fn the_two_mtp_shapes_are_charged_differently() {
        let s = summary("gemma4", 5376, 60, Some(1024));
        let mut i = inputs(240000, 512, true, 4);
        i.kv_unified = Some(true);
        let plain = host_overhead_bytes(&s, "gemma4", &i);

        i.mtp = true;
        let embedded = host_overhead_bytes(&s, "gemma4", &i);
        let draft_path = std::path::PathBuf::from("/d.gguf");
        i.draft_model = Some(&draft_path);
        let separate = host_overhead_bytes(&s, "gemma4", &i);

        assert!(plain < embedded && embedded < separate);
        assert_eq!(
            (separate - embedded) as f64 / MIB,
            (MTP_HOST_BYTES_SEPARATE_DRAFT - MTP_HOST_BYTES_EMBEDDED) as f64 / MIB
        );
        // The measured gemma-4 pair, which is the separate-draft shape.
        for (predicted, measured) in [(plain as f64 / MIB, 770.0), (separate as f64 / MIB, 1274.0)]
        {
            let ratio = measured / predicted;
            assert!(
                (0.8..=1.5).contains(&ratio),
                "predicted {predicted:.0} MiB against {measured:.0} MiB \
                 measured is a ratio of {ratio:.2}"
            );
        }
    }

    /// The prompt cache is charged at its cap, and only the operator can
    /// remove it.
    #[test]
    fn the_prompt_cache_is_charged_at_its_cap() {
        let mut i = inputs(4096, 512, true, 1);
        assert_eq!(
            prompt_cache_bytes(&i),
            DEFAULT_CACHE_RAM_MB as u64 * 1024 * 1024
        );
        i.cache_ram_mb = Some(0);
        assert_eq!(prompt_cache_bytes(&i), 0);
        i.cache_ram_mb = Some(512);
        assert_eq!(prompt_cache_bytes(&i), 512 * 1024 * 1024);
    }
}
