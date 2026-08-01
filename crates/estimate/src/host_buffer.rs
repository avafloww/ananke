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
//! The `Cpu` slot must not be charged the GPU-calibrated `compute_buffer_mb`:
//! that number is derived from `nvidia-smi` VRAM readings and says nothing
//! about a host backend.
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
//! `RollingBase::host_peak`.
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

// Every tuning constant below comes from `tuning.json`, which
// `ananke-calibrate`'s `emit` binary generates from the measurement dataset and
// `build.rs` turns into compile-time constants. Each carries its evidence in
// its own doc comment — how many models it rests on, and where it is weak.

use ananke_config::flags::cache_type;
use ananke_gguf::{GgufSummary, keys};

pub use crate::tuning::DEFAULT_CACHE_RAM_MB;
use crate::{
    tuning::{
        DEFAULT_UBATCH, GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN, IK_OP_OFFLOAD_MIN_BATCH,
        KV_CACHE_PAD, MAINLINE_LAYER_SPLIT_MASK_COPIES, MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD,
        MTP_HOST_BYTES_EMBEDDED, MTP_HOST_BYTES_SEPARATE_DRAFT, MTP_HOST_MIB_PER_1K,
        PINNED_EXTRA_BYTES, PROCESS_BASE_BYTES, PROCESS_BASE_BYTES_MOE,
        PROCESS_BASE_BYTES_PER_DEVICE, PROCESS_BASE_BYTES_PER_LAYER,
    },
    types::EstimatorInputs,
};

/// Host bytes the runtime is *predicted* to hold, excluding weights and KV.
///
/// This is the figure the rolling correction divides an observation by, so it
/// deliberately excludes the prompt-cache cap — see [`prompt_cache_bytes`].
pub fn host_overhead_bytes(summary: &GgufSummary, arch: &str, inputs: &EstimatorInputs<'_>) -> u64 {
    // Each visible CUDA device beyond the first costs host memory for its
    // context. Measured with placement pinned to the CPU so only the context
    // count varied: two models of very different size both moved by 20 MiB
    // between one card and two.
    let device_bytes =
        PROCESS_BASE_BYTES_PER_DEVICE * u64::from(inputs.visible_devices.saturating_sub(1));
    // The two MTP shapes cost materially different amounts of host memory,
    // and `spec_type` alone does not distinguish them — a separate draft GGUF
    // brings a whole second model, an embedded head brings only a context.
    // Both shapes are flat in the slot count and linear in context at 2 MiB
    // per 1024 — measured at 239, 243, and 240 MiB for Qwen3.6-27B at one,
    // two, and four slots, and 240, 274, 341 across ctx 32768, 65536, and
    // 131072. A flat constant is the wrong shape here, and wrong in opposite
    // directions at the ends of that range.
    let mtp = if inputs.mtp {
        let base = if inputs.draft_model.is_some() {
            MTP_HOST_BYTES_SEPARATE_DRAFT
        } else {
            MTP_HOST_BYTES_EMBEDDED
        };
        // Multiply before dividing: the rate is per 1024 tokens, and a context
        // that is not a whole multiple of that still pays for the remainder.
        let context_mib = MTP_HOST_MIB_PER_1K * u64::from(inputs.context) / 1024;
        base + context_mib * 1024 * 1024
    } else {
        0
    };
    // A tensor split costs host baseline a layer split does not — between 96
    // and 184 MiB, measured on every model that ran both at matching settings.
    // The operator runs several services this way, so omitting it under-predicts
    // each of them by that much.
    let tensor_split = if inputs.visible_devices > 1
        && inputs.split_mode == ananke_config::placement::SplitMode::Tensor
    {
        tensor_split_baseline(arch)
    } else {
        0
    };
    // A concurrently active slot costs host memory — 163 MiB on qwen35, 89 on
    // gemma3, 4 on llama — and that is measured and recorded in
    // `tuning.json`, but deliberately *not* charged here.
    //
    // This function is what the rolling correction divides an observation by,
    // so it has to model what a process actually holds rather than the worst
    // case it might. Slots that stay idle cost nothing, and most services are
    // idle in most slots: charging all four took the cells outside the
    // correction's band from 2 to 44, with ratios down to 0.34, because every
    // ordinary service then reads as a massive over-reservation and clamps
    // unreachably. Two stress cells inside the band is the better trade.
    //
    // A worst-case allowance belongs in the packer's slop, alongside the
    // prompt cache, which is reserved the same way and for the same reason.
    pinned_graph_bytes(summary, arch, inputs)
        .saturating_add(PINNED_EXTRA_BYTES)
        .saturating_add(process_base_bytes(
            summary,
            inputs.ik_llama,
            inputs.flash_attn.unwrap_or(false),
        ))
        .saturating_add(device_bytes)
        .saturating_add(tensor_split)
        .saturating_add(mtp)
}

/// What a tensor split adds to the host baseline, for this architecture.
fn tensor_split_baseline(arch: &str) -> u64 {
    lookup(
        crate::tuning::TENSOR_SPLIT_BASELINE,
        arch,
        crate::tuning::TENSOR_SPLIT_BASELINE_DEFAULT,
    )
}

/// The process's fixed host baseline for this model.
fn process_base_bytes(summary: &GgufSummary, ik_llama: bool, flash_attn: bool) -> u64 {
    let layers = summary.block_count.unwrap_or(0) as u64;
    let moe = if has_experts(summary) {
        PROCESS_BASE_BYTES_MOE
    } else {
        0
    };
    (PROCESS_BASE_BYTES + layers * PROCESS_BASE_BYTES_PER_LAYER + moe)
        .saturating_add_signed(baseline_offset(summary, ik_llama, flash_attn))
}

/// The measured correction the layer-count model above leaves behind.
///
/// Signed: it corrects a baseline that over-covers as well as one that
/// under-covers. An over-prediction is only safe while it stays inside the
/// band the rolling correction can travel, and gemma3 sat at 0.78 against a
/// floor of 0.8 — unreachable rather than safe.
///
/// Keyed on the architecture *and* the two variant distinctions that separate
/// models sharing one architecture string: within `gemma4`, the mixture of
/// experts is over-covered by 66 MiB, the dense model needs 17, and the
/// E-variant 107. Both discriminators are ones the estimator already reads, so
/// the key is one it can construct.
fn baseline_offset(summary: &GgufSummary, ik_llama: bool, flash_attn: bool) -> i64 {
    let mut key = variant_key(summary, &summary.architecture);
    // The two binaries do not share a baseline: grouped per runtime, ik's
    // resident cells are as consistent as mainline's and simply sit 24 to 192
    // MiB higher. Only this table is keyed on it — the flash-attention rates
    // must not be, since ik is excluded from that derivation and an
    // ik-suffixed key would inherit the worst rate as its default.
    if ik_llama {
        key.push_str("@ik");
    }
    // Flash attention shifts the baseline underneath the per-token arena term
    // as well: +21 to +33 MiB on most architectures and +131 on lfm2, which is
    // enough on a 190 MiB baseline to put that configuration outside the band.
    if !flash_attn {
        key.push_str("@nofa");
    }
    crate::tuning::BASELINE_OFFSET
        .iter()
        .find(|(name, _)| *name == key)
        .map_or(crate::tuning::BASELINE_OFFSET_DEFAULT, |(_, value)| *value)
}

/// Extra pinned bytes per batch token when flash attention is off.
///
/// Flat in context and proportional to batch: gemma-3-27B is 64 MiB over the
/// modelled arena at ctx 8192, 32768, and 131072 alike, and 256 MiB over at
/// ubatch 2048 in every one of them. The rate differs fourfold between
/// sliding-window architectures and the rest, which one representative value
/// could not carry.
fn no_flash_attn_rate(summary: &GgufSummary, arch: &str) -> u64 {
    // Never reached on the ik path, which returns before this term.
    lookup(
        crate::tuning::NO_FLASH_ATTN_RATES,
        &variant_key(summary, arch),
        crate::tuning::NO_FLASH_ATTN_RATE_DEFAULT,
    )
}

/// The architecture, plus the distinctions that split one architecture string.
///
/// `gemma4` covers three models whose host terms differ by more than the
/// rolling correction can travel: a mixture of experts, a dense model, and an
/// E-variant. Both discriminators are read from the GGUF, so the key costs
/// nothing beyond what is already loaded.
fn variant_key(summary: &GgufSummary, arch: &str) -> String {
    let mut key = arch.to_string();
    if has_experts(summary) {
        key.push_str("+moe");
    }
    if crate::compute_buffer::is_gemma_e_variant(summary) {
        key.push_str("+e");
    }
    key
}

/// The entry for `key`, or `fallback` for an architecture never measured.
fn lookup(table: &[(&str, u64)], key: &str, fallback: u64) -> u64 {
    table
        .iter()
        .find(|(name, _)| *name == key)
        .map_or(fallback, |(_, value)| *value)
}

/// Whether the model carries expert tensors, i.e. is a mixture of experts.
fn has_experts(summary: &GgufSummary) -> bool {
    summary.tensors.keys().any(|n| n.contains("_exps"))
}

/// Host memory the slots beyond the first cost when they are all busy.
///
/// Reserved, not predicted — the same treatment as the prompt cache and for
/// the same reason. A concurrently active slot costs real host memory (163 MiB
/// per slot on qwen35, 89 on gemma3, 4 on llama, measured across three
/// architectures), but an idle one costs nothing, and most services are idle
/// in most slots. Charging it in [`host_overhead_bytes`] would make every
/// ordinary service read as a large over-reservation and clamp the rolling
/// correction unreachably — measured at 44 of 259 cells outside the band
/// against 4 when it is left out.
///
/// So the packer reserves it as slop, where it protects a service that does
/// become busy, and the correction never divides by it.
pub fn slot_host_bytes(arch: &str, inputs: &EstimatorInputs<'_>) -> u64 {
    let slots = u64::from(inputs.parallel.unwrap_or(1).max(1));
    lookup(
        crate::tuning::PER_SLOT_HOST_BYTES,
        arch,
        crate::tuning::PER_SLOT_HOST_BYTES_DEFAULT,
    ) * slots.saturating_sub(1)
}

/// Host memory a prompt long enough to be checkpointed adds, per slot.
///
/// llama.cpp's server checkpoints a slot's state so a prompt can be rewound
/// rather than reprocessed, spaced by `--checkpoint-min-step` (8192 tokens).
/// A service serving real prompts holds more of them than a short probe does,
/// and how many more depends on the attention: the flag's other name is
/// `--swa-checkpoints`, and a sliding-window model needs them to rewind its
/// window. gemma-4-31B-QAT measures 524 MiB at a four-token prompt and 2138
/// at 16384; Qwen3.6-27B, without a sliding window, 778 and 928.
///
/// Reserved, not predicted — the same treatment as the prompt cache and the
/// per-slot cost, and for the same reason. Charging it in
/// [`host_overhead_bytes`] would make a service that never sees a long prompt
/// read as a 1.6 GiB over-reservation and clamp the correction unreachably.
pub fn checkpoint_headroom_bytes(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u64 {
    let slots = u64::from(inputs.parallel.unwrap_or(1).max(1));
    lookup(
        crate::tuning::CHECKPOINT_HEADROOM_BYTES,
        &variant_key(summary, &summary.architecture),
        crate::tuning::CHECKPOINT_HEADROOM_DEFAULT,
    ) * slots
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

    // `config::service_inputs::estimator_inputs` resolves `-fa auto` the way the runtime
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
        return arena + ik_moe_cpu_bytes(summary, arch, n_tokens, inputs.visible_devices);
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
        // Three window masks when several slots *share* one cache, otherwise
        // one.
        //
        // Measured on gemma3 and on gemma4 alike: both imply three at four
        // slots with `--kv-unified`, and one everywhere else. The condition is
        // the shared cache and not the slot count: gemma-4-31B-QAT at four
        // slots with a per-slot cache measures one, which rules the simpler
        // rule out.
        Some(window) => {
            let shared = inputs.parallel.unwrap_or(1) > 1 && inputs.kv_unified.unwrap_or(false);
            // How many batches the window spans, plus the batch's own mask.
            //
            // The count depends on the batch: at ubatch 512 a 1024-token
            // window spans two batches and the count is 3, while at 2048 the
            // same configuration measures 2 — a difference of exactly one mask
            // — 692.3 MiB against 740.0 modelled on gemma-3-27B at four
            // shared slots, where two masks give 692.0. Capped at the measured
            // range rather than extrapolated below ubatch 512.
            let masks = if shared {
                1 + (window as u64).div_ceil(n_tokens).min(2)
            } else {
                1
            };
            masks * pad_to_kv_cache(window as u64 + n_tokens) * n_tokens * element_bytes
        }
        None => 0,
    };

    let hidden_inputs = 2 * n_embd * n_tokens * 4;
    // Per token, per *device copy* — replicated under a layer split the same
    // way the masks are, and independent of the slot count.
    //
    // It must not be divided by the stream count. Single-card points measure 8
    // KiB per token against two-card cells at 32, which is the mask-copy factor
    // of 4 rather than the slot count: gemma3 measures 128.1 KiB per token and
    // qwen35 32.1 at both one slot and four, on two cards, identical to the
    // decimal.
    let no_fa_extra = if flash_attn {
        0
    } else {
        no_flash_attn_rate(summary, arch) * n_tokens
    };
    // A quantised KV cache costs more pinned memory than an f16 one, measured
    // in every one of 117 pairs differing in nothing else. The per-copy rate
    // varies by architecture — 160 bytes per batch token on a non-sliding-
    // window model, 6144 on deepseek4 — and is not predicted by head count,
    // head width or layer count, so the worst is charged to all of them. That
    // is 12 MiB at the largest batch measured, which is cheap against an
    // under-prediction whose mechanism is not understood.
    let quantised_cache = if quantised_kv(inputs) {
        quantised_cache_rate(arch) * n_tokens
    } else {
        0
    };

    // mainline replicates every mask when layers are split across more than
    // one device. Not under tensor split, where llama.cpp fuses the cards
    // into a single device, and not on ik at any card count. Measured at
    // 4.00-4.16 on six architectures, flat across context, batch, slot count
    // and cache mode — so it multiplies a term that scales, rather than being
    // a flat allowance (which at ctx 8192 comes to about +24 MiB).
    let masks = base_mask + indexer_masks + swa_mask + no_fa_extra;
    let masks = masks * mask_copies(inputs);

    masks
        + hidden_inputs
        + quantised_cache
        + gemma_e_variant_bytes(summary, arch, n_tokens)
        + mainline_tensor_moe_bytes(summary, arch, inputs, n_tokens)
}

/// mainline's host-resident MoE intermediates under tensor split.
///
/// A hybrid served with `--split-mode tensor` keeps per-token MoE buffers on
/// the host — the same shape as ik's term and at a higher rate. Under layer
/// split the same models show none of it, which is why this keys on the split
/// mode and not on the placement alone.
fn mainline_tensor_moe_bytes(
    summary: &GgufSummary,
    arch: &str,
    inputs: &EstimatorInputs<'_>,
    n_tokens: u64,
) -> u64 {
    if inputs.ik_llama
        || !inputs.host_resident_experts
        || inputs.split_mode != ananke_config::placement::SplitMode::Tensor
    {
        return 0;
    }
    MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD * embedding_length(summary, arch) * n_tokens
}

/// How many times the graph's attention masks are replicated.
///
/// Only when the whole model is split across devices. A hybrid — experts held
/// on the CPU — does not replicate them, measured at 1.00 on all three
/// (deepseek4, laguna, qwen35moe) against 4.00 on every fully-resident model
/// including a mixture of experts, which is what rules out MoE-ness as the
/// discriminator and leaves placement.
fn mask_copies(inputs: &EstimatorInputs<'_>) -> u64 {
    let split_across_devices = inputs.visible_devices > 1
        && matches!(
            inputs.split_mode,
            ananke_config::placement::SplitMode::Layer | ananke_config::placement::SplitMode::Row
        );
    if split_across_devices && !inputs.host_resident_experts {
        MAINLINE_LAYER_SPLIT_MASK_COPIES
    } else {
        1
    }
}

/// The Gemma 4 E-variant pins a per-layer embedding input alongside the
/// ordinary hidden-state buffers.
///
/// Measured flat at 1028 bytes per layer per batch token across three
/// contexts, two batch sizes, and both card counts — with two same-architecture
/// controls that are not E-variants showing none of it.
fn gemma_e_variant_bytes(summary: &GgufSummary, arch: &str, n_tokens: u64) -> u64 {
    if !crate::compute_buffer::is_gemma_e_variant(summary) {
        return 0;
    }
    let layers = summary.meta_u32(&keys::block_count(arch)).unwrap_or(0) as u64;
    GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN * layers * n_tokens
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
fn ik_moe_cpu_bytes(summary: &GgufSummary, arch: &str, n_tokens: u64, devices: u32) -> u64 {
    let experts = summary.meta_u32(&keys::expert_count(arch)).unwrap_or(0) as u64;
    let used = summary
        .meta_u32(&keys::expert_used_count(arch))
        .unwrap_or(0) as u64;
    if experts == 0 || used == 0 {
        return 0;
    }
    if n_tokens * used >= IK_OP_OFFLOAD_MIN_BATCH * experts {
        return 0;
    }
    // Proportional to hidden size, not flat: a flat 81 KiB/token, which is this
    // term at qwen35moe's `n_embd` of 2048, under-reserves GLM-5.2 — three
    // times that hidden size — threefold.
    ik_moe_rate(arch, devices) * embedding_length(summary, arch) * n_tokens
}

/// The per-token, per-unit-of-hidden-size rate for this architecture and
/// device count.
///
/// Both axes are measured. The three ik mixtures differ by a third from each
/// other — 41, 43 and 54 — and two of them differ again with the card count:
/// glm-dsa is 28 on one card and 43 on two at identical placement, laguna 36
/// and 54, while qwen35moe is 41 on both. A single constant would either
/// under-reserve the worst or over-reserve a single-card glm by 45 MiB.
fn ik_moe_rate(arch: &str, devices: u32) -> u64 {
    let table = crate::tuning::IK_MOE_RATES;
    let exact = format!("{arch}@{}", devices.max(1));
    if let Some((_, rate)) = table.iter().find(|(name, _)| *name == exact) {
        return *rate;
    }
    // No measurement at this device count. Take the worst seen for the
    // architecture: the rate rises with cards on both models where it varies
    // at all, and under-reserving is the direction that OOMs.
    let prefix = format!("{arch}@");
    table
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
        .map(|(_, rate)| *rate)
        .max()
        .unwrap_or(crate::tuning::IK_MOE_RATE_DEFAULT)
}

/// Round a cell count up to the KV cache's padding granularity.
fn pad_to_kv_cache(cells: u64) -> u64 {
    cells.div_ceil(KV_CACHE_PAD).max(1) * KV_CACHE_PAD
}

/// The model's hidden size. Zero when absent, which drops the hidden-input
/// term rather than guessing a width.
fn embedding_length(summary: &GgufSummary, arch: &str) -> u64 {
    summary.meta_u32(&keys::embedding_length(arch)).unwrap_or(0) as u64
}

/// Architectures whose attention compresses the KV cache into a latent
/// (multi-head latent attention), which halves the graph mask.
///
/// `glm-dsa` is *not* in this list despite also being MLA: it was measured
/// needing *more* than the plain law, not less. It gets an extra mask instead,
/// via [`extra_full_masks`].
pub(crate) fn is_mla(arch: &str) -> bool {
    // `deepseek4` only. The DeepSeek-V2/V3/R1 family (`deepseek2`) is also
    // MLA, but it has *no measurement here* and a materially different graph —
    // no NSA indexer, no sliding window. Halving its mask on the strength of a
    // different architecture's numbers is the mistake this module keeps
    // making; add it when someone measures it.
    arch == "deepseek4"
}

/// The model's sliding-window size, when it advertises one.
fn sliding_window(summary: &GgufSummary, arch: &str) -> Option<u32> {
    summary
        .meta_u32(&keys::attention_sliding_window(arch))
        .filter(|w| *w > 0)
}

#[cfg(test)]
mod tests {
    use ananke_gguf::{GgufSummary, GgufValue};
    use smol_str::SmolStr;

    use super::*;

    const MIB: f64 = (1024 * 1024) as f64;

    fn inputs<'a>(
        context: u32,
        ubatch: u32,
        flash_attn: bool,
        parallel: u32,
    ) -> EstimatorInputs<'a> {
        EstimatorInputs {
            visible_devices: 1,
            host_resident_experts: false,
            split_mode: ananke_config::placement::SplitMode::Layer,
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
        metadata.insert(keys::embedding_length(arch), GgufValue::U32(n_embd));
        if let Some(w) = swa {
            metadata.insert(keys::attention_sliding_window(arch), GgufValue::U32(w));
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
    /// is a change that stops describing the runtime.
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
        /// The architecture these points were measured on. Every term keyed on
        /// it — the flash-attention rate above all — reads a synthetic name as
        /// "never measured" and charges the table's worst default.
        arch: &'static str,
        /// `CUDA_Host compute buffer size`, as llama.cpp logged it.
        measured_mib: f64,
    }

    #[test]
    fn the_arena_model_reproduces_the_hardware_sweep() {
        #[rustfmt::skip]
        let points = [
            // Qwen3-4B: context sweep at a fixed batch.
            Point { n_embd: 2560, layers: 36, swa: None, context: 8192, ubatch: 512, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 12.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 16384, ubatch: 512, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 14.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 18.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 65536, ubatch: 512, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 26.01 },
            // … batch sweep at a fixed context.
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 1024, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 36.02 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 2048, flash_attn: true, parallel: 4, arch: "qwen3", measured_mib: 72.05 },
            // … slot count, which divides the cache and so the mask.
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 1, arch: "qwen3", measured_mib: 42.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: true, parallel: 2, arch: "qwen3", measured_mib: 26.01 },
            // … flash attention off: the mask doubles and a per-token term
            // appears.
            Point { n_embd: 2560, layers: 36, swa: None, context: 16384, ubatch: 512, flash_attn: false, parallel: 4, arch: "qwen3", measured_mib: 22.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 512, flash_attn: false, parallel: 4, arch: "qwen3", measured_mib: 30.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 65536, ubatch: 512, flash_attn: false, parallel: 4, arch: "qwen3", measured_mib: 46.01 },
            Point { n_embd: 2560, layers: 36, swa: None, context: 32768, ubatch: 1024, flash_attn: false, parallel: 4, arch: "qwen3", measured_mib: 60.02 },
            // Qwen3.6-27B: double the hidden size, which doubles the term the
            // mask alone cannot explain.
            Point { n_embd: 5120, layers: 64, swa: None, context: 8192, ubatch: 512, flash_attn: true, parallel: 4, arch: "qwen35", measured_mib: 22.02 },
            Point { n_embd: 5120, layers: 64, swa: None, context: 8192, ubatch: 1024, flash_attn: true, parallel: 4, arch: "qwen35", measured_mib: 44.04 },
            // Gemma-4-31B-QAT: interleaved SWA, so a second window-sized mask.
            Point { n_embd: 5376, layers: 60, swa: Some(1024), context: 8192, ubatch: 512, flash_attn: true, parallel: 4, arch: "gemma4", measured_mib: 24.52 },
        ];
        for p in points {
            let got = pinned_graph_bytes(
                &summary(p.arch, p.n_embd, p.layers, p.swa),
                p.arch,
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
            ananke_gguf::GgufTensor {
                name: SmolStr::new("blk.0.ffn_gate_exps.weight"),
                dtype: ananke_gguf::GgufType::F16,
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
            let predicted = process_base_bytes(&s, false, true) as f64 / MIB;
            let ratio = measured / predicted;
            assert!(
                (0.8..=1.5).contains(&ratio),
                "{layers}-layer MoE: predicted {predicted:.0} MiB against \
                 {measured:.0} MiB measured is a ratio of {ratio:.2}, outside \
                 the band the correction can travel"
            );
        }
        // A dense model of the same size gets no allowance.
        let dense = process_base_bytes(&summary("t", 2048, 41, None), false, true);
        let moe = process_base_bytes(&with_experts(summary("t", 2048, 41, None)), false, true);
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
        let small = process_base_bytes(&summary("t", 2560, 36, None), false, true) as f64 / MIB;
        assert!(
            (small - 233.0).abs() < 8.0,
            "36-layer baseline modelled at {small:.0} MiB against 233 MiB measured"
        );
        let large = process_base_bytes(&summary("t", 5120, 64, None), false, true) as f64 / MIB;
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

        // Both shapes cost more than none, and they cost *different* amounts —
        // which is the point of the assertion. Which of the two is larger is
        // not, and must not be assumed: subtracting one from the other in a
        // fixed order overflows the moment the order flips, and it overflows at
        // compile time, so `cargo test` reports no test failure at all. Hence
        // `abs_diff`.
        assert!(plain < embedded && plain < separate);
        assert_ne!(embedded, separate);
        assert_eq!(
            embedded.abs_diff(separate),
            MTP_HOST_BYTES_EMBEDDED.abs_diff(MTP_HOST_BYTES_SEPARATE_DRAFT)
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

#[cfg(test)]
mod measured_tests {
    use ananke_config::placement::SplitMode;

    use super::*;
    use crate::llama::test_support::{fake_summary, inputs};

    /// Two cards under layer split replicate the masks; one card does not,
    /// and neither does tensor split, where llama.cpp fuses the devices.
    ///
    /// Measured on six architectures at 4.00-4.16, flat across context and
    /// batch — so the difference has to scale with the mask rather than sit
    /// beside it as a flat per-card allowance.
    #[test]
    fn layer_split_replicates_masks_but_tensor_split_does_not() {
        let summary = fake_summary();
        let empty: [String; 0] = [];
        let one = pinned_graph_bytes(&summary, "llama", &inputs("f16", "f16", 32768, &empty));

        let mut two = inputs("f16", "f16", 32768, &empty);
        two.visible_devices = 2;
        two.split_mode = SplitMode::Layer;
        let layered = pinned_graph_bytes(&summary, "llama", &two);

        let mut fused = inputs("f16", "f16", 32768, &empty);
        fused.visible_devices = 2;
        fused.split_mode = SplitMode::Tensor;
        let tensored = pinned_graph_bytes(&summary, "llama", &fused);

        assert!(
            layered > one,
            "layer split across two cards must cost more than one card"
        );
        assert_eq!(
            tensored, one,
            "tensor split fuses the cards, so it must match the single-card figure"
        );
        // The replication applies to the masks and not to the hidden-state
        // buffers, so the ratio sits below the copy count.
        let ratio = layered as f64 / one as f64;
        assert!(
            (1.5..=4.0).contains(&ratio),
            "layer-split arena ratio {ratio:.2} outside the measured range"
        );
    }

    /// Each visible device beyond the first costs host memory for its CUDA
    /// context: 20 MiB, measured on two models with placement pinned to the
    /// CPU so nothing else varied.
    #[test]
    fn each_extra_visible_device_costs_host_memory() {
        // Tensor split throughout, so the mask replication that layer split
        // triggers cannot contaminate what is meant to be a measurement of
        // the per-device context cost alone. Other terms leaking into that
        // delta is exactly what this test guards against; two already have.
        let summary = fake_summary();
        let empty: [String; 0] = [];
        // Two devices against four, not one against four: the tensor-split
        // baseline is charged whenever more than one device is visible, so
        // starting from one would fold that term into the delta as well.
        let mut two = inputs("f16", "f16", 32768, &empty);
        two.visible_devices = 2;
        two.split_mode = SplitMode::Tensor;
        let mut four = inputs("f16", "f16", 32768, &empty);
        four.visible_devices = 4;
        four.split_mode = SplitMode::Tensor;

        let delta = host_overhead_bytes(&summary, "llama", &four)
            - host_overhead_bytes(&summary, "llama", &two);
        assert_eq!(delta, 2 * PROCESS_BASE_BYTES_PER_DEVICE);
    }

    /// The ik CPU-MoE term scales with hidden size rather than being a flat
    /// 81 KiB/token, which is this term at qwen35moe's `n_embd`.
    ///
    /// The rate is per architecture, because the three ik mixtures measured
    /// differ by a third — 41, 43 and 54 — and one number would either
    /// under-reserve the worst or over-reserve the rest.
    #[test]
    fn ik_moe_term_scales_with_hidden_size() {
        // qwen35moe's own rate, at its own hidden size.
        // qwen35moe measures the same rate on one card and two, which is what
        // makes it the right model to assert the hidden-size scaling against —
        // glm-dsa and laguna vary with the device count as well.
        let rate = ik_moe_rate("qwen35moe", 2);
        assert_eq!(
            rate,
            ik_moe_rate("qwen35moe", 1),
            "qwen35moe is measured card-independent"
        );
        let narrow = rate * 2048;
        let wide = rate * 6144;
        assert_eq!(
            wide,
            3 * narrow,
            "three times the hidden size, three times the term"
        );

        // At qwen35moe's hidden size its own rate should land near 81 KiB, the
        // flat figure this model alone yields — agreement there is what shows
        // the term is a hidden-size one.
        let drift = (narrow as f64 - 82_944.0).abs() / 82_944.0;
        assert!(
            drift < 0.10,
            "term at n_embd 2048 is {narrow}, {:.0}% from the flat constant it replaces",
            drift * 100.0
        );
    }
}

/// The per-token cost of a quantised cache for this architecture.
///
/// They span a factor of forty — 61 bytes per batch token on lfm2 against
/// 6144 on deepseek4 — so one value would either under-reserve the worst or
/// over-reserve everything else by about 3 MiB.
fn quantised_cache_rate(arch: &str) -> u64 {
    lookup(
        crate::tuning::QUANTISED_CACHE_RATES,
        arch,
        crate::tuning::QUANTISED_CACHE_RATE_DEFAULT,
    )
}

/// Whether either half of the KV cache is stored quantised.
///
/// A quantised cache costs more pinned host memory than an f16 one, measured
/// in every one of 117 pairs differing in nothing else. The partition is
/// [`cache_type::is_quantised`], shared with the calibration's `KvType` so the
/// fit and the estimate cannot classify a row differently.
pub(crate) fn quantised_kv(inputs: &EstimatorInputs<'_>) -> bool {
    [inputs.cache_type_k, inputs.cache_type_v]
        .iter()
        .flatten()
        .any(|t| cache_type::is_quantised(t))
}
