//! Multi-token-prediction (MTP / NextN) draft-context overhead.
//!
//! MTP ships in two shapes, and the estimator models both:
//!
//! **Embedded head (Qwen 3.6).** With `--spec-type draft-mtp` and no separate
//! draft GGUF, llama.cpp creates a second context against the *same* target
//! model. Its KV cache covers only the trailing `nextn_predict_layers` blocks
//! — the dense-attention MTP head — and uses the *draft* cache types, which
//! default to f16 regardless of the main context's `--cache-type-*`. No extra
//! weights load: the nextn-layer tensors live in the target GGUF and are
//! resident even without MTP. So the cost is `nextn KV (f16) + a roughly
//! constant compute buffer`. Calibrated against llama.cpp's own `[spec]
//! estimated memory usage of MTP context` figure on Qwen 3.6 27B (`qwen35`,
//! 4 KV heads) and 35B-A3B (`qwen35moe`, 2 KV heads) across context
//! (262144/524288) and parallelism (np 1/2): the KV term tracks `kv_heads ×
//! context` exactly, and the compute term sits at ~1.55–1.61 GiB independent
//! of both knobs (it is driven by the shared-tokenizer logit buffer at
//! `n_ubatch`, not model width).
//!
//! **Separate draft model (Gemma 4).** With `-md <file>` the MTP head is a
//! standalone GGUF (Gemma 4's `gemma4-assistant`, a 4-block model). Its
//! attention layers *share the target model's KV cache* — confirmed in the
//! load log (`llama_kv_cache: layer 3: sharing with layer 59`) — so it adds
//! no context-scaling KV. The whole cost is its GPU-resident weights
//! (everything but the CPU-side token embeddings) plus a small, roughly
//! constant draft compute/logit buffer. Calibrated against the production
//! 2×3090 run (peak 40858 MiB total) minus the target+mmproj estimate: the
//! draft contributes ~400 MiB, of which ~108 MiB is weights.

use ananke_gguf::GgufSummary;

use crate::{
    tuning::{
        DRAFT_MODEL_COMPUTE_MIB, DRAFT_MODEL_COMPUTE_MIB_PER_1K, MTP_DRAFT_COMPUTE_BASE_MIB,
        MTP_DRAFT_COMPUTE_BASE_MIB_DEFAULT, MTP_DRAFT_COMPUTE_MIB_PER_1K,
        MTP_DRAFT_COMPUTE_MIB_PER_1K_DEFAULT, MTP_UNACCOUNTED_MIB_PER_DEVICE,
    },
    types::EstimatorInputs,
};

/// GPU-resident weight bytes for a separate draft model: every tensor except
/// the token embeddings, which llama.cpp keeps on CPU (same rule as the target
/// model's `token_embd.weight`).
fn draft_model_gpu_weight_bytes(draft: &GgufSummary) -> u64 {
    let token_embd = draft
        .tensors
        .get("token_embd.weight")
        .map(|t| t.byte_size)
        .unwrap_or(0);
    draft.total_tensor_bytes.saturating_sub(token_embd)
}

/// Extra VRAM (bytes) a separate draft model (`-md`) adds: its GPU-resident
/// weights plus a fixed draft compute buffer. The draft's attention layers
/// reuse the target's KV cache, so there is no context-scaling KV term.
fn separate_draft_overhead_bytes(draft: &GgufSummary, context: u32) -> u64 {
    // The draft shares the target's KV cache — the load log shows `layer 3:
    // sharing with layer 59` — so there is no context-scaling cache term. Its
    // *compute* scales anyway: gemma-4-31B-QAT measures a driver delta of 724,
    // 788, and 920 MiB at ctx 32768, 65536, and 131072, where a flat constant
    // modelled 407 at each and so under-reserved by 317 to 513 MiB.
    //
    // Flat in the slot count, unlike an embedded head: 724, 724, and 728 MiB
    // at one, two, and four slots. That is the control confirming the embedded
    // head's slot scaling comes from keeping its own cache per slot.
    let compute =
        DRAFT_MODEL_COMPUTE_MIB + DRAFT_MODEL_COMPUTE_MIB_PER_1K * u64::from(context) / 1024;
    draft_model_gpu_weight_bytes(draft) + compute * 1024 * 1024
}

/// The share of [`mtp_overhead_bytes`] that is model tensors read from a GGUF
/// rather than memory the runtime allocates.
///
/// Non-zero only for a separate draft model (`-md`), whose weights are read
/// through its own mmap and therefore land in the process's file RSS the same
/// way the target model's do. An embedded MTP head allocates a KV cache and a
/// compute buffer and reads no additional tensors — its layers are resident as
/// part of the target model regardless.
///
/// The packer charges this part as `Charge`'s
/// weights so it reaches `gpu_weight_bytes`, which the host-pool observation
/// subtracts from a measured RSS peak. Left as runtime, the draft's weights
/// would inflate every host sample of an MTP service.
pub fn mtp_weight_bytes(draft: Option<&GgufSummary>, inputs: &EstimatorInputs<'_>) -> u64 {
    if !inputs.mtp {
        return 0;
    }
    draft.map(draft_model_gpu_weight_bytes).unwrap_or(0)
}

/// Extra VRAM (bytes) the MTP draft context adds, or 0 when MTP is off or the
/// model carries no MTP head (`{arch}.nextn_predict_layers` absent or zero and
/// no separate draft model).
///
/// When `draft` is `Some`, the service runs with a separate draft GGUF (`-md`)
/// and the overhead is read from that file. Otherwise the target model's
/// embedded MTP head is modelled.
///
/// `inputs.context` is the configured total context. The embedded-head KV
/// scales with it linearly the same way the main KV does — total KV tokens
/// equal the context budget whether the main cache is unified (np auto) or
/// split per-slot (np > 1), so the estimator does not need to know the
/// parallelism.
pub fn mtp_overhead_bytes(
    summary: &GgufSummary,
    draft: Option<&GgufSummary>,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if !inputs.mtp {
        return 0;
    }
    if let Some(draft) = draft {
        return separate_draft_overhead_bytes(draft, inputs.context);
    }
    let arch = summary.architecture.as_str();
    let nextn = meta_u32(summary, arch, "nextn_predict_layers").unwrap_or(0) as u64;
    if nextn == 0 {
        // `--spec-type draft-mtp` was requested but this model has no MTP
        // head; llama.cpp would refuse to draft, so there is no extra cost.
        return 0;
    }
    // The MTP head is a full-attention layer; `head_count_kv` is a scalar on
    // the qwen35 / qwen35moe families that ship MTP heads today.
    let n_kv_heads = meta_attn_u32(summary, arch, "head_count_kv").unwrap_or(0) as u64;
    if n_kv_heads == 0 {
        return 0;
    }
    let context = inputs.context as u64;

    // llama.cpp states this cost itself, and states it in two parts:
    //
    //     [spec] estimated memory usage of MTP context is N MiB
    //
    // is exactly the draft cache's *physical* size plus **one** device's share
    // of that context's compute buffer — verified to the second decimal on all
    // 13 measured MTP cells, across two architectures, four contexts, and slot
    // counts 1, 2, and 4. So the two terms are modelled separately, because
    // they land differently: the cache is split across the spanned devices
    // while the compute buffer is built on each of them.
    //
    // Neither term scales with the slot count. There is one MTP context, not
    // one per slot, and its cache covers the whole context budget however that
    // budget is divided: `mtpslot-*-mtp-np{1,2,4}` all report the same figure.
    let key_length = meta_attn_u32(summary, arch, "key_length").unwrap_or(0) as u64;
    let value_length = meta_attn_u32(summary, arch, "value_length").unwrap_or(0) as u64;
    // The MTP draft context always uses f16 for its KV cache, independent of
    // the main cache type.
    let bytes_per_element = crate::kv::kv_bytes_per_element("f16");
    let kv_bytes = nextn
        * n_kv_heads
        * (((key_length + value_length) as f64) * bytes_per_element) as u64
        * context;
    let devices = u64::from(inputs.visible_devices.max(1));
    kv_bytes
        + devices
            * (draft_compute_share_bytes(summary, arch, inputs)
                + MTP_UNACCOUNTED_MIB_PER_DEVICE * 1024 * 1024)
}

/// One device's share of the MTP draft context's compute buffer.
///
/// This is the draft context's own graph. Two other things MTP costs are
/// modelled elsewhere and must not be added here, or they are counted twice:
/// the recurrent module's speculative rollback copies, which
/// [`crate::recurrent`] scales by `parallel x (rs_seq + 1)` and which
/// account for most of the driver delta's growth with the slot count (188 MiB at
/// one slot against 754 at four, on Qwen3.6-35B-A3B), and the main context's own
/// compute, which the unified compute model covers.
///
/// The same two terms as the main context's — see
/// [`crate::compute_buffer::tensor_split_per_device`] — because it
/// is the same kind of graph over one layer instead of all of them: an f16 KQ
/// mask of two bytes per (batch token, cache token), plus a flat count of
/// hidden-width f32 intermediates per batch token.
///
/// The mask term is what makes this grow with context at all, and it accounts
/// for the growth exactly: Qwen3.6-27B's share net of the mask is 68 MiB at
/// contexts 65536, 131072, and 360448 alike, and Qwen3.6-35B-A3B's is 40 MiB
/// at 65536 and 524288. Both sit ~30 and ~4 MiB higher at ctx 32768, so the
/// per-architecture count is taken from the worst cell and over-reserves the
/// long contexts by that much.
fn draft_compute_share_bytes(
    _summary: &GgufSummary,
    arch: &str,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    let base = rate(
        MTP_DRAFT_COMPUTE_BASE_MIB,
        arch,
        MTP_DRAFT_COMPUTE_BASE_MIB_DEFAULT,
    );
    let slope = rate(
        MTP_DRAFT_COMPUTE_MIB_PER_1K,
        arch,
        MTP_DRAFT_COMPUTE_MIB_PER_1K_DEFAULT,
    );
    let mib = base + slope * u64::from(inputs.context / 1024) / 1000;
    mib * 1024 * 1024
}

/// A per-architecture rate, or the table's default.
fn rate(table: &[(&str, u64)], arch: &str, fallback: u64) -> u64 {
    table
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, v)| *v)
        .unwrap_or(fallback)
}

fn meta_u32(summary: &GgufSummary, arch: &str, key: &str) -> Option<u32> {
    summary
        .metadata
        .get(&*format!("{arch}.{key}"))
        .and_then(|v| v.as_u32())
}

fn meta_attn_u32(summary: &GgufSummary, arch: &str, key: &str) -> Option<u32> {
    summary
        .metadata
        .get(&*format!("{arch}.attention.{key}"))
        .and_then(|v| v.as_u32())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use ananke_gguf::types::{GgufSummary, GgufValue};
    use smol_str::SmolStr;

    use super::*;

    fn qwen35_summary(arch: &str, nextn: u32, kv_heads: u32) -> GgufSummary {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String(arch.into()),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.nextn_predict_layers")),
            GgufValue::U32(nextn),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.head_count_kv")),
            GgufValue::U32(kv_heads),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.key_length")),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.value_length")),
            GgufValue::U32(256),
        );
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata,
            block_count: Some(65),
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs(context: u32, mtp: bool, empty: &[String]) -> EstimatorInputs<'_> {
        EstimatorInputs {
            host_resident_experts: false,
            visible_devices: 1,
            split_mode: ananke_config::placement::SplitMode::Layer,
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            ubatch: None,
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            override_tensor: empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp,
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
    fn zero_when_mtp_disabled() {
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&s, None, &inputs(262144, false, &empty)),
            0
        );
    }

    #[test]
    fn zero_when_no_mtp_head() {
        // MTP requested but the model has nextn_predict_layers = 0.
        let s = qwen35_summary("qwen35", 0, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&s, None, &inputs(262144, true, &empty)),
            0
        );
    }

    /// Both production embedded-MTP cells, against what llama.cpp itself said
    /// the context would cost.
    ///
    /// `[spec] estimated memory usage of MTP context is N MiB` is the draft
    /// cache's physical size plus **one** device's compute share, so a two-card
    /// span pays the reported figure plus a second copy of the share. That is
    /// the whole model, and these are the only two configurations it has been
    /// measured against at production scale.
    #[test]
    fn embedded_head_reproduces_the_runtimes_own_figure() {
        // (arch, kv heads, n_embd, context, slots, reported MiB, cache MiB)
        let cells = [
            // prod-qwen36-27b: `is 1652.02 MiB`, of which 1408 is the cache.
            ("qwen35", 4u32, 5120u32, 360448u32, 2u32, 1652u64, 1408u64),
            // prod-qwen36-35b-a3b: `is 1320.02 MiB`, of which 1024 is cache.
            ("qwen35moe", 2, 2048, 524288, 2, 1320, 1024),
        ];
        let empty: Vec<String> = Vec::new();
        for (arch, kv_heads, n_embd, context, slots, reported, cache) in cells {
            let mut s = qwen35_summary(arch, 1, kv_heads);
            s.metadata.insert(
                SmolStr::new(format!("{arch}.embedding_length")),
                GgufValue::U32(n_embd),
            );
            let mut i = inputs(context, true, &empty);
            i.parallel = Some(slots);
            i.visible_devices = 2;
            let mib = mtp_overhead_bytes(&s, None, &i) / (1024 * 1024);

            // The cache term is exact: `nextn x heads x (kl + vl) x 2 x ctx`.
            let modelled_cache = u64::from(kv_heads) * 512 * 2 * u64::from(context) / (1024 * 1024);
            assert_eq!(modelled_cache, cache, "{arch} draft cache");

            // One share is `reported - cache`; two cards pay it twice. On top of
            // the runtime's own books sits the per-device remainder the driver
            // shows and the books do not — see
            // `MTP_UNACCOUNTED_MIB_PER_DEVICE`, derived as the residual of the
            // paired with/without deltas — so the reservation must exceed the
            // reported figure by about that much rather than match it.
            let share = reported - cache;
            let books = cache + 2 * share;
            let expected = books + 2 * MTP_UNACCOUNTED_MIB_PER_DEVICE;
            assert!(
                mib >= books,
                "{arch}: {mib} MiB must cover the cache plus both cards' \
                 {share} MiB share"
            );
            assert!(
                mib <= expected + 96,
                "{arch}: {mib} MiB over-reserves against {expected} MiB \
                 ({books} on the runtime's books plus the measured remainder)"
            );
        }
    }

    #[test]
    fn uses_f16_draft_cache_not_main_cache_type() {
        // The draft context always caches in f16. A q8_0 *main* cache — which
        // `inputs` sets — must not shrink this term.
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        let mib = mtp_overhead_bytes(&s, None, &inputs(262144, true, &empty)) / (1024 * 1024);
        // nextn 1 x 4 heads x (256 + 256) x 2 bytes x 262144 = 1024 MiB of
        // cache alone, which a q8_0 rate would have cut to 544.
        assert!(mib >= 1024, "the draft cache must be priced at f16: {mib}");
    }

    /// One MTP context, not one per slot. Its cache covers the whole context
    /// budget however that budget is divided, and llama.cpp reports the same
    /// figure at one, two, and four slots — the `mtpslot-*-mtp-np{1,2,4}` cells
    /// all read 258.02 MiB for the 27B.
    #[test]
    fn overhead_does_not_scale_with_slots() {
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        let at = |slots: u32| {
            let mut i = inputs(262144, true, &empty);
            i.parallel = Some(slots);
            i.kv_unified = Some(true);
            mtp_overhead_bytes(&s, None, &i)
        };
        assert_eq!(at(1), at(2));
        assert_eq!(at(1), at(4));
    }

    /// Build a separate-draft GGUF summary (Gemma 4's `gemma4-assistant`
    /// shape): a `token_embd.weight` kept on CPU plus the GPU-resident
    /// remainder, with `total_tensor_bytes` summing both.
    fn draft_summary(token_embd_mib: u64, gpu_weight_mib: u64) -> GgufSummary {
        use ananke_gguf::types::{GgufTensor, GgufType};
        let mut tensors = BTreeMap::new();
        let mk = |name: &str, bytes: u64| GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        };
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            mk("token_embd.weight", token_embd_mib * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("blk.0.attn_q.weight"),
            mk("blk.0.attn_q.weight", gpu_weight_mib * 1024 * 1024),
        );
        GgufSummary {
            path: "/fake-draft".into(),
            total_tensor_bytes: (token_embd_mib + gpu_weight_mib) * 1024 * 1024,
            tensors,
            metadata: BTreeMap::new(),
            block_count: Some(4),
            architecture: SmolStr::new("gemma4-assistant"),
            shards: vec!["/fake-draft".into()],
        }
    }

    #[test]
    fn separate_draft_counts_gpu_weights_plus_compute_not_kv() {
        // The target carries no embedded MTP head (gemma4, nextn = 0), so
        // without a draft model the overhead would be zero. With a separate
        // draft it is the draft's GPU-resident weights plus a compute term.
        let target = qwen35_summary("gemma4", 0, 4);
        let draft = draft_summary(144, 108);
        let empty: Vec<String> = Vec::new();
        let at = |context: u32| {
            mtp_overhead_bytes(&target, Some(&draft), &inputs(context, true, &empty))
                / (1024 * 1024)
        };
        let expected = |context: u64| {
            108 + DRAFT_MODEL_COMPUTE_MIB + DRAFT_MODEL_COMPUTE_MIB_PER_1K * context / 1024
        };
        assert_eq!(at(204800), expected(204800));

        // It grows with context — the compute buffer does, even though the KV
        // does not — but far too slowly to be a cache. A shared-KV draft adds
        // single-digit MiB per 1024 tokens where its own cache would add
        // hundreds of MiB over this range.
        let growth = at(409600) - at(204800);
        assert_eq!(growth, DRAFT_MODEL_COMPUTE_MIB_PER_1K * 200);
        assert!(
            growth < 108 + DRAFT_MODEL_COMPUTE_MIB,
            "grew like a KV cache"
        );
    }

    #[test]
    fn separate_draft_does_not_scale_with_slots() {
        // Measured flat at 724, 724, and 728 MiB across one, two, and four
        // slots, against an embedded head that doubles: the draft shares the
        // target's cache and so has none of its own to replicate per slot.
        let target = qwen35_summary("gemma4", 0, 4);
        let draft = draft_summary(144, 108);
        let empty: Vec<String> = Vec::new();
        let at = |slots: u32| {
            let mut i = inputs(32768, true, &empty);
            i.parallel = Some(slots);
            mtp_overhead_bytes(&target, Some(&draft), &i)
        };
        assert_eq!(at(1), at(4));
    }

    #[test]
    fn separate_draft_ignored_when_mtp_disabled() {
        let target = qwen35_summary("gemma4", 0, 4);
        let draft = draft_summary(144, 108);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&target, Some(&draft), &inputs(204800, false, &empty)),
            0
        );
    }
}
