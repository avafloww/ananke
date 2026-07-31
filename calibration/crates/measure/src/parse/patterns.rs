//! The line patterns a llama-server load log is read through.
//!
//! Every pattern here is a transcription of what one of the two runtimes
//! actually prints, and several carry a note about a shape that a looser pattern
//! mis-reads in silence. Treat the notes as part of the pattern: each names a
//! measurement that comes out wrong, and none of them are hypothetical.

use std::sync::LazyLock;

use regex::Regex;

/// Buffer-size lines the loaders emit, one pattern per figure.
///
/// The last occurrence wins because the loader logs a reserve pass first and
/// then the real graph, with the same figure.
pub(crate) static ARENA: LazyLock<Regex> =
    LazyLock::new(|| build(r"(?:CUDA_Host|CPU) compute buffer size *= *([0-9.]+)"));
pub(crate) static OUT_BUF: LazyLock<Regex> =
    LazyLock::new(|| build(r"(?:CUDA_Host|CPU) +output buffer size *= *([0-9.]+)"));
pub(crate) static CPU_KV: LazyLock<Regex> =
    LazyLock::new(|| build(r"CPU KV buffer size *= *([0-9.]+)"));
pub(crate) static CPU_MODEL: LazyLock<Regex> =
    LazyLock::new(|| build(r"CPU(?:_Mapped)? model buffer size *= *([0-9.]+)"));

/// The hyperparameter summary llama.cpp prints as `print_info:`.
///
/// The alternation order matters: `n_expert` is tried before `n_expert_used`
/// and `n_embd` before `n_embd_head_k`, and the longer key is only reached
/// because the shorter one fails on the `= ` that has to follow it.
///
/// `ssm_*` is the recurrent state's whole shape. `n_embd_r` (the rolling
/// convolution state) and `n_embd_s` (the SSM state) are computed by llama.cpp
/// from these, and both are GGUF metadata — which is what makes the RS term
/// predictable from the model file rather than something to measure.
///
/// `n_layer` is the layer span the contexts cover, which already excludes the
/// MTP head's trailing block — llama.cpp reports the full block count as
/// `n_layer_all`, deliberately *not* captured here because this parser already
/// spells a repeated key `<key>_all` and the two would collide.
/// `nextn_predict_layers` carries the same difference.
pub(crate) static META: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"(n_layer|n_embd|n_expert|n_expert_used|n_swa|n_vocab|n_head_kv|n_head",
        r"|n_embd_head_k|n_embd_head_v|n_ctx_train|n_ff",
        r"|ssm_d_conv|ssm_d_inner|ssm_d_state|ssm_n_group|ssm_dt_rank",
        r"|n_group_used) *= *(\d+)",
    ))
});

/// Every numeric GGUF metadata key the loader echoes.
///
/// The `print_info:` block above covers only the hyperparameters llama.cpp
/// keeps in `hparams`; keys it reads straight into an architecture-specific
/// path — `lfm2.shortconv.l_cache` sizes LFM2's rolling state and is printed
/// nowhere else — are only recoverable from the key/value dump.
pub(crate) static GGUF_KV: LazyLock<Regex> = LazyLock::new(|| {
    build(r"(?m)llama_model_loader: - kv +\d+: +([A-Za-z0-9_.]+) +[a-z0-9]+ += +(-?\d+)\s*$")
});

/// The embedded MTP head's depth, which sizes the modelled KV term.
///
/// It is a metadata key rather than one of llama.cpp's `n_* = ` summary lines,
/// so it needs its own pattern.
pub(crate) static NEXTN: LazyLock<Regex> =
    LazyLock::new(|| build(r"\.nextn_predict_layers[^=]*= *(\d+)"));

/// The tensor `compute_buffer::is_gemma_e_variant` keys on.
///
/// Recorded so the analysis can use the same discriminator the estimator does,
/// rather than a filename proxy that silently disagrees the moment an
/// E-variant ships under another name.
pub(crate) static PER_LAYER_EMBD: LazyLock<Regex> =
    LazyLock::new(|| build(r"per_layer_token_embd\.weight"));

/// llama.cpp's own figure for an MTP context.
///
/// It is the stated calibration source for the constants in
/// `estimator/mtp.rs`, and it is reported *per context* — flat across slot
/// counts while the real cost scales with them — so it is recorded to keep
/// that discrepancy visible rather than to fit to.
pub(crate) static MTP: LazyLock<Regex> =
    LazyLock::new(|| build(r"estimated memory usage of MTP context is *([0-9.]+)"));

/// llama.cpp's own memory-breakdown row, which splits each device into
/// model / context / compute.
///
/// The compute column is what the estimator's per-architecture GPU curves are
/// fitted against, so it has to be captured per device rather than as a total.
///
/// A device row carries every column, and the device is *not* always named
/// `CUDA<n>`: under `--split-mode tensor` llama.cpp fuses the cards and
/// reports a single `Meta()` device. Keying on the CUDA name silently records
/// zeros for every tensor-split run — which is to say for every production
/// configuration.
///
/// Every separator tolerates padding: the columns are right-aligned, so a
/// value with fewer digits than its column is preceded by more spaces. A
/// literal single space after the first `=` silently drops any row whose
/// free-memory figure is narrower than its neighbour's — one card of a
/// two-card breakdown, which leaves the surviving row misaligned against the
/// other card's driver reading and turns one cell's compute target into
/// 2042 MiB against a true 1032.
pub(crate) static BREAKDOWN: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"- (.+?) +\| *(\d+) *= *(\d+) *\+ *\( *(\d+) *= *(\d+) *\+ *(\d+) *\+ *(\d+)\)",
        r" *\+ *(\d+)",
    ))
});

/// The host row of the same table. It has no total/free and no unaccounted
/// column.
pub(crate) static BREAKDOWN_HOST: LazyLock<Regex> =
    LazyLock::new(|| build(r"- Host +\| *(\d+) = *(\d+) \+ *(\d+) \+ *(\d+)"));

pub(crate) static ARCH: LazyLock<Regex> = LazyLock::new(|| build(r"arch *= *([A-Za-z0-9_.-]+)"));

/// The end of one context's allocation log.
///
/// A server creates more than one context: the main one, a sliding-window
/// sibling on an interleaved-SWA model, and an MTP draft context under
/// `--spec-type draft-mtp`. Each prints its own memory pools and its own
/// compute reserve, and a whole-log sweep collapses them — which hides the MTP
/// context's compute buffer inside an opaque constant. So the log is segmented
/// into contexts first (each ends with its `graph nodes` line) and every figure
/// is attributed to the context that allocated it.
///
/// `graph splits` follows `graph nodes` on the next line, so a boundary that
/// stops at the latter attributes every split count to the *following* context.
pub(crate) static CONTEXT_END: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"(?m)^.*(?:sched_reserve|llama_init_from_model): graph nodes.*$",
        r"(?:\n^.*: graph splits.*$)?",
    ))
});

/// The attention cache's own summary line: the physical total across devices,
/// the cell count (per sequence), the layer count that actually allocates, the
/// sequence count, and the K/V split with the cache types in play.
///
/// Together these say exactly what llama.cpp sized, so a modelled KV can be
/// checked term by term instead of only in aggregate.
pub(crate) static KV_POOL: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"llama_kv_cache: size = *([0-9.]+) MiB *\( *(\d+) cells, *(\d+) layers, ",
        r"*(\d+)/(\d+) seqs\), K \(([a-z0-9_]+)\): *([0-9.]+) MiB, ",
        r"V \(([a-z0-9_]+)\): *([0-9.]+) MiB",
    ))
});

/// The recurrent module's equivalent.
///
/// `rs_seq` is the speculative rollback depth: the state is replicated
/// `n_seq × (rs_seq + 1)` times, and `rs_seq` is non-zero only under
/// speculative decoding.
pub(crate) static RS_POOL: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"llama_memory_recurrent: size = *([0-9.]+) MiB *\( *(\d+) cells, *(\d+) layers, ",
        r"*(\d+) seqs *(\d+) rs_seq\), R \([a-z0-9_]+\): *([0-9.]+) MiB, ",
        r"S \([a-z0-9_]+\): *([0-9.]+) MiB",
    ))
});

/// Per-device buffer lines, whatever the loader called the stage.
///
/// `Meta()` is the fused device a tensor split reports, and its figure is ONE
/// card's share. `llm_load_tensors` is ik_llama's spelling and it omits the
/// word `model` entirely — `CUDA0 buffer size = 6992.89` — so a pattern
/// demanding the kind records nothing at all for the fork, which is most of
/// what runs in production. The kind is taken from the stage when the line
/// does not name one.
pub(crate) static DEV_BUFFER: LazyLock<Regex> = LazyLock::new(|| {
    build(concat!(
        r"(load_tensors|llm_load_tensors|llama_kv_cache(?:_init)?|llama_memory_recurrent",
        r"|sched_reserve|llama_init_from_model|llama_context): +([A-Za-z0-9_()]+) +",
        r"(?:(model|KV|RS|compute|output) +)?buffer size *= *([0-9.]+)",
    ))
});

pub(crate) static GRAPH_SHAPE: LazyLock<Regex> =
    LazyLock::new(|| build(r"graph (nodes|splits) += *(\d+)"));

/// What a vision projector costs, as llama.cpp's own accounting states it.
///
/// The `fit_params_target` line is the whole per-device figure — the
/// projector's weights *and* its CLIP graph buffer — so pairing it with the
/// summed tensor sizes below isolates the graph term. The two are printed on
/// different lines and neither is derivable from the mmproj file's size (that
/// includes GGUF framing).
pub(crate) static MMPROJ_RESERVED: LazyLock<Regex> = LazyLock::new(|| {
    build(r"adding ([0-9.]+) MiB to fit_params_target for device ([A-Za-z0-9_()]+)")
});

pub(crate) static CLIP_TENSOR: LazyLock<Regex> =
    LazyLock::new(|| build(r"clip_model_loader: tensor\[\d+\]:.*tensor_size=(\d+)"));

/// The two vision settings that differ between the projectors measured, and so
/// the first candidates for what the graph buffer scales with.
///
/// Recorded rather than used: three cells across two configurations cannot
/// distinguish a rate in either of them from a constant.
pub(crate) static CLIP_IMAGE_SIZE: LazyLock<Regex> =
    LazyLock::new(|| build(r"image size = (\d+) x (\d+)"));
pub(crate) static CLIP_MERGE: LazyLock<Regex> = LazyLock::new(|| build(r"n_merge: *(\d+)"));

/// Every pattern in this module is a literal in the crate, so a failure to
/// compile one is a programming error rather than something a caller can act
/// on.
fn build(pattern: &str) -> Regex {
    Regex::new(pattern).expect("pattern is a well-formed literal")
}
