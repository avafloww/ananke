//! What the harness read back out of the runtime's own logs.
//!
//! The block is flat, and deliberately declared flat here rather than assembled
//! out of `serde(flatten)` groups. Flatten cannot coexist with
//! `deny_unknown_fields`, and `deny_unknown_fields` is the whole point of this
//! crate: a key the schema fails to name has to be a parse error, not a
//! silently dropped column. Two of the three readers this replaces reached the
//! per-card mirrors through an open `BTreeMap<String, Value>` catch-all, which
//! is exactly the shape that lets a column go missing without anyone noticing.
//!
//! Field order is the format — see [`crate::record`].
//!
//! Several figures come in pairs: `<key>` is the value to fit against, and
//! `<key>_all` lists every occurrence, written only when the occurrences
//! disagreed. A cell with `-md` or `--mmproj` loads two models, so a figure can
//! genuinely appear more than once with different values, and the list is what
//! lets a later reader tell the target's figure from the draft's.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use crate::parsed::log::{
    BufferRole, Context, DeviceRow, HostBreakdown, KvPool, Mapped, RsPool,
};

mod log;

/// The model's shape and its logged buffer sizes, as one record's `parsed`
/// block.
///
/// `serde(default)`: a run that failed to load logged nothing to parse, and
/// twenty-five committed rows carry a `parsed` block of one or two keys for
/// that reason. Absent means zero, which is what "the log did not say" has
/// always meant here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parsed {
    /// The pinned graph arena, which llama.cpp reports as its `CUDA_Host
    /// compute buffer size` and plain `CPU` where no GPU is present, in MiB.
    pub arena_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arena_mib_all: Option<Vec<f64>>,
    pub out_buf_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_buf_mib_all: Option<Vec<f64>>,
    /// The host's share of the attention cache, in MiB.
    pub cpu_kv_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_kv_mib_all: Option<Vec<f64>>,
    /// Weights the runtime placed on the host, in MiB.
    pub cpu_model_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model_mib_all: Option<Vec<f64>>,
    pub cpu_model_mapped: Mapped,

    /// The hyperparameters the loader echoes, in the order it prints them.
    ///
    /// Each is the *first* occurrence, not the last: the target model loads
    /// before any draft or projector, and last-wins recorded the draft's shape
    /// for exactly the MTP cells whose target shape the constants are fitted
    /// against.
    pub n_layer: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_layer_all: Option<Vec<u64>>,
    /// Hidden width.
    pub n_embd: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_embd_all: Option<Vec<u64>>,
    pub n_expert: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_expert_all: Option<Vec<u64>>,
    pub n_expert_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_expert_used_all: Option<Vec<u64>>,
    /// The sliding-attention window, zero on a model without one.
    pub n_swa: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_swa_all: Option<Vec<u64>>,
    pub n_vocab: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_vocab_all: Option<Vec<u64>>,
    /// Query heads. Zero where the GGUF omits `attention.head_count`, which is
    /// why the estimator's query-head count has a fallback.
    pub n_head: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_head_all: Option<Vec<u64>>,
    pub n_head_kv: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_head_kv_all: Option<Vec<u64>>,
    pub n_embd_head_k: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_embd_head_k_all: Option<Vec<u64>>,
    pub n_embd_head_v: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_embd_head_v_all: Option<Vec<u64>>,
    pub n_ctx_train: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx_train_all: Option<Vec<u64>>,
    pub n_ff: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ff_all: Option<Vec<u64>>,
    pub ssm_d_conv: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssm_d_conv_all: Option<Vec<u64>>,
    pub ssm_d_inner: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssm_d_inner_all: Option<Vec<u64>>,
    pub ssm_d_state: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssm_d_state_all: Option<Vec<u64>>,
    pub ssm_n_group: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssm_n_group_all: Option<Vec<u64>>,
    pub ssm_dt_rank: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssm_dt_rank_all: Option<Vec<u64>>,
    pub n_group_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_group_used_all: Option<Vec<u64>>,

    /// Every numeric GGUF metadata key the loader echoed, first occurrence
    /// winning.
    ///
    /// The second and last place the format admits a key no struct can name,
    /// and unlike the per-card mirrors it is genuinely unbounded: the keys are
    /// architecture-templated (`{arch}.attention.key_length`), so the
    /// architecture is data inside the key. It stays a map for that reason, but
    /// a *typed* one — the harness writes `i64`, every one of the 13285 values
    /// in the dataset is an integer, and a string-valued key would be a format
    /// change rather than something to absorb silently.
    pub gguf_kv: BTreeMap<String, i64>,
    /// The runtime's own buffer lines, one entry per context it created. The
    /// only route to a per-device figure where there is no breakdown table.
    pub contexts: Vec<Context>,

    /// What a vision projector cost, as llama.cpp's own accounting states it.
    ///
    /// The four keys travel together — the harness writes them from one
    /// `Option<Mmproj>` — but the format nests none of them, so they are four
    /// independently optional fields here rather than one flattened group.
    /// Per device: the projector's weights *and* its CLIP graph buffer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj_reserved_mib: Option<BTreeMap<String, f64>>,
    /// The summed `clip_model_loader` tensor sizes, which isolate the graph
    /// term out of the reservation above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj_tensor_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_image_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_n_merge: Option<u64>,

    /// llama.cpp's `[spec] estimated memory usage of MTP context is N MiB`,
    /// which it reports per context.
    pub mtp_context_mib: f64,
    /// `{arch}.nextn_predict_layers`: how many trailing blocks the embedded MTP
    /// head spans.
    pub nextn_predict_layers: u64,
    /// Set on the Gemma E-variants, which keep their embeddings per layer on
    /// the host. That is a different graph under the same architecture string,
    /// so it discriminates a fit variant.
    pub per_layer_token_embd: bool,
    pub arch: String,
    /// llama.cpp's memory breakdown table, one row per device — or a single
    /// fused `Meta()` row under a tensor split. Empty for a runtime that prints
    /// no table at all, which is ik.
    pub devices: Vec<DeviceRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_breakdown: Option<HostBreakdown>,

    /// Flat mirrors of the first four device rows.
    ///
    /// Redundant with [`Self::devices`], which is authoritative, and kept only
    /// because they are convenient to fit against. The card index lives in the
    /// key name, but the range is closed at four, so they are twenty named
    /// fields rather than a dynamic family: a fifth card is a schema change the
    /// round-trip test should catch.
    pub gpu0_model_mib: u64,
    pub gpu0_kv_mib: u64,
    pub gpu0_compute_mib: u64,
    pub gpu0_unaccounted_mib: u64,
    pub gpu0_self_mib: u64,
    pub gpu1_model_mib: u64,
    pub gpu1_kv_mib: u64,
    pub gpu1_compute_mib: u64,
    pub gpu1_unaccounted_mib: u64,
    pub gpu1_self_mib: u64,
    pub gpu2_model_mib: u64,
    pub gpu2_kv_mib: u64,
    pub gpu2_compute_mib: u64,
    pub gpu2_unaccounted_mib: u64,
    pub gpu2_self_mib: u64,
    pub gpu3_model_mib: u64,
    pub gpu3_kv_mib: u64,
    pub gpu3_compute_mib: u64,
    pub gpu3_unaccounted_mib: u64,
    pub gpu3_self_mib: u64,
}

impl Parsed {
    /// A GGUF metadata integer, absent reading as zero.
    pub fn gguf_int(&self, key: &str) -> i64 {
        self.gguf_kv.get(key).copied().unwrap_or(0)
    }
}
