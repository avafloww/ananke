//! The structured pieces the loader's own log yields: its memory-breakdown
//! table and its per-context buffer lines.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Whether the host-side weights landed in `RssFile` or in anonymous memory.
/// Read from the loader's own naming (`CPU_Mapped model buffer size` against
/// `CPU model buffer size`) rather than inferred from flags, because mainline
/// and ik_llama disagree on it for identical configurations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mapped {
    Yes,
    #[default]
    No,
}

/// One device's row of llama.cpp's memory breakdown table.
///
/// A tensor split reports a single fused row — recognised by a `device` name
/// starting `Meta` — whose `total`, `free`, and `self` are summed across cards
/// but whose `model`, `kv`, and `compute` are one card's share.
/// `unaccounted_mib` is what the driver reports minus what llama.cpp can
/// attribute, the term the GPU compute-buffer bases carry as a margin.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeviceRow {
    pub device: String,
    pub total_mib: u64,
    pub free_mib: u64,
    pub self_mib: u64,
    pub model_mib: u64,
    pub kv_mib: u64,
    pub compute_mib: u64,
    pub unaccounted_mib: u64,
}

/// The host row, which has no total/free and no unaccounted column.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HostBreakdown {
    pub self_mib: u64,
    pub model_mib: u64,
    pub kv_mib: u64,
    pub compute_mib: u64,
}

/// Everything the loader printed for one context, up to and including its
/// `graph nodes` line.
///
/// The first segment of a run belongs to llama.cpp's parameter-fitting dry run,
/// which reports the same shape with no weights loaded — so segments are kept
/// whole rather than merged, and a reader picks by the pools each holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Context {
    /// Per device, then per role, as the loader logged it. `Meta()` is the
    /// fused device a tensor split reports, and its figure is ONE card's share.
    /// The device names are backend names (`CUDA0`, `CUDA_Host`, `CPU_Mapped`)
    /// that vary with the build and the split; the roles are closed.
    pub buffers: BTreeMap<String, BTreeMap<BufferRole, f64>>,
    /// The attention caches this context allocated, one entry per span of
    /// layers sharing a pool.
    pub kv_pools: Vec<KvPool>,
    /// The recurrent-state module, on a hybrid or recurrent model. A context
    /// creates at most one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rs_pool: Option<RsPool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_nodes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_splits: Option<u64>,
}

/// What a per-device buffer line was holding. The variant order is the
/// serialized key order, since the map is a `BTreeMap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BufferRole {
    Model,
    Kv,
    Rs,
    Compute,
    Output,
}

/// One `llama_kv_cache` pool, as the load log summarises it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvPool {
    /// The physical total across devices.
    pub total_mib: f64,
    /// Cells per sequence.
    pub cells: u64,
    /// The layer count that actually allocates, which excludes an MTP head.
    pub layers: u64,
    pub seqs: u64,
    pub seqs_max: u64,
    pub k_type: String,
    pub k_mib: f64,
    pub v_type: String,
    pub v_mib: f64,
}

/// One `llama_memory_recurrent` module, as the load log summarises it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RsPool {
    pub total_mib: f64,
    pub cells: u64,
    /// The span the module covers. The attention layers inside it do not
    /// allocate, so this is larger than the number of recurrent layers.
    pub layers: u64,
    pub seqs: u64,
    /// llama.cpp's `n_rs_seq`: the speculative rollback depth. The state is
    /// replicated `seqs × (rs_seq + 1)` times, and this is non-zero only under
    /// speculative decoding.
    pub rs_seq: u64,
    pub r_mib: f64,
    pub s_mib: f64,
}
