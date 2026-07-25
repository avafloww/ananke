//! Fields and sub-tables shared across every template variant: the
//! `RawServiceCommon` block flattened into each service, plus its nested
//! tracking, filters, device-placement, and health tables.

use std::collections::BTreeMap;

use serde::Deserialize;
use smol_str::SmolStr;

use crate::config::parse::RawAutoRestart;

/// Fields shared by every template variant. Flattened into each variant so
/// users write `name = "x"` at the top level of `[[service]]` rather than
/// under a nested table.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceCommon {
    pub name: Option<SmolStr>,
    pub extends: Option<SmolStr>,
    pub migrate_from: Option<SmolStr>,
    pub port: Option<u16>,
    pub lifecycle: Option<SmolStr>,
    pub priority: Option<u8>,
    pub idle_timeout: Option<String>,
    pub description: Option<String>,
    /// What kind of OpenAI endpoint the service serves: `"chat"`
    /// (default) or `"embedding"`. Embedding services advertise
    /// themselves through `/v1/models` + `/api/services` with a typed
    /// `modality` field clients can filter on, and the frontend renders
    /// an `embedding` badge alongside the service name. Validated into
    /// the [`ananke_api::shared::modality::Modality`] enum during config validation; an
    /// unknown string is a hard config error rather than a silent fall
    /// back to chat.
    pub modality: Option<SmolStr>,
    pub filters: Option<RawFilters>,
    pub metadata: Option<BTreeMap<String, toml::Value>>,
    pub devices: Option<RawServiceDevices>,
    pub extra_args: Option<Vec<String>>,
    pub extra_args_append: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, String>>,
    /// Whether the child process inherits the daemon's environment
    /// (default `true`). When `false`, the child sees only the
    /// variables in `env` plus `CUDA_VISIBLE_DEVICES`.
    pub env_inherit: Option<bool>,
    pub health: Option<RawHealth>,
    pub drain_timeout: Option<String>,
    pub extended_stream_drain: Option<String>,
    pub max_request_duration: Option<String>,
    pub start_queue_depth: Option<usize>,
    pub tracking: Option<RawTracking>,
    /// Self-healing restart policy. See [`RawAutoRestart`]. Resolved as a
    /// whole block: a service that sets any `auto_restart` field replaces
    /// `[defaults.auto_restart]` entirely rather than merging field-by-field.
    pub auto_restart: Option<RawAutoRestart>,
}

/// `[service.tracking]` block. Optional per-service hints that adjust how
/// the snapshotter attributes observed VRAM/RSS to the service.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawTracking {
    /// Cgroup v2 path (e.g. `/system.slice/ananke-comfyui.slice`) under
    /// which the service's actual workload pids live. Used by services
    /// whose workload runs in a container and is therefore reparented
    /// out of the daemon's process tree, so descendant-pid attribution
    /// can't reach it. Pids whose `/proc/<pid>/cgroup` path equals this
    /// value or sits inside its subtree are summed into the service's
    /// observed peak.
    pub cgroup_parent: Option<SmolStr>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawFilters {
    pub strip_params: Option<Vec<String>>,
    pub set_params: Option<BTreeMap<String, toml::Value>>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceDevices {
    pub placement: Option<SmolStr>,
    pub gpu_allow: Option<Vec<u32>>,
    pub placement_override: Option<BTreeMap<String, u64>>,
    /// Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added
    /// on top of the global `[devices]` reserve. Lets a single model be packed
    /// more conservatively (headroom for KV growth, a co-resident service, or
    /// estimator slack) without bypassing the estimator.
    pub gpu_headroom_mb: Option<u64>,
    /// `--split-mode` for multi-GPU llama.cpp services: `"layer"` (default),
    /// `"row"`, or `"tensor"`. Validated into [`crate::config::SplitMode`].
    pub split: Option<SmolStr>,
    /// Optional per-GPU weights for the `--tensor-split` ratio in sharded
    /// (`row`/`tensor`) modes. One positive float per allowed GPU, in ascending
    /// GPU-id order. Unset keeps the historical equal `1,1,...` split.
    pub tensor_split_weights: Option<Vec<f32>>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawHealth {
    pub http: Option<String>,
    pub timeout: Option<String>,
    pub probe_interval: Option<String>,
}
