//! Fields and sub-tables shared across every template variant: the
//! `RawServiceCommon` block flattened into each service, plus its nested
//! tracking, filters, device-placement, and health tables.

use std::collections::BTreeMap;

use serde::Deserialize;
use smol_str::SmolStr;

use crate::parse::RawAutoRestart;

/// Fields shared by every template variant. Flattened into each variant so
/// users write `name = "x"` at the top level of `[[service]]` rather than
/// under a nested table.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceCommon {
    /// Service name, the key used by the API and CLI.
    pub name: Option<SmolStr>,
    /// Name of another service whose fields are merged under this one.
    pub extends: Option<SmolStr>,
    /// Previous name this service is migrated from, for database reparenting.
    pub migrate_from: Option<SmolStr>,
    /// Public port clients connect to.
    pub port: Option<u16>,
    /// When the service starts and stops: `"persistent"` or `"on_demand"`.
    pub lifecycle: Option<SmolStr>,
    /// Start priority; lower numbers start first.
    pub priority: Option<u8>,
    /// How long an idle on-demand service may stay up.
    pub idle_timeout: Option<String>,
    /// Free-form description surfaced through the management API.
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
    /// Request-scrubbing rules applied before forwarding.
    pub filters: Option<RawFilters>,
    /// Arbitrary key/value metadata attached to the service, exposed through
    /// the management API and usable for service discovery.
    pub metadata: Option<BTreeMap<String, toml::Value>>,
    /// Which GPUs and split the service is allowed to use.
    pub devices: Option<RawServiceDevices>,
    /// Extra arguments appended to the resolved command line.
    pub extra_args: Option<Vec<String>>,
    /// Extra arguments appended after any inherited `extra_args`.
    pub extra_args_append: Option<Vec<String>>,
    /// Environment variables set on the child process.
    pub env: Option<BTreeMap<String, String>>,
    /// Whether the child process inherits the daemon's environment
    /// (default `true`). When `false`, the child sees only the
    /// variables in `env` plus `CUDA_VISIBLE_DEVICES`.
    pub env_inherit: Option<bool>,
    /// HTTP readiness probe for the child.
    pub health: Option<RawHealth>,
    /// How long to wait for a draining service to exit before killing it.
    pub drain_timeout: Option<String>,
    /// How long to keep streaming drained connections open after the child exits.
    pub extended_stream_drain: Option<String>,
    /// Maximum wall-clock time for a single inference request.
    pub max_request_duration: Option<String>,
    /// Capacity of the queue for requests arriving while the service is starting.
    pub start_queue_depth: Option<usize>,
    /// Snapshotter attribution hints for this service.
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

/// `[service.filters]` request-scrubbing rules applied before a request is
/// forwarded to the service.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawFilters {
    /// Query parameters stripped from every proxied request.
    pub strip_params: Option<Vec<String>>,
    /// Query parameters injected (or overridden) on every proxied request.
    pub set_params: Option<BTreeMap<String, toml::Value>>,
}

/// `[service.devices]` block: placement mode and the GPU allow-list.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceDevices {
    /// Placement policy name from the global `[devices.placement]` table.
    pub placement: Option<SmolStr>,
    /// GPU indices this service may be placed on.
    pub gpu_allow: Option<Vec<u32>>,
    /// Per-GPU overrides of the placement policy's reservation weights.
    pub placement_override: Option<BTreeMap<String, u64>>,
    /// Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added
    /// on top of the global `[devices]` reserve. Lets a single model be packed
    /// more conservatively (headroom for KV growth, a co-resident service, or
    /// estimator slack) without bypassing the estimator.
    pub gpu_headroom_mb: Option<u64>,
    /// `--split-mode` for multi-GPU llama.cpp services: `"layer"` (default),
    /// `"row"`, or `"tensor"`. Validated into [`crate::SplitMode`].
    pub split: Option<SmolStr>,
    /// Optional per-GPU weights for the `--tensor-split` ratio in sharded
    /// (`row`/`tensor`) modes. One positive float per allowed GPU, in ascending
    /// GPU-id order. Unset keeps the historical equal `1,1,...` split.
    pub tensor_split_weights: Option<Vec<f32>>,
}

/// `[service.health]` block: HTTP readiness probe for the child.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawHealth {
    /// URL the daemon probes to declare the service ready.
    pub http: Option<String>,
    /// How long a probe may take before it counts as failed.
    pub timeout: Option<String>,
    /// How often the probe runs while the service is up.
    pub probe_interval: Option<String>,
}
