//! `GET /api/services/{name}/command` — launch command preview.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Whether a [`LaunchCommand`] describes a live process or a what-if.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LaunchCommandSource {
    /// The service is running; this configuration is what it was launched
    /// with (recomputed from the current config and placement, so it matches
    /// the live process unless the config was edited since it started).
    Running,
    /// The service is not running; this is the command it would launch with
    /// on the next start, given the current config and device state.
    Preview,
}

/// One environment variable ananke sets on the child process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct EnvVar {
    /// Variable name (e.g. `CUDA_VISIBLE_DEVICES`).
    pub key: String,
    /// Variable value.
    pub value: String,
}

/// Response from `GET /api/services/{name}/command`: the launch command
/// computed under two scenarios.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct LaunchCommandResponse {
    /// Command on an empty cluster — what the service would launch with
    /// if no other services held pledges. Always present when the service
    /// can fit on the hardware at all.
    pub on_empty: LaunchCommand,
    /// Command against the current device state and pledge book. `None`
    /// when the service can't fit alongside currently running services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<LaunchCommand>,
}

/// One launch command — argv and environment. Discriminated by the
/// optional `container` block: when `None` this is a native process (the
/// existing shape, byte-identical for process consumers); when `Some` it is
/// a containerized workload carrying the runtime create command plus the
/// in-container argv.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct LaunchCommand {
    /// Whether the service is running (`running`) or this is a preview of the
    /// next start (`preview`).
    pub source: LaunchCommandSource,
    /// The argv. For a native process `argv[0]` is the binary and the rest
    /// are its arguments. For a container this is the *in-container* argv,
    /// which for a `llama-cpp` service is flags only — the executable is
    /// the image's own entrypoint and does not appear here. Already split
    /// into tokens; no shell quoting is applied, so a client rendering a
    /// copy-pasteable line should quote as needed.
    pub argv: Vec<String>,
    /// Environment variables ananke sets or overrides for the child (notably
    /// `CUDA_VISIBLE_DEVICES`), sorted by key. Not the full inherited
    /// environment.
    pub env: Vec<EnvVar>,
    /// Whether the child process also inherits the daemon's full
    /// environment (`$PATH`, `$HOME`, locale, …) on top of `env`. When
    /// `false`, the child sees only the variables in `env`.
    pub env_inherit: bool,
    /// Container launch details when this is a containerized workload.
    /// `None` (and thus absent from JSON) for a native process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<LaunchContainer>,
}

/// Container-specific half of a launch preview. Present only when the
/// service carries a `[service.container]` block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct LaunchContainer {
    /// Container runtime (`docker` or `podman`).
    pub runtime: String,
    /// Container image reference.
    pub image: String,
    /// Generated name pattern (`ananke-<service>-<run-id>`; the run id is a
    /// runtime value and not expanded in a preview).
    pub name_pattern: String,
    /// In-container argv (the command that runs inside the container).
    pub argv: Vec<String>,
    /// Explicit environment keys and values set inside the container.
    pub env: Vec<EnvVar>,
    /// Environment variable names passed through from the host, without
    /// resolving their (possibly secret) values.
    pub env_passthrough: Vec<String>,
    /// Bind mounts.
    pub mounts: Vec<LaunchMount>,
    /// Network mode (`bridge` or `host`).
    pub network: String,
    /// Service port publication (`host_ip:host_port:container_port`) for
    /// bridge networking, or `None` for host networking.
    pub publication: Option<String>,
    /// IPC mode (`private` or `host`).
    pub ipc: String,
    /// Expanded CDI GPU device entries (one per allocated GPU).
    pub gpu_devices: Vec<String>,
    /// The exact, shell-free `create` argv ananke will invoke.
    pub create_argv: Vec<String>,
}

/// A bind mount in a container launch preview.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct LaunchMount {
    /// Absolute host path.
    pub source: String,
    /// Absolute container path.
    pub target: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
}
