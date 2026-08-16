//! The `command` template variant: `RawCommandService`, its allocation
//! table, and the OpenAI-proxy opt-in.

use std::path::PathBuf;

use serde::Deserialize;
use smol_str::SmolStr;

use crate::parse::{RawAllocation, RawContainerConfig, RawServiceCommon};

/// A raw `command`-template service: arbitrary argv plus optional
/// allocation and OpenAI-proxy blocks.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawCommandService {
    /// Fields shared with every template variant, flattened at top level.
    #[serde(flatten)]
    pub common: RawServiceCommon,
    /// argv to execute. Required; emptiness is caught by the validator.
    pub command: Option<Vec<String>>,
    /// Working directory the child is started in.
    pub workdir: Option<PathBuf>,
    /// How much memory the allocator reserves for this service.
    pub allocation: Option<RawAllocation>,
    /// Upstream port ananke's reverse proxy should forward to. When
    /// absent, ananke picks one from the daemon's private-port pool and
    /// substitutes it into `command` / `env` via the `{port}`
    /// placeholder. Set it explicitly when the external service binds a
    /// fixed port (e.g. a docker container exposing 18188 on the host).
    pub private_port: Option<u16>,
    /// Optional argv run at drain time after SIGTERM-then-SIGKILL
    /// completes. Useful for external services that don't stop via
    /// signal — e.g. a docker-run wrapper where SIGTERM reaches the
    /// host shell but the container needs an explicit `docker stop`.
    /// Accepts the same placeholder substitutions as `command`.
    /// Rejected when `[service.container]` is present because native
    /// runtime cleanup replaces it.
    pub shutdown_command: Option<Vec<String>>,
    /// Opt the command service into the OpenAI-compatible multiplexer.
    /// When present, the service shows up in `/v1/models` and accepts
    /// `/v1/chat/completions` and friends; the multiplexer rewrites the
    /// JSON `model` field to `upstream_model` before forwarding to the
    /// service's private port.
    pub openai_proxy: Option<RawOpenAiProxy>,
    /// Container configuration for Docker/Podman workloads. When present,
    /// the service runs inside a container rather than as a native process.
    pub container: Option<RawContainerConfig>,
}

/// `[service.openai_proxy]` block. Marks a `command` service as fronting
/// an upstream OpenAI-compatible API (vLLM, TGI, SGLang, …) so ananke's
/// allocator and lifecycle apply uniformly with the llama.cpp services.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawOpenAiProxy {
    /// Model name passed to the upstream's OpenAI API. The service's
    /// `name` is what clients see; this is what ananke writes into the
    /// JSON `model` field before forwarding. Required; the validator
    /// rejects an empty/missing value.
    pub upstream_model: Option<SmolStr>,
}
