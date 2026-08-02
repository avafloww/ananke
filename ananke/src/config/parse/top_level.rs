//! Whole-config sections: `[daemon]`, `[devices]`, `[openai_api]`,
//! `[defaults]`, and the top-level `RawConfig` that ties them together.

use std::{collections::BTreeMap, path::PathBuf};

use serde::Deserialize;
use smol_str::SmolStr;

use crate::config::parse::{RawAutoRestart, RawService};

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawAllocation {
    pub mode: Option<SmolStr>,
    /// Static allocation only: the reservation in GiB. Lands on whichever
    /// device the service is placed on — host RAM for a cpu-only command
    /// service, VRAM otherwise — hence the device-neutral name. `vram_gb` is
    /// accepted as a device-specific alias.
    #[serde(alias = "vram_gb")]
    pub reserve_gb: Option<f32>,
    /// Dynamic allocation only: the minimum reservation in GiB.
    #[serde(alias = "min_vram_gb")]
    pub min_reserve_gb: Option<f32>,
    /// Dynamic allocation only: the maximum reservation in GiB.
    #[serde(alias = "max_vram_gb")]
    pub max_reserve_gb: Option<f32>,
    /// Balloon resolver grace period (default 60s); dynamic only.
    pub min_borrower_runtime: Option<String>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub devices: DevicesConfig,
    #[serde(default)]
    pub openai_api: OpenAiApiConfig,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default, rename = "service")]
    pub services: Vec<RawService>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    #[serde(default = "default_management_listen")]
    pub management_listen: String,
    pub data_dir: Option<PathBuf>,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: String,
    #[serde(default)]
    pub allow_external_management: bool,
    /// Bind per-service reverse proxies on 0.0.0.0 instead of 127.0.0.1
    /// so direct hits to `<host>:<svc.port>` reach them over the network.
    /// The OpenAI multiplexed endpoint on `openai_api.listen` always
    /// honours its own bind address; this controls only the per-service
    /// proxies (one port per `[[service]]`). Same security posture as
    /// `allow_external_management`: unauthenticated, trust the perimeter.
    #[serde(default)]
    pub allow_external_services: bool,
    /// Inclusive lower bound of the loopback port range handed out to
    /// llama-server children for their private listener. Default: 40000.
    pub private_port_start: Option<u16>,
    /// Inclusive upper bound of the private-listener port range. Default:
    /// 59999. Override (together with `private_port_start`) when another
    /// process on the host occupies the default window.
    pub private_port_end: Option<u16>,
    /// Path (or `$PATH` lookup name) of the llama-server executable used
    /// when spawning llama-cpp services. Defaults to `"llama-server"`
    /// (looked up on `$PATH`). A per-service `llama_server` field
    /// overrides this. Useful when the llama-server binary lives outside
    /// `$PATH`, or when operators wrap it in a container/script that
    /// still accepts llama-server's CLI.
    pub llama_server: Option<PathBuf>,
}

fn default_management_listen() -> String {
    ananke_config::defaults::MANAGEMENT_LISTEN.into()
}

fn default_shutdown_timeout() -> String {
    "120s".into()
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DevicesConfig {
    #[serde(default)]
    pub gpu_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub gpu_reserved_mb: BTreeMap<String, u64>,
    #[serde(default)]
    pub default_gpu_reserved_mb: Option<u64>,
    #[serde(default)]
    pub cpu: CpuConfig,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub reserved_gb: Option<u64>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OpenAiApiConfig {
    pub listen: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    pub max_request_duration: Option<String>,
    /// Allow cross-origin requests from browsers. Defaults to `true`
    /// since ananke is unauthenticated and designed for trusted-network
    /// deployment; operators who want to block browser-based access to
    /// the OpenAI API can set `allow_cors = false`.
    #[serde(default = "default_true")]
    pub allow_cors: bool,
    /// Maximum request body size for the OpenAI endpoints, in mebibytes.
    /// Vision requests carry base64-encoded images that routinely exceed
    /// axum's 2 MiB default body limit, so ananke's default is generous.
    /// Raise it if a single request carries very large or many images.
    pub max_body_mb: Option<u64>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub idle_timeout: Option<String>,
    pub priority: Option<u8>,
    pub start_queue_depth: Option<u32>,
    /// Fleet-wide default auto-restart policy, applied to any service that
    /// does not set its own `[service.auto_restart]` block. See
    /// [`RawAutoRestart`].
    pub auto_restart: Option<RawAutoRestart>,
}
