//! The validated configuration tree: the daemon-global settings and the
//! per-service config each supervisor is built from.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke_api::shared::{metadata::AnankeMetadata, modality::Modality};
#[cfg(any(test, feature = "test-fakes"))]
use ananke_config::docs::DEFAULT_OPENAI_MAX_BODY_BYTES;
use smol_str::SmolStr;

use crate::config::{
    parse::{EstimationConfig, SamplingConfig},
    validate::{
        AllocationMode, AutoRestartSettings, DeviceReserves, DeviceSlot, Filters, HealthSettings,
        Lifecycle, NumaStrategy, OffloadMode, PlacementPolicy, RuntimeConfig, SplitMode, Template,
        TrackingSettings,
    },
};

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub daemon: DaemonSettings,
    pub services: Vec<ServiceConfig>,
}

#[derive(Debug, Clone)]
pub struct DaemonSettings {
    pub management_listen: String,
    pub openai_listen: String,
    pub data_dir: PathBuf,
    pub shutdown_timeout_ms: u64,
    pub allow_external_management: bool,
    pub allow_external_services: bool,
    pub openai_allow_cors: bool,
    pub openai_max_body_bytes: usize,
}

/// Neutral settings for test construction. Production always derives
/// `DaemonSettings` from validated config, never from `Default`, so this is
/// gated to test builds.
#[cfg(any(test, feature = "test-fakes"))]
impl Default for DaemonSettings {
    fn default() -> Self {
        Self {
            management_listen: String::new(),
            openai_listen: String::new(),
            data_dir: PathBuf::new(),
            shutdown_timeout_ms: 5_000,
            allow_external_management: false,
            allow_external_services: false,
            openai_allow_cors: false,
            openai_max_body_bytes: DEFAULT_OPENAI_MAX_BODY_BYTES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: SmolStr,
    pub port: u16,
    pub private_port: u16,
    pub lifecycle: Lifecycle,
    pub priority: u8,
    pub health: HealthSettings,
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    pub placement_policy: PlacementPolicy,
    pub gpu_allow: Vec<u32>,
    /// Inter-GPU split strategy for multi-GPU llama.cpp services. See
    /// [`SplitMode`]. Default [`SplitMode::Layer`] preserves the historical
    /// first-fit pipeline behaviour.
    pub split_mode: SplitMode,
    /// Optional per-GPU weights for the `--tensor-split` ratio in sharded
    /// (`row`/`tensor`) modes. One positive float per allowed GPU, in ascending
    /// GPU-id order. Unset keeps the historical equal `1,1,...` split.
    pub tensor_split_weights: Option<Vec<f32>>,
    /// Extra per-GPU VRAM (MiB) this service keeps free when packing, layered
    /// on top of [`Self::reserves`]. From `[service.devices] gpu_headroom_mb`.
    pub gpu_headroom_mb: u64,
    /// Global device reserves (resolved from `[devices]`), shared here so the
    /// packer reads them without a separate config handle. Identical across
    /// every service in a config, so the `Arc` is cloned per service rather than
    /// the map.
    pub reserves: Arc<DeviceReserves>,
    pub filters: Filters,
    pub idle_timeout_ms: u64,
    pub drain_timeout_ms: u64,
    pub extended_stream_drain_ms: u64,
    pub max_request_duration_ms: u64,
    /// Self-healing restart policy. Error-rate watchdog on by default;
    /// periodic restart off by default. See [`AutoRestartSettings`].
    pub auto_restart: AutoRestartSettings,
    pub allocation_mode: AllocationMode,
    pub openai_compat: bool,
    pub description: Option<String>,
    /// What kind of model the service exposes (chat or embedding).
    /// Default is [`Modality::Chat`] so configs and JSON shipped before
    /// the field landed are unchanged. Embedding services opt in with
    /// `modality = "embedding"` in their `[[service]]` block.
    pub modality: Modality,
    pub start_queue_depth: usize,
    pub extra_args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Whether the child process inherits the daemon's environment
    /// (default `true`). When `false`, the child sees only the
    /// variables in `env` plus `CUDA_VISIBLE_DEVICES`.
    pub env_inherit: bool,
    /// Per-service hints that adjust how the snapshotter attributes
    /// observed VRAM/RSS to this service. See [`TrackingSettings`].
    pub tracking: TrackingSettings,
    /// Passthrough entries from `[[service]] metadata.*`. Opaque to the
    /// daemon — these exist only to be echoed back through `/v1/models`
    /// and `/api/services` for clients (Discord rotation, residence
    /// flags, …).
    pub metadata: AnankeMetadata,
    pub template_config: TemplateConfig,
}

impl ServiceConfig {
    pub fn template(&self) -> Template {
        self.template_config.template()
    }

    /// Borrow the llama-cpp configuration, or `None` if this is a command
    /// service. Intended for code paths that are only reachable for
    /// llama-cpp services (estimator, llama-server argv rendering).
    pub fn llama_cpp(&self) -> Option<&LlamaCppConfig> {
        match &self.template_config {
            TemplateConfig::LlamaCpp(lc) => Some(lc.as_ref()),
            TemplateConfig::Command(_) => None,
        }
    }

    /// Borrow the command configuration, or `None` if this is a llama-cpp
    /// service.
    pub fn command(&self) -> Option<&CommandConfig> {
        match &self.template_config {
            TemplateConfig::LlamaCpp(_) => None,
            TemplateConfig::Command(cmd) => Some(cmd),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TemplateConfig {
    /// Boxed so the llama-cpp variant (~272 bytes) doesn't dominate the size of
    /// every `ServiceConfig`. Command services are ~48 bytes; boxing keeps the
    /// enum small for both.
    LlamaCpp(Box<LlamaCppConfig>),
    Command(CommandConfig),
}

impl TemplateConfig {
    pub fn template(&self) -> Template {
        match self {
            TemplateConfig::LlamaCpp(_) => Template::LlamaCpp,
            TemplateConfig::Command(_) => Template::Command,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Serving runtime (mainline vs ik_llama.cpp fork with its
    /// validated knobs). See [`RuntimeConfig`].
    pub runtime: RuntimeConfig,
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub context: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    /// MoE expert-offload policy. See [`OffloadMode`].
    pub expert_offload: OffloadMode,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<SmolStr>,
    pub cache_type_v: Option<SmolStr>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub parallel: Option<u32>,
    /// `--spec-type` value (e.g. `"draft-mtp"`). See
    /// [`crate::config::parse::RawLlamaCppService::spec_type`].
    pub spec_type: Option<SmolStr>,
    /// `--spec-draft-n-max` value.
    pub spec_draft_n_max: Option<u32>,
    /// Separate draft-model GGUF (`-md` / `--model-draft`). See
    /// [`crate::config::parse::RawLlamaCppService::draft_model`].
    pub draft_model: Option<PathBuf>,
    /// `-kvu` / `--kv-unified` unified KV pool toggle.
    pub kv_unified: Option<bool>,
    /// When `Some(false)`, emit `--no-cache-idle-slots`.
    pub cache_idle_slots: Option<bool>,
    /// Host RAM cap for the server's prompt cache (`-cram`, MiB). `None`
    /// means llama.cpp's default; the spawn path emits the resolved value
    /// either way so the reservation and the runtime cap agree.
    pub cache_ram_mb: Option<u32>,
    /// `--metrics` endpoint toggle.
    pub metrics: Option<bool>,
    /// `--slots` endpoint toggle.
    pub slots: Option<bool>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    /// `--numa` placement strategy. See [`NumaStrategy`].
    pub numa: Option<NumaStrategy>,
    pub jinja: Option<bool>,
    pub chat_template_file: Option<PathBuf>,
    pub override_tensor: Vec<String>,
    pub sampling: SamplingConfig,
    pub estimation: EstimationConfig,
    /// Resolved executable used to launch the service. Defaults to
    /// `"llama-server"` (looked up on `$PATH`); a per-service
    /// `llama_server` overrides that, falling back to the daemon-level
    /// `daemon.llama_server`. Ignored when [`Self::launcher`] is set —
    /// the launcher's first element becomes the executable.
    pub binary: PathBuf,
    /// Optional argv template that replaces the default
    /// `llama-server -m <model> …` invocation. `launcher[0]` becomes
    /// the executable; `launcher[1..]` is substituted with the standard
    /// placeholders and the splat `{args}` (which expands to every
    /// other llama-server flag ananke would have emitted).
    pub launcher: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct CommandConfig {
    pub command: Vec<String>,
    pub workdir: Option<PathBuf>,
    /// Optional argv to run after the SIGTERM/SIGKILL drain pipeline
    /// exits. Used for external services that can't stop via signal
    /// alone — e.g. a docker-run wrapper whose container needs an
    /// explicit `docker stop` sibling command.
    pub shutdown_command: Option<Vec<String>>,
    /// When `Some`, ananke's reverse proxy forwards to this port rather
    /// than one picked from the private-port pool. Lets operators point
    /// at a fixed upstream (docker host binding, a service managed
    /// externally, etc). `None` = auto-assign.
    pub private_port_override: Option<u16>,
    /// When `Some`, the service is fronted by the OpenAI-compatible
    /// multiplexer at `:7070`: it appears in `/v1/models`, accepts
    /// `/v1/chat/completions` (etc.) addressed to its service `name`,
    /// and the multiplexer rewrites the JSON `model` field to
    /// `upstream_model` before forwarding. `None` = ordinary command
    /// service that's only reachable via its per-service reverse proxy.
    pub openai_proxy: Option<OpenAiProxyConfig>,
}

#[derive(Debug, Clone)]
pub struct OpenAiProxyConfig {
    /// Model name written into the upstream's JSON `model` field. The
    /// service's `name` is what clients see in `/v1/models`; this is
    /// what the upstream (vLLM, TGI, …) is asked to serve.
    pub upstream_model: SmolStr,
}
