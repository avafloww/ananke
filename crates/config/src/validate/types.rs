//! The validated configuration tree: the daemon-global settings and the
//! per-service config each supervisor is built from.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke_api::shared::{metadata::AnankeMetadata, modality::Modality};
use smol_str::SmolStr;

#[cfg(any(test, feature = "test-fakes"))]
use crate::docs::DEFAULT_OPENAI_MAX_BODY_BYTES;
use crate::{
    parse::{EstimationConfig, SamplingConfig},
    validate::{
        AllocationMode, AutoRestartSettings, DeviceReserves, DeviceSlot, Filters, HealthSettings,
        Lifecycle, NumaStrategy, OffloadMode, PlacementPolicy, RuntimeConfig, SplitMode, Template,
        TrackingSettings,
    },
};

/// The validated config tree: daemon settings plus one config per service.
#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    /// Daemon-global settings.
    pub daemon: DaemonSettings,
    /// One fully validated config per `[[service]]` block.
    pub services: Vec<ServiceConfig>,
}

/// Daemon-global settings resolved from the `[daemon]`, `[openai_api]`,
/// and `[devices]` blocks.
#[derive(Debug, Clone)]
pub struct DaemonSettings {
    /// Address the management API listens on.
    pub management_listen: String,
    /// Address the OpenAI-compatible multiplexed endpoint listens on.
    pub openai_listen: String,
    /// Directory for the SQLite store and other persistent state.
    pub data_dir: PathBuf,
    /// How long a service gets to drain before a forced shutdown.
    pub shutdown_timeout_ms: u64,
    /// Whether the management API binds 0.0.0.0 instead of 127.0.0.1.
    pub allow_external_management: bool,
    /// Whether per-service reverse proxies bind 0.0.0.0 instead of 127.0.0.1.
    pub allow_external_services: bool,
    /// Whether the OpenAI endpoint answers cross-origin browser requests.
    pub openai_allow_cors: bool,
    /// Maximum request body size for the OpenAI endpoints, in bytes.
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

/// One service's fully resolved configuration, the basis for building its
/// supervisor.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Unique service name, the key used by the API and CLI.
    pub name: SmolStr,
    /// Public port clients connect to (per-service reverse proxy).
    pub port: u16,
    /// Port the child's private listener binds on.
    pub private_port: u16,
    /// When and how the service starts and stops relative to the daemon.
    pub lifecycle: Lifecycle,
    /// Start priority: lower numbers start first.
    pub priority: u8,
    /// HTTP readiness probe for the child.
    pub health: HealthSettings,
    /// Per-GPU overrides of the placement policy's reservation weights.
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    /// Packing policy that decides which device the service lands on.
    pub placement_policy: PlacementPolicy,
    /// GPU indices this service may be placed on.
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
    /// Query-parameter scrubbing rules applied before proxying.
    pub filters: Filters,
    /// How long an idle service may stay up before it is stopped.
    pub idle_timeout_ms: u64,
    /// How long a draining service gets before it is killed.
    pub drain_timeout_ms: u64,
    /// How long drained connections keep streaming after the child exits.
    pub extended_stream_drain_ms: u64,
    /// Maximum wall-clock time for a single proxied request.
    pub max_request_duration_ms: u64,
    /// Self-healing restart policy. Error-rate watchdog on by default;
    /// periodic restart off by default. See [`AutoRestartSettings`].
    pub auto_restart: AutoRestartSettings,
    /// How the allocator reserves memory: static or a dynamic balloon range.
    pub allocation_mode: AllocationMode,
    /// Whether the service is reachable through the OpenAI multiplexer.
    pub openai_compat: bool,
    /// Free-form description surfaced in `/v1/models` and `/api/services`.
    pub description: Option<String>,
    /// What kind of model the service exposes (chat or embedding).
    /// Defaults to [`Modality::Chat`], so a config that says nothing gets a
    /// chat service. Embedding services opt in with `modality = "embedding"`
    /// in their `[[service]]` block.
    pub modality: Modality,
    /// Capacity of the queue for requests arriving while the service is starting.
    pub start_queue_depth: usize,
    /// Extra arguments appended to the resolved command line.
    pub extra_args: Vec<String>,
    /// Environment variables set on the child process.
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
    /// The template-specific half of the service config.
    pub template_config: TemplateConfig,
}

impl ServiceConfig {
    /// The template this service was validated as: `llama-cpp` or `command`.
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

/// The validated template-specific half of a service config.
#[derive(Debug, Clone)]
pub enum TemplateConfig {
    /// Boxed so the llama-cpp variant (~272 bytes) doesn't dominate the size of
    /// every `ServiceConfig`. Command services are ~48 bytes; boxing keeps the
    /// enum small for both.
    LlamaCpp(Box<LlamaCppConfig>),
    /// A plain command service.
    Command(CommandConfig),
}

impl TemplateConfig {
    /// The [`Template`] this config was validated as.
    pub fn template(&self) -> Template {
        match self {
            TemplateConfig::LlamaCpp(_) => Template::LlamaCpp,
            TemplateConfig::Command(_) => Template::Command,
        }
    }
}

/// Validated llama-cpp service settings, the basis for the estimator and
/// llama-server argv rendering.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Serving runtime (mainline vs ik_llama.cpp fork with its
    /// validated knobs). See [`RuntimeConfig`].
    pub runtime: RuntimeConfig,
    /// Path to the model GGUF.
    pub model: PathBuf,
    /// Path to the multimodal projector GGUF, for vision models.
    pub mmproj: Option<PathBuf>,
    /// Context window size in tokens.
    pub context: Option<u32>,
    /// Number of GPU layers (negative offloads the last layers to CPU).
    pub n_gpu_layers: Option<i32>,
    /// MoE expert-offload policy. See [`OffloadMode`].
    pub expert_offload: OffloadMode,
    /// Whether to use flash attention.
    pub flash_attn: Option<bool>,
    /// KV cache quantization format for the K tensors.
    pub cache_type_k: Option<SmolStr>,
    /// KV cache quantization format for the V tensors.
    pub cache_type_v: Option<SmolStr>,
    /// Whether to memory-map the model file.
    pub mmap: Option<bool>,
    /// Whether to lock the model in RAM.
    pub mlock: Option<bool>,
    /// Number of parallel decoding slots.
    pub parallel: Option<u32>,
    /// `--spec-type` value (e.g. `"draft-mtp"`). See
    /// [`crate::parse::RawLlamaCppService::spec_type`].
    pub spec_type: Option<SmolStr>,
    /// `--spec-draft-n-max` value.
    pub spec_draft_n_max: Option<u32>,
    /// Separate draft-model GGUF (`-md` / `--model-draft`). See
    /// [`crate::parse::RawLlamaCppService::draft_model`].
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
    /// Context batch size (`-b`).
    pub batch_size: Option<u32>,
    /// Physical batch size (`-ub`).
    pub ubatch_size: Option<u32>,
    /// CPU threads for prompt processing.
    pub threads: Option<u32>,
    /// CPU threads for generation.
    pub threads_batch: Option<u32>,
    /// `--numa` placement strategy. See [`NumaStrategy`].
    pub numa: Option<NumaStrategy>,
    /// Whether to use Jinja chat templating.
    pub jinja: Option<bool>,
    /// Custom chat-template file override (`--chat-template-file`).
    pub chat_template_file: Option<PathBuf>,
    /// Per-tensor overrides of the model's tensor layout.
    pub override_tensor: Vec<String>,
    /// Sampling knobs forwarded as llama-server CLI flags.
    pub sampling: SamplingConfig,
    /// Estimator overrides (compute-buffer headroom, safety factor).
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

/// A validated `command`-template service: arbitrary argv plus optional
/// allocation and OpenAI-proxy blocks.
#[derive(Debug, Clone)]
pub struct CommandConfig {
    /// argv to execute.
    pub command: Vec<String>,
    /// Working directory the child is started in.
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

/// OpenAI-proxy settings for a command service fronted by the multiplexer.
#[derive(Debug, Clone)]
pub struct OpenAiProxyConfig {
    /// Model name written into the upstream's JSON `model` field. The
    /// service's `name` is what clients see in `/v1/models`; this is
    /// what the upstream (vLLM, TGI, …) is asked to serve.
    pub upstream_model: SmolStr,
}
