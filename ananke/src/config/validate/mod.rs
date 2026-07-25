//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

mod auto_restart_types;
mod auto_restart_validation;
mod command_validation;
mod common;
mod llama_cpp_validation;
mod metadata;
mod orchestrate;
mod placeholders;
mod placement;
mod port_pool;
mod restart_triggers;
mod runtime;
mod service_validation;
mod split_mode;
mod tracking;
mod types;

#[cfg(any(test, feature = "test-fakes"))]
pub mod test_fixtures;

// Config-default constants live in the `ananke-config` leaf crate so the
// xtask and CLI can reference them without pulling in the daemon's heavy
// deps. Re-exported here so existing `use crate::config::validate::DEFAULT_*`
// paths keep working.
pub use ananke_config::docs::{
    DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS, DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
    DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS, DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
    DEFAULT_AUTO_RESTART_MAX_RESTARTS, DEFAULT_AUTO_RESTART_MIN_REQUESTS,
    DEFAULT_AUTO_RESTART_MIN_UPTIME_MS, DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS, DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
    DEFAULT_AUTO_RESTART_TTFT_STALL_MS, DEFAULT_AUTO_RESTART_WINDOW_MS, DEFAULT_DRAIN_TIMEOUT_MS,
    DEFAULT_EXTENDED_STREAM_DRAIN_MS, DEFAULT_HEALTH_PROBE_INTERVAL_MS, DEFAULT_HEALTH_TIMEOUT_MS,
    DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_MIN_BORROWER_RUNTIME_MS,
    DEFAULT_OPENAI_MAX_BODY_BYTES, DEFAULT_OPENAI_MAX_BODY_MB, DEFAULT_PRIVATE_PORT_END,
    DEFAULT_PRIVATE_PORT_START, DEFAULT_SERVICE_PRIORITY,
};
pub use auto_restart_types::{
    AutoRestartSettings, DEFAULT_AUTO_RESTART_PERIODIC_MODE, ErrorRateTrigger, ErrorStatusClass,
    GenerationStallTrigger, PeriodicMode, PeriodicTrigger, SpecCollapseTrigger, TtftStallTrigger,
};
pub(crate) use auto_restart_validation::validate_auto_restart;
pub(crate) use command_validation::{command_uses_port_placeholder, validate_command};
pub use common::gib_to_mib;
pub(crate) use common::{fail, flag_variant, parse_duration_ms, variant_flag};
pub(crate) use llama_cpp_validation::validate_llama_cpp;
pub(crate) use metadata::{build_ananke_metadata, toml_value_to_json};
pub use orchestrate::validate;
pub(crate) use orchestrate::{DaemonValidationCtx, ServiceValidationState};
pub(crate) use placeholders::{check_launcher_placeholders, check_placeholders};
pub use placement::{
    AllocationMode, DeviceReserves, DeviceSlot, Filters, HealthSettings, Lifecycle,
    PlacementPolicy, Template,
};
pub(crate) use port_pool::{PrivatePortAllocator, PrivatePortRange};
pub(crate) use restart_triggers::{
    validate_error_rate, validate_generation_stall, validate_spec_collapse, validate_ttft_stall,
};
pub use runtime::{IkSettings, NumaStrategy, OffloadMode, Runtime};
pub(crate) use service_validation::validate_service;
pub use split_mode::SplitMode;
pub use tracking::TrackingSettings;
pub(crate) use tracking::validate_tracking;
pub use types::{
    CommandConfig, DaemonSettings, EffectiveConfig, LlamaCppConfig, OpenAiProxyConfig,
    ServiceConfig, TemplateConfig,
};
