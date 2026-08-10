//! Configuration loading, parsing, inheritance merging, and validation.
//!
//! The parse → merge → validate pipeline, the config manager, and the
//! file-source resolution now live in the `ananke-config` crate. This
//! module re-exports them so `crate::config::…` paths inside the daemon
//! are unchanged by the split, and supplies the daemon-side
//! placeholder dry-run checker that the pipeline injects.

pub mod service_inputs;
pub mod validate;

pub mod manager {
    pub use ananke_config::manager::{ApplyError, ConfigHash, ConfigManager};
}

pub mod parse {
    pub use ananke_config::parse::*;
}

use std::path::Path;

pub use ananke_config::{
    Migration, PathSources, RawConfig, RawService,
    manager::ConfigManager,
    merge::{resolve_inheritance, resolve_migrations},
    parse::parse_toml,
    resolve_config_path, resolve_from_env,
    validate::{
        AllocationMode, AutoRestartSettings, CommandConfig, DaemonSettings, DeviceReserves,
        DeviceSlot, EffectiveConfig, ErrorRateTrigger, ErrorStatusClass, Filters,
        GenerationStallTrigger, HealthSettings, IkSettings, Lifecycle, LlamaCppConfig,
        NumaStrategy, OffloadMode, PeriodicMode, PeriodicTrigger, PlacementPolicy, Runtime,
        RuntimeConfig, ServiceConfig, SpecCollapseTrigger, SplitMode, Template, TemplateConfig,
        TrackingSettings, TtftStallTrigger, validate, validate_with_checks,
    },
};
use ananke_config::{load_config_from_str_with_checks, preflight_ggufs};
use ananke_errors::ExpectedError;

/// Load, parse, merge, validate, and preflight a config file from disk.
pub fn load_config(path: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let fs = crate::system::LocalFs;
    let source = crate::system::Fs::read_to_string(&fs, path)
        .map_err(|_| ExpectedError::config_file_missing(path.to_path_buf()))?;
    load_config_with_fs(path, &fs, &source)
}

/// Variant of [`load_config`] that uses an explicit filesystem for the
/// GGUF preflight (but takes the TOML source directly rather than reading
/// it through the fs).
pub fn load_config_with_fs(
    origin: &Path,
    fs: &dyn crate::system::Fs,
    source: &str,
) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let (effective, migrations) = load_config_from_str_with_checks(
        source,
        origin,
        &crate::config::validate::DaemonPlaceholderChecker,
    )?;
    preflight_ggufs(&effective, fs)?;
    Ok((effective, migrations))
}

pub fn load_config_from_str(
    source: &str,
    origin: &Path,
) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    load_config_from_str_with_checks(
        source,
        origin,
        &crate::config::validate::DaemonPlaceholderChecker,
    )
}
