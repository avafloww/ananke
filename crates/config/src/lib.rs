//! Configuration defaults, vocabulary, parse/merge/validate pipeline, and
//! the config manager.
//!
//! The defaults, docs descriptors, placement vocabulary, and byte-unit
//! conversions form a leaf core (serde + smol_str only) so the xtask and
//! CLI can reference them without pulling in the daemon's heavy deps.
//! The parse → merge → validate pipeline and the config manager live
//! here too, with the placeholder dry-run checker injected by the daemon
//! (it needs template + allocation types that would create a cycle).

#![deny(missing_docs)]

pub mod defaults;
pub mod docs;
pub mod file;
pub mod flags;
pub mod manager;
pub mod merge;
pub mod parse;
pub mod placement;
pub mod runtime;
pub mod units;
pub mod validate;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
pub use merge::{Migration, resolve_inheritance, resolve_migrations};
pub use parse::{RawConfig, RawService, parse_toml};
pub use validate::{
    AllocationMode, AutoRestartSettings, CommandConfig, DaemonSettings, DeviceReserves, DeviceSlot,
    EffectiveConfig, ErrorRateTrigger, ErrorStatusClass, Filters, GenerationStallTrigger,
    HealthSettings, IkSettings, Lifecycle, LlamaCppConfig, NumaStrategy, OffloadMode, PeriodicMode,
    PeriodicTrigger, PlacementPolicy, Runtime, RuntimeConfig, ServiceConfig, SpecCollapseTrigger,
    SplitMode, Template, TemplateConfig, TrackingSettings, TtftStallTrigger, validate,
    validate_with_checks,
};

/// Load, parse, merge, validate, and preflight a config file from disk.
pub fn load_config(
    path: &std::path::Path,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    load_config_with_fs(
        path,
        &ananke_fs::LocalFs,
        &ananke_fs::Fs::read_to_string(&ananke_fs::LocalFs, path)
            .map_err(|_| ananke_errors::ExpectedError::config_file_missing(path.to_path_buf()))?,
    )
}

/// Variant of [`load_config`] that uses an explicit filesystem for the
/// GGUF preflight (but takes the TOML source directly rather than reading
/// it through the fs).
pub fn load_config_with_fs(
    origin: &std::path::Path,
    fs: &dyn ananke_fs::Fs,
    source: &str,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    let (effective, migrations) =
        load_config_from_str_with_checks(source, origin, &validate::NoopPlaceholderChecker)?;
    preflight_ggufs(&effective, fs)?;
    Ok((effective, migrations))
}

/// Parse, merge, and validate a TOML config from a string, with no GGUF
/// preflight. The daemon's full load path is [`load_config_with_fs`].
pub fn load_config_from_str(
    source: &str,
    origin: &std::path::Path,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    load_config_from_str_with_checks(source, origin, &validate::NoopPlaceholderChecker)
}

/// [`load_config_from_str`] with an injected placeholder dry-run checker.
pub fn load_config_from_str_with_checks(
    source: &str,
    origin: &std::path::Path,
    checker: &dyn validate::PlaceholderChecker,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    let mut raw = parse_toml(source, origin)?;
    resolve_inheritance(&mut raw)?;
    let migrations = resolve_migrations(&mut raw)?;
    let effective = validate_with_checks(&raw, checker)?;
    Ok((effective, migrations))
}

/// Walk every llama-cpp service's GGUF through `fs` and ensure the reader
/// can enumerate each tensor table.
pub fn preflight_ggufs(
    cfg: &EffectiveConfig,
    fs: &dyn ananke_fs::Fs,
) -> Result<(), ananke_errors::ExpectedError> {
    for svc in &cfg.services {
        let Some(lc) = svc.llama_cpp() else {
            continue;
        };
        ananke_gguf::read(fs, &lc.model).map_err(|e| {
            ananke_errors::ExpectedError::config_unparseable(
                std::path::PathBuf::from("<preflight>"),
                format!("service {}: {}", svc.name, e),
            )
        })?;
        if let Some(mmproj) = &lc.mmproj {
            ananke_gguf::read(fs, mmproj.as_path()).map_err(|e| {
                ananke_errors::ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<preflight>"),
                    format!("service {} mmproj: {}", svc.name, e),
                )
            })?;
        }
    }
    Ok(())
}
