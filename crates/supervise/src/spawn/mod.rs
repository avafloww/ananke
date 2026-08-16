//! Linux-only: render llama-server argv from an `EffectiveConfig` service
//! entry. Actual child spawning lives behind the
//! [`ananke_system::ProcessSpawner`] trait, with the production
//! [`ananke_system::LocalSpawner`] applying `prctl(PR_SET_PDEATHSIG, SIGTERM)`
//! so the child dies if the daemon exits unexpectedly.

mod command;
mod llama_cpp;

use ananke_config::validate::{ServiceConfig, TemplateConfig};
use ananke_devices::Allocation;
pub use ananke_spawn::SpawnConfig;
pub use command::render_shutdown_argv;
pub mod container;

/// Render the child command line plus env from a validated `ServiceConfig`,
/// its `Allocation`, and optional placement `CommandArgs`.
///
/// When `cmd_args` is `Some`, the placement engine has already computed
/// `-ngl`/`--tensor-split`/`-ot` values. Any existing `-ngl` flags from the
/// static config path are replaced by the placement-derived value.
pub fn render_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&ananke_allocator::placement::CommandArgs>,
) -> Result<SpawnConfig, ananke_templates::SubstituteError> {
    match &svc.template_config {
        TemplateConfig::LlamaCpp(lc) => llama_cpp::render_llama_cpp_argv(svc, lc, alloc, cmd_args),
        TemplateConfig::Command(_) => command::render_command_argv(svc, alloc),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn resolve_env_inherits_when_true() {
        let cfg = SpawnConfig {
            binary: "x".into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            env_inherit: true,
        };
        let mut inherited = BTreeMap::new();
        inherited.insert("PATH".into(), "/usr/bin".into());
        inherited.insert("HOME".into(), "/home/test".into());
        let resolved = cfg.resolve_env(&inherited);
        assert_eq!(resolved.get("PATH").unwrap(), "/usr/bin");
        assert_eq!(resolved.get("HOME").unwrap(), "/home/test");
    }

    #[test]
    fn resolve_env_excludes_inherited_when_false() {
        let cfg = SpawnConfig {
            binary: "x".into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            env_inherit: false,
        };
        let mut inherited = BTreeMap::new();
        inherited.insert("PATH".into(), "/usr/bin".into());
        let resolved = cfg.resolve_env(&inherited);
        assert!(resolved.is_empty());
    }

    #[test]
    fn resolve_env_self_env_overrides_inherited() {
        let mut env = BTreeMap::new();
        env.insert("PATH".into(), "/custom/bin".into());
        let cfg = SpawnConfig {
            binary: "x".into(),
            args: Vec::new(),
            env,
            env_inherit: true,
        };
        let mut inherited = BTreeMap::new();
        inherited.insert("PATH".into(), "/usr/bin".into());
        inherited.insert("HOME".into(), "/home/test".into());
        let resolved = cfg.resolve_env(&inherited);
        // Per-service override wins.
        assert_eq!(resolved.get("PATH").unwrap(), "/custom/bin");
        // Inherited key not overridden is preserved.
        assert_eq!(resolved.get("HOME").unwrap(), "/home/test");
    }
}
