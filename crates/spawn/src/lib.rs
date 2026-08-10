//! Child-process launch configuration shared by the system spawner and the
//! supervisor's argv renderers.

use std::collections::BTreeMap;

/// Resolved command line plus environment for a child process, produced by
/// the supervise spawn renderers and consumed by the process spawner.
#[derive(Debug, Clone)]
pub struct SpawnConfig {
    pub binary: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub env_inherit: bool,
}

impl SpawnConfig {
    /// Resolve the final environment map for the child process.
    ///
    /// When `env_inherit` is `true`, the child inherits the daemon's
    /// environment with per-service `env` entries overriding individual
    /// keys. When `false`, the child starts from a clean slate containing
    /// only the `env` entries (plus `CUDA_VISIBLE_DEVICES`, which is
    /// already folded into `self.env` by the render functions).
    pub fn resolve_env(&self, inherited: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        let mut env = if self.env_inherit {
            inherited.clone()
        } else {
            BTreeMap::new()
        };
        for (k, v) in &self.env {
            env.insert(k.clone(), v.clone());
        }
        env
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
