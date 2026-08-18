//! Child-process launch configuration shared by the system spawner and the
//! supervisor's argv renderers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Resolved command line plus environment for a child process, produced by
/// the supervise spawn renderers and consumed by the process spawner.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// ── Container workload specification ──────────────────────────────────────

/// Container runtime selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerRuntime {
    /// Docker container runtime.
    Docker,
    /// Podman container runtime.
    Podman,
}

impl ContainerRuntime {
    /// Canonical name: the `runtime` value in config, and the name of the
    /// binary that drives it. An explicit `runtime_executable` overrides
    /// the latter without changing the former.
    pub fn as_str(&self) -> &'static str {
        match self {
            ContainerRuntime::Docker => "docker",
            ContainerRuntime::Podman => "podman",
        }
    }

    /// Default CLI executable for this runtime.
    pub fn executable(&self) -> &'static str {
        self.as_str()
    }
}

/// Network mode for a containerized service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerNetwork {
    /// Bridge networking with loopback-only port publication.
    Bridge,
    /// Host networking with no port translation.
    Host,
}

/// IPC namespace policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerIpc {
    /// Private IPC namespace (default).
    Private,
    /// Host IPC namespace (shares /dev/shm with the host).
    Host,
}

/// A bind mount for a container workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerMount {
    /// Absolute host path.
    pub source: String,
    /// Absolute container path.
    pub target: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
    /// Optional SELinux relabel policy.
    pub selinux: Option<String>,
}

/// An additional port publication beyond the service endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPortPublication {
    /// Host IP to bind.
    pub host_ip: String,
    /// Host port.
    pub host_port: u16,
    /// Container port.
    pub container_port: u16,
    /// Protocol (`"tcp"` or `"udp"`).
    pub protocol: String,
}

/// Fully resolved container workload specification. Produced by the
/// supervisor's container renderer and consumed by the container runtime
/// seam in `ananke-system`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    /// Container runtime (Docker or Podman).
    pub runtime: ContainerRuntime,
    /// Runtime executable override (e.g., NixOS store path).
    pub runtime_executable: Option<String>,
    /// Container image reference.
    pub image: String,
    /// Explicit entrypoint override.
    pub entrypoint: Option<String>,
    /// Working directory inside the container.
    pub workdir: Option<String>,
    /// In-container argv (the command that runs inside the container).
    pub command: Vec<String>,
    /// Generated container name.
    pub name: String,
    /// Mandatory labels attached at creation time.
    pub labels: BTreeMap<String, String>,
    /// Network mode.
    pub network: ContainerNetwork,
    /// Container-side port for bridge networking.
    pub container_port: Option<u16>,
    /// Host-side private port (for bridge publication).
    pub host_port: Option<u16>,
    /// IPC namespace.
    pub ipc: ContainerIpc,
    /// CDI GPU device strings (expanded once per allocated GPU).
    pub gpu_devices: Vec<String>,
    /// Bind mounts.
    pub mounts: Vec<ContainerMount>,
    /// Additional port publications beyond the service endpoint.
    pub extra_publications: Vec<ContainerPortPublication>,
    /// Explicit container environment variables.
    pub env: BTreeMap<String, String>,
    /// Environment variable names passed through from the host.
    pub env_passthrough: Vec<String>,
}

// ── Discriminated workload spec ───────────────────────────────────────────

/// A workload specification: either a native process or a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadSpec {
    /// A native host process.
    Process(SpawnConfig),
    /// A container workload. Boxed because `ContainerSpec` runs large
    /// relative to `SpawnConfig`; boxing keeps the enum small for both.
    Container(Box<ContainerSpec>),
}

impl WorkloadSpec {
    /// Returns `true` when this is a container workload.
    pub fn is_container(&self) -> bool {
        matches!(self, WorkloadSpec::Container(_))
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

    #[test]
    fn container_spec_is_container() {
        let spec = WorkloadSpec::Container(Box::new(ContainerSpec {
            runtime: ContainerRuntime::Docker,
            runtime_executable: None,
            image: "test:latest".into(),
            entrypoint: None,
            workdir: None,
            command: vec!["run".into()],
            name: "ananke-test-abc123".into(),
            labels: BTreeMap::new(),
            network: ContainerNetwork::Bridge,
            container_port: Some(8000),
            host_port: Some(40000),
            ipc: ContainerIpc::Private,
            gpu_devices: Vec::new(),
            mounts: Vec::new(),
            extra_publications: Vec::new(),
            env: BTreeMap::new(),
            env_passthrough: Vec::new(),
        }));
        assert!(spec.is_container());
    }

    #[test]
    fn process_spec_is_not_container() {
        let spec = WorkloadSpec::Process(SpawnConfig {
            binary: "llama-server".into(),
            args: vec!["-m".into(), "/model.gguf".into()],
            env: BTreeMap::new(),
            env_inherit: true,
        });
        assert!(!spec.is_container());
    }
}
