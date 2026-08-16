//! Container configuration for Docker/Podman workloads: `RawContainerConfig`
//! and its nested mount and port-publication tables.

use std::collections::BTreeMap;

use serde::Deserialize;
use smol_str::SmolStr;

/// `[service.container]` block. When present, the service runs inside a
/// container rather than as a native host process. Legal on both
/// `template = "llama-cpp"` and `template = "command"` services.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawContainerConfig {
    /// Container runtime: `"docker"` or `"podman"`. Defaults to `"docker"`.
    pub runtime: Option<SmolStr>,
    /// Optional override for the runtime executable path. When absent,
    /// ananke uses the runtime name (`docker` or `podman`) and relies on
    /// `$PATH`. Set this when the binary lives outside standard locations
    /// (e.g., NixOS store paths).
    pub runtime_executable: Option<String>,
    /// Required container image reference (e.g., `vllm/vllm-openai:v0.26.0`).
    pub image: Option<String>,
    /// Optional explicit entrypoint that replaces the image's ENTRYPOINT.
    /// For llama-cpp services, ananke normally emits `--entrypoint` from
    /// the resolved `llama_server`; set this to override that behaviour.
    /// For command services, the configured `command` is passed after the
    /// image exactly as written; an explicit entrypoint replaces the image's.
    pub entrypoint: Option<String>,
    /// Working directory inside the container.
    pub workdir: Option<String>,
    /// Network mode: `"bridge"` (default) or `"host"`. Bridge mode publishes
    /// the service endpoint through a loopback-only port mapping; host mode
    /// binds directly on the host network with no port translation.
    pub network: Option<SmolStr>,
    /// Container-side port for bridge networking. In bridge mode, ananke
    /// publishes `127.0.0.1:<ananke_private_port>:<container_port>`. Required
    /// for bridge networking unless a template-specific default applies.
    pub container_port: Option<u16>,
    /// IPC namespace: `"private"` (default) or `"host"`. Host IPC shares
    /// the host's `/dev/shm`, which some GPU workloads require.
    pub ipc: Option<SmolStr>,
    /// Optional CDI GPU device template. A value like `nvidia.com/gpu=${id}`
    /// expands once per GPU selected by ananke placement. Unset means no
    /// GPU injection.
    pub gpu_device: Option<String>,
    /// Structured bind mounts. Each entry maps a host path to a container
    /// path. Llama-cpp typed path fields (`model`, `mmproj`, `draft_model`,
    /// `chat_template_file`) are automatically translated through matching
    /// mounts; opaque paths in `extra_args` are not.
    pub mounts: Option<Vec<RawContainerMount>>,
    /// Additional port publications beyond the service endpoint. Only valid
    /// in bridge mode.
    pub extra_publications: Option<Vec<RawPortPublication>>,
    /// Container-specific environment variables merged with the service's
    /// `env` map. The container does not inherit the daemon's environment
    /// implicitly.
    pub env: Option<BTreeMap<String, String>>,
    /// Environment variables passed through from the host environment into
    /// the container. Unlike `env_inherit` (which copies the full daemon
    /// environment for native processes), this is an explicit allowlist.
    pub env_passthrough: Option<Vec<String>>,
    /// User-defined labels merged with mandatory reserved labels. The entire
    /// `io.ananke.*` namespace is reserved and cannot be overridden.
    pub labels: Option<BTreeMap<String, String>>,
}

/// A structured bind mount for container configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RawContainerMount {
    /// Absolute host path. Must exist at container creation time.
    pub source: String,
    /// Absolute container path.
    pub target: String,
    /// Whether the mount is read-only. Defaults to `false`.
    #[serde(default)]
    pub read_only: bool,
    /// Optional SELinux relabel policy: `"z"` (shared unconfined) or
    /// `"Z"` (private unconfined). Omitted means no relabeling.
    pub selinux: Option<SmolStr>,
}

/// An additional port publication beyond the service endpoint.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RawPortPublication {
    /// Host IP to bind. Defaults to `"127.0.0.1"`.
    #[serde(default = "default_host_ip")]
    pub host_ip: String,
    /// Host port.
    pub host_port: u16,
    /// Container port.
    pub container_port: u16,
    /// Protocol: `"tcp"` (default) or `"udp"`.
    #[serde(default = "default_protocol")]
    pub protocol: SmolStr,
}

fn default_host_ip() -> String {
    "127.0.0.1".to_string()
}

fn default_protocol() -> SmolStr {
    SmolStr::new("tcp")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::parse::{RawService, parse_toml};

    #[test]
    fn parses_container_block_on_command_service() {
        let toml = r#"
[[service]]
name = "test"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
runtime = "docker"
image = "test:latest"
network = "bridge"
container_port = 8000
ipc = "host"
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::Command(cmd) = &cfg.services[0] else {
            panic!("expected Command");
        };
        assert!(cmd.container.is_some());
        let ct = cmd.container.as_ref().unwrap();
        assert_eq!(ct.runtime.as_deref(), Some("docker"));
        assert_eq!(ct.image.as_deref(), Some("test:latest"));
        assert_eq!(ct.network.as_deref(), Some("bridge"));
        assert_eq!(ct.container_port, Some(8000));
        assert_eq!(ct.ipc.as_deref(), Some("host"));
    }

    #[test]
    fn parses_container_block_on_llama_cpp_service() {
        let toml = r#"
[[service]]
name = "test"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.container]
runtime = "podman"
image = "ghcr.io/ggml-org/llama.cpp:server-cuda"
network = "host"
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::LlamaCpp(lc) = &cfg.services[0] else {
            panic!("expected LlamaCpp");
        };
        assert!(lc.container.is_some());
        let ct = lc.container.as_ref().unwrap();
        assert_eq!(ct.runtime.as_deref(), Some("podman"));
        assert_eq!(
            ct.image.as_deref(),
            Some("ghcr.io/ggml-org/llama.cpp:server-cuda")
        );
        assert_eq!(ct.network.as_deref(), Some("host"));
    }

    #[test]
    fn parses_mounts() {
        let toml = r#"
[[service]]
name = "test"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
mounts = [
  { source = "/host/path", target = "/container/path", read_only = true },
  { source = "/cache", target = "/root/.cache", selinux = "z" },
]

"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::Command(cmd) = &cfg.services[0] else {
            panic!("expected Command");
        };
        let ct = cmd.container.as_ref().unwrap();
        assert_eq!(ct.mounts.as_ref().map(|m| m.len()), Some(2));
        let mounts = ct.mounts.as_ref().unwrap();
        assert_eq!(mounts[0].source, "/host/path");
        assert!(mounts[0].read_only);
        assert_eq!(mounts[1].selinux.as_deref(), Some("z"));
    }

    #[test]
    fn parses_extra_publications() {
        let toml = r#"
[[service]]
name = "test"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
container_port = 8000
extra_publications = [
  { host_port = 9000, container_port = 9000 },
  { host_port = 9001, container_port = 9001, protocol = "udp" },
]
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::Command(cmd) = &cfg.services[0] else {
            panic!("expected Command");
        };
        let ct = cmd.container.as_ref().unwrap();
        assert_eq!(ct.extra_publications.as_ref().map(|p| p.len()), Some(2));
        let pubs = ct.extra_publications.as_ref().unwrap();
        assert_eq!(pubs[0].host_ip, "127.0.0.1");
        assert_eq!(pubs[0].protocol.as_str(), "tcp");
        assert_eq!(pubs[1].protocol.as_str(), "udp");
    }
}
