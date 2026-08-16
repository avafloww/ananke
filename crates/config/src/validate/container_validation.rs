//! Validate container configuration: runtime, image, networking, mounts,
//! CDI devices, and labels.

use std::collections::BTreeMap;

use ananke_errors::ExpectedError;
use smol_str::SmolStr;

use crate::{
    parse::RawContainerConfig,
    validate::{
        ContainerConfig, ContainerIpc, ContainerMount, ContainerNetwork, ContainerPortPublication,
        ContainerRuntime, fail,
    },
};

/// Reserved label prefix. User labels cannot override these.
const RESERVED_LABEL_PREFIX: &str = "io.ananke.";

pub(crate) fn validate_container(
    name: &SmolStr,
    raw: &RawContainerConfig,
) -> Result<ContainerConfig, ExpectedError> {
    // Image is mandatory.
    let image = raw.image.as_ref().ok_or_else(|| {
        fail(format!(
            "service {name}: container.image is required when [service.container] is present"
        ))
    })?;
    if image.is_empty() {
        return Err(fail(format!(
            "service {name}: container.image must not be empty"
        )));
    }

    // Runtime defaults to docker.
    let runtime = match raw.runtime.as_deref() {
        None | Some("docker") => ContainerRuntime::Docker,
        Some("podman") => ContainerRuntime::Podman,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: container.runtime `{other}` is invalid (expected `docker` or `podman`)"
            )));
        }
    };

    // Network mode defaults to bridge.
    let network = match raw.network.as_deref() {
        None | Some("bridge") => ContainerNetwork::Bridge,
        Some("host") => ContainerNetwork::Host,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: container.network `{other}` is invalid (expected `bridge` or `host`)"
            )));
        }
    };

    // IPC defaults to private.
    let ipc = match raw.ipc.as_deref() {
        None | Some("private") => ContainerIpc::Private,
        Some("host") => ContainerIpc::Host,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: container.ipc `{other}` is invalid (expected `private` or `host`)"
            )));
        }
    };

    // Bridge mode requires container_port.
    if matches!(network, ContainerNetwork::Bridge) && raw.container_port.is_none() {
        return Err(fail(format!(
            "service {name}: container.container_port is required for bridge networking"
        )));
    }

    // Host mode rejects extra_publications (no port mapping in host mode).
    if matches!(network, ContainerNetwork::Host)
        && !raw
            .extra_publications
            .as_ref()
            .map(|p| p.is_empty())
            .unwrap_or(true)
    {
        return Err(fail(format!(
            "service {name}: container.extra_publications is not allowed with host networking"
        )));
    }

    // Validate mounts.
    let mounts = match &raw.mounts {
        None => Vec::new(),
        Some(m) => {
            let mut out = Vec::with_capacity(m.len());
            for (i, mount) in m.iter().enumerate() {
                if !mount.source.starts_with('/') {
                    return Err(fail(format!(
                        "service {name}: container.mounts[{i}].source must be an absolute path (got `{}`)",
                        mount.source
                    )));
                }
                if !mount.target.starts_with('/') {
                    return Err(fail(format!(
                        "service {name}: container.mounts[{i}].target must be an absolute path (got `{}`)",
                        mount.target
                    )));
                }
                if let Some(ref sel) = mount.selinux
                    && sel != "z"
                    && sel != "Z"
                {
                    return Err(fail(format!(
                        "service {name}: container.mounts[{i}].selinux `{sel}` is invalid (expected `z` or `Z`)"
                    )));
                }
                out.push(ContainerMount {
                    source: mount.source.clone(),
                    target: mount.target.clone(),
                    read_only: mount.read_only,
                    selinux: mount.selinux.clone(),
                });
            }
            out
        }
    };

    // Check for duplicate mount targets.
    {
        let mut seen = BTreeMap::new();
        for (i, m) in mounts.iter().enumerate() {
            if let Some(&first) = seen.get(&m.target) {
                return Err(fail(format!(
                    "service {name}: container.mounts has duplicate target `{}` (first at index {first}, duplicate at {i})",
                    m.target
                )));
            }
            seen.insert(m.target.clone(), i);
        }
    }

    // Validate extra publications.
    let extra_publications = match &raw.extra_publications {
        None => Vec::new(),
        Some(pubs) => {
            let mut out = Vec::with_capacity(pubs.len());
            for (i, pub_entry) in pubs.iter().enumerate() {
                if pub_entry.protocol.as_str() != "tcp" && pub_entry.protocol.as_str() != "udp" {
                    return Err(fail(format!(
                        "service {name}: container.extra_publications[{i}].protocol `{}` is invalid (expected `tcp` or `udp`)",
                        pub_entry.protocol
                    )));
                }
                if pub_entry.host_ip.trim().is_empty() {
                    return Err(fail(format!(
                        "service {name}: container.extra_publications[{i}].host_ip is empty, which publishes on every interface — state `127.0.0.1` explicitly, or the address you mean"
                    )));
                }
                out.push(ContainerPortPublication {
                    host_ip: pub_entry.host_ip.clone(),
                    host_port: pub_entry.host_port,
                    container_port: pub_entry.container_port,
                    protocol: pub_entry.protocol.clone(),
                });
            }
            out
        }
    };

    // Validate CDI GPU device template.
    let gpu_device = match &raw.gpu_device {
        None => None,
        Some(tmpl) => {
            // Must contain {id} exactly once.
            let count = tmpl.matches("${id}").count();
            if count == 0 {
                return Err(fail(format!(
                    "service {name}: container.gpu_device must contain `${{id}}` placeholder (got `{tmpl}`)"
                )));
            }
            if count > 1 {
                return Err(fail(format!(
                    "service {name}: container.gpu_device must contain exactly one `${{id}}` placeholder (got {count} in `{tmpl}`)"
                )));
            }
            Some(tmpl.clone())
        }
    };

    // Validate and merge labels. Reject reserved prefix overrides.
    let labels = match &raw.labels {
        None => BTreeMap::new(),
        Some(some) => {
            let mut out = BTreeMap::new();
            for (k, v) in some {
                if k.starts_with(RESERVED_LABEL_PREFIX) {
                    return Err(fail(format!(
                        "service {name}: container.labels key `{k}` is reserved (the `{RESERVED_LABEL_PREFIX}` namespace is managed by ananke)"
                    )));
                }
                out.insert(k.clone(), v.clone());
            }
            out
        }
    };

    // Merge container env with service env (container env takes precedence).
    let mut env = BTreeMap::new();
    if let Some(container_env) = &raw.env {
        for (k, v) in container_env {
            env.insert(k.clone(), v.clone());
        }
    }

    // Passthrough entries are variable *names*. Docker and Podman also
    // accept `-e NAME=value`, so an entry containing `=` would be rendered
    // verbatim into the create argv — and from there into the launch
    // preview, the management API, and the persisted spec. That is the one
    // way a secret can escape a design that otherwise never resolves these
    // values daemon-side.
    let env_passthrough = raw.env_passthrough.clone().unwrap_or_default();
    for (i, entry) in env_passthrough.iter().enumerate() {
        if entry.contains('=') {
            return Err(fail(format!(
                "service {name}: container.env_passthrough[{i}] must be a variable name, not `NAME=value` — the value would be rendered into the launch preview and the API. Put the value in `container.env` if it is not a secret"
            )));
        }
        if entry.trim().is_empty() {
            return Err(fail(format!(
                "service {name}: container.env_passthrough[{i}] is empty"
            )));
        }
    }

    Ok(ContainerConfig {
        runtime,
        runtime_executable: raw.runtime_executable.clone(),
        image: image.clone(),
        entrypoint: raw.entrypoint.clone(),
        workdir: raw.workdir.clone(),
        network,
        container_port: raw.container_port,
        ipc,
        gpu_device,
        mounts,
        extra_publications,
        env,
        env_passthrough,
        labels,
    })
}

/// Reject a bridge-networked `command` service whose argv and environment
/// never consume the resolved endpoint.
///
/// In bridge mode ananke publishes `127.0.0.1:<private_port>` onto
/// `<container_port>`, which only works if the command actually binds
/// `{listen_host}` (`0.0.0.0`) on `{listen_port}` (the container port).
/// A command that hardcodes its own interface or port silently binds
/// somewhere the publication does not reach, so the proxy would forward
/// into a dead endpoint. Opaque bridge commands are out of scope: ananke
/// cannot guarantee reachability it was never told about.
pub(crate) fn validate_bridge_command_endpoint(
    name: &SmolStr,
    container: &ContainerConfig,
    command: &[String],
    service_env: &BTreeMap<String, String>,
) -> Result<(), ExpectedError> {
    if !matches!(container.network, ContainerNetwork::Bridge) {
        return Ok(());
    }

    let consumes = |needles: &[&str]| {
        let hit = |s: &String| needles.iter().any(|n| s.contains(n));
        command.iter().any(hit) || service_env.values().any(hit) || container.env.values().any(hit)
    };

    if !consumes(&["${listen_host}"]) {
        return Err(fail(format!(
            "service {name}: bridge-networked container command must consume `${{listen_host}}` in its command or env, so it binds the interface the published port maps onto (use `network = \"host\"` for a command that binds its own address)"
        )));
    }
    if !consumes(&["${listen_port}", "${port}"]) {
        return Err(fail(format!(
            "service {name}: bridge-networked container command must consume `${{listen_port}}` in its command or env, so it binds the container port `container_port` publishes to"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::validate::{test_fixtures::parse_and_merge, validate};

    #[test]
    fn container_config_accepts_both_templates_and_runtimes() {
        // Docker on command template.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "cmd-docker"
template = "command"
port = 8500
command = ["run", "--host", "${listen_host}", "--port", "${listen_port}"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
runtime = "docker"
image = "test:latest"
network = "bridge"
container_port = 8000
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].container.is_some());
        let ct = ec.services[0].container.as_ref().unwrap();
        assert!(matches!(ct.runtime, super::ContainerRuntime::Docker));

        // Podman on llama-cpp template.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "lc-podman"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.container]
runtime = "podman"
image = "ghcr.io/ggml-org/llama.cpp:server-cuda"
network = "host"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].container.is_some());
        let ct = ec.services[0].container.as_ref().unwrap();
        assert!(matches!(ct.runtime, super::ContainerRuntime::Podman));
    }

    #[test]
    fn container_rejects_missing_image() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("container.image"));
    }

    #[test]
    fn container_rejects_invalid_runtime() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
runtime = "containerd"
image = "test:latest"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("container.runtime"));
    }

    #[test]
    fn container_rejects_reserved_label() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
labels = { "io.ananke.managed" = "false" }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("reserved"));
    }

    #[test]
    fn container_rejects_bad_cdi_template() {
        // Missing {id} — use host network so we skip the container_port requirement.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
gpu_device = "nvidia.com/gpu"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("${id}"));
    }

    #[test]
    fn bridge_requires_container_port() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("container_port"));
    }

    #[test]
    fn container_rejects_network_publish_conflict() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
extra_publications = [
  { host_port = 9000, container_port = 9000 },
]
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("extra_publications"));
    }

    #[test]
    fn container_rejects_relative_mount_source() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "bad"
template = "command"
port = 8500
command = ["run"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
mounts = [
  { source = "relative/path", target = "/container/path" },
]
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("absolute path"));
    }

    #[test]
    fn container_rejects_launcher_and_shutdown_command() {
        let llama = parse_and_merge(
            r#"
[[service]]
name = "lc"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
launcher = ["/wrap.sh", "${model}", "${args}"]

[service.container]
image = "llama:latest"
network = "host"
"#,
        );
        let err = validate(&llama).unwrap_err();
        assert!(
            format!("{err}").contains("launcher and [service.container] are mutually exclusive")
        );

        let command = parse_and_merge(
            r#"
[[service]]
name = "cmd"
template = "command"
port = 8500
command = ["run"]
shutdown_command = ["/wrap.sh", "--stop"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
"#,
        );
        let err = validate(&command).unwrap_err();
        assert!(
            format!("{err}").contains("shutdown_command is not allowed with [service.container]")
        );
    }

    #[test]
    fn bridge_command_without_listen_placeholders_is_rejected() {
        // Neither placeholder: the interface complaint comes first.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "opaque"
template = "command"
port = 8500
command = ["serve", "--host", "0.0.0.0", "--port", "8000"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
container_port = 8000
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("must consume `${listen_host}`"));

        // Interface consumed, port hardcoded.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "half"
template = "command"
port = 8500
command = ["serve", "--host", "${listen_host}", "--port", "8000"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
container_port = 8000
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("must consume `${listen_port}`"));
    }

    #[test]
    fn bridge_command_listen_placeholders_in_env_are_accepted() {
        // The endpoint may reach the workload through the environment rather
        // than argv; `{port}` remains a legal spelling of `{listen_port}`.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "via-env"
template = "command"
port = 8500
command = ["serve"]
env = { BIND_HOST = "${listen_host}", BIND_PORT = "${port}" }
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
container_port = 8000
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].container.is_some());

        // The container's own env block counts too.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "via-container-env"
template = "command"
port = 8501
command = ["serve"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "bridge"
container_port = 8000
env = { BIND_HOST = "${listen_host}", BIND_PORT = "${listen_port}" }
"#,
        );
        assert!(validate(&cfg).unwrap().services[0].container.is_some());
    }

    #[test]
    fn host_network_command_needs_no_listen_placeholders() {
        // Host networking has no publication to bridge, so an opaque command
        // that binds the private port itself stays legal.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "host-opaque"
template = "command"
port = 8500
command = ["serve", "--port", "${port}"]
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "test:latest"
network = "host"
"#,
        );
        assert!(validate(&cfg).unwrap().services[0].container.is_some());
    }

    #[test]
    fn native_service_has_no_container() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "native"
template = "command"
port = 8500
command = ["/bin/true"]
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].container.is_none());
    }
}
