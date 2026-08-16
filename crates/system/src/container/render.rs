//! Pure argv rendering for Docker/Podman container lifecycle commands.
//!
//! Every function returns argv vectors (never shell strings) so the
//! supervisor and reconciliation paths can't accidentally shell out.
//! Docker/Podman differences live here as explicit branches.

use ananke_spawn::{ContainerIpc, ContainerNetwork, ContainerSpec};

/// The binary to invoke for `spec`: its explicit override, else the name of
/// the runtime the service asked for.
///
/// Never the adapter's own default — one shared engine serves every
/// service, so defaulting to it would run `docker` for a service configured
/// with `runtime = "podman"`, having already rendered Podman's flags.
pub fn executable_for(spec: &ContainerSpec) -> String {
    spec.runtime_executable
        .clone()
        .unwrap_or_else(|| spec.runtime.executable().to_string())
}

/// Render the argv for `create`. The trailing image + command are included
/// so the output is a complete, shell-free create invocation.
pub fn render_create_argv(executable: &str, spec: &ContainerSpec) -> Vec<String> {
    let mut argv = vec![executable.to_string(), "create".to_string()];

    // Name.
    argv.push("--name".into());
    argv.push(spec.name.clone());

    // IPC namespace. Orthogonal to networking: a host-networked container
    // can still want the host's /dev/shm, and vice versa.
    match spec.ipc {
        ContainerIpc::Host => {
            argv.push("--ipc".into());
            argv.push("host".into());
        }
        ContainerIpc::Private => {}
    }

    // Network mode and its dependent publications. Host networking shares
    // the host's namespace outright, so it has nothing to publish.
    match spec.network {
        ContainerNetwork::Host => {
            argv.push("--network".into());
            argv.push("host".into());
        }
        ContainerNetwork::Bridge => {
            // Service port publication: loopback-only mapping.
            if let (Some(hp), Some(cp)) = (spec.host_port, spec.container_port) {
                argv.push("-p".into());
                argv.push(format!("127.0.0.1:{hp}:{cp}"));
            }

            // Extra publications. Both runtimes spell the protocol as a
            // `/`-suffix on the container port, not a fourth colon field.
            for pub_entry in &spec.extra_publications {
                argv.push("-p".into());
                argv.push(format!(
                    "{}:{}:{}/{}",
                    pub_entry.host_ip,
                    pub_entry.host_port,
                    pub_entry.container_port,
                    pub_entry.protocol
                ));
            }
        }
    }

    // Entrypoint override.
    if let Some(ep) = &spec.entrypoint {
        argv.push("--entrypoint".into());
        argv.push(ep.clone());
    }

    // Working directory.
    if let Some(wd) = &spec.workdir {
        argv.push("-w".into());
        argv.push(wd.clone());
    }

    // GPU devices (CDI).
    for device in &spec.gpu_devices {
        argv.push("--device".into());
        argv.push(device.clone());
    }

    // Bind mounts.
    for mount in &spec.mounts {
        let mut mount_arg = format!("{}:{}", mount.source, mount.target);
        if mount.read_only {
            mount_arg.push_str(":ro");
        }
        if let Some(sel) = &mount.selinux {
            mount_arg.push(':');
            mount_arg.push_str(sel);
        }
        argv.push("-v".into());
        argv.push(mount_arg);
    }

    // Environment variables (explicit values).
    for (k, v) in &spec.env {
        argv.push("-e".into());
        argv.push(format!("{k}={v}"));
    }

    // Passthrough environment variables (names only — values come from host).
    for var in &spec.env_passthrough {
        argv.push("-e".into());
        argv.push(var.clone());
    }

    // Labels.
    for (k, v) in &spec.labels {
        argv.push("-l".into());
        argv.push(format!("{k}={v}"));
    }

    // Image, then in-container command argv.
    argv.push(spec.image.clone());
    argv.extend(spec.command.iter().cloned());

    argv
}

/// Render the argv for `start`.
pub fn render_start_argv(executable: &str, id: &str) -> Vec<String> {
    vec![executable.to_string(), "start".to_string(), id.to_string()]
}

/// Render the argv for `logs --follow`.
pub fn render_logs_argv(executable: &str, id: &str) -> Vec<String> {
    vec![
        executable.to_string(),
        "logs".to_string(),
        "--follow".to_string(),
        id.to_string(),
    ]
}

/// Render the argv for `wait`.
pub fn render_wait_argv(executable: &str, id: &str) -> Vec<String> {
    vec![executable.to_string(), "wait".to_string(), id.to_string()]
}

/// Render the argv for `kill --signal <signal>`.
pub fn render_kill_argv(executable: &str, id: &str, signal: &str) -> Vec<String> {
    vec![
        executable.to_string(),
        "kill".to_string(),
        "--signal".to_string(),
        signal.to_string(),
        id.to_string(),
    ]
}

/// Render the argv for `rm -f -v`.
///
/// `-v` removes the anonymous volumes the runtime created for this
/// container, which an image declaring `VOLUME` gets one of per run; without
/// it they accumulate unreferenced, one per start, forever. Named volumes
/// and bind mounts are not anonymous and are left alone.
pub fn render_remove_argv(executable: &str, id: &str) -> Vec<String> {
    vec![
        executable.to_string(),
        "rm".to_string(),
        "-f".to_string(),
        "-v".to_string(),
        id.to_string(),
    ]
}

/// The one label ananke reads back off a container.
pub const OWNER_LABEL: &str = "io.ananke.owner";

/// Render the argv for `inspect --format`.
pub fn render_inspect_argv(executable: &str, id: &str) -> Vec<String> {
    vec![
        executable.to_string(),
        "inspect".to_string(),
        "--format".to_string(),
        format!(
            "{{{{.Id}}}}|{{{{.Name}}}}|{{{{.State.Status}}}}|{{{{.State.ExitCode}}}}|{{{{.State.Pid}}}}|{{{{index .Config.Labels \"{OWNER_LABEL}\"}}}}"
        ),
        id.to_string(),
    ]
}

/// Render the argv for `ps -a --format` with label filters.
pub fn render_list_argv(executable: &str, filters: &[String]) -> Vec<String> {
    let mut argv = vec![
        executable.to_string(),
        "ps".to_string(),
        "-a".to_string(),
        // Ids only, untruncated. The runtime applies the label filters, and
        // each id is inspected afterwards for the fields that differ in
        // shape between runtimes.
        "--no-trunc".to_string(),
        "--format".to_string(),
        "{{.ID}}".to_string(),
    ];
    for f in filters {
        argv.push("--filter".to_string());
        argv.push(f.clone());
    }
    argv
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ananke_spawn::{
        ContainerMount, ContainerNetwork, ContainerPortPublication, ContainerRuntime, ContainerSpec,
    };

    use super::*;

    fn bridge_spec() -> ContainerSpec {
        ContainerSpec {
            runtime: ContainerRuntime::Docker,
            runtime_executable: None,
            image: "vllm/vllm-openai:v0.26.0".into(),
            entrypoint: None,
            workdir: None,
            command: vec!["--model".into(), "nvidia/diffusiongemma".into()],
            name: "ananke-diffusiongemma-abc123".into(),
            labels: BTreeMap::from([
                ("io.ananke.managed".into(), "true".into()),
                ("io.ananke.owner".into(), "uuid-123".into()),
            ]),
            network: ContainerNetwork::Bridge,
            container_port: Some(8000),
            host_port: Some(40000),
            ipc: ContainerIpc::Host,
            gpu_devices: vec!["nvidia.com/gpu=0".into()],
            mounts: vec![ContainerMount {
                source: "/cache".into(),
                target: "/root/.cache".into(),
                read_only: false,
                selinux: None,
            }],
            extra_publications: Vec::new(),
            env: BTreeMap::from([("OMP_NUM_THREADS".into(), "1".into())]),
            env_passthrough: vec!["HF_TOKEN".into()],
        }
    }

    fn host_spec() -> ContainerSpec {
        ContainerSpec {
            runtime: ContainerRuntime::Podman,
            runtime_executable: None,
            image: "ninfer:local".into(),
            entrypoint: None,
            workdir: None,
            command: vec!["ninfer-serve".into(), "/artifacts/model.ninfer".into()],
            name: "ananke-ninfer-xyz789".into(),
            labels: BTreeMap::from([("io.ananke.managed".into(), "true".into())]),
            network: ContainerNetwork::Host,
            container_port: None,
            host_port: None,
            ipc: ContainerIpc::Private,
            gpu_devices: vec!["nvidia.com/gpu=0".into()],
            mounts: vec![ContainerMount {
                source: "/home/philpax/ai/ninfer".into(),
                target: "/artifacts".into(),
                read_only: true,
                selinux: None,
            }],
            extra_publications: Vec::new(),
            env: BTreeMap::new(),
            env_passthrough: Vec::new(),
        }
    }

    #[test]
    fn docker_container_lifecycle_argv() {
        // The whole create invocation, in order: a `contains` check would
        // miss a duplicated or misplaced flag, which is exactly the failure
        // mode that produces a container bound to the wrong endpoint.
        let mut spec = bridge_spec();
        spec.entrypoint = Some("python3".into());
        spec.workdir = Some("/work".into());
        spec.extra_publications = vec![ContainerPortPublication {
            host_ip: "127.0.0.1".into(),
            host_port: 9001,
            container_port: 9001,
            protocol: "udp".into(),
        }];
        assert_eq!(
            render_create_argv("docker", &spec),
            [
                "docker",
                "create",
                "--name",
                "ananke-diffusiongemma-abc123",
                "--ipc",
                "host",
                "-p",
                "127.0.0.1:40000:8000",
                "-p",
                "127.0.0.1:9001:9001/udp",
                "--entrypoint",
                "python3",
                "-w",
                "/work",
                "--device",
                "nvidia.com/gpu=0",
                "-v",
                "/cache:/root/.cache",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "HF_TOKEN",
                "-l",
                "io.ananke.managed=true",
                "-l",
                "io.ananke.owner=uuid-123",
                "vllm/vllm-openai:v0.26.0",
                "--model",
                "nvidia/diffusiongemma",
            ]
        );

        // Docker takes no Podman hardening flag.
        assert!(
            !render_create_argv("docker", &bridge_spec())
                .contains(&"--no-new-privileges".to_string())
        );

        assert_eq!(
            render_start_argv("docker", "cid"),
            ["docker", "start", "cid"]
        );
        assert_eq!(
            render_logs_argv("docker", "cid"),
            ["docker", "logs", "--follow", "cid"]
        );
        assert_eq!(render_wait_argv("docker", "cid"), ["docker", "wait", "cid"]);
        assert_eq!(
            render_kill_argv("docker", "cid", "TERM"),
            ["docker", "kill", "--signal", "TERM", "cid"]
        );
        assert_eq!(
            render_kill_argv("docker", "cid", "KILL"),
            ["docker", "kill", "--signal", "KILL", "cid"]
        );
        assert_eq!(
            render_remove_argv("docker", "cid"),
            // `-v` takes the anonymous volumes with it.
            ["docker", "rm", "-f", "-v", "cid"]
        );
    }

    #[test]
    fn podman_container_lifecycle_argv() {
        let spec = host_spec();
        assert_eq!(
            render_create_argv("podman", &spec),
            [
                "podman",
                "create",
                "--name",
                "ananke-ninfer-xyz789",
                "--network",
                "host",
                "--device",
                "nvidia.com/gpu=0",
                "-v",
                "/home/philpax/ai/ninfer:/artifacts:ro",
                "-l",
                "io.ananke.managed=true",
                "ninfer:local",
                "ninfer-serve",
                "/artifacts/model.ninfer",
            ]
        );

        // The runtime executable may be an absolute store path.
        assert_eq!(
            render_create_argv("/nix/store/x/bin/podman", &spec)[0],
            "/nix/store/x/bin/podman"
        );

        assert_eq!(
            render_start_argv("podman", "cid"),
            ["podman", "start", "cid"]
        );
        assert_eq!(
            render_logs_argv("podman", "cid"),
            ["podman", "logs", "--follow", "cid"]
        );
        assert_eq!(render_wait_argv("podman", "cid"), ["podman", "wait", "cid"]);
        assert_eq!(
            render_kill_argv("podman", "cid", "TERM"),
            ["podman", "kill", "--signal", "TERM", "cid"]
        );
        assert_eq!(
            render_remove_argv("podman", "cid"),
            ["podman", "rm", "-f", "-v", "cid"]
        );
    }

    #[test]
    fn runtime_selects_its_own_binary() {
        // A service asking for Podman must invoke `podman`, not whichever
        // binary the shared adapter happens to default to.
        let mut spec = host_spec();
        spec.runtime = ContainerRuntime::Podman;
        spec.runtime_executable = None;
        assert_eq!(executable_for(&spec), "podman");

        spec.runtime = ContainerRuntime::Docker;
        assert_eq!(executable_for(&spec), "docker");

        // An explicit override still wins, for a store path or a wrapper.
        spec.runtime_executable = Some("/nix/store/x/bin/podman".into());
        assert_eq!(executable_for(&spec), "/nix/store/x/bin/podman");
    }

    #[test]
    fn host_network_publishes_nothing_but_keeps_ipc() {
        // IPC is orthogonal to networking: a host-networked container that
        // asked for the host's /dev/shm must still get it.
        let mut spec = host_spec();
        spec.ipc = ContainerIpc::Host;
        let argv = render_create_argv("podman", &spec);
        assert!(argv.windows(2).any(|w| w == ["--ipc", "host"]));
        assert!(!argv.contains(&"-p".to_string()));
    }

    #[test]
    fn container_argv_preserves_spaces() {
        let spec = ContainerSpec {
            runtime: ContainerRuntime::Docker,
            runtime_executable: None,
            image: "test:latest".into(),
            entrypoint: None,
            workdir: None,
            command: vec![
                "run".into(),
                "--path".into(),
                "/path with spaces/file".into(),
            ],
            name: "test".into(),
            labels: BTreeMap::new(),
            network: ContainerNetwork::Host,
            container_port: None,
            host_port: None,
            ipc: ContainerIpc::Private,
            gpu_devices: Vec::new(),
            mounts: vec![ContainerMount {
                source: "/home/user/my models".into(),
                target: "/models".into(),
                read_only: true,
                selinux: None,
            }],
            extra_publications: Vec::new(),
            env: BTreeMap::new(),
            env_passthrough: Vec::new(),
        };

        let argv = render_create_argv("docker", &spec);
        assert!(argv.contains(&"/path with spaces/file".into()));
        assert!(argv.contains(&"/home/user/my models:/models:ro".into()));
    }

    #[test]
    fn lifecycle_argv_shapes() {
        assert_eq!(
            render_start_argv("docker", "abc"),
            ["docker", "start", "abc"]
        );
        assert_eq!(
            render_logs_argv("docker", "abc"),
            ["docker", "logs", "--follow", "abc"]
        );
        assert_eq!(render_wait_argv("docker", "abc"), ["docker", "wait", "abc"]);
        assert_eq!(
            render_kill_argv("docker", "abc", "TERM"),
            ["docker", "kill", "--signal", "TERM", "abc"]
        );
        assert_eq!(
            render_remove_argv("docker", "abc"),
            ["docker", "rm", "-f", "-v", "abc"]
        );
    }
}
