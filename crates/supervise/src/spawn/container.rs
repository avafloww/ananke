//! Render a containerized service's resolved [`ContainerSpec`]: the
//! runtime, image, name/labels, in-container argv, mounts, networking,
//! publications, IPC, and CDI devices.
//!
//! This module translates the template-specific semantics (llama-cpp's
//! generated server argv, or a command's arbitrary argv) into a
//! container-shaped launch specification. Native-process rendering remains
//! in `spawn::command` / `spawn::llama_cpp`.

use std::collections::BTreeMap;

use ananke_config::validate::{
    ContainerIpc, ContainerNetwork, ContainerRuntime, ServiceConfig, TemplateConfig,
};
use ananke_devices::Allocation;
use ananke_spawn::{ContainerMount, ContainerPortPublication, ContainerSpec};
use ananke_templates::{PlaceholderContext, SubstituteError};

use crate::spawn;

/// Errors that can surface while translating a service config into a
/// container spec.
#[derive(Debug)]
pub enum ContainerRenderError {
    /// Placeholder substitution failed in the command argv.
    Substitute(SubstituteError),
    /// A typed llama.cpp path was not covered by any bind mount.
    UnmappedPath { field: &'static str, path: String },
    /// A CDI device template was malformed (missing `{id}`).
    BadCdiTemplate(String),
    /// The mount list is ambiguous (longest-prefix collision) or has a
    /// duplicate target.
    AmbiguousMount(String),
}

impl std::fmt::Display for ContainerRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerRenderError::Substitute(e) => write!(f, "{e}"),
            ContainerRenderError::UnmappedPath { field, path } => {
                write!(
                    f,
                    "containerized path {path:?} for {field} is not covered by any bind mount"
                )
            }
            ContainerRenderError::BadCdiTemplate(t) => {
                write!(
                    f,
                    "malformed CDI device template {t:?} (must contain ${{id}} exactly once)"
                )
            }
            ContainerRenderError::AmbiguousMount(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ContainerRenderError {}

impl From<SubstituteError> for ContainerRenderError {
    fn from(e: SubstituteError) -> Self {
        ContainerRenderError::Substitute(e)
    }
}

/// Sanitize a service name into a container-name component: lowercase, with
/// every non `[a-z0-9_.-]` run collapsed to a single dash. Dots are valid
/// in Docker/Podman container names and preserved so `qwen3.6` stays
/// recognisable.
fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.chars() {
        let ch = ch.to_ascii_lowercase();
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Collision-resistant container name: `ananke-<sanitized-service>-<run-id>`.
pub fn container_name(service_name: &str, run_id: i64) -> String {
    format!("ananke-{}-{}", sanitize_name(service_name), run_id)
}

/// The same name with the run id left unexpanded, for a preview that has no
/// run to name yet.
pub fn container_name_pattern(service_name: &str) -> String {
    format!("ananke-{}-<run-id>", sanitize_name(service_name))
}

/// Expand a CDI device template (`nvidia.com/gpu=${id}`) once per selected
/// GPU id.
pub fn expand_cdi_devices(
    template: &str,
    gpu_ids: &[u32],
) -> Result<Vec<String>, ContainerRenderError> {
    if template.matches("${id}").count() != 1 {
        return Err(ContainerRenderError::BadCdiTemplate(template.to_string()));
    }
    Ok(gpu_ids
        .iter()
        .map(|id| template.replace("${id}", &id.to_string()))
        .collect())
}

/// Translate a host path through a set of bind mounts using lexical,
/// path-component-aware longest-source-prefix matching. Returns the
/// container-side path, or `None` if no mount covers it.
fn translate_path(mounts: &[ContainerMount], host_path: &str) -> Option<String> {
    let mut best: Option<(&str, String)> = None;
    for m in mounts {
        // Match on component boundary: either exact or prefix + `/`.
        if host_path == m.source || host_path.starts_with(&format!("{}/", m.source)) {
            // Longest source wins.
            match &best {
                Some((best_src, _)) if best_src.len() >= m.source.len() => {}
                _ => {
                    let relative = host_path.strip_prefix(&m.source).unwrap_or_default();
                    let target = format!("{}{}", m.target, relative);
                    best = Some((&m.source, target));
                }
            }
        }
    }
    best.map(|(_, target)| target)
}

/// Translate the four typed llama.cpp path fields through the configured
/// mounts. Returns an error with the exact unmapped path when a mount
/// doesn't cover it.
fn translate_llama_paths(
    svc: &ServiceConfig,
    lc: &ananke_config::validate::LlamaCppConfig,
    mounts: &[ContainerMount],
) -> Result<(), ContainerRenderError> {
    let fields: [(&'static str, Option<&std::path::PathBuf>); 4] = [
        ("model", Some(&lc.model)),
        ("mmproj", lc.mmproj.as_ref()),
        ("draft_model", lc.draft_model.as_ref()),
        ("chat_template_file", lc.chat_template_file.as_ref()),
    ];
    for (field, path) in fields {
        let Some(path) = path else { continue };
        let host = path.to_string_lossy();
        if translate_path(mounts, &host).is_none() {
            return Err(ContainerRenderError::UnmappedPath {
                field,
                path: host.into_owned(),
            });
        }
    }
    // Silence the unused-variable warning for `svc`; it's part of the
    // exhaustive contract that future typed path fields register here.
    let _ = svc;
    Ok(())
}

/// Translate the llama.cpp typed paths in-place within a rendered argv.
fn translate_argv_paths(
    argv: &mut [String],
    svc: &ServiceConfig,
    mounts: &[ContainerMount],
) -> Result<(), ContainerRenderError> {
    let TemplateConfig::LlamaCpp(lc) = &svc.template_config else {
        return Ok(());
    };
    // The four typed path fields owned by ananke. Opaque `extra_args` are
    // never guessed.
    let typed: Vec<String> = std::iter::empty()
        .chain(std::iter::once(lc.model.to_string_lossy().into_owned()))
        .chain(lc.mmproj.iter().map(|p| p.to_string_lossy().into_owned()))
        .chain(
            lc.draft_model
                .iter()
                .map(|p| p.to_string_lossy().into_owned()),
        )
        .chain(
            lc.chat_template_file
                .iter()
                .map(|p| p.to_string_lossy().into_owned()),
        )
        .collect();
    for entry in argv.iter_mut() {
        if typed.iter().any(|t| t == entry)
            && let Some(target) = translate_path(mounts, entry)
        {
            *entry = target;
        }
    }
    Ok(())
}

/// The interface and port a service's workload should bind.
///
/// A native process, and a host-networked container, bind the ananke private
/// port on loopback. A bridge-networked container binds all interfaces on
/// `container_port`, because ananke publishes `127.0.0.1:<private_port>`
/// onto that port — a loopback listener inside the container's own network
/// namespace would be unreachable from the host side of the publication.
///
/// Both templates resolve their endpoint here: `command` feeds it to
/// `{listen_host}`/`{listen_port}`, and `llama-cpp` emits it directly as
/// `--host`/`--port`.
pub fn listen_endpoint(svc: &ServiceConfig) -> (&'static str, u16) {
    match svc
        .container
        .as_ref()
        .map(|c| (c.network, c.container_port))
    {
        Some((ContainerNetwork::Bridge, container_port)) => {
            ("0.0.0.0", container_port.unwrap_or(svc.private_port))
        }
        _ => ("127.0.0.1", svc.private_port),
    }
}

/// The environment the container receives: the service's own `env` with
/// `container.env` layered over it, placeholders resolved.
///
/// A container never inherits the daemon's environment (`env_inherit`
/// governs host processes only), so this map plus the `env_passthrough`
/// allowlist is the whole of what the workload can see. Dropping the
/// service-level half would silently strip settings like vLLM's
/// `PYTORCH_CUDA_ALLOC_CONF`, which fails as a performance regression
/// rather than an error.
///
/// `CUDA_VISIBLE_DEVICES` is deliberately not set: CDI already injects
/// exactly the GPUs the allocator picked, and they are renumbered from zero
/// inside the container, so a host-indexed value would be wrong.
fn container_env(
    svc: &ServiceConfig,
    alloc: &Allocation,
    container: &ananke_config::validate::ContainerConfig,
) -> Result<BTreeMap<String, String>, ContainerRenderError> {
    let mut merged = svc.env.clone();
    merged.extend(container.env.iter().map(|(k, v)| (k.clone(), v.clone())));

    let (listen_host, port) = listen_endpoint(svc);
    let ctx = PlaceholderContext {
        name: &svc.name,
        port,
        model: None,
        allocation: alloc,
        static_reserve_mb: match svc.allocation_mode {
            ananke_config::validate::AllocationMode::Static { reserve_mb } => Some(reserve_mb),
            _ => None,
        },
        listen_host: Some(listen_host),
        host_port: svc.private_port,
    };
    let mut out = BTreeMap::new();
    for (k, v) in &merged {
        out.insert(k.clone(), ananke_templates::substitute(v, &ctx)?);
    }
    Ok(out)
}

/// Build the container argv for a command service, resolving the same
/// placeholders as native process rendering but with the container-aware
/// listen host/port context.
fn command_container_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Result<Vec<String>, ContainerRenderError> {
    let TemplateConfig::Command(cmd) = &svc.template_config else {
        return Ok(Vec::new());
    };
    let (listen_host, port) = listen_endpoint(svc);
    let static_reserve_mb = match svc.allocation_mode {
        ananke_config::validate::AllocationMode::Static { reserve_mb } => Some(reserve_mb),
        _ => None,
    };
    let ctx = PlaceholderContext {
        name: &svc.name,
        port,
        model: None,
        allocation: alloc,
        static_reserve_mb,
        listen_host: Some(listen_host),
        host_port: svc.private_port,
    };
    let user_env = svc.env.clone();
    let (args, _) = ananke_templates::substitute_argv(&cmd.command, &user_env, &ctx)?;
    Ok(args)
}

/// Render the full container spec for a service.
pub fn render_container_spec(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&ananke_allocator::placement::CommandArgs>,
    run_id: i64,
    owner_uuid: &str,
) -> Result<ContainerSpec, ContainerRenderError> {
    let Some(container) = &svc.container else {
        unreachable!("render_container_spec called without a container block");
    };

    let name = container_name(&svc.name, run_id);

    // Mandatory labels merged with user labels; reserved namespace already
    // rejected at validation.
    let mut labels = container.labels.clone();
    labels.insert("io.ananke.managed".to_string(), "true".to_string());
    labels.insert("io.ananke.owner".to_string(), owner_uuid.to_string());
    labels.insert("io.ananke.service".to_string(), svc.name.to_string());
    labels.insert("io.ananke.run".to_string(), run_id.to_string());

    // CDI devices from allocated GPU ids.
    let gpu_ids = alloc.gpu_ids();
    let gpu_devices = match &container.gpu_device {
        Some(tmpl) => expand_cdi_devices(tmpl, &gpu_ids)?,
        None => Vec::new(),
    };

    // Mounts: convert validated config mounts to spawn-spec mounts.
    let mounts: Vec<ContainerMount> = container
        .mounts
        .iter()
        .map(|m| ContainerMount {
            source: m.source.clone(),
            target: m.target.clone(),
            read_only: m.read_only,
            selinux: m.selinux.clone().map(|s| s.to_string()),
        })
        .collect();

    // Validate no duplicate targets (defensive; validator already did this).
    let mut seen = BTreeMap::new();
    for (i, m) in mounts.iter().enumerate() {
        if let Some(&first) = seen.get(&m.target) {
            return Err(ContainerRenderError::AmbiguousMount(format!(
                "duplicate mount target {} at indices {} and {}",
                m.target, first, i
            )));
        }
        seen.insert(m.target.clone(), i);
    }

    // Container argv and entrypoint depend on template.
    let (entrypoint, command) = match &svc.template_config {
        TemplateConfig::LlamaCpp(lc) => {
            translate_llama_paths(svc, lc, &mounts)?;
            // Render the native llama argv, then translate typed paths
            // through the mounts. Only flags follow the image: `llama_server`
            // answers where llama-server lives *on the host*, and its
            // `llama-server` default is a `$PATH` lookup that means nothing
            // in an image — the official one ships `/app/llama-server` with
            // `/app` off `$PATH`. So the image's own ENTRYPOINT stands
            // unless `container.entrypoint` overrides it.
            let spawn_cfg = spawn::render_argv(svc, alloc, cmd_args)
                .map_err(ContainerRenderError::Substitute)?;
            let mut args = spawn_cfg.args;
            translate_argv_paths(&mut args, svc, &mounts)?;
            (container.entrypoint.clone(), args)
        }
        TemplateConfig::Command(_) => {
            let args = command_container_argv(svc, alloc)?;
            (container.entrypoint.clone(), args)
        }
    };

    // Only translate the model-prefixed paths for llama-cpp; command services
    // use container paths directly via placeholders.
    let extra_publications: Vec<ContainerPortPublication> = container
        .extra_publications
        .iter()
        .map(|p| ContainerPortPublication {
            host_ip: p.host_ip.clone(),
            host_port: p.host_port,
            container_port: p.container_port,
            protocol: p.protocol.to_string(),
        })
        .collect();

    let (network, container_port, host_port) = match container.network {
        ContainerNetwork::Bridge => (
            ananke_spawn::ContainerNetwork::Bridge,
            container.container_port,
            Some(svc.private_port),
        ),
        ContainerNetwork::Host => (ananke_spawn::ContainerNetwork::Host, None, None),
    };

    let ipc = match container.ipc {
        ContainerIpc::Host => ananke_spawn::ContainerIpc::Host,
        ContainerIpc::Private => ananke_spawn::ContainerIpc::Private,
    };

    let runtime = match container.runtime {
        ContainerRuntime::Docker => ananke_spawn::ContainerRuntime::Docker,
        ContainerRuntime::Podman => ananke_spawn::ContainerRuntime::Podman,
    };

    Ok(ContainerSpec {
        runtime,
        runtime_executable: container.runtime_executable.clone(),
        image: container.image.clone(),
        entrypoint,
        workdir: container.workdir.clone(),
        command,
        name,
        labels,
        network,
        container_port,
        host_port,
        ipc,
        gpu_devices,
        mounts,
        extra_publications,
        env: container_env(svc, alloc, container)?,
        env_passthrough: container.env_passthrough.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_lowercases_and_dashes() {
        assert_eq!(sanitize_name("Muse Glimmer"), "muse-glimmer");
        assert_eq!(
            sanitize_name("qwen3.6-35b-a3b-ninfer-dflash7"),
            "qwen3.6-35b-a3b-ninfer-dflash7"
        );
        assert_eq!(sanitize_name("A_B_C"), "a_b_c");
    }

    #[test]
    fn container_name_is_collision_resistant() {
        let n = container_name("Muse Glimmer", 12345);
        assert_eq!(n, "ananke-muse-glimmer-12345");
    }

    #[test]
    fn cdi_expansion_repeats_per_gpu() {
        let out = expand_cdi_devices("nvidia.com/gpu=${id}", &[0, 1]).unwrap();
        assert_eq!(out, ["nvidia.com/gpu=0", "nvidia.com/gpu=1"]);
    }

    #[test]
    fn cdi_rejects_missing_id() {
        assert!(matches!(
            expand_cdi_devices("nvidia.com/gpu", &[0]),
            Err(ContainerRenderError::BadCdiTemplate(_))
        ));
    }

    #[test]
    fn mount_translation_uses_longest_component_prefix() {
        let mounts = nested_mounts();
        assert_eq!(
            translate_path(&mounts, "/home/user/models/x.gguf").as_deref(),
            Some("/models/x.gguf")
        );
        assert_eq!(
            translate_path(&mounts, "/home/user/other").as_deref(),
            Some("/root/other")
        );
        // An exact source match translates to the bare target.
        assert_eq!(
            translate_path(&mounts, "/home/user/models").as_deref(),
            Some("/models")
        );
        // Unrelated path is not covered.
        assert_eq!(translate_path(&mounts, "/etc/hosts"), None);
    }

    #[test]
    fn mount_translation_rejects_partial_component_match() {
        let mounts = nested_mounts();
        // `/home/user2` shares a textual prefix with `/home/user` but not a
        // path component, so it must not be silently rewritten to `/root2`.
        assert_eq!(translate_path(&mounts, "/home/user2/x"), None);
        // `/home/user/models2` shares a textual prefix with the `/models`
        // mount's source but not a component, so it falls back to the
        // enclosing `/home/user` mount rather than landing under `/models`.
        assert_eq!(
            translate_path(&mounts, "/home/user/models2/x.gguf").as_deref(),
            Some("/root/models2/x.gguf")
        );
    }

    fn nested_mounts() -> Vec<ContainerMount> {
        vec![
            ContainerMount {
                source: "/home/user".into(),
                target: "/root".into(),
                read_only: false,
                selinux: None,
            },
            ContainerMount {
                source: "/home/user/models".into(),
                target: "/models".into(),
                read_only: true,
                selinux: None,
            },
        ]
    }

    // ── whole-spec rendering ────────────────────────────────────────────

    use std::path::PathBuf;

    use ananke_config::validate::{
        AllocationMode, DeviceSlot, PlacementPolicy,
        test_fixtures::{
            container_mount, expect_command, expect_llama_cpp, minimal_command_service,
            minimal_container_config, minimal_llama_cpp_service,
        },
    };
    use ananke_devices::Allocation;

    /// A GPU-0 allocation with a static reservation, the shape every
    /// container service in the field uses.
    fn gpu0_alloc() -> (BTreeMap<DeviceSlot, u64>, Allocation) {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 23_500);
        let alloc = Allocation::from_override(&placement);
        (placement, alloc)
    }

    /// A bridge-networked vLLM-shaped command service on container port 8000.
    fn bridge_command_service() -> (ServiceConfig, Allocation) {
        let (placement, alloc) = gpu0_alloc();
        let mut svc = minimal_command_service(
            "vllm",
            vec![
                "--model".into(),
                "some/model".into(),
                "--host".into(),
                "${listen_host}".into(),
                "--port".into(),
                "${listen_port}".into(),
            ],
        );
        svc.port = 8200;
        svc.private_port = 48200;
        svc.placement_override = placement;
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.allocation_mode = AllocationMode::Static { reserve_mb: 23_500 };
        let mut ct = minimal_container_config("vllm/vllm-openai:v0.26.0");
        ct.network = ananke_config::validate::ContainerNetwork::Bridge;
        ct.container_port = Some(8000);
        svc.container = Some(ct);
        (svc, alloc)
    }

    /// A host-networked llama.cpp service with its models under one mount.
    fn host_llama_service() -> (ServiceConfig, Allocation) {
        let (placement, alloc) = gpu0_alloc();
        let mut svc = minimal_llama_cpp_service("muse-glimmer");
        svc.port = 8202;
        svc.private_port = 48202;
        svc.placement_override = placement;
        svc.placement_policy = PlacementPolicy::GpuOnly;
        {
            let lc = expect_llama_cpp(&mut svc);
            lc.binary = PathBuf::from("llama-server");
            lc.model = PathBuf::from("/host/models/muse-glimmer.gguf");
            lc.mmproj = Some(PathBuf::from("/host/models/mmproj.gguf"));
            lc.draft_model = Some(PathBuf::from("/host/models/dflash.gguf"));
            lc.chat_template_file = Some(PathBuf::from("/host/models/template.jinja"));
        }
        let mut ct = minimal_container_config("ghcr.io/ggml-org/llama.cpp:server-cuda");
        ct.mounts = vec![container_mount("/host/models", "/models")];
        svc.container = Some(ct);
        (svc, alloc)
    }

    #[test]
    fn bridge_command_binds_container_interface() {
        let (svc, alloc) = bridge_command_service();
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert!(
            spec.command.iter().any(|a| a == "0.0.0.0"),
            "bridge command must bind all interfaces inside the container: {:?}",
            spec.command
        );
    }

    #[test]
    fn bridge_command_uses_container_port() {
        let (svc, alloc) = bridge_command_service();
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        // The argv binds the container side; the host side is the private port.
        assert!(
            spec.command.iter().any(|a| a == "8000"),
            "expected the container port in argv: {:?}",
            spec.command
        );
        assert!(
            !spec.command.iter().any(|a| a == "48200"),
            "the host private port must not leak into the container argv: {:?}",
            spec.command
        );
    }

    #[test]
    fn bridge_publish_maps_private_to_container_port() {
        let (svc, alloc) = bridge_command_service();
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert_eq!(spec.container_port, Some(8000));
        assert_eq!(spec.host_port, Some(48200));
    }

    #[test]
    fn host_network_preserves_private_port_and_loopback() {
        let (mut svc, alloc) = bridge_command_service();
        let ct = svc.container.as_mut().unwrap();
        ct.network = ananke_config::validate::ContainerNetwork::Host;
        ct.container_port = None;
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert!(spec.command.iter().any(|a| a == "127.0.0.1"));
        assert!(spec.command.iter().any(|a| a == "48200"));
        // Host networking has no publication at all.
        assert_eq!(spec.container_port, None);
        assert_eq!(spec.host_port, None);
    }

    #[test]
    fn bridge_llama_binds_all_interfaces() {
        let (mut svc, alloc) = host_llama_service();
        {
            let ct = svc.container.as_mut().unwrap();
            ct.network = ananke_config::validate::ContainerNetwork::Bridge;
            ct.container_port = Some(8080);
        }
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert!(
            spec.command.iter().any(|a| a == "0.0.0.0"),
            "bridge llama.cpp must bind all interfaces: {:?}",
            spec.command
        );
        assert!(spec.command.iter().any(|a| a == "8080"));
    }

    #[test]
    fn typed_llama_paths_all_translate() {
        let (svc, alloc) = host_llama_service();
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        for translated in [
            "/models/muse-glimmer.gguf",
            "/models/mmproj.gguf",
            "/models/dflash.gguf",
            "/models/template.jinja",
        ] {
            assert!(
                spec.command.iter().any(|a| a == translated),
                "expected {translated} in argv: {:?}",
                spec.command
            );
        }
        assert!(
            !spec.command.iter().any(|a| a.starts_with("/host/")),
            "a host path survived translation: {:?}",
            spec.command
        );
        // The image's own ENTRYPOINT stands, and no executable token leaks
        // into the flags that follow the image.
        assert_eq!(spec.entrypoint, None);
        assert!(!spec.command.iter().any(|a| a == "llama-server"));
        assert_eq!(spec.command.first().map(String::as_str), Some("-m"));
    }

    #[test]
    fn llama_entrypoint_comes_only_from_the_container_block() {
        // `llama_server` is a host `$PATH` lookup; emitting it as the
        // container entrypoint assumes an image layout the official one
        // does not have (`/app/llama-server`, `/app` off `$PATH`).
        let (mut svc, alloc) = host_llama_service();
        expect_llama_cpp(&mut svc).binary = PathBuf::from("llama-server");
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert_eq!(spec.entrypoint, None);

        // An image needing something else says so explicitly.
        svc.container.as_mut().unwrap().entrypoint = Some("/app/llama-server".into());
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert_eq!(spec.entrypoint.as_deref(), Some("/app/llama-server"));
    }

    #[test]
    fn container_rejects_unmapped_llama_path() {
        let (mut svc, alloc) = host_llama_service();
        expect_llama_cpp(&mut svc).mmproj = Some(PathBuf::from("/elsewhere/mmproj.gguf"));
        let err = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("/elsewhere/mmproj.gguf"), "{msg}");
        assert!(msg.contains("mmproj"), "{msg}");
    }

    #[test]
    fn extra_args_paths_are_not_guessed() {
        let (mut svc, alloc) = host_llama_service();
        // An opaque path in extra_args is passed through untouched: ananke
        // only owns the four typed path fields.
        svc.extra_args = vec!["--lora".into(), "/host/models/adapter.gguf".into()];
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert!(
            spec.command
                .iter()
                .any(|a| a == "/host/models/adapter.gguf"),
            "extra_args path must survive verbatim: {:?}",
            spec.command
        );
    }

    #[test]
    fn container_cdi_uses_allocated_gpus() {
        let (mut svc, _) = host_llama_service();
        svc.container.as_mut().unwrap().gpu_device = Some("nvidia.com/gpu=${id}".into());
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 10_000);
        placement.insert(DeviceSlot::Gpu(2), 10_000);
        svc.placement_override = placement.clone();
        let alloc = Allocation::from_override(&placement);
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert_eq!(spec.gpu_devices, ["nvidia.com/gpu=0", "nvidia.com/gpu=2"]);
    }

    #[test]
    fn placement_args_reach_the_container_argv() {
        // The packer's derived flags have to survive the whole launch path.
        // Dropping them renders single-GPU defaults instead — a different
        // command from the one the packer planned, and one that quietly
        // ignores a multi-GPU split.
        let (svc, alloc) = host_llama_service();
        let cmd_args = ananke_allocator::placement::CommandArgs {
            ngl: Some(40),
            tensor_split: Some(vec![30, 10]),
            override_tensor: Vec::new(),
            split_mode: None,
            main_gpu: None,
            n_cpu_moe: None,
        };
        let spec = render_container_spec(&svc, &alloc, Some(&cmd_args), 1, "owner").unwrap();

        assert_eq!(
            endpoint_of_flag(&spec.command, "-ngl").as_deref(),
            Some("40"),
            "packer ngl must win over the default: {:?}",
            spec.command
        );
        assert_eq!(
            endpoint_of_flag(&spec.command, "--tensor-split").as_deref(),
            Some("30,10"),
            "a multi-GPU split must reach the argv: {:?}",
            spec.command
        );
    }

    /// The value following `flag` in an argv.
    fn endpoint_of_flag(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    #[test]
    fn native_service_render_regression() {
        // Dropping the container block must return both templates to the
        // pre-container rendering exactly: loopback on the private port,
        // untranslated host paths, and the binary as argv[0].
        let (mut llama, alloc) = host_llama_service();
        llama.container = None;
        let cfg = crate::spawn::render_argv(&llama, &alloc, None).unwrap();
        assert_eq!(cfg.binary, "llama-server");
        assert_eq!(
            endpoint_of(&cfg.args),
            (Some("127.0.0.1".to_string()), Some("48202".to_string()))
        );
        assert!(
            cfg.args
                .iter()
                .any(|a| a == "/host/models/muse-glimmer.gguf"),
            "a native service must keep its host paths: {:?}",
            cfg.args
        );

        let (mut command, alloc) = bridge_command_service();
        command.container = None;
        expect_command(&mut command).command = vec![
            "serve".into(),
            "--host".into(),
            "${listen_host}".into(),
            "--port".into(),
            "${port}".into(),
        ];
        let cfg = crate::spawn::render_argv(&command, &alloc, None).unwrap();
        assert_eq!(cfg.binary, "serve");
        assert_eq!(
            endpoint_of(&cfg.args),
            (Some("127.0.0.1".to_string()), Some("48200".to_string()))
        );
    }

    /// The values following `--host` and `--port` in a rendered argv.
    fn endpoint_of(args: &[String]) -> (Option<String>, Option<String>) {
        let after = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        (after("--host"), after("--port"))
    }

    #[test]
    fn service_env_reaches_the_container() {
        // A container gets no daemon environment at all, so anything the
        // service declared is the only thing it has. Dropping the
        // service-level half would strip settings like vLLM's allocator
        // tuning without any error to show for it.
        let (mut svc, alloc) = bridge_command_service();
        svc.env = BTreeMap::from([
            ("VLLM_NO_USAGE_STATS".to_string(), "1".to_string()),
            ("OMP_NUM_THREADS".to_string(), "1".to_string()),
            ("SHADOWED".to_string(), "from-service".to_string()),
        ]);
        {
            let ct = svc.container.as_mut().unwrap();
            ct.env = BTreeMap::from([
                ("SHADOWED".to_string(), "from-container".to_string()),
                (
                    "BIND".to_string(),
                    "${listen_host}:${listen_port}".to_string(),
                ),
            ]);
        }
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();

        assert_eq!(
            spec.env.get("VLLM_NO_USAGE_STATS").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            spec.env.get("OMP_NUM_THREADS").map(String::as_str),
            Some("1")
        );
        // The container's own block wins the overlap.
        assert_eq!(
            spec.env.get("SHADOWED").map(String::as_str),
            Some("from-container")
        );
        // Env values resolve the same endpoint the argv does.
        assert_eq!(
            spec.env.get("BIND").map(String::as_str),
            Some("0.0.0.0:8000")
        );
    }

    #[test]
    fn json_command_arguments_survive_rendering() {
        // vLLM's JSON flags are arguments, not placeholders; a container
        // launch must not fail on them.
        let (mut svc, alloc) = bridge_command_service();
        expect_command(&mut svc).command = vec![
            "--diffusion-config".into(),
            r#"{"canvas_length": 256}"#.into(),
            "--limit-mm-per-prompt".into(),
            r#"{"image": 7}"#.into(),
            "--host".into(),
            "${listen_host}".into(),
            "--port".into(),
            "${listen_port}".into(),
        ];
        let spec = render_container_spec(&svc, &alloc, None, 1, "owner").unwrap();
        assert!(
            spec.command
                .iter()
                .any(|a| a == r#"{"canvas_length": 256}"#)
        );
        assert!(spec.command.iter().any(|a| a == r#"{"image": 7}"#));
        assert!(spec.command.iter().any(|a| a == "0.0.0.0"));
    }

    #[test]
    fn reserved_labels_are_always_applied() {
        let (svc, alloc) = host_llama_service();
        let spec = render_container_spec(&svc, &alloc, None, 42, "owner-uuid").unwrap();
        assert_eq!(
            spec.labels.get("io.ananke.managed").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            spec.labels.get("io.ananke.owner").map(String::as_str),
            Some("owner-uuid")
        );
        assert_eq!(
            spec.labels.get("io.ananke.service").map(String::as_str),
            Some("muse-glimmer")
        );
        assert_eq!(
            spec.labels.get("io.ananke.run").map(String::as_str),
            Some("42")
        );
        assert_eq!(spec.name, "ananke-muse-glimmer-42");
    }
}
