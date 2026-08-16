//! Descriptors for the `[service.container]` block and its nested mount
//! and extra-publication tables.
//!
//! The block is shared by both templates, so it is documented once as a
//! top-level section rather than duplicated under `llama-cpp` and
//! `command`.

use crate::docs::{SectionDoc, code_values, field};

/// Return the container field-reference section and its sub-tables.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "container",
            title: "Container workloads",
            fields: vec![
                field(
                    "runtime",
                    "string",
                    "`docker`",
                    format!(
                        "Container runtime to drive. One of {}.",
                        code_values(&["docker", "podman"])
                    ),
                ),
                field(
                    "runtime_executable",
                    "path",
                    "the runtime name",
                    "Absolute path to the runtime binary. Set this when the binary isn't on the daemon's `$PATH` (e.g. a Nix store path). The value is recorded in the launch intent, so a container stays reconcilable across a runtime change.",
                ),
                field(
                    "image",
                    "string",
                    "*required*",
                    "Image reference to run, e.g. `vllm/vllm-openai:v0.26.0`. Must already be in the runtime's local store: ananke neither pulls nor builds, and a missing image fails the start with the runtime's own error.",
                ),
                field(
                    "entrypoint",
                    "string",
                    "the image's own",
                    "Replaces the image ENTRYPOINT. Unset, the image's own applies — including for `llama-cpp` services, whose `llama_server` is a host-side path and is not used inside the container.",
                ),
                field(
                    "workdir",
                    "path",
                    "the image's own",
                    "Working directory inside the container.",
                ),
                field(
                    "network",
                    "string",
                    "`bridge`",
                    format!(
                        "Network mode, one of {}. See [Networking](#networking) for what each resolves the service endpoint to.",
                        code_values(&["bridge", "host"])
                    ),
                ),
                field(
                    "container_port",
                    "u16",
                    "*required in bridge mode*",
                    "Port the workload binds inside the container. ananke publishes `127.0.0.1:<private_port>:<container_port>`. Rejected as meaningless under host networking.",
                ),
                field(
                    "ipc",
                    "string",
                    "`private`",
                    format!(
                        "IPC namespace, one of {}. `host` shares the host `/dev/shm`, which vLLM and other multi-worker runtimes need.",
                        code_values(&["private", "host"])
                    ),
                ),
                field(
                    "gpu_device",
                    "string",
                    "none",
                    "CDI device template expanded once per GPU ananke's placement picked, e.g. `nvidia.com/gpu={id}`. Must contain `{id}` exactly once. Unset means no GPU is injected — there is no `/dev/nvidia*` fallback.",
                ),
                field(
                    "env",
                    "table",
                    "none",
                    "Explicit environment for the container, merged over the service's own `env`. Container services never inherit the daemon's environment: `env_inherit` governs host processes only.",
                ),
                field(
                    "env_passthrough",
                    "array of string",
                    "none",
                    "Names of host environment variables forwarded into the container by name, e.g. `[\"HF_TOKEN\"]`. Values are read from the daemon's environment at launch and never rendered into the API, previews, or logs.",
                ),
                field(
                    "labels",
                    "table",
                    "none",
                    "User labels applied to the container. The whole `io.ananke.*` namespace is reserved for ananke's ownership labels and rejected here.",
                ),
                field(
                    "mounts",
                    "array of table",
                    "none",
                    "Bind mounts (see [Mounts](#mounts)).",
                ),
                field(
                    "extra_publications",
                    "array of table",
                    "none",
                    "Port publications beyond the service endpoint (see [Extra publications](#extra-publications)). Bridge mode only.",
                ),
            ],
        },
        SectionDoc {
            id: "container_mounts",
            title: "Mounts",
            fields: vec![
                field(
                    "source",
                    "path",
                    "*required*",
                    "Absolute host path. Must exist when the container is created.",
                ),
                field(
                    "target",
                    "path",
                    "*required*",
                    "Absolute container path. Two mounts may not share a target.",
                ),
                field("read_only", "bool", "`false`", "Mount read-only."),
                field(
                    "selinux",
                    "string",
                    "none",
                    format!(
                        "SELinux relabel policy, one of {} ({} is shared, {} is private). Omit on systems without SELinux.",
                        code_values(&["z", "Z"]),
                        "`z`",
                        "`Z`"
                    ),
                ),
            ],
        },
        SectionDoc {
            id: "container_extra_publications",
            title: "Extra publications",
            fields: vec![
                field(
                    "host_ip",
                    "string",
                    "`127.0.0.1`",
                    "Host address to bind. The default keeps the port off the network, matching the service endpoint's own publication.",
                ),
                field("host_port", "u16", "*required*", "Host-side port."),
                field(
                    "container_port",
                    "u16",
                    "*required*",
                    "Container-side port.",
                ),
                field(
                    "protocol",
                    "string",
                    "`tcp`",
                    format!("One of {}.", code_values(&["tcp", "udp"])),
                ),
            ],
        },
    ]
}
