//! Descriptors for the top-level `[daemon]`, `[openai_api]`, `[defaults]`,
//! and `[devices]` config sections.

use crate::{
    defaults::{MANAGEMENT_LISTEN, OPENAI_LISTEN},
    docs::{
        DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_OPENAI_MAX_BODY_MB,
        DEFAULT_PRIVATE_PORT_END, DEFAULT_PRIVATE_PORT_START, DEFAULT_SERVICE_PRIORITY,
        DEFAULT_START_QUEUE_DEPTH, SectionDoc, bt, bt_dur, field,
    },
};

/// Return the daemon, OpenAI API, defaults, and devices sections.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "daemon",
            title: "Daemon Settings",
            fields: vec![
                field(
                    "management_listen",
                    "string",
                    bt(MANAGEMENT_LISTEN),
                    "Bind address for the management API. Non-loopback requires `allow_external_management = true`.",
                ),
                field(
                    "allow_external_management",
                    "bool",
                    "`false`",
                    "Must be `true` when `management_listen` is non-loopback.",
                ),
                field(
                    "allow_external_services",
                    "bool",
                    "`false`",
                    "Bind per-service reverse proxies on `0.0.0.0` instead of `127.0.0.1`. Controls only the per-service proxies, not the OpenAI multiplexer (which honours `openai_api.listen`).",
                ),
                field(
                    "data_dir",
                    "path",
                    "`$XDG_DATA_HOME/ananke` (or `~/.local/share/ananke`)",
                    "Directory for the SQLite database and runtime state.",
                ),
                field(
                    "shutdown_timeout",
                    "duration string",
                    "`120s`",
                    "Max time to wait for services to drain on daemon shutdown.",
                ),
                field(
                    "private_port_start",
                    "u16",
                    bt(DEFAULT_PRIVATE_PORT_START),
                    "Inclusive lower bound of the loopback port range handed to llama-server children for their private listener.",
                ),
                field(
                    "private_port_end",
                    "u16",
                    bt(DEFAULT_PRIVATE_PORT_END),
                    "Inclusive upper bound of the private-listener port range. Override when another process occupies the default window.",
                ),
                field(
                    "llama_server",
                    "path",
                    "`llama-server` (from `$PATH`)",
                    "Default llama-server executable for every llama-cpp service. Overridable per-service.",
                ),
            ],
        },
        SectionDoc {
            id: "openai_api",
            title: "OpenAI API Settings",
            fields: vec![
                field(
                    "listen",
                    "string",
                    bt(OPENAI_LISTEN),
                    "Bind address for the OpenAI-compatible API.",
                ),
                field(
                    "enabled",
                    "bool",
                    "`true`",
                    "Set to `false` to disable the OpenAI API entirely.",
                ),
                field(
                    "max_request_duration",
                    "duration string",
                    bt_dur(DEFAULT_MAX_REQUEST_DURATION_MS),
                    "Max wall-clock duration per proxied request.",
                ),
                field(
                    "allow_cors",
                    "bool",
                    "`true`",
                    "Allow cross-origin requests from browsers. Set to `false` to block browser-based access.",
                ),
                field(
                    "max_body_mb",
                    "u64",
                    bt(DEFAULT_OPENAI_MAX_BODY_MB),
                    "Max request body size in MiB. Raise for large or many images (vision payloads are base64-encoded).",
                ),
            ],
        },
        SectionDoc {
            id: "defaults",
            title: "Global Defaults",
            fields: vec![
                field(
                    "idle_timeout",
                    "duration string",
                    bt_dur(DEFAULT_IDLE_TIMEOUT_MS),
                    "Default idle timeout for on-demand services.",
                ),
                field(
                    "priority",
                    "u8",
                    bt(DEFAULT_SERVICE_PRIORITY),
                    "Default eviction priority (higher wins eviction contests).",
                ),
                field(
                    "start_queue_depth",
                    "u32",
                    bt(DEFAULT_START_QUEUE_DEPTH),
                    "Default concurrency cap on pending start requests waiting for the same supervisor before they are rejected with `QueueFull`.",
                ),
            ],
        },
        SectionDoc {
            id: "devices",
            title: "Device Configuration",
            fields: vec![
                field(
                    "gpu_ids",
                    "array of u32",
                    "all visible GPUs",
                    "Only probe these GPUs.",
                ),
                field(
                    "default_gpu_reserved_mb",
                    "u64",
                    "`0`",
                    "VRAM (MiB) kept free on every GPU that lacks a `gpu_reserved_mb` entry.",
                ),
                field(
                    "gpu_reserved_mb",
                    "map string → u64",
                    "empty",
                    "Per-GPU VRAM reserve (MiB), keyed by GPU id string.",
                ),
                field(
                    "cpu.enabled",
                    "bool",
                    "`true`",
                    "Allow CPU placement for services.",
                ),
                field(
                    "cpu.reserved_gb",
                    "u64",
                    "`0`",
                    "Host RAM (GiB) the daemon keeps free. Bounds how much expert weight a hybrid MoE service may offload to the CPU; a placement that would exceed it is rejected.",
                ),
            ],
        },
    ]
}
