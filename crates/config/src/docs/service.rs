//! Descriptors for the per-service config sections shared by every service
//! template: common fields, placement, health checks, resource allocation,
//! request filters, and tracking.

use crate::docs::{
    DEFAULT_DRAIN_TIMEOUT_MS, DEFAULT_EXTENDED_STREAM_DRAIN_MS, DEFAULT_HEALTH_PROBE_INTERVAL_MS,
    DEFAULT_HEALTH_TIMEOUT_MS, DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_MAX_REQUEST_DURATION_MS,
    DEFAULT_MIN_BORROWER_RUNTIME_MS, DEFAULT_SERVICE_PRIORITY, DEFAULT_START_QUEUE_DEPTH,
    SectionDoc, bt, bt_dur, code_values, field,
};

/// Return the common-fields, placement, health-check, allocation, filter,
/// and tracking sections shared by every service template.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "service_common",
            title: "Common Fields",
            fields: vec![
                field("name", "string", "*required*", "Unique service identifier."),
                field(
                    "template",
                    "string",
                    "*required*",
                    "`\"llama-cpp\"` or `\"command\"`.",
                ),
                field(
                    "port",
                    "u16",
                    "*required*",
                    "Public-facing port for the service's reverse proxy.",
                ),
                field(
                    "lifecycle",
                    "string",
                    "`\"on_demand\"`",
                    "`\"on_demand\"` or `\"persistent\"` (see [Lifecycle](#lifecycle)).",
                ),
                field(
                    "priority",
                    "u8",
                    format!("{} (or `[defaults]` value)", bt(DEFAULT_SERVICE_PRIORITY)),
                    "Eviction priority; higher wins eviction contests.",
                ),
                field(
                    "idle_timeout",
                    "duration string",
                    format!(
                        "{} (or `[defaults]` value)",
                        bt_dur(DEFAULT_IDLE_TIMEOUT_MS)
                    ),
                    "Idle timeout for on-demand services.",
                ),
                field(
                    "description",
                    "string",
                    "none",
                    "Human-readable description exposed through `/v1/models` and `/api/services`.",
                ),
                field(
                    "modality",
                    "string",
                    "`\"chat\"`",
                    "`\"chat\"` or `\"embedding\"` (see [Embedding Services](#embedding-services)). On `llama-cpp` services, `\"embedding\"` also passes `--embeddings` to llama-server. Any other string is a hard config error.",
                ),
                field(
                    "extra_args",
                    "array of string",
                    "none",
                    "Extra argv appended to the service's launch command.",
                ),
                field(
                    "extra_args_append",
                    "array of string",
                    "none",
                    "Extra argv appended to the inherited list (use with `extends`; concatenated with parent's list).",
                ),
                field(
                    "env",
                    "map string → string",
                    "none",
                    "Environment variables set on the spawned process. Accepts `{port}`, `{gpu_ids}`, `{reserve_mb}`, `{model}`, `{name}` placeholders.",
                ),
                field(
                    "env_inherit",
                    "bool",
                    "`true`",
                    "Whether the child process inherits the daemon's environment (`$PATH`, `$HOME`, locale, …). Per-service `env` entries override individual inherited keys. Set `false` to start with a clean environment containing only the variables in `env` plus `CUDA_VISIBLE_DEVICES`.",
                ),
                field(
                    "drain_timeout",
                    "duration string",
                    bt_dur(DEFAULT_DRAIN_TIMEOUT_MS),
                    "Drain timeout before the supervisor escalates to SIGKILL.",
                ),
                field(
                    "extended_stream_drain",
                    "duration string",
                    bt_dur(DEFAULT_EXTENDED_STREAM_DRAIN_MS),
                    "Extra grace granted to in-flight streaming requests during drain.",
                ),
                field(
                    "max_request_duration",
                    "duration string",
                    bt_dur(DEFAULT_MAX_REQUEST_DURATION_MS),
                    "Cap on wall-clock duration of a single proxied request.",
                ),
                field(
                    "start_queue_depth",
                    "u32",
                    format!("{} (or `[defaults]` value)", bt(DEFAULT_START_QUEUE_DEPTH)),
                    "Concurrency cap on pending start requests before `QueueFull` rejection.",
                ),
                field(
                    "extends",
                    "string",
                    "none",
                    "Name of a parent service to inherit from. See [Service Inheritance](#service-inheritance).",
                ),
                field(
                    "migrate_from",
                    "string",
                    "none",
                    "Old service name to preserve database history from. See [Service Migration](#service-migration).",
                ),
            ],
        },
        SectionDoc {
            id: "service_devices",
            title: "Placement",
            fields: vec![
                field(
                    "placement",
                    "string",
                    "`\"gpu-only\"`",
                    "Placement policy (see below).",
                ),
                field(
                    "gpu_allow",
                    "array of u32",
                    "all `[devices]` GPUs",
                    "Restrict the service to these GPU ids.",
                ),
                field(
                    "gpu_headroom_mb",
                    "u64",
                    "`0`",
                    "Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added on top of the global `[devices]` reserve. Lets a single model be packed more conservatively without bypassing the estimator.",
                ),
                field(
                    "placement_override",
                    "map string → u64",
                    "none",
                    "Hand-pin VRAM (MiB) per device slot. Keys: `\"cpu\"` or `\"gpu:N\"`. Overrides the estimator's per-slot distribution. Must be non-empty if present; zero values and `cpu` keys under `gpu-only` are rejected.",
                ),
                field(
                    "split",
                    "string",
                    "`\"layer\"`",
                    format!(
                        "Multi-GPU split mode for llama.cpp services: {}. Maps to llama.cpp's `--split-mode`. See [Multi-GPU split modes](#multi-gpu-split-modes) for constraints.",
                        code_values(crate::flags::split_mode::ALL)
                    ),
                ),
                field(
                    "tensor_split_weights",
                    "array of f32",
                    "none",
                    "Optional per-GPU weights for the `--tensor-split` ratio in sharded (`row`/`tensor`) modes. One positive weight per allowed GPU, in ascending GPU-id order. Unset gives an equal `1,1,...` split. Use this for heterogeneous GPUs (e.g. weight by relative memory bandwidth). Weights are meaningful to four decimal places; additional precision is rounded when converting to the integer `--tensor-split` ratio. See [Multi-GPU split modes](#multi-gpu-split-modes).",
                ),
            ],
        },
        SectionDoc {
            id: "service_health",
            title: "Health Checks",
            fields: vec![
                field(
                    "http",
                    "string",
                    "`/v1/models`",
                    "HTTP path to probe for readiness. Set to `\"\"` (empty string) to disable the health check entirely - the service transitions to Running immediately after spawn, with no readiness probe.",
                ),
                field(
                    "timeout",
                    "duration string",
                    bt_dur(DEFAULT_HEALTH_TIMEOUT_MS),
                    "Per-probe timeout before a health check fails.",
                ),
                field(
                    "probe_interval",
                    "duration string",
                    bt_dur(DEFAULT_HEALTH_PROBE_INTERVAL_MS),
                    "Cadence between health probes.",
                ),
            ],
        },
        SectionDoc {
            id: "service_allocation",
            title: "Resource Allocation",
            fields: vec![
                field(
                    "mode",
                    "string",
                    "*required* (command only)",
                    "`\"static\"` or `\"dynamic\"`. Required on every `command` service. Optional on a `llama-cpp` one, where it replaces the estimator entirely — the only way to place a model whose architecture ananke does not recognise.",
                ),
                field(
                    "reserve_gb",
                    "f32",
                    "none",
                    "`static` only. Memory to reserve, in GiB — host RAM for a cpu-only service, VRAM otherwise. Required for `static`. Accepted as `vram_gb` for pre-rename configs.",
                ),
                field(
                    "min_reserve_gb",
                    "f32",
                    "none",
                    "`dynamic` only. Minimum reservation in GiB. Required for `dynamic`. Accepted as `min_vram_gb` for pre-rename configs.",
                ),
                field(
                    "max_reserve_gb",
                    "f32",
                    "none",
                    "`dynamic` only. Maximum reservation in GiB. Required for `dynamic`; must be > `min_reserve_gb`. Accepted as `max_vram_gb` for pre-rename configs.",
                ),
                field(
                    "min_borrower_runtime",
                    "duration string",
                    bt_dur(DEFAULT_MIN_BORROWER_RUNTIME_MS),
                    "`dynamic` only. Balloon resolver grace period: minimum runtime a borrower must accumulate before it may be fast-killed.",
                ),
            ],
        },
        SectionDoc {
            id: "service_filters",
            title: "Request Filters",
            fields: vec![
                field(
                    "strip_params",
                    "array of string",
                    "none",
                    "JSON keys to remove from the request body before forwarding.",
                ),
                field(
                    "set_params",
                    "map string → toml value",
                    "none",
                    "JSON key/value pairs to set on the request body before forwarding.",
                ),
            ],
        },
        SectionDoc {
            id: "service_tracking",
            title: "Tracking",
            fields: vec![field(
                "cgroup_parent",
                "string",
                "none",
                "Cgroup v2 path under which the service's actual workload pids live. Used by services whose workload runs in a container and is therefore reparented out of the daemon's process tree, so descendant-pid attribution can't reach it. Pids whose `/proc/<pid>/cgroup` path equals this value or sits inside its subtree are summed into the service's observed peak. Must be an absolute cgroup path (no trailing slash).",
            )],
        },
    ]
}
