//! Descriptors for the `llama-cpp` service template: the top-level field
//! reference plus its `estimation` and `sampling` sub-tables.

use crate::docs::{SectionDoc, code_values, field};

/// Return the llama-cpp field-reference, estimation-overrides, and sampling
/// sections.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "llama_cpp",
            title: "llama-cpp field reference",
            fields: vec![
                field(
                    "model",
                    "path",
                    "*required*",
                    "Path to the GGUF model file.",
                ),
                field(
                    "mmproj",
                    "path",
                    "none",
                    "Path to an optional vision projector GGUF. Services with an `mmproj` render a purple `vision` badge.",
                ),
                field(
                    "context",
                    "u32",
                    "`4096` (estimator default)",
                    "Context window size. If unset, a warning is logged and the estimator defaults to 4096 tokens.",
                ),
                field(
                    "n_gpu_layers",
                    "i32",
                    "`-1`",
                    "Number of layers to offload to GPU. `-1` (default) offloads all layers. Must be `0` under `placement = \"cpu-only\"`.",
                ),
                field(
                    "expert_offload",
                    "string or u32",
                    "`\"off\"`",
                    "MoE expert-offload policy (see [MoE Expert Offload](#moe-expert-offload)).",
                ),
                field(
                    "runtime",
                    "table",
                    "mainline llama.cpp",
                    "Serving runtime, tagged by `kind`: `{ kind = \"ik-llama\", mla = 1, dsa = true, attn_max_batch = 512, runtime_repack = false }` selects the ik_llama.cpp fork and its options (`-mla`, `-dsa -fidx`, `-amb`, `-rtr`). Absent means mainline. `dsa` requires f16 KV. Point `llama_server` at a matching binary. ik services use ik's `--spec-type` dialect (`\"mtp:n_max=4,p_min=0.5\"`).",
                ),
                field(
                    "flash_attn",
                    "bool",
                    "`false`",
                    "Enable flash attention. Required for quantised KV cache types (`cache_type_k`/`cache_type_v` other than `f16`) on mainline llama.cpp; ik_llama handles quantised caches without this flag.",
                ),
                field(
                    "cache_type_k",
                    "string",
                    "`f16`",
                    "KV cache type for keys. Non-`f16` values require `flash_attn = true` (mainline only; ik_llama is exempt).",
                ),
                field(
                    "cache_type_v",
                    "string",
                    "`f16`",
                    "KV cache type for values. Non-`f16` values require `flash_attn = true` (mainline only; ik_llama is exempt).",
                ),
                field("mmap", "bool", "`true`", "Memory-map the model file."),
                field(
                    "mlock",
                    "bool",
                    "`false`",
                    "Lock the model in RAM (prevents swapping).",
                ),
                field(
                    "parallel",
                    "u32",
                    "`1`",
                    "Request parallelism (`-np`). With a non-unified KV this splits the context budget across slots, so each request caps at `context / parallel`.",
                ),
                field(
                    "spec_type",
                    "string",
                    "none",
                    "Speculative-decoding type passed to `--spec-type` (e.g. `\"draft-mtp\"` for multi-token prediction).",
                ),
                field(
                    "spec_draft_n_max",
                    "u32",
                    "none",
                    "Max draft tokens per step (`--spec-draft-n-max`). Only meaningful when `spec_type` is set.",
                ),
                field(
                    "draft_model",
                    "path",
                    "none",
                    "Separate draft-model GGUF for speculative decoding (`-md` / `--model-draft`). Requires `spec_type` to be set.",
                ),
                field(
                    "kv_unified",
                    "bool",
                    "`false`",
                    "Use a single unified KV cache pool shared across all parallel slots (`-kvu` / `--kv-unified`). With `parallel > 1`, idle slots lend their share to active ones; total KV footprint is unchanged.",
                ),
                field(
                    "cache_idle_slots",
                    "bool",
                    "`true`",
                    "When `false`, pass `--no-cache-idle-slots` so idle slots' prompt-cache state is dropped (a stability mitigation).",
                ),
                field(
                    "cache_ram_mb",
                    "int (MiB)",
                    "`8192`",
                    "Host RAM cap for llama-server's prompt cache (`-cram`), which holds serialized evicted prompts so a returning conversation skips reprocessing. Always passed through explicitly, so the packer's host reservation and the runtime's cap are the same number; `0` disables the cache and frees the reservation with it.",
                ),
                field(
                    "metrics",
                    "bool",
                    "`false`, but auto-enabled while the `generation_stall` watchdog is on",
                    "Expose llama-server's Prometheus `/metrics` endpoint. The generation-stall watchdog needs it and passes `--metrics` automatically while active; an explicit `metrics = false` suppresses the flag and disables that watchdog.",
                ),
                field(
                    "slots",
                    "bool",
                    "`false`",
                    "Expose the `/slots` introspection endpoint. Note: reveals prompt contents - avoid on network-reachable ports.",
                ),
                field("batch_size", "u32", "none", "Context batch size (`-b`)."),
                field("ubatch_size", "u32", "none", "Physical batch size (`-ub`)."),
                field("threads", "u32", "none", "Number of CPU threads (`-t`)."),
                field(
                    "threads_batch",
                    "u32",
                    "none",
                    "Number of CPU threads for batch processing (`-tb`).",
                ),
                field(
                    "numa",
                    "string",
                    "none",
                    format!(
                        "NUMA thread-and-memory placement strategy (`--numa`): {}. Unset leaves llama.cpp's default.",
                        code_values(crate::flags::numa::ALL)
                    ),
                ),
                field("jinja", "bool", "`false`", "Use Jinja chat templates."),
                field(
                    "chat_template_file",
                    "path",
                    "none",
                    "Path to a custom chat template file.",
                ),
                field(
                    "override_tensor",
                    "array of string",
                    "none",
                    "Manual tensor placement rules (e.g. `[ \".ffn_(up|down)_exps.=CPU\" ]`). Incompatible with sharded split modes (`row`/`tensor`).",
                ),
                field(
                    "sampling",
                    "table",
                    "none",
                    "Sampling parameters (see [Sampling](#sampling)).",
                ),
                field(
                    "estimation",
                    "table",
                    "none",
                    "Estimator overrides (see [Estimation Overrides](#estimation-overrides)).",
                ),
                field(
                    "llama_server",
                    "path",
                    "daemon's `llama_server` or `$PATH`",
                    "Per-service override of the llama-server executable. Has no effect when `launcher` is set.",
                ),
                field(
                    "launcher",
                    "array of string",
                    "none",
                    "Full argv template that replaces the default `llama-server -m <model> ...` invocation (see [Custom llama-server Binary or Wrapper](#custom-llama-server-binary-or-wrapper)).",
                ),
            ],
        },
        SectionDoc {
            id: "llama_cpp_estimation",
            title: "Estimation overrides",
            fields: vec![
                field(
                    "compute_buffer_mb",
                    "u32",
                    "none",
                    "Override the estimated compute buffer size (MiB).",
                ),
                field(
                    "safety_factor",
                    "f32",
                    "none",
                    "Multiplier applied to the estimated VRAM footprint.",
                ),
            ],
        },
        SectionDoc {
            id: "llama_cpp_sampling",
            title: "Sampling",
            fields: vec![
                field("temperature", "f32", "none", "Sampling temperature."),
                field("top_p", "f32", "none", "Nucleus sampling threshold."),
                field("top_k", "u32", "none", "Top-k sampling limit."),
                field("min_p", "f32", "none", "Minimum-p sampling threshold."),
                field(
                    "repeat_penalty",
                    "f32",
                    "none",
                    "Repeat penalty applied to generated tokens.",
                ),
            ],
        },
    ]
}
