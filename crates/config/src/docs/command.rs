//! Descriptors for the `command` service template's field reference and its
//! `openai_proxy` sub-table.

use crate::docs::{SectionDoc, field};

/// Return the command field-reference and OpenAI-proxy sections.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "command",
            title: "command field reference",
            fields: vec![
                field(
                    "command",
                    "array of string",
                    "*required*",
                    "argv to execute. Accepts placeholders (see below).",
                ),
                field(
                    "workdir",
                    "path",
                    "none",
                    "Working directory for the spawned process.",
                ),
                field(
                    "allocation",
                    "table",
                    "none",
                    "Memory reservation (see [Resource Allocation](#resource-allocation)). Required for command services.",
                ),
                field(
                    "private_port",
                    "u16",
                    "auto-assigned",
                    "Upstream port ananke's reverse proxy should forward to. When absent, ananke picks one from the daemon's private-port pool and substitutes it into `command`/`env` via the `${port}` placeholder. Set explicitly when the external service binds a fixed port (e.g. a docker container exposing 18188 on the host).",
                ),
                field(
                    "shutdown_command",
                    "array of string",
                    "none",
                    "Optional argv run at drain time after SIGTERM-then-SIGKILL completes. Useful for external services that don't stop via signal - e.g. a docker-run wrapper where SIGTERM reaches the host shell but the container needs an explicit `docker stop`. Accepts the same placeholder substitutions as `command`.",
                ),
                field(
                    "openai_proxy",
                    "table",
                    "none",
                    "Opt the service into the OpenAI-compatible multiplexer (see [OpenAI Proxy](#openai-proxy)).",
                ),
            ],
        },
        SectionDoc {
            id: "openai_proxy",
            title: "OpenAI proxy",
            fields: vec![field(
                "upstream_model",
                "string",
                "none",
                "Model name the upstream server was started with (e.g. via `--served-model-name`). ananke rewrites the JSON `model` field to this value before forwarding.",
            )],
        },
    ]
}
