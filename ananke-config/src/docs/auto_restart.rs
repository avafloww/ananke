//! Descriptors for the `[services.*.auto_restart]` watchdog section and its
//! per-trigger sub-tables (error-rate, periodic, TTFT stall, generation
//! stall, spec-collapse).

use crate::docs::{
    DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS, DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
    DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS, DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
    DEFAULT_AUTO_RESTART_MAX_RESTARTS, DEFAULT_AUTO_RESTART_MIN_REQUESTS,
    DEFAULT_AUTO_RESTART_MIN_UPTIME_MS, DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS, DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
    DEFAULT_AUTO_RESTART_TTFT_STALL_MS, DEFAULT_AUTO_RESTART_WINDOW_MS, SectionDoc, bt, bt_dur,
    field,
};

/// Return the auto-restart section and its per-trigger sub-tables.
pub(crate) fn sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "service_auto_restart",
            title: "Auto-restart",
            fields: vec![
                field(
                    "error_rate",
                    "table | `false`",
                    "on, with the defaults below",
                    "Error-rate watchdog. `false` disables it; a table enables it and overrides individual thresholds.",
                ),
                field(
                    "periodic",
                    "table | `false`",
                    "off",
                    "Periodic restart. Absent or `false` disables it; a table (with an `interval`) enables it.",
                ),
                field(
                    "ttft_stall",
                    "table | `false`",
                    "on, with the defaults below",
                    "Time-to-first-token stall watchdog. `false` disables it; a table enables it and overrides the timeout. Catches a wedged child that accepts a streaming request but never emits a frame — a failure the error-rate watchdog cannot see, because the request never completes. Restarts only when the whole service has gone silent, so it never fights healthy concurrent traffic.",
                ),
                field(
                    "generation_stall",
                    "table | bool",
                    "on for `llama-cpp` services, off for `command` services",
                    "Generation-stall watchdog. Polls the child's `/metrics` progress counters and restarts when they stay flat while requests are in flight — the wedge `ttft_stall` cannot see, because non-streaming requests give the proxy nothing to watch. Needs the child's `--metrics` endpoint; see the generation-stall trigger section below.",
                ),
                field(
                    "spec_collapse",
                    "table | bool",
                    "on for `llama-cpp` services, off for `command` services",
                    "Speculative-decoding collapse watchdog. Fires when a run that previously accepted draft tokens stops accepting any across a full window of drafting requests, which indicates corrupted inference state (e.g. all-NaN logits) that still returns HTTP 200 and is invisible to the other watchdogs. On by default only when `spec_type` is set; an explicit per-service enable without `spec_type` is rejected. See the spec-collapse trigger section below.",
                ),
                field(
                    "min_uptime",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_MIN_UPTIME_MS),
                    "Minimum uptime a fresh run must reach before an error-rate, generation-stall, or spec-collapse restart may fire — the anti-flap cooldown.",
                ),
                field(
                    "max_restarts",
                    "u32",
                    bt(DEFAULT_AUTO_RESTART_MAX_RESTARTS),
                    "Watchdog restarts (error-rate, stall, generation-stall, and spec-collapse) tolerated within `flap_window` before the service is disabled with reason `auto_restart_loop` instead of restarted again. Periodic restarts are intentional and do not count toward this cap.",
                ),
                field(
                    "flap_window",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS),
                    "Sliding window over which `max_restarts` is counted.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_error_rate",
            title: "Error-rate trigger",
            fields: vec![
                field(
                    "window",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_WINDOW_MS),
                    "Rolling window over which the error rate is measured. Scoped to the current run, so a fresh process starts from zero.",
                ),
                field(
                    "max_error_rate",
                    "float (0.0–1.0]",
                    bt(DEFAULT_AUTO_RESTART_MAX_ERROR_RATE),
                    "Fraction of requests in the window that must be errors to trigger.",
                ),
                field(
                    "min_requests",
                    "u32",
                    bt(DEFAULT_AUTO_RESTART_MIN_REQUESTS),
                    "Minimum request count in the window before the ratio is trusted — stops a 2-of-2-failed service from restarting.",
                ),
                field(
                    "poll_interval",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS),
                    "How often the watchdog queries the metrics store.",
                ),
                field(
                    "error_statuses",
                    "`\"5xx\"` | `\"4xx+5xx\"`",
                    "`5xx`",
                    "Which HTTP statuses count as errors. `5xx` (server errors only) is the default because a 4xx is usually the client's fault, not the service's. `4xx+5xx` counts any status ≥ 400.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_periodic",
            title: "Periodic trigger",
            fields: vec![
                field(
                    "interval",
                    "duration string",
                    "required",
                    "How long a run may live before a periodic restart is due, measured from when it entered `Running`.",
                ),
                field(
                    "mode",
                    "`\"immediate\"` | `\"on-idle\"` | `\"on-request\"`",
                    "`on-request`",
                    "How the restart is timed once the interval elapses. `immediate` drains and respawns at once (interrupting in-flight traffic gracefully). `on-idle` waits for a quiet window with no in-flight requests, then restarts — zero disruption, but may never fire under continuous load. `on-request` marks the run stale and lets the next request drive the restart, blocking that request on the fresh process; it guarantees the restart happens even under continuous load.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_ttft_stall",
            title: "Stall trigger",
            fields: vec![field(
                "timeout",
                "duration string",
                bt_dur(DEFAULT_AUTO_RESTART_TTFT_STALL_MS),
                "How long a streaming request may stay in-flight with no response frame before the service is restarted. A restart fires only if the *whole service* produced no frame in that window — a request merely queued behind a healthy generation does not trip it. Only streaming requests are watched (non-streaming and embeddings are bounded by `max_request_duration` instead). Does not gate on `min_uptime`; the flap cap still applies.",
            )],
        },
        SectionDoc {
            id: "service_auto_restart_generation_stall",
            title: "Generation-stall trigger",
            fields: vec![
                field(
                    "timeout",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_GENERATION_STALL_MS),
                    "How long the child's `/metrics` progress counters may stay flat, with at least one request in flight, before the service is restarted. Healthy prefill and decode both advance the counters every batch, so the default is unambiguous under load. An idle service (nothing in flight) never trips it.",
                ),
                field(
                    "poll_interval",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS),
                    "How often the child's `/metrics` endpoint is polled.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_spec_collapse",
            title: "Spec-collapse trigger",
            fields: vec![
                field(
                    "window",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS),
                    "Rolling window over which draft acceptance is measured, keyed on request completion time and scoped to the current run. Only requests that actually drafted (`draft_n > 0` in the engine's `timings`) count; a single accepted draft token anywhere in the window vetoes the restart.",
                ),
                field(
                    "min_draft_tokens",
                    "u64",
                    bt(DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS),
                    "Minimum count of drafted tokens in the window before an all-zero acceptance is trusted. Counted in tokens rather than requests so that slow-arriving long generations — which draft thousands of tokens each — reach the floor on their own; one short unlucky generation stays under it.",
                ),
                field(
                    "poll_interval",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS),
                    "How often the watchdog queries the metrics store.",
                ),
            ],
        },
    ]
}
