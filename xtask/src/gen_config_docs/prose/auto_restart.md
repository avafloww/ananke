Self-healing for a `Running` service that is alive but degraded — the process has not exited, so the crash-detection path never fires, yet every request is failing, hanging, or returning garbage. Five independent triggers all feed the existing drain → respawn cycle. The error-rate, TTFT-stall, and periodic triggers observe HTTP traffic at the proxy and apply to any service; the generation-stall and spec-collapse triggers read llama.cpp-specific surfaces and are gated to services that provide them. [docs/auto-restart.md](auto-restart.md) explains the reasoning behind each trigger, the exact engine coverage, and the incidents that motivated them.

```toml
[service.auto_restart]
# Error-rate watchdog (on by default; write `error_rate = false` to opt out):
error_rate = { window = "2m", max_error_rate = 0.5, min_requests = 20, poll_interval = "30s", error_statuses = "5xx" }
# Periodic restart (off by default; a table with an interval enables it):
periodic = { interval = "6h", mode = "on-request" }
# Speculative-decoding collapse watchdog (on by default when spec_type is set):
spec_collapse = { window = "2m", min_requests = 10, poll_interval = "30s" }
# Anti-flap guardrails, shared by all triggers:
min_uptime = "5m"
max_restarts = 3
flap_window = "30m"
```

The block is resolved as a **whole unit**: a service that sets any `auto_restart` field replaces `[defaults.auto_restart]` entirely rather than merging field-by-field. The same `[defaults.auto_restart]` block is accepted for fleet-wide defaults.
