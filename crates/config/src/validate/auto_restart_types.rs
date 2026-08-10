//! The resolved self-healing restart policy and its per-trigger settings.

use crate::docs::{
    DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS, DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
    DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS, DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
    DEFAULT_AUTO_RESTART_MAX_RESTARTS, DEFAULT_AUTO_RESTART_MIN_REQUESTS,
    DEFAULT_AUTO_RESTART_MIN_UPTIME_MS, DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS, DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
    DEFAULT_AUTO_RESTART_TTFT_STALL_MS, DEFAULT_AUTO_RESTART_WINDOW_MS,
};

/// Default periodic-restart interval — only consulted when a service enables
/// periodic restarts without spelling out an interval, which is rejected;
/// present for completeness alongside the other knobs.
pub const DEFAULT_AUTO_RESTART_PERIODIC_MODE: PeriodicMode = PeriodicMode::OnRequest;

/// Resolved self-healing restart policy for a service. Both triggers are
/// independent `Option`s: the error-rate watchdog is `Some` by default, the
/// periodic timer `None` by default. The guardrail fields apply to whichever
/// trigger fires.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoRestartSettings {
    /// Error-rate watchdog, or `None` if disabled (`error_rate = false`).
    pub error_rate: Option<ErrorRateTrigger>,
    /// Periodic restart, or `None` if disabled (the default).
    pub periodic: Option<PeriodicTrigger>,
    /// Time-to-first-token stall watchdog, or `None` if disabled
    /// (`ttft_stall = false`). `Some` by default.
    pub ttft_stall: Option<TtftStallTrigger>,
    /// Generation-stall watchdog (child `/metrics` progress polling), or
    /// `None` if disabled. `Some` by default on the `llama-cpp` template;
    /// `None` by default on the `command` template, where ananke cannot
    /// inject `--metrics` into an argv it does not build.
    pub generation_stall: Option<GenerationStallTrigger>,
    /// Speculative-decoding collapse watchdog, or `None` if disabled.
    /// `Some` by default only for llama-cpp services with `spec_type` set;
    /// no other service produces draft counts, and an explicit per-service
    /// enable elsewhere is rejected at validation.
    pub spec_collapse: Option<SpecCollapseTrigger>,
    /// Anti-flap cooldown: minimum uptime of a fresh run before another
    /// auto-restart may fire.
    pub min_uptime_ms: u64,
    /// Auto-restarts tolerated within [`Self::flap_window_ms`] before the
    /// service is disabled with `AutoRestartLoop`.
    pub max_restarts: u32,
    /// Sliding window over which [`Self::max_restarts`] is counted.
    pub flap_window_ms: u64,
}

impl AutoRestartSettings {
    /// Whether any trigger is active — a cheap gate the supervisor uses to
    /// skip watchdog setup entirely for services that opted fully out.
    pub fn any_enabled(&self) -> bool {
        self.error_rate.is_some()
            || self.periodic.is_some()
            || self.ttft_stall.is_some()
            || self.generation_stall.is_some()
            || self.spec_collapse.is_some()
    }

    /// All triggers off, guardrails at their defaults. Used by test
    /// fixtures so a supervisor under test gets no watchdog unless the test
    /// opts in explicitly.
    pub fn disabled() -> Self {
        Self {
            error_rate: None,
            periodic: None,
            ttft_stall: None,
            generation_stall: None,
            spec_collapse: None,
            ..Self::default()
        }
    }
}

impl Default for AutoRestartSettings {
    /// The `llama-cpp`-template defaults. `validate_auto_restart` overrides
    /// `generation_stall` to `None` for `command`-template services, where it
    /// is off unless explicitly enabled.
    fn default() -> Self {
        Self {
            error_rate: Some(ErrorRateTrigger::default()),
            periodic: None,
            ttft_stall: Some(TtftStallTrigger::default()),
            generation_stall: Some(GenerationStallTrigger::default()),
            spec_collapse: Some(SpecCollapseTrigger::default()),
            min_uptime_ms: DEFAULT_AUTO_RESTART_MIN_UPTIME_MS,
            max_restarts: DEFAULT_AUTO_RESTART_MAX_RESTARTS,
            flap_window_ms: DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS,
        }
    }
}

/// Resolved time-to-first-token stall watchdog settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TtftStallTrigger {
    /// How long a request may stay in-flight with no response token before
    /// the service is restarted.
    pub timeout_ms: u64,
}

impl Default for TtftStallTrigger {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_AUTO_RESTART_TTFT_STALL_MS,
        }
    }
}

/// Resolved generation-stall watchdog settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationStallTrigger {
    /// How long the child's `/metrics` progress counters may stay flat, with
    /// at least one request in flight, before the service is restarted.
    pub timeout_ms: u64,
    /// How often the child's `/metrics` endpoint is polled.
    pub poll_interval_ms: u64,
}

impl Default for GenerationStallTrigger {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
            poll_interval_ms: DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS,
        }
    }
}

/// Resolved speculative-decoding collapse watchdog thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecCollapseTrigger {
    /// Rolling window over which draft acceptance is measured.
    pub window_ms: u64,
    /// Minimum count of drafted tokens in the window before an all-zero
    /// acceptance is trusted. Tokens rather than requests: long generations
    /// arrive slowly but draft thousands of tokens each, so a request floor
    /// would starve on exactly the traffic a garbage wedge produces.
    pub min_draft_tokens: u64,
    /// How often the watchdog queries the metrics store.
    pub poll_interval_ms: u64,
}

impl Default for SpecCollapseTrigger {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
            min_draft_tokens: DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
            poll_interval_ms: DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS,
        }
    }
}

/// Resolved error-rate watchdog thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorRateTrigger {
    /// Rolling window over which the error rate is measured.
    pub window_ms: u64,
    /// Fraction of requests in the window that must be errors to trigger.
    pub max_error_rate: f64,
    /// Minimum request count in the window before the ratio is trusted.
    pub min_requests: u32,
    /// How often the watchdog queries the metrics store.
    pub poll_interval_ms: u64,
    /// Which statuses count as errors.
    pub statuses: ErrorStatusClass,
}

impl Default for ErrorRateTrigger {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_AUTO_RESTART_WINDOW_MS,
            max_error_rate: DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
            min_requests: DEFAULT_AUTO_RESTART_MIN_REQUESTS,
            poll_interval_ms: DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
            statuses: ErrorStatusClass::ServerOnly,
        }
    }
}

/// Which HTTP statuses the error-rate watchdog counts as errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStatusClass {
    /// Server errors only (500–599). The default — a wedged upstream 5xxs,
    /// whereas 4xx is usually the client's fault and should not self-restart.
    ServerOnly,
    /// Any status ≥ 400 (client and server errors alike).
    ClientAndServer,
}

impl ErrorStatusClass {
    /// Whether a recorded status code counts as an error under this class.
    pub fn is_error(self, status: u16) -> bool {
        match self {
            ErrorStatusClass::ServerOnly => (500..600).contains(&status),
            ErrorStatusClass::ClientAndServer => status >= 400,
        }
    }

    /// The inclusive lower bound on status codes that count as errors, for the
    /// SQL `status_code >= ?` predicate. There are no ≥ 600 statuses in
    /// practice, so `ServerOnly`'s upper bound needs no separate clause.
    pub fn min_status_code(self) -> u16 {
        match self {
            ErrorStatusClass::ServerOnly => 500,
            ErrorStatusClass::ClientAndServer => 400,
        }
    }
}

/// Resolved periodic-restart settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicTrigger {
    /// How long a run may live before a periodic restart is due.
    pub interval_ms: u64,
    /// How the restart is timed once the interval elapses.
    pub mode: PeriodicMode,
}

/// How a periodic restart is timed once the interval elapses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicMode {
    /// Drain and respawn the moment the interval elapses, interrupting any
    /// in-flight traffic (gracefully, via the normal drain pipeline).
    Immediate,
    /// Wait for a quiet window (no in-flight requests) after the interval
    /// elapses, then restart. Zero disruption, but may never fire under
    /// continuous load.
    OnIdle,
    /// Mark the run stale when the interval elapses; the next request
    /// triggers the restart and blocks on the fresh process. Guarantees the
    /// restart happens even under continuous load.
    OnRequest,
}
