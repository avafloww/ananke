//! Per-service runtime tracking: activity timestamps, in-flight counters,
//! live memory observations, and rolling safety factors.

pub mod activity;
pub mod inflight;
pub mod progress;
pub mod rolling;
pub mod sampler;

/// Wall-clock helpers, re-exported for callers that reach for them via
/// `ananke_time::now_unix_ms()`.
pub use ananke_time::{now_unix_ms, now_unix_ms_u64};
