//! Per-dynamic-service balloon resolver.
//!
//! Each dynamic service gets one resolver task. The task samples the service's
//! current usage on the device its reservation sits on every `SAMPLE_INTERVAL`,
//! maintains a rolling window, and takes action when either the ceiling is
//! breached for too long or growth pressure is detected while contention with
//! a borrower exists.

mod ceiling;
mod contention;
mod growth;
mod pledge;
mod resolver;
#[cfg(test)]
mod test_support;
mod window;

pub use growth::detect_growth;
pub use pledge::{BalloonConfig, pledge_from_window, should_update_pledge};
pub use resolver::{ResolverDeps, spawn_resolver};

/// Number of recent samples the growth/pledge window retains.
pub(crate) const WINDOW_SIZE: usize = 6;
