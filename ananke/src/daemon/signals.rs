//! Linux-only: signal handling via `tokio::signal::unix`.
//! SIGTERM/SIGINT → graceful drain, SIGQUIT → emergency.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::time::Duration;

use tokio::signal::unix::{SignalKind, signal};
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownKind {
    Graceful,
    Emergency,
}

/// Blocks until a shutdown signal arrives.
pub async fn await_shutdown() -> ShutdownKind {
    // Invariant: registering the three standard shutdown signals is a fixed
    // OS capability; failure would leave the daemon unable to drain.
    let mut term = signal(SignalKind::terminate())
        .unwrap_or_else(|_| unreachable!("SIGTERM handler registration"));
    let mut int = signal(SignalKind::interrupt())
        .unwrap_or_else(|_| unreachable!("SIGINT handler registration"));
    let mut quit =
        signal(SignalKind::quit()).unwrap_or_else(|_| unreachable!("SIGQUIT handler registration"));
    tokio::select! {
        _ = term.recv() => { info!("SIGTERM received"); ShutdownKind::Graceful }
        _ = int.recv() => { info!("SIGINT received"); ShutdownKind::Graceful }
        _ = quit.recv() => { info!("SIGQUIT received"); ShutdownKind::Emergency }
    }
}

/// Grace given to in-flight work on SIGTERM/SIGINT before child kill.
const GRACEFUL_SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// Grace given on SIGQUIT before we escalate to SIGKILL for every child.
const EMERGENCY_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

pub fn grace_for(kind: ShutdownKind) -> Duration {
    match kind {
        ShutdownKind::Graceful => GRACEFUL_SHUTDOWN_GRACE,
        ShutdownKind::Emergency => EMERGENCY_SHUTDOWN_GRACE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grace_is_shorter_for_emergency() {
        assert!(grace_for(ShutdownKind::Emergency) < grace_for(ShutdownKind::Graceful));
    }
}
