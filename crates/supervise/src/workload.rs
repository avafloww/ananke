//! Common managed-workload interface shared by native process and container
//! instances.
//!
//! The supervisor's health/running/drain state machine consumes a
//! [`ManagedWorkload`] so it does not need two parallel copies of every
//! select arm. Process and container launch paths remain separate right up
//! until each has produced a running handle; they converge here.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::os::unix::process::ExitStatusExt;

use ananke_system::{DynAsyncRead, ManagedChild, container::ManagedContainer};
use tracing::warn;

/// Exit outcome unified across process and container workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadExit {
    /// Exited with a numeric status code (process `code()`, or a container's
    /// `wait` exit code).
    Code(i32),
    /// Terminated by a signal (native processes only — a container `wait`
    /// returns a numeric code, not a signal).
    Signal(i32),
}

/// A running managed workload: either a native child or a running container.
///
/// Both variants expose the operations the health/running/drain state
/// machine needs — wait, graceful terminate, force kill, and host PID —
/// without leaking the differing log-reader shapes (split stdout/stderr for
/// processes, a merged combined stream for containers).
pub enum ManagedWorkload {
    /// A native child process.
    Process(Box<dyn ManagedChild>),
    /// A running container.
    Container(Box<dyn ManagedContainer>),
}

impl ManagedWorkload {
    /// The host PID when known. `None` for a container without an inspected
    /// host PID, or a process that has already been reaped.
    pub fn host_pid(&self) -> Option<u32> {
        match self {
            ManagedWorkload::Process(child) => child.id(),
            ManagedWorkload::Container(c) => c.host_pid(),
        }
    }

    /// Take the stdout reader (native processes only). Returns `None` for a
    /// container, which has a single merged `combined` stream instead.
    pub fn take_stdout(&mut self) -> Option<DynAsyncRead> {
        match self {
            ManagedWorkload::Process(child) => child.take_stdout(),
            ManagedWorkload::Container(_) => None,
        }
    }

    /// Take the stderr reader (native processes only). Returns `None` for a
    /// container, which has a single merged `combined` stream instead.
    pub fn take_stderr(&mut self) -> Option<DynAsyncRead> {
        match self {
            ManagedWorkload::Process(child) => child.take_stderr(),
            ManagedWorkload::Container(_) => None,
        }
    }

    /// Take the `combined` readers (containers only). Empty for a native
    /// process, which exposes split stdout/stderr instead.
    ///
    /// More than one because the log follower has its own stdout and
    /// stderr; every line from either is tagged `combined`.
    pub fn take_combined(&mut self) -> Vec<DynAsyncRead> {
        match self {
            ManagedWorkload::Process(_) => Vec::new(),
            ManagedWorkload::Container(c) => c.logs(),
        }
    }

    /// Await exit, returning a unified outcome. Safe to re-await across
    /// `tokio::select!` cancellations for both variants.
    pub async fn wait(&mut self) -> std::io::Result<WorkloadExit> {
        match self {
            ManagedWorkload::Process(child) => {
                let status = child.wait().await?;
                Ok(status_to_exit(status))
            }
            ManagedWorkload::Container(c) => {
                // The container runtime's `wait` returns the authoritative
                // numeric exit status. Errors surface as I/O errors here so
                // the outer select arms treat them as a failure uniformly.
                match c.wait().await {
                    Ok(code) => Ok(WorkloadExit::Code(code)),
                    Err(e) => {
                        warn!(error = %e, "container wait failed");
                        Err(std::io::Error::other(e.to_string()))
                    }
                }
            }
        }
    }

    /// Request graceful termination (SIGTERM). Idempotent.
    pub async fn terminate(&mut self) -> std::io::Result<()> {
        match self {
            ManagedWorkload::Process(child) => child.sigterm().await,
            ManagedWorkload::Container(c) => c
                .terminate()
                .await
                .map_err(|e| std::io::Error::other(e.to_string())),
        }
    }

    /// Force kill (SIGKILL). Idempotent.
    pub async fn kill(&mut self) -> std::io::Result<()> {
        match self {
            ManagedWorkload::Process(child) => child.sigkill().await,
            ManagedWorkload::Container(c) => c
                .kill()
                .await
                .map_err(|e| std::io::Error::other(e.to_string())),
        }
    }

    /// Idempotent removal/cleanup after exit. Native processes are cleaned
    /// up by their drop handles (`kill_on_drop`); this is a no-op. Containers
    /// are explicitly removed as the final step of drain.
    pub async fn cleanup(&mut self) -> std::io::Result<()> {
        match self {
            ManagedWorkload::Process(_) => Ok(()),
            ManagedWorkload::Container(c) => c
                .remove()
                .await
                .map_err(|e| std::io::Error::other(e.to_string())),
        }
    }
}

fn status_to_exit(status: std::process::ExitStatus) -> WorkloadExit {
    if let Some(sig) = status.signal() {
        WorkloadExit::Signal(sig)
    } else {
        WorkloadExit::Code(status.code().unwrap_or(0))
    }
}
