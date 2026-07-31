//! Waiting for a supervisor to reach `Running`, and the structured failure
//! types the reservation/eviction planner and the HTTP layers use to talk
//! about why a service couldn't be placed or started.

use std::time::Duration;

use crate::{
    allocator::placement::PackError,
    supervise::handle::{
        EnsureResponse, EnsureSource, StartFailureKind, StartOutcome, SupervisorHandle,
    },
};

/// Outcome of waiting for a supervisor to reach Running.
///
/// Both the per-service transparent proxy and the OpenAI-compat router need
/// the same "ensure + wait-on-bus + map to error kind" dance before
/// forwarding. [`await_ensure`] packages it once; each caller renders its
/// own error response from the [`EnsureFailure`] kinds.
pub enum EnsureOutcome {
    /// Service is Running (or was already).
    ///
    /// `was_already_running` is `true` when the supervisor was already in the
    /// Running state at the time of the call; `false` when the call triggered
    /// an Idle → Starting transition and the caller waited for the child to
    /// become healthy.
    Ready { was_already_running: bool },
    /// Service cannot serve the request.
    Failed(EnsureFailure),
}

/// Why `compute_reservation_map` couldn't produce a reservation for the
/// next spawn. Each variant carries the structured inner error it wraps
/// so callers can inspect the specific cause without parsing a message.
#[derive(Debug, Clone)]
pub(crate) enum ReservationFailure {
    /// Service config is missing something required to even try estimation
    /// (e.g. no model path on a llama-cpp service). Not recoverable by
    /// eviction — the service needs a config fix first.
    Misconfigured(MisconfiguredKind),
    /// The estimator refused or failed on this GGUF. Carries the concrete
    /// [`estimator::EstimatorError`] so callers can tell an unknown-arch
    /// from a GGUF-read failure.
    EstimatorError(crate::estimator::EstimatorError),
    /// The packer couldn't lay the model down given current reservations.
    /// Carries the structured [`placement::PackError`] — the supervisor
    /// branches on this specifically to retry with eviction.
    PackFailed(crate::allocator::placement::PackError),
}

/// Concrete ways a service's config can prevent the estimator from even
/// running. Expands over time as new check surfaces are added.
#[derive(Debug, Clone)]
pub(crate) enum MisconfiguredKind {
    /// Llama-cpp service without a `model` path. Should have been caught
    /// at config validation but the supervisor double-checks defensively.
    NoModelPath,
    /// `tensor_split_weights` count doesn't match the spanned GPU count at
    /// pack time. Carries the structured [`PackError`] so the operator sees
    /// the expected/got counts. Routed here (not `PackFailed`) because eviction
    /// cannot fix a config error.
    InvalidTensorSplitWeights(crate::allocator::placement::PackError),
}

impl std::fmt::Display for MisconfiguredKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModelPath => f.write_str("no model path configured"),
            Self::InvalidTensorSplitWeights(e) => write!(f, "{e}"),
        }
    }
}

impl ReservationFailure {
    /// Flatten to the string the proxy / OpenAI layer shows operators. The
    /// shape is stable so log scrapes keep working; inspectors should match
    /// on the enum instead.
    pub(crate) fn message(self) -> String {
        match self {
            Self::Misconfigured(k) => k.to_string(),
            Self::EstimatorError(e) => format!("estimator: {e}"),
            Self::PackFailed(p) => format!("placement: {p}"),
        }
    }
}

/// Convert a [`PackError`] into the appropriate [`ReservationFailure`].
///
/// A [`PackError::InvalidTensorSplitWeights`] is a configuration error —
/// eviction cannot fix a weight-count mismatch — so it is routed to
/// [`ReservationFailure::Misconfigured`], which skips the eviction-retry loop
/// and disables the service. All other pack errors are capacity problems
/// routed to [`ReservationFailure::PackFailed`], which the supervisor retries
/// with eviction.
pub(crate) fn pack_err_to_reservation_failure(
    e: crate::allocator::placement::PackError,
) -> ReservationFailure {
    match e {
        e @ PackError::InvalidTensorSplitWeights { .. } => {
            ReservationFailure::Misconfigured(MisconfiguredKind::InvalidTensorSplitWeights(e))
        }
        other => ReservationFailure::PackFailed(other),
    }
}

/// Outcome of the retry-with-eviction pack attempt. The supervisor branches
/// on the `WaitForBusy` variant to queue the request (poll until a busy peer
/// idles out) rather than 503'ing immediately.
#[derive(Debug, Clone)]
pub(crate) enum RetryPackFailure {
    /// Every current candidate is busy at our priority or below; none is
    /// evictable right now, but each will be once its in-flight request
    /// finishes. `busy_peers` is the set the caller should watch: when any
    /// of their inflight counters drops to zero, the queue is woken up to
    /// retry the pack.
    WaitForBusy { busy_peers: Vec<smol_str::SmolStr> },
    /// Pack is infeasible no matter what: either there is no peer to evict,
    /// all peers are higher priority, or the optimistic pack still fails
    /// after treating every evictable peer as gone. Reject outright.
    NotPossible(String),
}

/// Semantic bucket of an [`EnsureOutcome::Failed`]. Callers map these onto
/// their own error-response surface (OpenAI error JSON, proxy error body).
#[derive(Debug, Clone)]
pub enum EnsureFailure {
    /// The start fit-check rejected or the child got OOM-killed.
    InsufficientCapacity(String),
    /// The service is Disabled (config or health) or otherwise unavailable.
    ServiceDisabled(String),
    /// The supervisor's start queue is saturated.
    StartQueueFull,
    /// The start itself failed (launch error, health timeout, bus closed,
    /// overall timeout, or the supervisor task is gone).
    StartFailed(String),
    /// The request parked in the start queue for [`QUEUE_BLOCKED_GRACE`]
    /// without a single watched peer idling. Carries the structured
    /// list of blocking peer names — wire-layer renderers turn this
    /// into a 503 + `service_blocked` body that names each blocker.
    /// Distinct from `InsufficientCapacity` because the *fit* is fine — the
    /// planner just can't displace the current occupant on its own.
    Blocked { busy_peers: Vec<smol_str::SmolStr> },
}

/// Ensure the service is Running, waiting up to `max_request_duration` for
/// an in-flight start to finish. Used by every HTTP path that forwards to
/// a supervised child.
pub async fn await_ensure(
    handle: &SupervisorHandle,
    max_request_duration: Duration,
) -> EnsureOutcome {
    let rx = match handle.ensure(EnsureSource::UserRequest).await {
        Some(EnsureResponse::AlreadyRunning) => {
            return EnsureOutcome::Ready {
                was_already_running: true,
            };
        }
        Some(EnsureResponse::Waiting { rx }) => rx,
        Some(EnsureResponse::QueueFull) => {
            return EnsureOutcome::Failed(EnsureFailure::StartQueueFull);
        }
        Some(EnsureResponse::Unavailable(failure)) => {
            return EnsureOutcome::Failed(failure);
        }
        None => {
            return EnsureOutcome::Failed(EnsureFailure::StartFailed(
                "supervisor unreachable".into(),
            ));
        }
    };
    await_start_bus(rx, max_request_duration).await
}

async fn await_start_bus(
    mut rx: tokio::sync::broadcast::Receiver<StartOutcome>,
    max_request_duration: Duration,
) -> EnsureOutcome {
    match tokio::time::timeout(max_request_duration, rx.recv()).await {
        Ok(Ok(StartOutcome::Ok)) => EnsureOutcome::Ready {
            was_already_running: false,
        },
        Ok(Ok(StartOutcome::Err(f))) => EnsureOutcome::Failed(match f.kind {
            StartFailureKind::NoFit | StartFailureKind::Oom => {
                EnsureFailure::InsufficientCapacity(f.message)
            }
            StartFailureKind::Disabled => EnsureFailure::ServiceDisabled(f.message),
            StartFailureKind::HealthTimeout => {
                EnsureFailure::StartFailed("health check timed out".into())
            }
            StartFailureKind::LaunchFailed => EnsureFailure::StartFailed(f.message),
            StartFailureKind::Blocked { busy_peers } => EnsureFailure::Blocked { busy_peers },
        }),
        Ok(Err(e)) => EnsureOutcome::Failed(EnsureFailure::StartFailed(format!(
            "start broadcast closed: {e}"
        ))),
        Err(_) => EnsureOutcome::Failed(EnsureFailure::StartFailed("start timed out".into())),
    }
}

#[cfg(test)]
mod pack_err_tests {
    use super::*;
    use crate::allocator::placement::PackError;

    #[test]
    fn pack_err_routes_invalid_tensor_split_to_misconfigured() {
        let e = PackError::InvalidTensorSplitWeights {
            expected: 2,
            got: 3,
        };
        match pack_err_to_reservation_failure(e) {
            ReservationFailure::Misconfigured(MisconfiguredKind::InvalidTensorSplitWeights(
                PackError::InvalidTensorSplitWeights {
                    expected: 2,
                    got: 3,
                },
            )) => {}
            other => panic!("expected Misconfigured(InvalidTensorSplitWeights), got {other:?}"),
        }
    }

    #[test]
    fn pack_err_routes_capacity_errors_to_pack_failed() {
        let e = PackError::WeightsDoNotFit {
            shortfalls: Vec::new(),
        };
        assert!(matches!(
            pack_err_to_reservation_failure(e),
            ReservationFailure::PackFailed(PackError::WeightsDoNotFit { .. })
        ));
    }
}
