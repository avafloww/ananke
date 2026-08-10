//! `POST /api/services/{name}/start` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::shared::errors::ApiErrorBody;

/// `POST /api/services/{name}/start` response body.
///
/// The `status` tag tells the caller the outcome of the start request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StartResponse {
    /// The service was already running when the start request arrived.
    AlreadyRunning,
    /// The service entered `Starting` and this spawning run's id.
    Started {
        /// The id of the run that was spawned.
        run_id: i64,
    },
    /// The start was queued instead of run: the service has a
    /// `start_queue_depth` cap and the queue was full.
    QueueFull,
    /// The supervisor declined to start: it didn't fit, the service
    /// is disabled, etc. The embedded [`ApiErrorBody`] carries the
    /// same typed slug, message, and kind that a 503 `ApiError`
    /// response would (so clients can switch on `error.code` instead
    /// of pattern-matching a freeform `reason` string). The 202 status
    /// is preserved because this is a "controlled outcome" of the
    /// start request, not a server-side fault.
    Unavailable {
        /// The typed error the supervisor produced, matching what a 503
        /// `ApiError` response would carry.
        error: ApiErrorBody,
    },
}
