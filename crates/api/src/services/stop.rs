//! `POST /api/services/{name}/stop` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/stop` response body.
///
/// The `status` tag tells the caller the outcome of the stop request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StopResponse {
    /// The service was not running, so the stop was a no-op.
    NotRunning,
    /// The service was drained: its in-flight work finished, then its
    /// children were stopped and the reservation released.
    Drained,
}
