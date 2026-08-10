//! `POST /api/services/{name}/enable` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/enable` response body.
///
/// The `status` tag tells the caller the outcome of the enable request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EnableResponse {
    /// The service was enabled by this request.
    Enabled,
    /// The service was not disabled, so enabling it was a no-op.
    NotDisabled,
    /// The service was already enabled before this request.
    AlreadyEnabled,
}
