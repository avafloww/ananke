//! `POST /api/services/{name}/disable` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/disable` response body.
///
/// The `status` tag tells the caller the outcome of the disable request. All
/// variants are success states; a request that fails validation or targets a
/// missing service comes back as an error response instead.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DisableResponse {
    /// The service was disabled by this request.
    Disabled,
    /// The service was already disabled before this request.
    AlreadyDisabled,
}
