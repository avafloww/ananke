//! `POST /api/services/{name}/disable` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/disable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
// `DisableResponse` is a wire enum; the status tag is the API contract and
// the unit variants are self-describing.
#[expect(missing_docs)]
pub enum DisableResponse {
    Disabled,
    AlreadyDisabled,
}
