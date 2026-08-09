//! `POST /api/services/{name}/enable` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/enable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
// `EnableResponse` is a wire enum; the status tag is the API contract and
// the unit variants are self-describing.
#[expect(missing_docs)]
pub enum EnableResponse {
    Enabled,
    NotDisabled,
    AlreadyEnabled,
}
