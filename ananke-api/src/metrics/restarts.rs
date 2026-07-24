//! `GET /api/restarts` — auto-restart firings within a time window.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/restarts` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct RestartsResponse {
    /// Firings within the requested window, oldest first.
    pub restarts: Vec<ServiceRestartEntry>,
}

/// One auto-restart watchdog firing, with the service it belongs to.
/// The service-detail endpoint serves the same records scoped to one
/// service; this shape adds the `service` label for cross-service views.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ServiceRestartEntry {
    /// Service name.
    pub service: String,
    /// Wall-clock timestamp of the firing (ms since epoch).
    pub at_ms: i64,
    /// Which watchdog fired: `"error_rate"`, `"ttft_stall"`,
    /// `"generation_stall"`, `"spec_collapse"`, or `"periodic"`.
    pub trigger: String,
    /// Human-readable reason carried by the event.
    pub detail: String,
    /// The run that was drained by the firing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<i64>,
}
