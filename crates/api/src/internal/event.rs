//! WebSocket event envelope published on `/api/events`.
//!
//! `Event` is also used as a broadcast bus message in `supervise/`,
//! `allocator/`, and `config/`, so it lives in the internal module rather
//! than under a single endpoint.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use utoipa::ToSchema;

/// One event delivered over `/api/events`. The `type` tag discriminates the
/// variant; `at_ms` is present on every variant except `overflow`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
// Wire-serialized event variants; the snake_case `type` tag is the API
// contract, and each variant's field names are self-describing in JSON.
#[expect(missing_docs)]
pub enum Event {
    StateChanged {
        #[schema(value_type = String)]
        service: SmolStr,
        from: String,
        to: String,
        at_ms: i64,
    },
    AllocationChanged {
        #[schema(value_type = String)]
        service: SmolStr,
        reservations: BTreeMap<String, u64>,
        at_ms: i64,
    },
    ConfigReloaded {
        at_ms: i64,
        #[schema(value_type = Vec<String>)]
        changed_services: Vec<SmolStr>,
    },
    /// One of a service's rolling estimator corrections moved by more than 5%.
    /// `class` is the memory pool it belongs to — `"vram"` or `"host"` — since
    /// the two are learned independently from separate observations.
    EstimatorDrift {
        #[schema(value_type = String)]
        service: SmolStr,
        class: String,
        rolling_mean: f32,
        at_ms: i64,
    },
    /// A `Running` service was drained and respawned by its auto-restart
    /// policy. `trigger` is `"error_rate"`, `"periodic"`, `"ttft_stall"`,
    /// `"generation_stall"`, or `"spec_collapse"`; `detail` is a
    /// human-readable reason (e.g. the observed error rate and window).
    AutoRestarted {
        #[schema(value_type = String)]
        service: SmolStr,
        trigger: String,
        detail: String,
        at_ms: i64,
    },
    Overflow {
        dropped: u64,
    },
}
