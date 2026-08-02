//! Whether a service's estimated placement fits under current conditions.
//!
//! Used by the supervisor's placement engine (`supervise/preview.rs`) as
//! well as the `/api/services` and `/api/services/:name` endpoints, so it
//! lives in the internal module rather than under a single endpoint.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Whether a service's estimated placement fits under current conditions.
///
/// Serialised as a `kind`-tagged union so a failing verdict can carry the
/// per-device numbers that explain it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FitVerdict {
    /// Starts now in currently-free memory — no eviction needed.
    Fits,
    /// Fits within the hardware, but currently-free memory is insufficient, so
    /// the daemon would reclaim or evict lower-priority peers to make room.
    NeedsEviction,
    /// No placement is possible under current conditions, even with every other
    /// service gone. `shortfalls` names the devices that came up short and by
    /// how much. Empty when there is no device to point at: a configuration
    /// error that no amount of freed memory would fix, or a host with no
    /// eligible device for this service at all.
    DoesNotFit {
        /// The devices that came up short, and by how much.
        shortfalls: Vec<DeviceShortfall>,
    },
}

/// One device's contribution to a placement failure.
///
/// The binding constraint is often host RAM rather than GPU VRAM — a MoE with
/// expert offload routinely spills the bulk of its weights to the CPU — so the
/// device is named rather than the failure being sorted into a category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct DeviceShortfall {
    /// Device id string, e.g. `"gpu:0"` or `"cpu"`. Matches `DeviceSummary::id`
    /// from `GET /api/devices`, so a shortfall can be cross-referenced against
    /// the live device view.
    pub device: String,
    /// Bytes the placement needed on this device.
    pub requested_bytes: u64,
    /// Bytes the device could offer.
    pub available_bytes: u64,
}
