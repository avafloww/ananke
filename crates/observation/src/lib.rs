//! Per-service observed memory and the shared device snapshot.
//!
//! The snapshotter (`ananke-devices`) samples NVML + `/proc` each tick,
//! writes the current [`SharedSnapshot`] atomically, and feeds per-service
//! observations into [`ObservationTable`]. The allocator and tracking
//! read both back. The types live here (rather than in `devices` or
//! `tracking`) so both can depend on this crate without reaching into
//! each other.

mod observation;

use std::sync::Arc;

pub use ananke_placement::devices::DeviceSnapshot;
pub use observation::{ObservationTable, read_rss};
use parking_lot::RwLock;

/// Atomically-replaced snapshot of the current device state, shared with
/// the allocator, the management API, and the device-sample db writer.
/// Readers never block the sampler; the sampler replaces the whole
/// snapshot in one write.
pub type SharedSnapshot = Arc<RwLock<DeviceSnapshot>>;

/// Build a fresh shared snapshot initialized to default (all-empty) state.
pub fn new_shared() -> SharedSnapshot {
    Arc::new(RwLock::new(DeviceSnapshot::default()))
}
