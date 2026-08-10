//! Linux-only: device snapshotter.
//!
//! The sampling loop lives in `ananke-devices`; the shared snapshot and
//! the observation table live in `ananke-observation`. This module
//! re-exports them so `crate::devices::snapshotter::…` paths inside the
//! daemon keep resolving.

pub use ananke_devices::snapshotter::spawn;
pub use ananke_observation::{SharedSnapshot, new_shared};
