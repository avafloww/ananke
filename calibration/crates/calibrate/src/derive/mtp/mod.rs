//! Multi-token prediction: what a second context costs, on the host and on the device.

pub mod device;
pub mod host;
pub mod pairs;

pub use device::{DraftComputeFit, draft_compute_slope, mtp_draft_compute, mtp_unaccounted};
pub use host::{HostFit, mtp_host_embedded, mtp_host_fit, mtp_host_separate, mtp_host_slope};
pub use pairs::{MtpPair, SAME_SITTING_SECONDS, mtp_pairs, mtp_slot_scaling};
