//! Ananke — GPU/CPU-aware model proxy daemon.

pub mod allocator;
pub mod api;
pub mod config;
pub mod daemon;
pub mod db;
pub mod devices;
pub mod errors;
/// The VRAM estimator, re-exported so `crate::estimator::…` paths are unchanged.
pub use ananke_estimate as estimator;
/// The GGUF reader, re-exported so `crate::gguf::…` paths are unchanged.
pub use ananke_gguf as gguf;
pub mod oneshot;
pub mod supervise;
pub mod system;
pub mod templates;
pub mod tracking;
