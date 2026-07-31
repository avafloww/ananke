//! Ananke — GPU/CPU-aware model proxy daemon.

pub mod allocator;
pub mod api;
pub mod config;
pub mod daemon;
pub mod db;
pub mod devices;
pub mod errors;
/// The VRAM estimator, reachable as `crate::estimator::…`.
pub use ananke_estimate as estimator;
/// The GGUF reader, reachable as `crate::gguf::…`.
pub use ananke_gguf as gguf;
pub mod oneshot;
pub mod supervise;
pub mod system;
pub mod templates;
pub mod tracking;
