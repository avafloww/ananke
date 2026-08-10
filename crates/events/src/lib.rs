//! Daemon-wide broadcast channel and the shared estimate cache.
//!
//! `EventBus` is the infallible-publisher broadcast bus every subsystem
//! (config reloads, tracking, the allocator, supervision, the HTTP surface)
//! publishes lifecycle events to. The estimate cache is shared between the
//! supervisor's spawn-time estimator run and the management `ServiceDetail`
//! handler so successive detail polls don't re-parse the GGUF.

pub mod estimate_cache;
pub mod events;

pub use estimate_cache::{CacheEntry, EstimateCache};
pub use events::EventBus;
