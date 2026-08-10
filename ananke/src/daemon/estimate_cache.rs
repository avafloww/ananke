//! Per-service cache of GGUF metadata + estimator output.
//!
//! The cache container and the GGUF → entry projection live in
//! [`crate::supervise::estimate_cache`]; this module re-exports them so
//! `crate::daemon::estimate_cache::…` paths keep resolving.

pub use ananke_events::{CacheEntry, EstimateCache};

pub use crate::supervise::estimate_cache::{
    EstimateCacheEntry, EstimateCacheHandle, build_cache_entry,
};
