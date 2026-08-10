//! GGUF → cache-entry projection for the shared estimate cache.
//!
//! The cache container itself lives in `ananke-events` (generic over the
//! full-estimate payload). This module instantiates it with the daemon's
//! `Estimate` type and owns the projection from a fresh estimator run —
//! GGUF summary + `Estimate` → `ModelInfo` + `EstimateSummary` sumary —
//! so the management detail handler and the supervisor's spawn-time cache
//! warming produce byte-identical entries.

use std::path::PathBuf;

use ananke_api::services::detail::{EstimateSummary, ModelInfo};
use ananke_events::CacheEntry;
use ananke_gguf::keys;

use crate::{
    estimator::Estimate,
    gguf::{GgufSummary, GgufValue},
};

/// Concrete cache entry for the daemon's `Estimate` payload.
pub type EstimateCacheEntry = CacheEntry<Estimate>;

/// Concrete estimate cache for the daemon's `Estimate` payload.
pub type EstimateCacheHandle = ananke_events::EstimateCache<Estimate>;

/// Build a cache entry from a fresh estimator run. Centralises
/// the GGUF → `ModelInfo` and `Estimate` → `EstimateSummary`
/// projections so the management detail handler and the
/// supervisor's spawn-time cache warming produce byte-identical
/// entries from the same inputs.
pub fn build_cache_entry(
    summary: &GgufSummary,
    estimate: &Estimate,
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    config_fingerprint: u64,
) -> EstimateCacheEntry {
    let file_name = model_path
        .file_name()
        .map(|os| os.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.to_string_lossy().to_string());
    let trained_context_key = format!("{}.context_length", summary.architecture);
    let trained_context_length = summary
        .metadata
        .get(trained_context_key.as_str())
        .and_then(|v| v.as_u32());
    let model_name = summary
        .metadata
        .get(keys::NAME)
        .and_then(GgufValue::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let license = summary
        .metadata
        .get(keys::LICENSE)
        .and_then(GgufValue::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let parameter_count = summary
        .metadata
        .get(keys::PARAMETER_COUNT)
        .and_then(GgufValue::as_u64);

    let has_mmproj = mmproj_path.is_some();
    let model_info = ModelInfo {
        architecture: summary.architecture.to_string(),
        model_name,
        license,
        parameter_count,
        total_tensor_bytes: summary.total_tensor_bytes,
        block_count: summary.block_count,
        shard_count: summary.shards.len() as u32,
        trained_context_length,
        file_name,
        has_mmproj,
    };

    let kv_bytes_for_context = estimate
        .kv_per_token
        .saturating_mul(estimate.context as u64);
    let compute_buffer_bytes_per_device = (estimate.buffers.compute_mb as u64) * 1024 * 1024;
    let estimate_summary = EstimateSummary {
        weights_bytes: estimate.weights_bytes,
        kv_per_token: estimate.kv_per_token,
        configured_context: estimate.context,
        kv_bytes_for_context,
        compute_buffer_bytes_per_device,
    };

    CacheEntry {
        model_path,
        mmproj_path,
        config_fingerprint,
        model_info,
        estimate: estimate_summary,
        estimate_full: estimate.clone(),
    }
}
