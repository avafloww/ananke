//! Per-service reverse HTTP proxy.
//!
//! Re-exported from `ananke-proxy` so `crate::api::proxy::…` paths
//! inside the daemon are unchanged by the split.

pub use ananke_proxy::{
    ApiErrorCode, ErasedRecorder, InflightGuard, MetricsBody, NoRecorder, ProxyBody, ProxyError,
    ProxyMetrics, RecorderFactory, WebSocketLifecycle, error_response, serve, serve_with_activity,
};
