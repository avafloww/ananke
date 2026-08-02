//! `/api/metrics`, `/api/restarts`, and `/metrics` endpoint types.

pub mod get;
pub mod prometheus;
pub mod restarts;

pub use get::{MetricBucketResponse, MetricsResponse};
pub use restarts::{RestartsResponse, ServiceRestartEntry};
