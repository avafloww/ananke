//! Response-body metrics wrapper.
//!
//! Re-exported from `ananke-proxy`; the generic `MetricsBody<B, R>`
//! wraps any body and drives the recorder. The daemon instantiates it
//! with its OpenAI `MetricsRecorder` and `Database`.

pub use ananke_proxy::MetricsBody;
