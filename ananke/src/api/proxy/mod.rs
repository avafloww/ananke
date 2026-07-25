//! Per-service reverse HTTP proxy.

mod request;
mod serve;
mod upgrade;

use bytes::Bytes;
use hyper::Response;
pub use request::error_response;
pub(crate) use request::handle;
pub use serve::{ProxyMetrics, WebSocketLifecycle, serve, serve_with_activity};
pub(crate) use upgrade::handle_upgrade;

/// Boxed body type used for both upstream requests and downstream responses.
pub type ProxyBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

/// Short-circuit error response returned by a `before_request` hook. Alias of
/// `Response<ProxyBody>`; kept as a separate name so callers that build an
/// error reply need not reach into `ProxyBody` themselves.
pub type ProxyError = Response<ProxyBody>;
