//! API error projections.
//!
//! `ApiErrorCode` itself lives in `ananke-proxy` (shared with the hyper
//! data plane), including its axum `IntoResponse` impl. This module
//! re-exports it so `crate::api::errors::…` paths keep resolving.

pub use ananke_api::shared::errors::ApiError;
pub use ananke_proxy::ApiErrorCode;
