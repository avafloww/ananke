//! Single-request proxying: map a request to its upstream, forward it, and
//! (for the token-generating endpoints) wrap the response body with a
//! metrics recorder. Upgrade requests are delegated to the sibling
//! `upgrade` module.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::{convert::Infallible, error::Error, net::SocketAddr, time::Instant};

use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::{
    Request, Response,
    body::{Frame, Incoming},
    header,
};
use hyper_util::client::legacy::Client;
use tracing::warn;

use crate::api::{
    errors::ApiErrorCode,
    openai::metrics::{MetricsBody, MetricsRecorder},
    proxy::{ProxyBody, ProxyError, ProxyMetrics, WebSocketLifecycle, handle_upgrade},
};

/// Map a request path to the token-generating endpoint whose responses carry
/// `usage`/`timings`. Returns the matched `&'static str` for the metric's
/// `endpoint` column, or `None` for every other path (`/health`, `/metrics`,
/// `/v1/models`, upgrades, …) so those are forwarded without recording.
fn metrics_endpoint(path: &str) -> Option<&'static str> {
    match path {
        "/v1/chat/completions" => Some("/v1/chat/completions"),
        "/v1/completions" => Some("/v1/completions"),
        _ => None,
    }
}

/// Build the standard JSON error response on the hyper proxy data
/// plane. Pairs with `ApiErrorCode::into_response` on the axum side —
/// both serialise the same `ApiError` body the typed code projects to,
/// so a client gets a byte-identical body regardless of which surface
/// emitted the failure.
pub fn error_response(code: ApiErrorCode) -> ProxyError {
    let status = code.status();
    let body: ananke_api::shared::errors::ApiError = code.into();
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let full: ProxyBody = Full::new(Bytes::from(body_bytes))
        .map_err(|never| -> Box<dyn Error + Send + Sync> { match never {} })
        .boxed();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(full)
        // Invariant: the status and single content-type header are fixed
        // valid values, so the response builder cannot fail.
        .unwrap_or_else(|_| unreachable!("response with fixed status and headers builds"))
}

/// Reverse-proxy a single request to the upstream, returning an infallible result.
///
/// All upstream and protocol errors are translated into HTTP error responses so that
/// `serve_connection` never sees a service error (avoiding the lifetime trouble with
/// `Box<dyn Error + Send + Sync>` and the `From` blanket impl).
pub(crate) async fn handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
    metrics: Option<ProxyMetrics>,
) -> Result<Response<ProxyBody>, Infallible> {
    match try_handle(req, client, upstream_port, peer, ws_lifecycle, metrics).await {
        Ok(resp) => Ok(resp),
        Err(e) => {
            warn!(error = %e, peer = %peer, "proxy error");
            Ok(error_response(ApiErrorCode::ProxyInternal {
                reason: e.to_string(),
            }))
        }
    }
}

async fn try_handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
    metrics: Option<ProxyMetrics>,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    // Upgrade requests (WebSocket and friends) need a raw byte splice between
    // the client and the upstream after the 101 — the pooled HTTP client
    // can't model that, and stripping the response's `Connection` header
    // would make aiohttp's WebSocket client reject the handshake. Upgrades
    // never carry token usage, so they bypass the metrics path entirely.
    if is_upgrade_request(&req) {
        return handle_upgrade(req, upstream_port, peer, ws_lifecycle).await;
    }

    // Only the token-generating endpoints get recorded; every other path is a
    // pure passthrough. Match on the path (sans query) before it is consumed.
    let metric_endpoint = metrics
        .as_ref()
        .and_then(|_| metrics_endpoint(req.uri().path()));
    // Wall-clock start for the request, captured before the upstream round-trip
    // so TTFT and total duration cover the full server-visible latency.
    let request_start = Instant::now();

    let (parts, body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}").parse::<hyper::Uri>()?;

    let mut upstream_req = Request::builder().method(parts.method.clone()).uri(uri);
    for (k, v) in parts.headers.iter() {
        if k == header::HOST {
            continue;
        }
        upstream_req = upstream_req.header(k, v);
    }
    let body_bytes = body.collect().await?.to_bytes();
    let upstream_body: ProxyBody = http_body_util::Full::new(body_bytes)
        .map_err(|never| match never {})
        .boxed();
    let upstream_req = upstream_req.body(upstream_body)?;

    let resp = match client.request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream request failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };

    let (parts, body) = resp.into_parts();
    // Stream the body back without buffering — critical for SSE.
    let stream = body.into_data_stream().map_ok(Frame::data);
    let boxed: ProxyBody = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();

    // Record per-request token metrics for the token-generating endpoints,
    // mirroring the OpenAI multiplexer's `MetricsBody` wrap. `is_streaming` is
    // inferred from the upstream `Content-Type` (`text/event-stream`); the
    // recorder handles both SSE and plain JSON, so this only affects TTFT
    // bookkeeping. Any other endpoint (or an absent context) forwards the body
    // verbatim.
    let body: ProxyBody = match (metric_endpoint, metrics) {
        (Some(endpoint), Some(metrics)) => {
            let status_code = parts.status.as_u16();
            let is_streaming = parts
                .headers
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.starts_with("text/event-stream"))
                .unwrap_or(false);
            let recorder = MetricsRecorder::new(
                request_start,
                metrics.service_id,
                (metrics.run_id)(),
                metrics.model.to_string(),
                endpoint,
                is_streaming,
            );
            MetricsBody::new(boxed, recorder, metrics.db.clone(), status_code).boxed()
        }
        _ => boxed,
    };

    let mut out = Response::from_parts(parts, body);
    out.headers_mut().remove(header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    Ok(out)
}

/// True if `req` carries an HTTP/1.1 `Upgrade` handshake (WebSocket, h2c, …).
///
/// Per RFC 7230 §6.7 the request must list the `upgrade` token in
/// `Connection` and name the target protocol in `Upgrade`. Both checks are
/// case-insensitive; `Connection` may contain other tokens alongside
/// `upgrade` (e.g. `keep-alive, Upgrade`).
fn is_upgrade_request(req: &Request<Incoming>) -> bool {
    let has_upgrade_token = req
        .headers()
        .get(header::CONNECTION)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|tok| tok.trim().eq_ignore_ascii_case("upgrade"))
        })
        .unwrap_or(false);
    let has_upgrade_target = req
        .headers()
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    has_upgrade_token && has_upgrade_target
}
