//! Shared upstream-forwarding core for the OpenAI multiplexer endpoints:
//! the ensure-ready gate, the guarded response body, and the hyper POST
//! data plane with inflight/stall/metrics plumbing. The JSON endpoints
//! (`forward_json_post`) and the multipart audio endpoint both terminate
//! here so the request-lifecycle bookkeeping exists exactly once.

use std::{
    sync::Arc,
    task::Poll,
    time::{Duration, Instant},
};

use ananke_tracking::{inflight::InflightGuard, progress::ProgressCell};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, SizeHint};
use tracing::warn;

use crate::{
    api::openai::{
        errors,
        metrics::{MetricsBody, MetricsRecorder, RequestMetricsRecorder},
        stall::{self, StallDisarm},
    },
    config::ServiceConfig,
    daemon::app_state::AppState,
    supervise::{EnsureFailure, EnsureOutcome, SupervisorHandle, await_ensure},
};

/// Everything [`forward_upstream`] needs to POST a fully-prepared body to a
/// service's private port and stream the response back with the request
/// lifecycle accounted for (inflight guard, stall watchdog, metrics).
pub(crate) struct UpstreamPost<'a> {
    pub state: &'a AppState,
    pub svc: &'a ServiceConfig,
    pub handle: &'a Arc<SupervisorHandle>,
    /// Client-facing model name (the service name); used for logging,
    /// error bodies, and metric attribution.
    pub model: &'a str,
    /// Endpoint path, forwarded verbatim to the upstream.
    pub path: &'static str,
    /// Client request headers, copied through minus host/content-length.
    pub headers: &'a HeaderMap,
    /// Content type for the upstream request. JSON endpoints force
    /// `application/json`; the audio endpoint passes the client's
    /// original multipart content type so the boundary survives.
    pub content_type: HeaderValue,
    pub body: Bytes,
    pub is_streaming: bool,
    pub request_start: Instant,
}

/// Ensure the service is running (coalescing concurrent first-requests),
/// mapping every failure to its wire error response.
pub(crate) async fn ensure_ready(
    handle: &SupervisorHandle,
    svc: &ServiceConfig,
    model: &str,
    path: &'static str,
) -> Result<(), Response> {
    let max_request_duration = Duration::from_millis(svc.max_request_duration_ms);
    match await_ensure(handle, max_request_duration).await {
        EnsureOutcome::Ready { .. } => Ok(()),
        EnsureOutcome::Failed(EnsureFailure::InsufficientCapacity(msg)) => {
            warn!(model = %model, endpoint = path, reason = %msg, "request rejected: insufficient_capacity");
            Err(errors::insufficient_capacity(model, &msg))
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(msg)) => {
            warn!(model = %model, endpoint = path, reason = %msg, "request rejected: service_disabled");
            Err(errors::service_disabled(model, &msg))
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => {
            warn!(model = %model, endpoint = path, "request rejected: start_queue_full");
            Err(errors::start_queue_full(model))
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(msg)) => {
            warn!(model = %model, endpoint = path, reason = %msg, "request rejected: start_failed");
            Err(errors::start_failed(model, &msg))
        }
        EnsureOutcome::Failed(EnsureFailure::Blocked { busy_peers }) => {
            warn!(model = %model, endpoint = path, ?busy_peers, "request rejected: service_blocked");
            Err(errors::service_blocked(model, &busy_peers))
        }
    }
}

/// POST the prepared body to the service's private port and return the
/// proxied response. Owns the whole request-lifecycle tail: activity ping,
/// inflight guard, stall watchdog arming, hyper client, guarded body, and
/// metrics recording.
pub(crate) async fn forward_upstream(req: UpstreamPost<'_>) -> Response {
    let UpstreamPost {
        state,
        svc,
        handle,
        model,
        path,
        headers,
        content_type,
        body,
        is_streaming,
        request_start,
    } = req;

    // Bump activity and acquire an in-flight guard before forwarding.
    state.activity.ping(&svc.name);
    let counter = state.inflight.counter(&svc.name);
    let guard = InflightGuard::new(counter);

    // Per-service last-frame stamp for the stall watchdog. Present whenever the
    // watchdog is enabled; every forwarded frame (streaming or not) bumps it,
    // so a stalled request can tell "the whole service is silent" (wedge) from
    // "another request is streaming fine" (just queued).
    let progress = svc
        .auto_restart
        .ttft_stall
        .map(|_| state.progress.stamp(&svc.name));

    // Arm the stall watchdog only for streaming requests. A streaming upstream
    // returns headers before any token, so silence on the body is the wedge
    // signal. Non-streaming responses only arrive once fully buffered, making
    // TTFT indistinguishable from a slow-but-healthy generation — those are
    // bounded by `max_request_duration`, not this watchdog. Armed before the
    // request is sent so it also covers a child that never returns headers.
    let mut stall = match (
        svc.auto_restart.ttft_stall,
        is_streaming,
        handle.peek().run_id,
    ) {
        (Some(trigger), true, Some(run_id)) => progress.clone().map(|prog| {
            stall::arm(
                Arc::clone(handle),
                run_id,
                Duration::from_millis(trigger.timeout_ms),
                prog,
            )
        }),
        _ => None,
    };

    // Build hyper client and forward to the upstream service.
    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build_http::<http_body_util::combinators::BoxBody<
        Bytes,
        Box<dyn std::error::Error + Send + Sync>,
    >>();

    let uri = format!("http://127.0.0.1:{}{}", svc.private_port, path)
        .parse::<hyper::Uri>()
        .unwrap();
    let mut hreq = hyper::Request::builder().method("POST").uri(uri);
    for (k, v) in headers.iter() {
        if k == hyper::header::HOST || k == hyper::header::CONTENT_LENGTH {
            continue;
        }
        hreq = hreq.header(k, v);
    }
    hreq = hreq.header(hyper::header::CONTENT_TYPE, content_type);
    hreq = hreq.header(hyper::header::CONTENT_LENGTH, body.len());
    let upstream_body = http_body_util::Full::new(body)
        .map_err(|never| match never {})
        .boxed();
    let hreq = match hreq.body(upstream_body) {
        Ok(r) => r,
        Err(e) => {
            // Never left the ground — disarm so the timer doesn't fire against
            // a healthy run five minutes from now.
            if let Some(s) = stall.as_mut() {
                s.disarm();
            }
            return errors::bad_request(format!("build request: {e}"));
        }
    };

    let resp = match client.request(hreq).await {
        Ok(r) => r,
        Err(e) => {
            // A synchronous upstream failure (e.g. connection reset during a
            // drain race) is not a stall — disarm before returning.
            if let Some(s) = stall.as_mut() {
                s.disarm();
            }
            warn!(error = %e, model = %model, "upstream request failed");
            return errors::start_failed(model, "upstream unavailable");
        }
    };

    let (parts, upstream_body) = resp.into_parts();
    let status_code = parts.status.as_u16();
    // Convert the upstream body into a stream of data frames for axum to proxy.
    // Wrap in GuardedBody so the in-flight counter stays elevated for the full
    // duration of the response, including SSE streams. Then wrap in MetricsBody
    // to extract usage/TTFT and record per-request metrics on stream completion.
    let stream = upstream_body.into_data_stream().map_ok(Frame::data);
    let boxed: http_body_util::combinators::BoxBody<
        Bytes,
        Box<dyn std::error::Error + Send + Sync>,
    > = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();
    let guarded = GuardedBody {
        body: boxed,
        _guard: guard,
        stall,
        progress,
    };

    // Resolve the service_id and run_id for metric recording. If the
    // service can't be resolved (not yet in the DB), skip metrics —
    // the proxy still works, we just don't record this request.
    let service_id = state.db.resolve_service_id(model).await.ok().flatten();
    let run_id = handle.peek().run_id;

    let axum_body = if let Some(service_id) = service_id {
        let recorder = MetricsRecorder::new(
            request_start,
            service_id,
            run_id,
            model.to_string(),
            path,
            is_streaming,
        );
        let metrics_body = MetricsBody::new(
            guarded,
            Box::new(RequestMetricsRecorder {
                recorder,
                db: state.db.clone(),
            }),
            status_code,
        );
        Body::new(metrics_body)
    } else {
        Body::new(guarded)
    };

    let mut out = Response::from_parts(parts, axum_body);
    // Strip hop-by-hop headers. These are per-connection directives
    // between the proxy and its upstream (llama.cpp); forwarding them
    // to the browser is incorrect and can cause the browser to close
    // the connection prematurely (e.g. llama.cpp sends
    // `keep-alive: timeout=5`, which the browser interprets as a
    // 5-second inactivity deadline — cutting off slow streaming
    // responses like image inference that takes >5s to first token).
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    out.headers_mut().remove("keep-alive");
    out
}

pin_project_lite::pin_project! {
    /// Wraps a body and holds an [`InflightGuard`] so the counter stays elevated
    /// until the full response body (including SSE streams) has been consumed.
    /// On each forwarded data frame it stamps the per-service `progress` cell
    /// (feeding the stall watchdog's run-level liveness check) and, on the
    /// first frame, disarms this request's stall timer.
    struct GuardedBody<B> {
        #[pin]
        body: B,
        _guard: InflightGuard,
        stall: Option<StallDisarm>,
        progress: Option<ProgressCell>,
    }
}

impl<B: hyper::body::Body> hyper::body::Body for GuardedBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        let polled = this.body.poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &polled
            && frame.data_ref().is_some()
        {
            // Any data frame is proof the child is producing output: record
            // service-level progress, and disarm this request's stall timer.
            if let Some(progress) = this.progress.as_ref() {
                progress.record();
            }
            if let Some(stall) = this.stall.as_mut() {
                stall.disarm();
            }
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}
