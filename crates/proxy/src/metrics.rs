//! Type-erased per-request metrics recording for the proxy data plane.
//!
//! The daemon's OpenAI surface owns the concrete recorder (parses SSE /
//! JSON token usage and writes `RequestMetric` rows). The proxy only
//! needs an object-safe recorder plus a factory that mints one per
//! request; the db handle is captured inside the recorder by the
//! factory, so the data plane never names it.

use std::{pin::Pin, sync::Arc, time::Instant};

use bytes::Bytes;
use hyper::body::Frame;

/// The per-request token-usage recorder surface the data plane drives.
///
/// Implemented by the daemon's OpenAI metrics recorder adapter, which
/// captures the db handle at construction so the proxy never sees it.
pub trait ErasedRecorder: Send + Sync + 'static {
    /// Feed response bytes to the recorder (SSE line accumulation, JSON
    /// buffering, TTFT bookkeeping).
    fn ingest(&mut self, data: &Bytes);

    /// Write the accumulated metric, consuming the recorder.
    fn finish(self: Box<Self>, status_code: u16);
}

/// Mints a fresh recorder for each token-generating request.
pub type RecorderFactory = Arc<
    dyn Fn(Instant, i64, Option<i64>, String, &'static str, bool) -> Box<dyn ErasedRecorder>
        + Send
        + Sync,
>;

/// Always-None factory used by the metrics-free `serve` path.
pub fn no_recorder_factory() -> RecorderFactory {
    Arc::new(|_, _, _, _, _, _| Box::new(NoRecorder))
}

/// Recorder placeholder for the metrics-free [`super::serve`] path.
pub struct NoRecorder;

impl ErasedRecorder for NoRecorder {
    fn ingest(&mut self, _data: &Bytes) {}
    fn finish(self: Box<Self>, _status_code: u16) {}
}

/// Wraps a body with an [`ErasedRecorder`]. Passes all data through
/// unchanged while feeding bytes to the recorder. When the stream ends
/// (or the body is dropped), the recorder writes the metric.
pub struct MetricsBody<B> {
    body: B,
    recorder: Option<Box<dyn ErasedRecorder>>,
    status_code: u16,
    recorded: bool,
}

impl<B> MetricsBody<B> {
    pub fn new(body: B, recorder: Box<dyn ErasedRecorder>, status_code: u16) -> Self {
        Self {
            body,
            recorder: Some(recorder),
            status_code,
            recorded: false,
        }
    }

    fn record_metric(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        if let Some(recorder) = self.recorder.take() {
            recorder.finish(self.status_code);
        }
    }
}

impl<B> Drop for MetricsBody<B> {
    fn drop(&mut self) {
        self.record_metric();
    }
}

impl<B: hyper::body::Body<Data = Bytes>> hyper::body::Body for MetricsBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // Safety: we never move `body` out of self, and the other fields
        // are not pinned (only `body` needs pinning for the Body trait).
        // This is the standard manual pin-projection pattern.
        let this = unsafe { self.get_unchecked_mut() };
        let body = unsafe { Pin::new_unchecked(&mut this.body) };
        match body.poll_frame(cx) {
            std::task::Poll::Ready(None) => {
                this.record_metric();
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref()
                    && let Some(recorder) = this.recorder.as_mut()
                {
                    recorder.ingest(data);
                }
                std::task::Poll::Ready(Some(Ok(frame)))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                this.record_metric();
                std::task::Poll::Ready(Some(Err(e)))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
