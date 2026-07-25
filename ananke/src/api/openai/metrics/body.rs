//! `hyper::body::Body` wrapper that feeds bytes to a [`MetricsRecorder`]
//! and records the metric once the stream ends or the body is dropped.

use std::pin::Pin;

use bytes::Bytes;
use hyper::body::Frame;

use crate::{api::openai::metrics::MetricsRecorder, db::Database};

/// Wraps a body with a [`MetricsRecorder`]. Passes all data through
/// unchanged while feeding bytes to the recorder. When the stream ends
/// (or the body is dropped), the recorder writes the metric to the
/// database via a spawned task.
pub struct MetricsBody<B> {
    body: B,
    recorder: Option<MetricsRecorder>,
    db: Option<Database>,
    status_code: u16,
    recorded: bool,
}

impl<B> MetricsBody<B> {
    pub fn new(body: B, recorder: MetricsRecorder, db: Database, status_code: u16) -> Self {
        Self {
            body,
            recorder: Some(recorder),
            db: Some(db),
            status_code,
            recorded: false,
        }
    }

    fn record_metric(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        if let (Some(recorder), Some(db)) = (self.recorder.take(), self.db.take()) {
            recorder.finish(db, self.status_code);
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

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}
