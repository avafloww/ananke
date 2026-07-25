//! Response body inspection for per-request metrics recording.
//!
//! Wraps the proxied response body to extract `usage` (token counts) and
//! time-to-first-token from the SSE stream (or JSON body) as it passes
//! through to the client. When the stream ends, the recorded data is
//! written to the `request_metrics` table via a spawned task.

mod body;
mod recorder;

pub use body::MetricsBody;
pub use recorder::MetricsRecorder;
