//! Incremental extraction of usage, timing, and speculative-decode counters
//! from a proxied response's SSE stream or JSON body.

use std::time::Instant;

use ananke_api::openai::response::{ChatCompletionChunk, Timings};
use ananke_db::{Database, models::RequestMetric};
use bytes::Bytes;
use tracing::warn;

/// Collects metrics from a proxied response body. For streaming responses,
/// parses SSE `data:` lines incrementally. For non-streaming, buffers the
/// body and parses as JSON on completion.
pub struct MetricsRecorder {
    start: Instant,
    service_id: i64,
    run_id: Option<i64>,
    model: String,
    endpoint: &'static str,
    is_streaming: bool,
    first_token_at: Option<Instant>,
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    /// Engine-reported count of prompt tokens actually evaluated during
    /// prefill (`timings.prompt_n`). Excludes cache-served tokens, so it is
    /// the correct numerator for prefill throughput.
    prompt_eval_tokens: Option<i64>,
    /// Engine-reported prefill time from the response `timings` object
    /// (llama.cpp). Preferred over proxy-observed TTFT for the input/output
    /// TPS split when present.
    prompt_ms: Option<i64>,
    /// Engine-reported decode time from the response `timings` object.
    predicted_ms: Option<i64>,
    /// Engine-reported count of speculative draft tokens proposed
    /// (`timings.draft_n`). Present only when the engine ran speculative
    /// decoding for the request.
    draft_tokens: Option<i64>,
    /// Engine-reported count of draft tokens the target accepted
    /// (`timings.draft_n_accepted`). The spec_collapse watchdog's signal.
    draft_tokens_accepted: Option<i64>,
    /// For SSE parsing: accumulates a partial line that spans a chunk
    /// boundary. For non-streaming: accumulates the full body (capped).
    buf: String,
    /// Cap on the buffer size. For non-streaming, most responses fit in
    /// well under this. For streaming, the `usage` field is in the final
    /// SSE chunk, so we never need more than the last chunk — but keeping
    /// the full buffer simplifies parsing. 256 KiB is generous.
    buf_cap: usize,
}

impl MetricsRecorder {
    pub fn new(
        start: Instant,
        service_id: i64,
        run_id: Option<i64>,
        model: String,
        endpoint: &'static str,
        is_streaming: bool,
    ) -> Self {
        Self {
            start,
            service_id,
            run_id,
            model,
            endpoint,
            is_streaming,
            first_token_at: None,
            prompt_tokens: None,
            completion_tokens: None,
            prompt_eval_tokens: None,
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            buf: String::new(),
            buf_cap: 256 * 1024,
        }
    }

    /// Feed response data bytes into the recorder. Passes through
    /// unchanged — the caller still sends `data` to the client.
    pub fn ingest(&mut self, data: &Bytes) {
        if self.buf.len() >= self.buf_cap {
            // Past the cap: for streaming, clear and keep only new data
            // (usage is in the final chunk). For non-streaming, stop
            // accumulating — we won't be able to parse the full JSON.
            if self.is_streaming {
                self.buf.clear();
            } else {
                return;
            }
        }
        self.buf.push_str(&String::from_utf8_lossy(data));
        if self.is_streaming {
            self.process_sse_lines();
        }
    }

    /// Split the buffer on newlines and process each complete SSE line.
    fn process_sse_lines(&mut self) {
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].to_string();
            self.buf.replace_range(..=pos, "");
            self.process_sse_line(&line);
        }
    }

    fn process_sse_line(&mut self, line: &str) {
        let Some(data) = line.trim().strip_prefix("data: ") else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) else {
            return;
        };
        // Extract usage (usually in the final chunk).
        if let Some(usage) = &chunk.usage {
            self.prompt_tokens = usage.prompt_tokens;
            self.completion_tokens = usage.completion_tokens;
        }
        self.extract_timings(chunk.timings.as_ref());
        // Record TTFT on the first content or reasoning chunk.
        if self.first_token_at.is_none() {
            let has_content = chunk
                .delta()
                .and_then(|d| d.any_text())
                .is_some_and(|s| !s.is_empty());
            if has_content {
                self.first_token_at = Some(Instant::now());
            }
        }
    }

    /// Extract llama.cpp's engine-reported phase timings from a response
    /// object, if present. The `timings` object sits next to `usage` and
    /// carries `prompt_ms` (prefill) and `predicted_ms` (decode) as
    /// floating-point milliseconds; both are rounded to whole ms. It also
    /// carries `prompt_n`, the number of prompt tokens actually evaluated
    /// (cache misses) — the correct prefill-throughput numerator, since
    /// `usage.prompt_tokens` counts the cached prefix too. Absent for engines
    /// that do not emit it, in which case the fields stay null.
    fn extract_timings(&mut self, timings: Option<&Timings>) {
        let Some(timings) = timings else {
            return;
        };
        if let Some(ms) = timings.prompt_ms {
            self.prompt_ms = Some(ms.round() as i64);
        }
        if let Some(ms) = timings.predicted_ms {
            self.predicted_ms = Some(ms.round() as i64);
        }
        if let Some(n) = timings.prompt_n {
            self.prompt_eval_tokens = Some(n);
        }
        if let Some(n) = timings.draft_n {
            self.draft_tokens = Some(n);
        }
        if let Some(n) = timings.draft_n_accepted {
            self.draft_tokens_accepted = Some(n);
        }
    }

    /// Called when the response stream ends. Extracts any remaining
    /// data (e.g. usage from a non-streaming JSON body) and spawns a
    /// task to write the metric to the database.
    pub fn finish(self, db: Database, status_code: u16) {
        let mut rec = self;
        // For non-streaming, parse the accumulated body as JSON.
        if !rec.is_streaming
            && !rec.buf.is_empty()
            && let Ok(body) = serde_json::from_str::<ChatCompletionChunk>(&rec.buf)
        {
            if let Some(usage) = &body.usage {
                rec.prompt_tokens = usage.prompt_tokens;
                rec.completion_tokens = usage.completion_tokens;
            }
            rec.extract_timings(body.timings.as_ref());
        }

        let duration_ms = rec.start.elapsed().as_millis() as i64;
        let ttft_ms = rec
            .first_token_at
            .map(|t| t.duration_since(rec.start).as_millis() as i64);
        let timestamp_ms = ananke_time::now_unix_ms() - duration_ms;

        let metric = RequestMetric {
            metric_id: 0,
            service_id: rec.service_id,
            run_id: rec.run_id,
            timestamp_ms,
            endpoint: rec.endpoint.to_string(),
            model: rec.model.clone(),
            prompt_tokens: rec.prompt_tokens,
            completion_tokens: rec.completion_tokens,
            prompt_eval_tokens: rec.prompt_eval_tokens,
            duration_ms: Some(duration_ms),
            ttft_ms,
            prompt_ms: rec.prompt_ms,
            predicted_ms: rec.predicted_ms,
            draft_tokens: rec.draft_tokens,
            draft_tokens_accepted: rec.draft_tokens_accepted,
            status_code: status_code as i64,
        };

        tokio::spawn(async move {
            if let Err(e) = db.insert_request_metric(&metric).await {
                warn!(error = %e, "failed to record request metric");
            }
        });
    }
}

impl ananke_proxy::ErasedRecorder for MetricsRecorder {
    fn ingest(&mut self, data: &Bytes) {
        self.ingest(data);
    }

    fn finish(self: Box<Self>, _status_code: u16) {
        unreachable!("plain MetricsRecorder has no db; use RequestMetricsRecorder")
    }
}

/// Erased recorder for the proxy data plane: pairs the token-usage
/// recorder with the db handle it writes to at finish time.
pub struct RequestMetricsRecorder {
    pub recorder: MetricsRecorder,
    pub db: Database,
}

impl ananke_proxy::ErasedRecorder for RequestMetricsRecorder {
    fn ingest(&mut self, data: &Bytes) {
        self.recorder.ingest(data);
    }

    fn finish(self: Box<Self>, status_code: u16) {
        self.recorder.finish(self.db, status_code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recorder(is_streaming: bool) -> MetricsRecorder {
        MetricsRecorder::new(
            Instant::now(),
            1,
            Some(1),
            "demo".into(),
            "/v1/chat/completions",
            is_streaming,
        )
    }

    /// Streaming: usage is extracted from the final SSE chunk that
    /// carries a `usage` object.
    #[test]
    fn streaming_extracts_usage_from_final_chunk() {
        let mut rec = make_recorder(true);

        // Content chunk with delta.
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        ));
        assert!(rec.first_token_at.is_some(), "TTFT should be set");

        // Final chunk with usage.
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":7}}\n\n",
        ));
        rec.ingest(&Bytes::from("data: [DONE]\n\n"));

        assert_eq!(rec.prompt_tokens, Some(42));
        assert_eq!(rec.completion_tokens, Some(7));
    }

    /// Streaming: TTFT is only set on the first content-bearing chunk,
    /// not on chunks that carry only role or empty content.
    #[test]
    fn streaming_ttft_ignores_empty_content() {
        let mut rec = make_recorder(true);

        // First chunk has role but no content — should NOT set TTFT.
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        ));
        assert!(rec.first_token_at.is_none());

        // Second chunk has content — should set TTFT.
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        ));
        assert!(rec.first_token_at.is_some());
    }

    /// Streaming: reasoning_content also triggers TTFT (DeepSeek-style).
    #[test]
    fn streaming_ttft_on_reasoning_content() {
        let mut rec = make_recorder(true);
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n\n",
        ));
        assert!(rec.first_token_at.is_some());
    }

    /// Streaming: a `data:` line split across two chunks is reassembled
    /// before parsing — the partial line buffer handles the boundary.
    #[test]
    fn streaming_handles_split_data_line() {
        let mut rec = make_recorder(true);

        rec.ingest(&Bytes::from("data: {\"choices\":[{\"delta\":{\"co"));
        rec.ingest(&Bytes::from("ntent\":\"hi\"}}]}\n\n"));

        assert!(rec.first_token_at.is_some());
    }

    /// Non-streaming: the entire body is buffered and parsed as JSON.
    #[test]
    fn non_streaming_extracts_usage_from_json() {
        let mut rec = make_recorder(false);
        rec.ingest(&Bytes::from(
            r#"{"id":"chatcmpl-1","choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#,
        ));

        // finish() does the JSON parse — simulate it by calling the
        // non-streaming parse path directly. We can't call finish()
        // because it spawns a tokio task, so we replicate the check.
        assert!(!rec.buf.is_empty());
        let body: ChatCompletionChunk = serde_json::from_str(&rec.buf).unwrap();
        let usage = body.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(3));
    }

    /// Streaming: llama.cpp's `timings` object in the final chunk yields
    /// engine-reported prefill and decode times, rounded to whole ms.
    #[test]
    fn streaming_extracts_timings_from_final_chunk() {
        let mut rec = make_recorder(true);
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        ));
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":7},\
             \"timings\":{\"prompt_ms\":123.4,\"predicted_ms\":456.6,\"prompt_n\":8}}\n\n",
        ));
        assert_eq!(rec.prompt_ms, Some(123));
        assert_eq!(rec.predicted_ms, Some(457));
        // prompt_n (8 evaluated) differs from billed prompt_tokens (42) — the
        // prompt was mostly cache-served.
        assert_eq!(rec.prompt_eval_tokens, Some(8));
    }

    /// Non-streaming: a buffered JSON body carrying `timings` populates the
    /// engine timing fields — the tier-1 source that works without a stream.
    #[test]
    fn non_streaming_extracts_timings_from_json() {
        let mut rec = make_recorder(false);
        let body = r#"{"choices":[{"message":{"content":"hi"}}],
            "usage":{"prompt_tokens":10,"completion_tokens":3},
            "timings":{"prompt_ms":50.0,"predicted_ms":200.0,"prompt_n":10}}"#;
        rec.ingest(&Bytes::from(body));

        // Replicate finish()'s non-streaming parse path (finish spawns a task).
        let body: ChatCompletionChunk = serde_json::from_str(&rec.buf).unwrap();
        rec.extract_timings(body.timings.as_ref());
        assert_eq!(rec.prompt_ms, Some(50));
        assert_eq!(rec.predicted_ms, Some(200));
        assert_eq!(rec.prompt_eval_tokens, Some(10));
    }

    /// Streaming: a `timings` object carrying speculative draft counts
    /// (llama.cpp with `--spec-type`) populates the draft fields. The values
    /// mirror a real collapsed response: 59 drafted, 0 accepted.
    #[test]
    fn streaming_extracts_draft_counts() {
        let mut rec = make_recorder(true);
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":32},\
             \"timings\":{\"prompt_ms\":135.5,\"predicted_ms\":873.5,\"prompt_n\":13,\
             \"draft_n\":59,\"draft_n_accepted\":0}}\n\n",
        ));
        assert_eq!(rec.draft_tokens, Some(59));
        assert_eq!(rec.draft_tokens_accepted, Some(0));
    }

    /// A `timings` object without draft counts (speculative decoding off)
    /// leaves the draft fields null so the row never feeds the
    /// spec_collapse watchdog.
    #[test]
    fn missing_draft_counts_leave_fields_null() {
        let mut rec = make_recorder(true);
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1},\
             \"timings\":{\"prompt_ms\":50.0,\"predicted_ms\":100.0,\"prompt_n\":1}}\n\n",
        ));
        assert_eq!(rec.draft_tokens, None);
        assert_eq!(rec.draft_tokens_accepted, None);
    }

    /// A response without a `timings` object leaves the engine timing
    /// fields null, so the query layer falls back to TTFT or aggregate.
    #[test]
    fn missing_timings_leaves_fields_null() {
        let mut rec = make_recorder(true);
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
        ));
        assert_eq!(rec.prompt_ms, None);
        assert_eq!(rec.predicted_ms, None);
    }

    /// Buffer cap: once the cap is reached for a non-streaming response,
    /// further data is dropped. For streaming, the buffer clears so only
    /// recent data (containing usage) is kept.
    #[test]
    fn streaming_clears_buffer_past_cap() {
        let mut rec = make_recorder(true);
        rec.buf_cap = 50; // tiny cap for testing
        rec.buf = "x".repeat(50);

        // Ingest more data — the buffer should clear (streaming path).
        rec.ingest(&Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
        ));

        // The old buffer was cleared and replaced with new data.
        assert!(rec.buf.len() < 60);
        assert!(rec.first_token_at.is_some(), "TTFT should still be set");
    }
}
