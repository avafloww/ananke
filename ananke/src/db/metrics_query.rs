//! Pre-bucketed time-series queries over `request_metrics` for the
//! `/api/metrics` endpoint. The write and retention side lives in the
//! sibling `metrics` module.

use rusqlite::params;

use crate::{db::Database, errors::ExpectedError};

impl Database {
    /// Query aggregated request metrics for the JSON `/api/metrics` endpoint.
    /// Returns pre-bucketed time-series data — the frontend doesn't aggregate.
    /// Each bucket is scoped to a single service so the frontend can distinguish
    /// per-service contributions when no service filter is given.
    pub async fn query_request_metrics(
        &self,
        service_id: Option<i64>,
        since_ms: i64,
        until_ms: i64,
        bucket_ms: i64,
    ) -> Result<Vec<MetricBucket>, ExpectedError> {
        let conn = self.conn.lock();
        // LEFT JOIN services so that orphaned metrics (service deleted but
        // rows remain) still appear with service = NULL. Grouping by
        // service_id in addition to the bucket ensures per-service breakdown
        // when no filter is given.
        // The input/output TPS split is sourced tier-by-tier per row:
        //   - output interval = engine `predicted_ms` if present, else the
        //     proxy-observed decode window (`duration_ms - ttft_ms`) when the
        //     response streamed, else null (no boundary → no split);
        //   - input interval = engine `prompt_ms` if present, else `ttft_ms`.
        // The input numerator uses the engine's evaluated prompt-token count
        // `prompt_eval_tokens` when present, falling back to the billed
        // `prompt_tokens`. The billed count includes tokens served from the KV
        // cache, so dividing it by the cache-aware `prompt_ms` would wildly
        // overstate prefill throughput.
        // Effective TPS is completion tokens over wall-clock `duration_ms`:
        // end-to-end generation throughput (prefill, TTFT, and queue wait all
        // count against it), always computable whenever a request has a
        // duration. It is the tier-3 (non-streaming, no engine timings) fall
        // back where no decode window exists to derive `output_tps`. It counts
        // only generated tokens — a prompt-only request (e.g. embeddings)
        // contributes zero and drops out rather than spiking the line.
        let out_interval = "COALESCE(rm.predicted_ms, CASE WHEN rm.ttft_ms IS NOT NULL \
             AND rm.duration_ms IS NOT NULL THEN rm.duration_ms - rm.ttft_ms END)";
        let in_interval = "COALESCE(rm.prompt_ms, rm.ttft_ms)";
        let sql = format!(
            "SELECT
                (rm.timestamp_ms / ?1) * ?1 AS bucket,
                s.name AS service_name,
                COUNT(*) AS request_count,
                COALESCE(SUM(rm.prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(rm.completion_tokens), 0) AS completion_tokens,
                AVG(rm.duration_ms) AS avg_duration_ms,
                SUM(CASE WHEN rm.status_code >= 400 THEN 1 ELSE 0 END) AS error_count,
                AVG(rm.ttft_ms) AS avg_ttft_ms,
                COALESCE(SUM(CASE WHEN {out} IS NOT NULL
                                  THEN rm.completion_tokens ELSE 0 END), 0) AS output_tokens,
                COALESCE(SUM({out}), 0) AS output_ms,
                COALESCE(SUM(CASE WHEN {inp} IS NOT NULL
                                  THEN COALESCE(rm.prompt_eval_tokens, rm.prompt_tokens) ELSE 0 END), 0) AS input_tokens,
                COALESCE(SUM({inp}), 0) AS input_ms,
                COALESCE(SUM(CASE WHEN rm.duration_ms IS NOT NULL
                                  THEN COALESCE(rm.completion_tokens, 0)
                                  ELSE 0 END), 0) AS effective_tokens,
                COALESCE(SUM(CASE WHEN rm.duration_ms IS NOT NULL
                                  THEN rm.duration_ms ELSE 0 END), 0) AS effective_ms
             FROM request_metrics rm
             LEFT JOIN services s ON s.service_id = rm.service_id
             WHERE rm.timestamp_ms >= ?2 AND rm.timestamp_ms <= ?3
               AND (?4 IS NULL OR rm.service_id = ?4)
             GROUP BY bucket, rm.service_id
             ORDER BY bucket, service_name",
            out = out_interval,
            inp = in_interval,
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![bucket_ms, since_ms, until_ms, service_id], |row| {
                // A rate needs both a positive interval and tokens actually
                // produced. Zero tokens over a non-zero interval is not
                // "0 tok/s" — it is the absence of throughput (e.g. every
                // request in the bucket stalled or errored before emitting a
                // token), which must read as null so the chart breaks the line
                // rather than pinning it to the floor. Without the token guard
                // the effective tier (completion / wall-clock duration) reports
                // 0 for a wedged run, while the decode/prefill tiers correctly
                // go null — an inconsistency across the three lines.
                let tps = |tokens: i64, ms: i64| {
                    (ms > 0 && tokens > 0).then(|| tokens as f64 / (ms as f64 / 1000.0))
                };
                let output_tokens: i64 = row.get(8)?;
                let output_ms: i64 = row.get(9)?;
                let input_tokens: i64 = row.get(10)?;
                let input_ms: i64 = row.get(11)?;
                let effective_tokens: i64 = row.get(12)?;
                let effective_ms: i64 = row.get(13)?;
                Ok(MetricBucket {
                    service: row.get::<_, Option<String>>(1)?,
                    bucket_start: row.get(0)?,
                    request_count: row.get(2)?,
                    prompt_tokens: row.get(3)?,
                    completion_tokens: row.get(4)?,
                    avg_duration_ms: row.get::<_, Option<f64>>(5)?,
                    error_count: row.get(6)?,
                    avg_ttft_ms: row.get::<_, Option<f64>>(7)?,
                    output_tps: tps(output_tokens, output_ms),
                    input_tps: tps(input_tokens, input_ms),
                    effective_tps: tps(effective_tokens, effective_ms),
                })
            })
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }
}

/// One time bucket of aggregated request metrics, scoped to a single service.
pub struct MetricBucket {
    /// Service name, or `None` if the service was deleted but metric rows remain.
    pub service: Option<String>,
    pub bucket_start: i64,
    pub request_count: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub avg_duration_ms: Option<f64>,
    pub error_count: i64,
    /// Average time-to-first-token in milliseconds (streaming requests only).
    pub avg_ttft_ms: Option<f64>,
    /// Output tokens per second during decode: completion tokens divided
    /// by total decode time. `None` if no timed requests in the bucket.
    pub output_tps: Option<f64>,
    /// Input tokens per second during prompt processing: prompt tokens
    /// divided by total TTFT. `None` if no timed requests in the bucket.
    pub input_tps: Option<f64>,
    /// End-to-end effective generation throughput: completion tokens divided
    /// by total wall-clock duration (so prefill, TTFT, and queue wait all
    /// count against it). Always available whenever the bucket has any request
    /// with a recorded duration, including non-streaming requests with no
    /// engine timings where no decode window exists to derive `output_tps`.
    /// This is *not* a decode rate — it is always ≤ `output_tps`. A bucket
    /// that generated no tokens (e.g. only embeddings or stalled requests) is
    /// `None`, not zero.
    pub effective_tps: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models::RequestMetric;

    #[tokio::test]
    async fn request_metrics_insert_and_query() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 1000).await.unwrap();

        // Insert two requests 5 minutes apart, one success + one error.
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 10_000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            duration_ms: Some(1200),
            ttft_ms: Some(200),
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 200,
        })
        .await
        .unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 10_000 + 5 * 60_000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(200),
            completion_tokens: Some(80),
            duration_ms: Some(800),
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 500,
        })
        .await
        .unwrap();

        // Query with a 10-minute bucket — both should land in the same bucket.
        let buckets = db
            .query_request_metrics(Some(svc), 0, 20 * 60_000, 10 * 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 1);
        let b = &buckets[0];
        assert_eq!(b.service.as_deref(), Some("demo"));
        assert_eq!(b.request_count, 2);
        assert_eq!(b.prompt_tokens, 300);
        assert_eq!(b.completion_tokens, 130);
        assert!((b.avg_duration_ms.unwrap() - 1000.0).abs() < 0.1);
        assert_eq!(b.error_count, 1);
    }

    /// The input/output split is sourced tier-by-tier: engine timings
    /// (tier 1), proxy TTFT for streaming (tier 2), or neither (tier 3,
    /// which yields only effective TPS). Each tier lives in its own bucket.
    #[tokio::test]
    async fn request_metrics_tps_tiers() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let base = |timestamp_ms| RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 200,
        };

        // Tier 1 (engine timings) at t=0: input 100 tok / 1000 ms = 100 tps,
        // output 50 tok / 500 ms = 100 tps, effective 50 tok / 2000 ms = 25.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            duration_ms: Some(2000),
            prompt_ms: Some(1000),
            predicted_ms: Some(500),
            ..base(0)
        })
        .await
        .unwrap();

        // Tier 2 (streaming TTFT) at t=60s: input 100 tok / 200 ms = 500 tps,
        // output 80 tok / (1000 - 200) ms = 100 tps, effective 80 / 1000 = 80.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(100),
            completion_tokens: Some(80),
            duration_ms: Some(1000),
            ttft_ms: Some(200),
            ..base(60_000)
        })
        .await
        .unwrap();

        // Tier 3 (non-streaming, no timings) at t=120s: no split, effective
        // 100 tok / 1000 ms = 100 tps.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(200),
            completion_tokens: Some(100),
            duration_ms: Some(1000),
            ..base(120_000)
        })
        .await
        .unwrap();

        let buckets = db
            .query_request_metrics(Some(svc), 0, 200_000, 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 3);
        let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 0.01;

        assert!(close(buckets[0].input_tps, 100.0));
        assert!(close(buckets[0].output_tps, 100.0));
        assert!(close(buckets[0].effective_tps, 25.0));

        assert!(close(buckets[1].input_tps, 500.0));
        assert!(close(buckets[1].output_tps, 100.0));
        assert!(close(buckets[1].effective_tps, 80.0));

        assert_eq!(buckets[2].input_tps, None);
        assert_eq!(buckets[2].output_tps, None);
        assert!(close(buckets[2].effective_tps, 100.0));
    }

    /// A bucket whose requests ran (non-null `duration_ms`) but produced no
    /// tokens — the signature of a wedged run whose cancelled requests each
    /// held the connection for their full duration — must report null for
    /// every TPS tier, not a floor-pinned effective figure of `0 tok/s`.
    /// Otherwise the effective line stays drawn across a stall while the
    /// decode/prefill lines correctly break.
    #[tokio::test]
    async fn zero_token_bucket_reports_null_tps() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let base = |timestamp_ms| RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 200,
        };

        // Two cancelled requests: each ran ~300s and emitted nothing.
        for i in 0..2 {
            db.insert_request_metric(&RequestMetric {
                duration_ms: Some(300_000),
                ..base(i)
            })
            .await
            .unwrap();
        }

        let buckets = db
            .query_request_metrics(Some(svc), 0, 60_000, 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].request_count, 2);
        assert_eq!(buckets[0].effective_tps, None, "effective must be null");
        assert_eq!(buckets[0].input_tps, None);
        assert_eq!(buckets[0].output_tps, None);
    }

    /// Prompt caching: `prompt_tokens` is the full billed prompt (8000) but
    /// only `prompt_eval_tokens` (100) were actually evaluated. Input TPS must
    /// use the evaluated count, not the billed one — else 8000 / 200 ms =
    /// 40 000 tok/s instead of the true 100 / 200 ms = 500. Effective TPS
    /// sidesteps this entirely: it counts only completion tokens.
    #[tokio::test]
    async fn tps_uses_evaluated_prompt_tokens_not_billed() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: Some(100),
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 0,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(8000),
            completion_tokens: Some(50),
            duration_ms: Some(1000),
            ttft_ms: None,
            prompt_ms: Some(200),
            predicted_ms: Some(500),
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 200,
        })
        .await
        .unwrap();

        let b = &db
            .query_request_metrics(Some(svc), 0, 60_000, 60_000)
            .await
            .unwrap()[0];
        let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 0.01;
        // input = 100 evaluated / 0.2 s = 500 (not 8000 / 0.2 = 40 000).
        assert!(close(b.input_tps, 500.0), "input_tps = {:?}", b.input_tps);
        // effective = completion only: 50 / 1 s = 50 (prompt tokens, cached or
        // billed, never enter the effective throughput).
        assert!(
            close(b.effective_tps, 50.0),
            "effective_tps = {:?}",
            b.effective_tps
        );
        // output is unaffected by prompt caching: 50 / 0.5 s = 100.
        assert!(
            close(b.output_tps, 100.0),
            "output_tps = {:?}",
            b.output_tps
        );
        // The displayed prompt-token total stays the billed count.
        assert_eq!(b.prompt_tokens, 8000);
    }

    #[tokio::test]
    async fn request_metrics_filter_by_service() {
        let db = Database::open_in_memory().await.unwrap();
        let svc_a = db.upsert_service("alpha", 1000).await.unwrap();
        let svc_b = db.upsert_service("beta", 2000).await.unwrap();

        for svc_id in [svc_a, svc_b] {
            db.insert_request_metric(&RequestMetric {
                metric_id: 0,
                prompt_eval_tokens: None,
                service_id: svc_id,
                run_id: Some(1),
                timestamp_ms: 1000,
                endpoint: "/v1/chat/completions".into(),
                model: "test".into(),
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                duration_ms: None,
                ttft_ms: None,
                prompt_ms: None,
                predicted_ms: None,
                draft_tokens: None,
                draft_tokens_accepted: None,
                status_code: 200,
            })
            .await
            .unwrap();
        }

        // Query all services (None) — should get one bucket per service.
        let all = db
            .query_request_metrics(None, 0, 10_000, 60_000)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        // Buckets are ordered by (bucket, service_name) — alpha before beta.
        assert_eq!(all[0].service.as_deref(), Some("alpha"));
        assert_eq!(all[0].request_count, 1);
        assert_eq!(all[1].service.as_deref(), Some("beta"));
        assert_eq!(all[1].request_count, 1);

        // Query only svc_a.
        let filtered = db
            .query_request_metrics(Some(svc_a), 0, 10_000, 60_000)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].service.as_deref(), Some("alpha"));
        assert_eq!(filtered[0].request_count, 1);
    }
}
