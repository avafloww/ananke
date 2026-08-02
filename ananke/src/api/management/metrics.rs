//! `GET /api/metrics` and `GET /api/devices/samples` handlers.

use ananke_api::{
    devices::samples::{DeviceSampleResponse, DeviceSamplesResponse},
    metrics::{
        get::{MetricBucketResponse, MetricsResponse},
        restarts::{RestartsResponse, ServiceRestartEntry},
    },
    shared::errors::ApiError,
};
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Deserialize;

use crate::{api::errors::ApiErrorCode, daemon::app_state::AppState, db::MetricBucket};

#[derive(Debug, Deserialize)]
pub struct MetricsQuery {
    pub service: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub bucket: Option<String>,
}

#[utoipa::path(
    summary = "Get request metrics (time-bucketed)",
    get,
    path = "/api/metrics",
    params(
        ("service" = Option<String>, Query, description = "Filter to one service name"),
        ("since" = Option<i64>, Query, description = "Earliest timestamp_ms, inclusive (default: 1h ago)"),
        ("until" = Option<i64>, Query, description = "Latest timestamp_ms, inclusive (default: now)"),
        ("bucket" = Option<String>, Query, description = "Bucket size: \"1m\", \"5m\", \"1h\" (default: \"5m\")"),
    ),
    responses((status = 200, body = MetricsResponse), (status = 400, body = ApiError, description = "invalid_request_error"), (status = 500, body = ApiError, description = "query_failed"))
)]
pub async fn get_metrics(State(state): State<AppState>, Query(q): Query<MetricsQuery>) -> Response {
    let now = crate::tracking::now_unix_ms();
    let until = q.until.unwrap_or(now);
    let since = q.since.unwrap_or(now - 3_600_000); // default: last 1h

    // Parse bucket size. Default 5 minutes.
    let bucket_str = q.bucket.as_deref().unwrap_or("5m");
    let bucket_ms = match crate::config::validate::parse_duration_ms(bucket_str) {
        Ok(ms) => ms.max(1000) as i64, // minimum 1s bucket
        Err(_) => {
            return ApiErrorCode::InvalidRequest {
                reason: format!("invalid bucket: {bucket_str}"),
            }
            .into_response();
        }
    };

    let service_id = if let Some(name) = &q.service {
        state.db.resolve_service_id(name).await.ok().flatten()
    } else {
        None
    };

    match state
        .db
        .query_request_metrics(service_id, since, until, bucket_ms)
        .await
    {
        Ok(buckets) => {
            let resp = MetricsResponse {
                buckets: buckets
                    .into_iter()
                    .map(|b: MetricBucket| MetricBucketResponse {
                        service: b.service,
                        bucket_start: b.bucket_start,
                        request_count: b.request_count,
                        prompt_tokens: b.prompt_tokens,
                        completion_tokens: b.completion_tokens,
                        avg_duration_ms: b.avg_duration_ms,
                        error_count: b.error_count,
                        avg_ttft_ms: b.avg_ttft_ms,
                        output_tps: b.output_tps,
                        input_tps: b.input_tps,
                        effective_tps: b.effective_tps,
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => ApiErrorCode::QueryFailed {
            reason: e.to_string(),
        }
        .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct RestartsQuery {
    pub service: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

#[utoipa::path(
    summary = "Get auto-restart firings",
    get,
    path = "/api/restarts",
    params(
        ("service" = Option<String>, Query, description = "Filter to one service name"),
        ("since" = Option<i64>, Query, description = "Earliest at_ms, inclusive (default: 1h ago)"),
        ("until" = Option<i64>, Query, description = "Latest at_ms, inclusive (default: now)"),
    ),
    responses((status = 200, body = RestartsResponse), (status = 500, body = ApiError, description = "query_failed"))
)]
pub async fn get_restarts(
    State(state): State<AppState>,
    Query(q): Query<RestartsQuery>,
) -> Response {
    let now = crate::tracking::now_unix_ms();
    let until = q.until.unwrap_or(now);
    let since = q.since.unwrap_or(now - 3_600_000);

    let service_id = if let Some(name) = &q.service {
        state.db.resolve_service_id(name).await.ok().flatten()
    } else {
        None
    };
    // An unknown service name matches nothing rather than everything.
    if q.service.is_some() && service_id.is_none() {
        let resp = RestartsResponse { restarts: vec![] };
        return (StatusCode::OK, Json(resp)).into_response();
    }

    match state
        .db
        .query_service_restarts(service_id, since, until)
        .await
    {
        Ok(rows) => {
            let resp = RestartsResponse {
                restarts: rows
                    .into_iter()
                    .map(|(service, r)| ServiceRestartEntry {
                        service,
                        at_ms: r.at_ms,
                        trigger: r.trigger,
                        detail: r.detail,
                        run_id: r.run_id,
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => ApiErrorCode::QueryFailed {
            reason: e.to_string(),
        }
        .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct DeviceSamplesQuery {
    pub device: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

#[utoipa::path(
    summary = "Get device memory samples",
    get,
    path = "/api/devices/samples",
    params(
        ("device" = Option<String>, Query, description = "Filter to one device (e.g. \"gpu:0\", \"cpu\")"),
        ("since" = Option<i64>, Query, description = "Earliest timestamp_ms (default: 1h ago)"),
        ("until" = Option<i64>, Query, description = "Latest timestamp_ms (default: now)"),
    ),
    responses((status = 200, body = DeviceSamplesResponse), (status = 500, body = ApiError, description = "query_failed"))
)]
pub async fn get_device_samples(
    State(state): State<AppState>,
    Query(q): Query<DeviceSamplesQuery>,
) -> Response {
    let now = crate::tracking::now_unix_ms();
    let until = q.until.unwrap_or(now);
    let since = q.since.unwrap_or(now - 3_600_000);

    match state
        .db
        .query_device_samples(q.device.as_deref(), since, until)
        .await
    {
        Ok(samples) => {
            let resp = DeviceSamplesResponse {
                samples: samples
                    .into_iter()
                    .map(|s| DeviceSampleResponse {
                        device: s.device,
                        timestamp_ms: s.timestamp_ms,
                        total_bytes: s.total_bytes,
                        free_bytes: s.free_bytes,
                        used_bytes: s.used_bytes,
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => ApiErrorCode::QueryFailed {
            reason: e.to_string(),
        }
        .into_response(),
    }
}

pub fn register(router: Router, state: AppState) -> Router {
    let mgmt: Router = Router::new()
        .route("/api/metrics", get(get_metrics))
        .route("/api/restarts", get(get_restarts))
        .route("/api/devices/samples", get(get_device_samples))
        .with_state(state);
    router.merge(mgmt)
}
