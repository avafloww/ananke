//! Handlers for /v1/models and the three POST body-rewriting endpoints.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::time::Instant;

use ananke_api::{
    openai::{
        ChatCompletionEnvelope, CompletionEnvelope, EmbeddingEnvelope, ModelListing, ModelsResponse,
    },
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{Router, get, post},
};
use bytes::Bytes;
use serde_json::Value;
use tracing::{info, warn};

use crate::{
    api::openai::{
        audio, errors, filters,
        forward::{UpstreamPost, ensure_ready, forward_upstream},
    },
    daemon::app_state::AppState,
    supervise::state::ServiceState,
};

pub fn register(router: Router, state: AppState) -> Router {
    // Build the main routes against AppState, collapse to Router<()>, then
    // merge the unimplemented stubs (which already return Router<()>) and
    // the caller's router. The literal /v1/audio/transcriptions route takes
    // precedence over the stubs' /v1/audio/*rest wildcard (static paths win
    // in axum's matchit router), so the other audio endpoints keep
    // returning 501.
    let implemented: Router = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route(
            "/v1/audio/transcriptions",
            post(audio::audio_transcriptions),
        )
        .with_state(state.clone());
    let stubs: Router = crate::api::openai::unimplemented::register(Router::new(), state);
    router.merge(implemented).merge(stubs)
}

#[utoipa::path(
    summary = "List available models (OpenAI-compatible)",get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn list_models(State(state): State<AppState>) -> Response {
    let mut data = Vec::new();
    let effective = state.config.effective();
    for (name, handle) in state.registry.all() {
        // Hide services whose template doesn't speak OpenAI. llama-cpp
        // does; `command` services opt in via openai_proxy or the
        // transcription modality.
        let Some(svc) = effective.services.iter().find(|s| s.name == name) else {
            continue;
        };
        if !svc.openai_compat {
            continue;
        }
        match handle.peek_state() {
            ServiceState::Idle | ServiceState::Starting | ServiceState::Running => {
                data.push(ModelListing::new(
                    name.to_string(),
                    svc.modality,
                    svc.metadata.clone(),
                ));
            }
            _ => {}
        }
    }
    let body = ModelsResponse::new(data);
    (StatusCode::OK, Json(body)).into_response()
}

#[utoipa::path(
    summary = "Chat completion (OpenAI-compatible proxy)",
    post,
    path = "/v1/chat/completions",
    request_body = ChatCompletionEnvelope,
    responses(
        (status = 200, description = "Proxied from upstream"),
        (status = 400, body = ApiError, description = "invalid_request_error"),
        (status = 404, body = ApiError, description = "model_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked"),
        (status = 502, body = ApiError, description = "upstream_unavailable")
    )
)]
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/chat/completions", state, headers, body).await
}

#[utoipa::path(
    summary = "Text completion (OpenAI-compatible proxy)",
    post,
    path = "/v1/completions",
    request_body = CompletionEnvelope,
    responses(
        (status = 200, description = "Proxied from upstream"),
        (status = 400, body = ApiError, description = "invalid_request_error"),
        (status = 404, body = ApiError, description = "model_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked"),
        (status = 502, body = ApiError, description = "upstream_unavailable")
    )
)]
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/completions", state, headers, body).await
}

#[utoipa::path(
    summary = "Embeddings (OpenAI-compatible proxy)",
    post,
    path = "/v1/embeddings",
    request_body = EmbeddingEnvelope,
    responses(
        (status = 200, description = "Proxied from upstream"),
        (status = 400, body = ApiError, description = "invalid_request_error"),
        (status = 404, body = ApiError, description = "model_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked"),
        (status = 502, body = ApiError, description = "upstream_unavailable")
    )
)]
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/embeddings", state, headers, body).await
}

async fn forward_json_post(
    path: &'static str,
    state: AppState,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let request_start = Instant::now();
    // Untyped on purpose: the body belongs to the client and is forwarded
    // verbatim. The daemon reads `model` and `stream` and writes back at
    // most `model` and `timings_per_token`; every other key has to survive
    // the round trip unexamined, which a struct would not let it do.
    let mut parsed: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            warn!(endpoint = path, error = %e, "request rejected: invalid JSON body");
            return errors::bad_request(format!("invalid JSON body: {e}"));
        }
    };
    let model = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => {
            warn!(endpoint = path, "request rejected: missing `model` field");
            return errors::bad_request("request body missing `model` field");
        }
    };

    info!(model = %model, endpoint = path, "openai request received");

    let handle = match state.registry.get(&model) {
        Some(h) => h,
        None => {
            warn!(model = %model, endpoint = path, "request rejected: model not found in registry");
            return errors::not_found_model(&model);
        }
    };

    let eff = state.config.effective();
    let svc = eff.services.iter().find(|s| s.name == model);
    let Some(svc) = svc else {
        warn!(model = %model, endpoint = path, "request rejected: model not found in effective config");
        return errors::not_found_model(&model);
    };

    if let Err(resp) = ensure_ready(&handle, svc, &model, path).await {
        return resp;
    }

    // Apply filters.
    filters::apply(&mut parsed, &svc.filters);
    // For openai-proxy command services, rewrite the JSON `model` field
    // to the upstream's expected name. Runs after filters so the proxy
    // rewrite is the last word — operators can use filters to mangle
    // other JSON keys, but `model` is whatever `openai_proxy.upstream_model`
    // says.
    if let Some(cmd) = svc.command()
        && let Some(proxy) = &cmd.openai_proxy
    {
        parsed["model"] = Value::String(proxy.upstream_model.to_string());
    }
    // For spec-decoding llama-cpp services, ask llama.cpp to attach its
    // cumulative `timings` (including the draft counts) to every streamed
    // chunk instead of only the final one. An aborted stream never receives
    // the final chunk, and a garbage-generation wedge produces exactly that
    // traffic — long generations cut off by client timeouts — so without
    // per-chunk timings the spec-collapse watchdog gets no evidence from
    // the requests that matter most. An explicit client value wins.
    if parsed
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        && svc.llama_cpp().is_some_and(|lc| lc.spec_type.is_some())
        && let Some(obj) = parsed.as_object_mut()
    {
        obj.entry("timings_per_token").or_insert(Value::Bool(true));
    }
    let new_body = match serde_json::to_vec(&parsed) {
        Ok(b) => b,
        Err(e) => return errors::bad_request(format!("re-serialise failed: {e}")),
    };

    let is_streaming = parsed
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    forward_upstream(UpstreamPost {
        state: &state,
        svc,
        handle: &handle,
        model: &model,
        path,
        headers: &headers,
        content_type: HeaderValue::from_static("application/json"),
        body: Bytes::from(new_body),
        is_streaming,
        request_start,
    })
    .await
}
