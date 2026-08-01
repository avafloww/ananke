//! `POST /v1/audio/transcriptions` — the multipart-aware proxy route.
//!
//! Unlike the JSON endpoints, the request body is multipart/form-data, so
//! the `model` field used for routing is a form part rather than a JSON
//! key. The body is parsed only far enough to read that field and is then
//! forwarded byte-for-byte with its original content type: every supported
//! upstream (parakeet-server, whisper-server) ignores the `model` part, so
//! no rewrite — and therefore no multipart re-encoding — is needed.

use std::time::Instant;

use ananke_api::{
    openai::TranscriptionEnvelope,
    shared::{errors::ApiError, modality::Modality},
};
use axum::{extract::State, http::HeaderMap, response::Response};
use bytes::Bytes;
use tracing::{info, warn};

use crate::{
    api::openai::{
        errors,
        forward::{UpstreamPost, ensure_ready, forward_upstream},
    },
    daemon::app_state::AppState,
};

#[utoipa::path(
    summary = "Audio transcription (OpenAI-compatible proxy)",
    post,
    path = "/v1/audio/transcriptions",
    request_body(content = TranscriptionEnvelope, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Proxied from upstream"),
        (status = 400, body = ApiError, description = "invalid_request_error"),
        (status = 404, body = ApiError, description = "model_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked"),
        (status = 502, body = ApiError, description = "upstream_unavailable")
    )
)]
pub async fn audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    const PATH: &str = "/v1/audio/transcriptions";
    let request_start = Instant::now();

    let Some(content_type) = headers.get(hyper::header::CONTENT_TYPE).cloned() else {
        warn!(endpoint = PATH, "request rejected: missing content-type");
        return errors::bad_request("missing content-type header (expected multipart/form-data)");
    };
    let boundary = match content_type
        .to_str()
        .map_err(|e| e.to_string())
        .and_then(|ct| multer::parse_boundary(ct).map_err(|e| e.to_string()))
    {
        Ok(b) => b,
        Err(e) => {
            warn!(endpoint = PATH, error = %e, "request rejected: not multipart");
            return errors::bad_request(format!(
                "invalid content type (expected multipart/form-data with a boundary): {e}"
            ));
        }
    };

    let model = match extract_model_field(body.clone(), boundary).await {
        Ok(Some(m)) => m,
        Ok(None) => {
            warn!(endpoint = PATH, "request rejected: missing `model` field");
            return errors::bad_request("multipart body missing `model` field");
        }
        Err(e) => {
            warn!(endpoint = PATH, error = %e, "request rejected: malformed multipart body");
            return errors::bad_request(format!("malformed multipart body: {e}"));
        }
    };

    info!(model = %model, endpoint = PATH, "openai request received");

    let handle = match state.registry.get(&model) {
        Some(h) => h,
        None => {
            warn!(model = %model, endpoint = PATH, "request rejected: model not found in registry");
            return errors::not_found_model(&model);
        }
    };

    let eff = state.config.effective();
    let Some(svc) = eff.services.iter().find(|s| s.name == model) else {
        warn!(model = %model, endpoint = PATH, "request rejected: model not found in effective config");
        return errors::not_found_model(&model);
    };
    if svc.modality != Modality::Transcription {
        warn!(model = %model, endpoint = PATH, "request rejected: not a transcription service");
        return errors::bad_request(format!(
            "model `{model}` does not serve audio transcription (service modality is not `transcription`)"
        ));
    }

    if let Err(resp) = ensure_ready(&handle, svc, &model, PATH).await {
        return resp;
    }

    forward_upstream(UpstreamPost {
        state: &state,
        svc,
        handle: &handle,
        model: &model,
        path: PATH,
        headers: &headers,
        content_type,
        body,
        is_streaming: false,
        request_start,
    })
    .await
}

/// Parse the multipart body just far enough to find the `model` form field
/// and return its text. `Ok(None)` when no `model` part exists. The `Bytes`
/// clone in the caller is a refcount bump, so parsing here never copies the
/// (potentially large) audio payload.
async fn extract_model_field(
    body: Bytes,
    boundary: String,
) -> Result<Option<String>, multer::Error> {
    let stream = futures::stream::once(async move { Ok::<Bytes, std::convert::Infallible>(body) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    while let Some(field) = multipart.next_field().await? {
        if field.name() == Some("model") {
            return Ok(Some(field.text().await?));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDARY: &str = "test-boundary";

    fn multipart_body(fields: &[(&str, &str)]) -> Bytes {
        let mut out = String::new();
        for (name, value) in fields {
            out.push_str(&format!(
                "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        out.push_str(&format!("--{BOUNDARY}--\r\n"));
        Bytes::from(out)
    }

    #[tokio::test]
    async fn finds_model_field() {
        let body = multipart_body(&[("model", "parakeet"), ("response_format", "json")]);
        let got = extract_model_field(body, BOUNDARY.into()).await.unwrap();
        assert_eq!(got.as_deref(), Some("parakeet"));
    }

    #[tokio::test]
    async fn finds_model_after_file_part() {
        // OpenAI clients commonly put the file part first; the parser must
        // walk past it rather than stopping at the first field.
        let mut out = format!(
            "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\ncontent-type: audio/wav\r\n\r\nRIFFxxxxWAVE\r\n"
        );
        out.push_str(&format!(
            "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"model\"\r\n\r\nparakeet\r\n--{BOUNDARY}--\r\n"
        ));
        let got = extract_model_field(Bytes::from(out), BOUNDARY.into())
            .await
            .unwrap();
        assert_eq!(got.as_deref(), Some("parakeet"));
    }

    #[tokio::test]
    async fn missing_model_field_is_none() {
        let body = multipart_body(&[("response_format", "json")]);
        let got = extract_model_field(body, BOUNDARY.into()).await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn malformed_body_is_an_error() {
        let body = Bytes::from_static(b"this is not multipart at all");
        let got = extract_model_field(body, BOUNDARY.into()).await;
        assert!(got.is_err());
    }
}
