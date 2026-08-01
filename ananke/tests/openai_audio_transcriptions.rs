#![cfg(feature = "test-fakes")]
mod common;

use ananke::api::openai;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use bytes::Bytes;
use common::{build_harness, minimal_llama_service, minimal_transcription_service};
use tower::util::ServiceExt;

const BOUNDARY: &str = "ananke-test-boundary";

/// Build a multipart transcription body with a fake WAV part and, when
/// `model` is `Some`, a trailing `model` field (mirroring OpenAI clients,
/// which commonly put the file first).
fn multipart_body(model: Option<&str>) -> Bytes {
    let mut out = format!(
        "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\ncontent-type: audio/wav\r\n\r\nRIFF0000WAVEfmt \r\n"
    );
    if let Some(m) = model {
        out.push_str(&format!(
            "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"model\"\r\n\r\n{m}\r\n"
        ));
    }
    out.push_str(&format!("--{BOUNDARY}--\r\n"));
    Bytes::from(out)
}

fn multipart_request(body: Bytes) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn transcription_routes_and_forwards_body_verbatim() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let body = multipart_body(Some("parakeet"));
    let resp = app.oneshot(multipart_request(body.clone())).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(parsed["text"], "the quick brown fox");

    // The upstream must receive the body byte-for-byte with its original
    // multipart content type — no rewrite, no re-encoding.
    let sunk = h.echo_state.raw_sink.lock().clone();
    assert_eq!(sunk.len(), 1);
    assert_eq!(
        sunk[0].0,
        format!("multipart/form-data; boundary={BOUNDARY}")
    );
    assert_eq!(sunk[0].1, body);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn transcription_missing_model_field_400() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let resp = app
        .oneshot(multipart_request(multipart_body(None)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn transcription_unknown_model_404() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let resp = app
        .oneshot(multipart_request(multipart_body(Some("nope"))))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn transcription_rejects_non_transcription_service() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());
    let resp = app
        .oneshot(multipart_request(multipart_body(Some("alpha"))))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn transcription_non_multipart_body_400() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"parakeet"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    h.cleanup().await;
}

/// The literal route now owns the path: a GET is a method mismatch (405),
/// not the wildcard stub's 501.
#[tokio::test(flavor = "current_thread")]
async fn transcription_get_is_405_not_501() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let req = Request::builder()
        .method("GET")
        .uri("/v1/audio/transcriptions")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    h.cleanup().await;
}

/// Sibling audio endpoints stay on the 501 stub.
#[tokio::test(flavor = "current_thread")]
async fn audio_translations_still_501() {
    let h = build_harness(vec![minimal_transcription_service("parakeet", 0)]).await;
    let app = openai::router(h.state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/translations")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    h.cleanup().await;
}
