#![cfg(feature = "test-fakes")]
mod common;

use ananke::api::management;
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn openapi_json_is_valid() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/openapi.json")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), 10 * 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["openapi"].as_str().unwrap_or("").chars().next(),
        Some('3')
    );
    let paths = parsed["paths"].as_object().expect("paths object");
    assert!(paths.contains_key("/v1/models"));
    assert!(paths.contains_key("/api/services"));
    assert!(paths.contains_key("/api/devices"));

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn openapi_contains_container_preview_variants() {
    // The frontend consumes generated types, so a container field that never
    // reaches the schema is a field the UI cannot render no matter what the
    // daemon sends.
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/openapi.json")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 10 * 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let schemas = parsed["components"]["schemas"]
        .as_object()
        .expect("component schemas");

    for name in [
        "LaunchCommand",
        "LaunchContainer",
        "LaunchMount",
        "ContainerDetail",
    ] {
        assert!(schemas.contains_key(name), "schema `{name}` is missing");
    }

    // `container` discriminates the launch-preview union, and is optional so
    // the process variant stays wire-compatible.
    let launch = &schemas["LaunchCommand"];
    assert!(
        launch["properties"].get("container").is_some(),
        "LaunchCommand must expose the container discriminator"
    );
    let required: Vec<&str> = launch["required"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    assert!(
        !required.contains(&"container"),
        "container must stay optional so process consumers are unaffected"
    );

    // Every container preview field the UI renders is described.
    let container = &schemas["LaunchContainer"]["properties"];
    for field in [
        "runtime",
        "image",
        "name_pattern",
        "argv",
        "env",
        "env_passthrough",
        "mounts",
        "network",
        "publication",
        "ipc",
        "gpu_devices",
        "create_argv",
    ] {
        assert!(
            container.get(field).is_some(),
            "LaunchContainer is missing `{field}`"
        );
    }

    h.cleanup().await;
}
