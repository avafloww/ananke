//! Integration test: llama.cpp-native endpoints (`/tokenize`,
//! `/apply-template`, `/props`, `/slots`, …) reached through the main
//! OpenAI listener are forwarded verbatim to the upstream of a
//! `llama-cpp` template service.
//!
//! Assertions cover the routing rules: a `model` field in the JSON body
//! picks the service; a request without one falls back to the sole
//! llama-cpp service; naming a non-llama-cpp service or running without
//! any llama-cpp service is a structured error. The echo server records
//! every request it receives, so "verbatim" is checked exactly (method,
//! path, and body bytes reach the upstream untouched).
#![cfg(feature = "test-fakes")]

mod common;

use std::collections::BTreeMap;

use ananke::{
    api::openai,
    config::{
        AllocationMode, CommandConfig, TemplateConfig,
        parse::DEFAULT_START_QUEUE_DEPTH,
        validate::{
            AutoRestartSettings, DeviceReserves, DeviceSlot, Filters, HealthSettings, Lifecycle,
            PlacementPolicy, ServiceConfig, SplitMode,
        },
    },
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

fn command_service(name: &str) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Cpu, 100);
    ServiceConfig {
        name: SmolStr::new(name),
        port: 0,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: None,
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: placement,
        placement_policy: PlacementPolicy::CpuOnly,
        gpu_allow: Vec::new(),
        split_mode: SplitMode::Layer,
        tensor_split_weights: None,
        gpu_headroom_mb: 0,
        reserves: std::sync::Arc::new(DeviceReserves::default()),
        idle_timeout_ms: 60_000,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        auto_restart: AutoRestartSettings::disabled(),
        filters: Filters::default(),
        allocation_mode: AllocationMode::Static { reserve_mb: 100 },
        openai_compat: false,
        description: None,
        modality: ananke_api::shared::Modality::Chat,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        env_inherit: true,
        tracking: ananke::config::TrackingSettings::default(),
        metadata: ananke_api::shared::AnankeMetadata::new(),
        template_config: TemplateConfig::Command(CommandConfig {
            command: vec!["true".into()],
            workdir: None,
            shutdown_command: None,
            private_port_override: None,
            openai_proxy: None,
        }),
        container: None,
    }
}

async fn raw(app: axum::Router, method: &str, path: &str, body: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn native_request_forwards_verbatim_to_unique_llama_service() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());

    // A JSON body with no `model` falls back to the sole llama-cpp service.
    let (st, body) = raw(app.clone(), "POST", "/tokenize", r#"{"content":"hi"}"#).await;
    assert_eq!(
        st,
        StatusCode::OK,
        "tokenize must reach the upstream: {body}"
    );
    assert_eq!(body, "hello");

    // The upstream saw the exact method, path, and bytes — nothing rewritten.
    let seen = h
        .echo_state
        .requests
        .lock()
        .iter()
        .find(|r| r.path == "/tokenize")
        .cloned()
        .expect("echo must have received the /tokenize request");
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.body, r#"{"content":"hi"}"#);

    // The multiparameter and introspection GETs work the same way.
    let (st, _) = raw(app, "GET", "/props", "").await;
    assert_eq!(st, StatusCode::OK, "props must be forwarded");
}

#[tokio::test(flavor = "multi_thread")]
async fn model_field_selects_between_multiple_llama_services() {
    let h = build_harness(vec![
        minimal_llama_service("alpha", 0),
        minimal_llama_service("beta", 0),
    ])
    .await;
    let app = openai::router(h.state.clone());

    let (st, body) = raw(
        app.clone(),
        "POST",
        "/tokenize",
        r#"{"model":"beta","content":"x"}"#,
    )
    .await;
    assert_eq!(
        st,
        StatusCode::OK,
        "a body model must select the service: {body}"
    );

    // Without a `model`, two candidates is ambiguous and must be named.
    let (st, body) = raw(app, "POST", "/tokenize", r#"{"content":"x"}"#).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("multiple llama-cpp services")
            && body.contains("alpha")
            && body.contains("beta"),
        "ambiguity error should name both services: {body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn model_naming_a_command_service_is_rejected() {
    let h = build_harness(vec![
        minimal_llama_service("alpha", 0),
        command_service("beta"),
    ])
    .await;
    let app = openai::router(h.state.clone());

    let (st, body) = raw(
        app.clone(),
        "POST",
        "/tokenize",
        r#"{"model":"beta","content":"x"}"#,
    )
    .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("does not use the llama-cpp template"),
        "command-template model must be rejected: {body}"
    );

    // The single llama-cpp service still serves model-less requests.
    let (st, _) = raw(app, "POST", "/tokenize", r#"{"content":"x"}"#).await;
    assert_eq!(st, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn unmapped_model_is_404_and_slots_routes_coexist_with_v1() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());

    let (st, body) = raw(app.clone(), "POST", "/tokenize", r#"{"model":"nope"}"#).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "unknown model: {body}");

    // `/slots/:id` (slot save/load) and the OpenAI surface both still work.
    let (st, _) = raw(app.clone(), "POST", "/slots/0", "{}").await;
    assert_eq!(st, StatusCode::OK);
    let (st, _) = raw(app.clone(), "GET", "/slots", "").await;
    assert_eq!(st, StatusCode::OK);
    let (st, _) = raw(
        app,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"alpha","messages":[]}"#,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "the OpenAI surface must be unaffected");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_llama_cpp_service_is_a_structured_error() {
    let h = build_harness(vec![command_service("beta")]).await;
    let app = openai::router(h.state.clone());

    let (st, body) = raw(app.clone(), "POST", "/tokenize", r#"{"content":"x"}"#).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(body.contains("no llama-cpp service is configured"));

    let (st, _) = raw(app, "GET", "/props", "").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}
