//! Integration test: draining a service folds that run's observed peaks into
//! its rolling corrections, one pool at a time.
//!
//! This is the wiring the unit tests can't reach — `capture_rolling_base` at
//! the reservation commit, through the run, to `record_rolling_observation` at
//! the drain — and the place a regression would be invisible: every pool's
//! `update` is a no-op on a zero base, so a broken capture looks exactly like
//! a service that legitimately holds nothing.
//!
//! The service is `cpu-only` against a synthetic GGUF, which is the shape that
//! exercises the host pool: the whole model is host-resident weight, so it
//! clears the host learning floor, and the VRAM pool has no base at all.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use ananke::{
    api::openai,
    config::{PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{CpuSnapshot, DeviceSnapshot},
    supervise::drain::DrainReason,
    system::Fs,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness_with_snapshot, synth_gguf};
use smol_str::SmolStr;
use tower::util::ServiceExt;

const GIB: u64 = 1024 * 1024 * 1024;

fn service(model_path: PathBuf) -> ServiceConfig {
    let mut svc = common::minimal_llama_service("drainy", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::CpuOnly;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model_path;
    svc
}

/// A GGUF large enough that its host-resident weight clears the host pool's
/// learning floor — below that the ratio would be dominated by the process
/// footprint no reservation models, and the daemon deliberately declines to
/// learn from it.
fn big_cpu_model() -> Vec<u8> {
    // f16 → 2 bytes per element; 5 G elements over 4 tensors ≈ 20 GiB.
    let per_tensor = 2 * 1024 * 1024 * 1024 / 2;
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 8)
        .kv_u32("qwen3.context_length", 8192)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", per_tensor * 5)
        .tensor_f16("blk.1.attn_q.weight", per_tensor * 5)
        .tensor_f16("blk.2.attn_q.weight", per_tensor * 5)
        .tensor_f16("output.weight", per_tensor)
        .tensor_f16("token_embd.weight", per_tensor)
        .build()
}

#[tokio::test(flavor = "current_thread")]
async fn draining_records_the_host_pool_against_the_placement_that_ran() {
    let model_path = Path::new("/fake/drainy.gguf");
    let snapshot = DeviceSnapshot {
        gpus: vec![],
        cpu: Some(CpuSnapshot {
            total_bytes: 256 * GIB,
            available_bytes: 256 * GIB,
        }),
        taken_at_ms: 0,
    };

    let h = build_harness_with_snapshot(vec![service(model_path.to_path_buf())], snapshot).await;
    h.fs.write(model_path, &big_cpu_model()).unwrap();
    let svc = SmolStr::new("drainy");

    // Run the service, which commits a reservation and captures its base.
    let openai = openai::router(h.state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"drainy","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    assert_eq!(
        openai.oneshot(req).await.unwrap().status(),
        StatusCode::OK,
        "the service must start before there is anything to observe"
    );

    let reserved = h
        .state
        .allocations
        .lock()
        .get(&svc)
        .and_then(|row| row.get(&ananke::config::DeviceSlot::Cpu).copied())
        .expect("a cpu-only service must hold a CPU reservation")
        * 1024
        * 1024;
    // Comfortably above the host pool's learning floor, which is private to
    // the daemon; the margin is what keeps this test from tracking its value.
    assert!(
        reserved > 32 * GIB,
        "the fixture must reserve enough host weight to clear the learning floor ({reserved})"
    );

    // Stand in for the snapshotter: the service used 10% more host memory than
    // predicted. No VRAM was ever attributed, which is the truth for a
    // cpu-only service and must not be recorded as a ratio of zero.
    let observed = reserved + reserved / 10;
    h.state.observation.record_sample(&svc, 0, observed);

    assert_eq!(
        h.state.rolling.get(&svc).host.samples,
        0,
        "nothing is learned until the run ends"
    );

    let handle = h.state.registry.get(&svc).expect("registered supervisor");
    handle.begin_drain(DrainReason::UserKilled).await;

    let rc = h.state.rolling.get(&svc);
    assert_eq!(
        rc.vram.samples, 0,
        "a cpu-only service holds no VRAM, so its VRAM pool must stay untrained"
    );
    assert_eq!(
        rc.host.samples, 1,
        "the drain must record exactly one host sample"
    );
    assert!(
        rc.host.mean > 1.0,
        "a run that used more host memory than predicted must read as an \
         under-estimate, not the 0.8 floor an all-device base produced (mean {})",
        rc.host.mean
    );

    h.cleanup().await;
}
