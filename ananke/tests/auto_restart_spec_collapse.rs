//! Integration test: the spec-collapse watchdog drains and respawns a
//! service whose speculative draft acceptance has collapsed to zero.
//!
//! Mirrors the 2026-07-24 production incident: gemma-4-31b-it-qat served
//! healthy MTP traffic (60–100 % per-request acceptance) for an hour, then
//! flipped mid-run into an all-NaN-logits state — every completion an
//! endless `<unused49>` stream, every response's `timings` reporting
//! `draft_n > 0` with `draft_n_accepted = 0`. HTTP stayed 200 and tokens
//! kept flowing, so the error-rate and stall watchdogs never fired. Here
//! the wedge is simulated by recording the same metric shapes in
//! `request_metrics` for the running run; the watchdog's poll observes the
//! collapse and self-heals. Runs under `start_paused` so the poll interval
//! and cooldown advance virtually.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{
    api::openai,
    config::{AutoRestartSettings, ServiceConfig, SpecCollapseTrigger, TemplateConfig},
    supervise::state::ServiceState,
};
use ananke_db::models::RequestMetric;
use ananke_system::FakeProcessState;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap()
}

/// A llama-cpp service with speculative decoding configured and only the
/// spec-collapse watchdog enabled, with short spans so the virtual clock
/// reaches them quickly.
fn spec_service(min_draft_tokens: u64) -> ServiceConfig {
    let mut svc = minimal_llama_service("alpha", 0);
    // Keep the idle timeout well out of the way so only the watchdog can drain.
    svc.idle_timeout_ms = 600_000;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        panic!("minimal_llama_service is a llama-cpp service");
    };
    lc.spec_type = Some("draft-mtp".into());
    svc.auto_restart = AutoRestartSettings {
        spec_collapse: Some(SpecCollapseTrigger {
            window_ms: 120_000,
            min_draft_tokens,
            poll_interval_ms: 500,
        }),
        min_uptime_ms: 1_000,
        max_restarts: 3,
        flap_window_ms: 1_800_000,
        ..AutoRestartSettings::disabled()
    };
    svc
}

/// Record `n` drafting requests against `(service_id, run_id)`, each with
/// `draft` proposed and `accepted` accepted tokens, stamped ~now so they
/// land inside a default-length window.
async fn inject_drafting_requests(
    db: &ananke_db::Database,
    service_id: i64,
    run_id: i64,
    n: i64,
    draft: i64,
    accepted: i64,
) {
    inject_drafting_requests_at(db, service_id, run_id, n, draft, accepted, 0).await;
}

/// As [`inject_drafting_requests`], but stamped `age_ms` in the past.
/// `timestamp_ms` is wall-clock (`tracking::now_unix_ms`), which the paused
/// tokio clock does not advance — so a test that needs rows to sit outside
/// the watchdog's window must backdate them explicitly.
async fn inject_drafting_requests_at(
    db: &ananke_db::Database,
    service_id: i64,
    run_id: i64,
    n: i64,
    draft: i64,
    accepted: i64,
    age_ms: i64,
) {
    let now = ananke_time::now_unix_ms() - age_ms;
    for i in 0..n {
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            service_id,
            run_id: Some(run_id),
            timestamp_ms: now + i,
            endpoint: "/v1/chat/completions".into(),
            model: "alpha".into(),
            prompt_tokens: Some(13),
            completion_tokens: Some(32),
            prompt_eval_tokens: Some(13),
            duration_ms: Some(1000),
            ttft_ms: None,
            prompt_ms: Some(136),
            predicted_ms: Some(873),
            draft_tokens: Some(draft),
            draft_tokens_accepted: Some(accepted),
            status_code: 200,
        })
        .await
        .unwrap();
    }
}

async fn cold_start(h: &common::TestHarness, app: &axum::Router) -> (i64, i64) {
    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let sup = &h.supervisors[0];
    assert!(matches!(sup.peek_state(), ServiceState::Running));
    let run_id = sup.peek().run_id.expect("running has a run_id");
    let service_id = h
        .state
        .db
        .resolve_service_id("alpha")
        .await
        .unwrap()
        .unwrap();
    (service_id, run_id)
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn spec_collapse_watchdog_restarts_wedged_service() {
    let h = build_harness(vec![spec_service(200)]).await;
    let app = openai::router(h.state.clone());
    let (service_id, run_id) = cold_start(&h, &app).await;
    let sup = &h.supervisors[0];

    // The healthy hour, now aged past the 120 s window: acceptance well
    // above zero. In production these rows veto the watchdog until they age
    // out — roughly one window length after the flip — which is why they are
    // backdated here rather than co-resident with the collapsed rows.
    inject_drafting_requests_at(&h.state.db, service_id, run_id, 20, 59, 45, 200_000).await;

    // The flip: from 15:38:47 every drafting request rejects everything.
    // 48 wholly-rejected requests, the shape of the incident's probe traffic.
    inject_drafting_requests(&h.state.db, service_id, run_id, 48, 59, 0).await;

    // Advance past the cooldown and a poll tick; the watchdog fires and drains.
    let mut drained = false;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "spec-collapse watchdog did not drain to Idle; state = {:?}",
        sup.peek_state()
    );

    // The wedged child was terminated, and nothing respawns without a request.
    let children = h.process_spawner.children();
    assert_eq!(
        children.len(),
        1,
        "no respawn should happen without traffic"
    );
    assert!(
        matches!(
            children[0].state,
            FakeProcessState::SigTerm | FakeProcessState::SigKill
        ),
        "wedged child was not terminated; state = {:?}",
        children[0].state
    );

    // The firing is durably recorded, not just broadcast: the service-detail
    // surface reads this history back after the fact.
    let restarts = h
        .state
        .db
        .recent_service_restarts(service_id, 10)
        .await
        .unwrap();
    assert_eq!(restarts.len(), 1, "expected one persisted firing");
    assert_eq!(restarts[0].trigger, "spec_collapse");
    assert_eq!(restarts[0].run_id, Some(run_id));
    assert!(
        restarts[0].detail.contains("drafted tokens accepted"),
        "detail should carry the reason; got {:?}",
        restarts[0].detail
    );

    // A fresh request spawns a new run — the self-heal is complete, and the
    // fresh run's metrics start from zero so the watchdog does not re-fire.
    let resp2 = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    let new_run = sup.peek().run_id.expect("respawned run_id");
    assert_ne!(new_run, run_id, "expected a fresh run after auto-restart");

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn healthy_acceptance_never_triggers() {
    let h = build_harness(vec![spec_service(200)]).await;
    let app = openai::router(h.state.clone());
    let (service_id, run_id) = cold_start(&h, &app).await;
    let sup = &h.supervisors[0];

    // Sustained healthy traffic, including short generations that accept
    // little — but never a fully-zero window.
    for _ in 0..30 {
        tokio::time::advance(Duration::from_secs(5)).await;
        tokio::task::yield_now().await;
        inject_drafting_requests(&h.state.db, service_id, run_id, 3, 59, 0).await;
        inject_drafting_requests(&h.state.db, service_id, run_id, 1, 14, 12).await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "healthy traffic must not restart; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn all_zero_below_min_draft_tokens_never_triggers() {
    let h = build_harness(vec![spec_service(200)]).await;
    let app = openai::router(h.state.clone());
    let (service_id, run_id) = cold_start(&h, &app).await;
    let sup = &h.supervisors[0];

    // The run has accepted before (so the collapse precondition holds), but
    // the all-zero window sums to 99 drafted tokens — plausible for a few
    // unlucky short generations, and below the 200-token floor that makes
    // all-zero trustworthy.
    inject_drafting_requests_at(&h.state.db, service_id, run_id, 5, 59, 45, 200_000).await;
    inject_drafting_requests(&h.state.db, service_id, run_id, 9, 11, 0).await;
    for _ in 0..10 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "below-floor all-zero must not restart; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn run_that_never_accepted_never_triggers() {
    // A workload whose draft acceptance is legitimately zero from the start
    // of the run — e.g. grammar-constrained speculative decoding, where the
    // draft's proposals are rejected for violating the grammar — must not
    // trip the watchdog: zero is its baseline, not a collapse.
    let h = build_harness(vec![spec_service(200)]).await;
    let app = openai::router(h.state.clone());
    let (service_id, run_id) = cold_start(&h, &app).await;
    let sup = &h.supervisors[0];

    inject_drafting_requests(&h.state.db, service_id, run_id, 48, 59, 0).await;
    for _ in 0..20 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "a run with no prior acceptance must not restart; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn non_spec_service_never_polls_even_with_trigger_configured() {
    // The trigger is configured (as the llama-cpp default will be) but the
    // service has no `spec_type` — the runtime gate must keep the watchdog
    // off entirely. Belt: even if rows with draft counts somehow appeared,
    // no poll runs to see them.
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = AutoRestartSettings {
        spec_collapse: Some(SpecCollapseTrigger {
            window_ms: 120_000,
            min_draft_tokens: 200,
            poll_interval_ms: 500,
        }),
        min_uptime_ms: 1_000,
        max_restarts: 3,
        flap_window_ms: 1_800_000,
        ..AutoRestartSettings::disabled()
    };

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let (service_id, run_id) = cold_start(&h, &app).await;
    let sup = &h.supervisors[0];

    inject_drafting_requests(&h.state.db, service_id, run_id, 48, 59, 0).await;
    for _ in 0..20 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "a service without spec_type must never spec-collapse-restart; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn repeated_collapses_trip_flap_cap_and_disable() {
    let mut svc = spec_service(200);
    // One restart tolerated; the second collapse disables instead.
    svc.auto_restart.max_restarts = 1;

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let sup = &h.supervisors[0];

    // First cycle: cold-start, healthy acceptance, then collapse, and
    // confirm the watchdog restarts.
    let (service_id, run_a) = cold_start(&h, &app).await;
    inject_drafting_requests_at(&h.state.db, service_id, run_a, 5, 59, 45, 200_000).await;
    inject_drafting_requests(&h.state.db, service_id, run_a, 12, 59, 0).await;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            break;
        }
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Idle),
        "first collapse should restart to Idle; state = {:?}",
        sup.peek_state()
    );

    // Second cycle: respawn, healthy acceptance again, then a second
    // collapse. This trips the flap cap and the service is disabled rather
    // than restarted a second time. (Without renewed acceptance the fresh
    // run could not re-fire at all — that is the transition requirement's
    // flap protection, covered by `run_that_never_accepted_never_triggers`.)
    let resp2 = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let run_b = sup.peek().run_id.expect("run B");
    assert_ne!(run_b, run_a);
    inject_drafting_requests_at(&h.state.db, service_id, run_b, 5, 59, 45, 200_000).await;
    inject_drafting_requests(&h.state.db, service_id, run_b, 12, 59, 0).await;

    let mut disabled = false;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Disabled { .. }) {
            disabled = true;
            break;
        }
    }
    assert!(
        disabled,
        "second collapse should trip the flap cap and disable; state = {:?}",
        sup.peek_state()
    );
    assert!(
        matches!(
            sup.peek_state(),
            ServiceState::Disabled {
                reason: ananke::supervise::state::DisableReason::AutoRestartLoop
            }
        ),
        "expected AutoRestartLoop disable reason; state = {:?}",
        sup.peek_state()
    );

    // Both firings are persisted, including the one that tripped the cap —
    // it is the one that took the service down, so it is the one an operator
    // goes looking for. It is recorded but not broadcast as `auto_restarted`,
    // since nothing was restarted; the `Disabled` state change carries that.
    let restarts = h
        .state
        .db
        .recent_service_restarts(service_id, 10)
        .await
        .unwrap();
    assert_eq!(restarts.len(), 2, "expected both firings persisted");
    assert!(
        restarts[0].detail.contains("flap cap reached"),
        "newest firing should record the disable; got {:?}",
        restarts[0].detail
    );
    assert_eq!(restarts[0].run_id, Some(run_b));
    assert_eq!(restarts[1].run_id, Some(run_a));

    h.cleanup().await;
}
