//! Tests for the three placement-preview paths in
//! [`crate::supervise::preview::verdict`]: the estimator packer, a manual
//! `placement_override`, and a command-template GPU pick.

use std::collections::BTreeMap;

use ananke_gguf::Architecture;
use smol_str::SmolStr;

use super::*;
use crate::{
    config::validate::test_fixtures::{minimal_command_service, minimal_service},
    devices::{CpuSnapshot, GpuSnapshot},
    estimator::NonLayer,
    supervise::preview::preview_command,
    system::SystemDeps,
};

fn two_gpu_snapshot() -> DeviceSnapshot {
    gpus_with_free(24)
}

/// Two 24 GiB GPUs with `free_gib` free each.
fn gpus_with_free(free_gib: u64) -> DeviceSnapshot {
    let free = free_gib << 30;
    DeviceSnapshot {
        gpus: vec![
            GpuSnapshot {
                id: 0,
                name: "gpu0".into(),
                total_bytes: 24 << 30,
                free_bytes: free,
            },
            GpuSnapshot {
                id: 1,
                name: "gpu1".into(),
                total_bytes: 24 << 30,
                free_bytes: free,
            },
        ],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 << 30,
            available_bytes: 64 << 30,
        }),
        taken_at_ms: 0,
    }
}

/// A GPU-only llama service (the fixture defaults to CPU-only with an
/// override; clear both so placement actually packs onto GPUs).
fn llama_svc() -> ServiceConfig {
    let mut s = minimal_service("m");
    s.placement_override.clear();
    s.placement_policy = PlacementPolicy::GpuOnly;
    s
}

/// `n_layers × per_gib` GiB of pure layer weights — no KV, MTP, or
/// compute buffer — so the fit maths is easy to reason about.
fn estimate_gib(n_layers: u32, per_gib: u64) -> Estimate {
    let per = per_gib << 30;
    Estimate {
        weights_bytes: per * n_layers as u64,
        kv_per_token: 0,
        compute_buffer_mb: 0,
        output_buffer_bytes: 0,
        mtp_bytes: 0,
        mtp_weight_bytes: 0,
        mmproj_graph_bytes: 0,
        mtp_head_expert_layers: 0,
        tensor_split_replicated_bytes: 0,
        host_overhead_bytes: 0,
        host_cache_bytes: 0,
        host_slot_bytes: 0,
        host_checkpoint_bytes: 0,
        per_layer_bytes: Some(vec![per; n_layers as usize]),
        attention_layers: None,
        non_layer: NonLayer::default(),
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_tensors: None,
        context: 4096,
        architecture: Architecture::Qwen3,
    }
}

/// A command-template preview picks the GPU with headroom in the pledge
/// book, renders the chosen device into `CUDA_VISIBLE_DEVICES`, and
/// substitutes argv placeholders — without spawning anything.
#[test]
fn command_preview_picks_free_gpu_and_renders_env() {
    let mut svc = minimal_command_service(
        "comfy",
        vec!["comfyui-start".into(), "--port".into(), "{port}".into()],
    );
    svc.placement_override.clear();
    svc.placement_policy = PlacementPolicy::GpuOnly;
    svc.allocation_mode = AllocationMode::Static { reserve_mb: 4096 };
    svc.private_port = 8200;

    // Peer holds 23 GiB on GPU 0 (in MB), so only GPU 1 can fit the 4 GiB
    // reservation — the optimistic planner reads the pledge book.
    let mut table = AllocationTable::new();
    let mut peer = BTreeMap::new();
    peer.insert(DeviceSlot::Gpu(0), 23_000u64);
    table.insert(SmolStr::new("peer"), peer);

    let snap = two_gpu_snapshot();
    let (deps, _fakes) = SystemDeps::fake();
    let cfg = preview_command(&svc, &snap, &table, deps.fs.as_ref(), Corrections::NEUTRAL)
        .expect("command preview must succeed");

    assert_eq!(cfg.binary, "comfyui-start");
    assert_eq!(
        cfg.env.get("CUDA_VISIBLE_DEVICES").map(String::as_str),
        Some("1"),
        "must pick GPU 1 (GPU 0 is pledged out); env={:?}",
        cfg.env
    );
    assert!(
        cfg.args.contains(&"8200".to_string()),
        "the {{port}} placeholder must be substituted; got {:?}",
        cfg.args
    );
}

/// 20 GiB across two empty 24 GiB cards fits in currently-free VRAM.
#[test]
fn placement_fits_in_free_vram() {
    let out = preview_placement(
        &llama_svc(),
        &estimate_gib(20, 1),
        &gpus_with_free(24),
        &AllocationTable::new(),
        false,
        Corrections::NEUTRAL,
    );
    assert_eq!(out.verdict, FitVerdict::Fits);
    assert!(!out.devices.is_empty(), "a fitting placement names devices");
}

/// Same 20 GiB, but the cards are nearly full (1 GiB free each): it fits
/// within total capacity, so the daemon would reclaim/evict — and the
/// would-be shape is still reported.
#[test]
fn placement_needs_eviction_when_free_is_low() {
    let out = preview_placement(
        &llama_svc(),
        &estimate_gib(20, 1),
        &gpus_with_free(1),
        &AllocationTable::new(),
        false,
        Corrections::NEUTRAL,
    );
    assert_eq!(out.verdict, FitVerdict::NeedsEviction);
    assert!(
        !out.devices.is_empty(),
        "needs-eviction still shows where it would land"
    );
}

/// 60 GiB can't fit on two 24 GiB cards even with everything else gone.
#[test]
fn placement_does_not_fit_when_too_large() {
    let out = preview_placement(
        &llama_svc(),
        &estimate_gib(60, 1),
        &gpus_with_free(24),
        &AllocationTable::new(),
        false,
        Corrections::NEUTRAL,
    );
    let FitVerdict::DoesNotFit { shortfalls } = &out.verdict else {
        panic!("expected DoesNotFit, got {:?}", out.verdict);
    };
    // Both allowed cards are named, with ids that cross-reference
    // `GET /api/devices`, and each is short of what the layer needed.
    assert_eq!(
        shortfalls
            .iter()
            .map(|s| s.device.as_str())
            .collect::<Vec<_>>(),
        vec!["gpu:0", "gpu:1"],
        "got {shortfalls:?}"
    );
    assert!(
        shortfalls
            .iter()
            .all(|s| s.available_bytes < s.requested_bytes),
        "every shortfall reports less available than requested, got {shortfalls:?}"
    );
    assert!(
        out.devices.is_empty(),
        "no valid placement names no devices"
    );
}

/// A hybrid model whose spill exceeds host RAM is unplaceable for a reason
/// that has nothing to do with the GPUs — the deepseek-v4-flash case, where
/// both cards were empty and RAM was the binding constraint. The verdict
/// must name `cpu`, not leave the operator inferring a VRAM problem.
#[test]
fn placement_that_overruns_host_ram_names_the_cpu() {
    let mut svc = llama_svc();
    svc.placement_policy = PlacementPolicy::Hybrid;
    // 60 GiB over two empty 24 GiB cards spills ~20 GiB to a host with 2.
    let mut snap = gpus_with_free(24);
    snap.cpu = Some(CpuSnapshot {
        total_bytes: 2 << 30,
        available_bytes: 2 << 30,
    });

    let out = preview_placement(
        &svc,
        &estimate_gib(60, 1),
        &snap,
        &AllocationTable::new(),
        false,
        Corrections::NEUTRAL,
    );
    let FitVerdict::DoesNotFit { shortfalls } = &out.verdict else {
        panic!("expected DoesNotFit, got {:?}", out.verdict);
    };
    assert_eq!(
        shortfalls
            .iter()
            .map(|s| s.device.as_str())
            .collect::<Vec<_>>(),
        vec!["cpu"],
        "host RAM is the binding constraint, not the GPUs; got {shortfalls:?}"
    );
    assert!(
        shortfalls[0].requested_bytes > shortfalls[0].available_bytes,
        "got {shortfalls:?}"
    );
}

/// A running service is reported as fitting even when free VRAM is low —
/// it is demonstrably placed, and the low free is its own resident VRAM.
#[test]
fn running_service_always_fits() {
    let out = preview_placement(
        &llama_svc(),
        &estimate_gib(20, 1),
        &gpus_with_free(1),
        &AllocationTable::new(),
        true,
        Corrections::NEUTRAL,
    );
    assert_eq!(out.verdict, FitVerdict::Fits);
}

/// An override service reports the override as its per-device split, with a
/// verdict checked against the live snapshot the same way the packer path
/// is — without running the estimator at all.
#[test]
fn override_placement_uses_override_and_checks_fit() {
    let mut svc = minimal_service("ov");
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Gpu(1), 8000); // MB
    let table = AllocationTable::new();

    let fits = preview_override_placement(&svc, &gpus_with_free(24), &table, false);
    assert_eq!(fits.verdict, FitVerdict::Fits);
    assert_eq!(
        fits.devices.get(&DeviceId::Gpu(1)).copied(),
        Some(8000 * 1024 * 1024),
        "override MB is reported as bytes on the declared device"
    );

    // Cards nearly full: fits the hardware but not currently-free VRAM.
    let tight = preview_override_placement(&svc, &gpus_with_free(1), &table, false);
    assert_eq!(tight.verdict, FitVerdict::NeedsEviction);

    // A running override service is always reported as fitting.
    let live = preview_override_placement(&svc, &gpus_with_free(1), &table, true);
    assert_eq!(live.verdict, FitVerdict::Fits);
}

/// A dynamic command-template service (e.g. ComfyUI) reserves its `min_mb`
/// on the GPU with headroom — here GPU 1, since GPU 0 is nearly full.
#[test]
fn command_placement_reserves_min_on_picked_gpu() {
    let mut svc = minimal_command_service("comfy", vec!["comfyui".into()]);
    svc.placement_override.clear();
    svc.placement_policy = PlacementPolicy::GpuOnly;
    svc.allocation_mode = AllocationMode::Dynamic {
        min_mb: 2048,
        max_mb: 20480,
        min_borrower_runtime_ms: 0,
    };
    // GPU 0 has 1 GiB free (can't hold 2 GiB min), GPU 1 has 24 GiB.
    let mut snap = gpus_with_free(24);
    snap.gpus[0].free_bytes = 1 << 30;

    let out = preview_command_placement(&svc, &snap, &AllocationTable::new(), false)
        .expect("a reserving command service has a placement");
    assert_eq!(out.verdict, FitVerdict::Fits);
    assert_eq!(
        out.devices.get(&DeviceId::Gpu(1)).copied(),
        Some(2048 * 1024 * 1024),
        "min_mb is reserved on the picked GPU"
    );
    assert!(!out.devices.contains_key(&DeviceId::Gpu(0)));
}

/// A command service that reserves no VRAM has no placement to show.
#[test]
fn command_placement_without_reservation_is_none() {
    let svc = minimal_command_service("ext", vec!["external".into()]);
    // The fixture's allocation_mode defaults to `None` (no reservation).
    let out = preview_command_placement(&svc, &gpus_with_free(24), &AllocationTable::new(), false);
    assert!(out.is_none());
}

/// An override larger than the card can't fit even on bare hardware.
#[test]
fn override_placement_too_large_does_not_fit() {
    let mut svc = minimal_service("ov");
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Gpu(0), 30_000); // 30 GB > 24 GiB
    let out = preview_override_placement(&svc, &gpus_with_free(24), &AllocationTable::new(), false);
    let FitVerdict::DoesNotFit { shortfalls } = &out.verdict else {
        panic!("expected DoesNotFit, got {:?}", out.verdict);
    };
    // The overflowing slot is named, so the operator sees that gpu:0 — not
    // the GPUs collectively — is where the 30 GB override overran.
    assert_eq!(shortfalls.len(), 1, "got {shortfalls:?}");
    assert_eq!(shortfalls[0].device, "gpu:0");
    assert_eq!(shortfalls[0].requested_bytes, 30_000 * 1024 * 1024);
    assert_eq!(shortfalls[0].available_bytes, 24 << 30);
}
