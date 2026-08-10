//! Integration test: when a model is too large for a single GPU, `placement::pack`
//! distributes layers across both GPUs and emits a `tensor_split` argument.
//!
//! This test calls `placement::pack` directly rather than going through the
//! daemon, because the fake spawn_child doesn't invoke llama-server and the
//! allocations are therefore never reported back.
#![cfg(feature = "test-fakes")]

mod common;

use std::collections::BTreeMap;

use ananke::config::{PlacementPolicy, ServiceConfig};
use ananke_allocator::AllocationTable;
use ananke_estimate::{Estimate, Layout};
use ananke_gguf::Architecture;
use ananke_placement::{
    self as placement,
    devices::{DeviceId, DeviceSnapshot, GpuSnapshot},
};

fn two_gpu_svc() -> ServiceConfig {
    // No placement_override — the test drives pack() directly.
    let mut svc = common::minimal_llama_service("split-svc", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::GpuOnly;
    svc
}

fn two_gpu_snapshot() -> DeviceSnapshot {
    // Two GPUs with 8 GB free each. The model needs 14+ GB total, so a single
    // GPU cannot hold it all.
    DeviceSnapshot {
        gpus: vec![
            GpuSnapshot {
                id: 0,
                name: "GPU 0".into(),
                total_bytes: 16 * 1024 * 1024 * 1024,
                free_bytes: 8 * 1024 * 1024 * 1024,
            },
            GpuSnapshot {
                id: 1,
                name: "GPU 1".into(),
                total_bytes: 16 * 1024 * 1024 * 1024,
                free_bytes: 8 * 1024 * 1024 * 1024,
            },
        ],
        cpu: None,
        taken_at_ms: 0,
    }
}

fn large_estimate() -> Estimate {
    // 20 layers at 600 MiB each = 12 GiB total. Won't fit on one 8 GB GPU
    // but fits across two.
    let n_layers: usize = 20;
    let per_layer_bytes = 600 * 1024 * 1024u64;
    Estimate {
        weights_bytes: per_layer_bytes * n_layers as u64,
        kv_per_token: 0,
        layout: Layout {
            per_layer_bytes: Some(vec![per_layer_bytes; n_layers]),
            ..Layout::default()
        },
        ..Estimate::empty(Architecture::Qwen3, 4096)
    }
}

#[test]
fn multi_gpu_split_produces_tensor_split_and_both_gpus_allocated() {
    let svc = two_gpu_svc();
    let snap = two_gpu_snapshot();
    let reserved = AllocationTable::new();
    let est = large_estimate();

    let packed = placement::pack(
        &est,
        &ananke::config::placement_inputs(&svc),
        &snap,
        &reserved,
    )
    .expect("placement must succeed across two GPUs");

    // Both GPUs should carry some allocation.
    assert!(
        packed.allocation.bytes.contains_key(&DeviceId::Gpu(0)),
        "GPU 0 must be allocated"
    );
    assert!(
        packed.allocation.bytes.contains_key(&DeviceId::Gpu(1)),
        "GPU 1 must be allocated"
    );

    // tensor_split must be emitted when more than one GPU is used.
    let ts = packed
        .args
        .tensor_split
        .as_ref()
        .expect("tensor_split must be present for multi-GPU placement");
    assert_eq!(ts.len(), 2, "tensor_split must have one entry per GPU");
    assert!(ts[0] > 0, "GPU 0 must carry at least one layer");
    assert!(ts[1] > 0, "GPU 1 must carry at least one layer");
}
