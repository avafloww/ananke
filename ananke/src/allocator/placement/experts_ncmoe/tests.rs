//! Tests for the `--n-cpu-moe` expert-offload path in
//! [`crate::allocator::placement::experts_ncmoe`]: auto and manual offload
//! counts, the pooled-overflow rejection, and multi-GPU expert spread.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::*;
use crate::{
    allocator::{
        AllocationTable,
        placement::{
            entry::{NGL_OFFLOAD_ALL, pack},
            test_support::{GIB, MIB, cpu_bytes, moe_estimate, moe_svc, snapshot},
        },
    },
    devices::{CpuSnapshot, DeviceId},
    estimator::{Estimate, ExpertKind, ExpertTensor, NonLayer},
};

#[test]
fn expert_offload_auto_spills_surplus_experts_to_cpu() {
    // 10 layers: 100 MiB non-expert + 900 MiB experts each (10 GiB total),
    // non-expert only ≈ 1 GiB. A 4 GiB card holds all attention but not all
    // experts.
    let e = moe_estimate(10, 100, 300);
    let snap = snapshot(&[4]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

    assert_eq!(
        packed.args.ngl,
        Some(NGL_OFFLOAD_ALL),
        "-ngl 999: all layers on GPU, then -ncmoe pulls trailing experts back"
    );
    assert!(cpu_bytes(&packed) > 0, "surplus experts land on the CPU");
    assert!(packed.expert_offload_bytes > 0);
    assert!(packed.expert_offload_layers > 0);
    assert!(
        matches!(packed.args.n_cpu_moe, Some(n) if n > 0),
        "coarse whole-layer offload via --n-cpu-moe, got {:?}",
        packed.args.n_cpu_moe
    );
    assert!(
        packed.args.override_tensor.is_empty(),
        "no per-tensor expert -ot is synthesised, got {:?}",
        packed.args.override_tensor
    );
    // The GPU pledge must stay within the card.
    let gpu = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(0))
        .copied()
        .unwrap_or(0);
    assert!(gpu <= 24 * GIB);
}

/// When the whole model fits, the expert-aware path offloads nothing and
/// emits no synthesised rule — identical shape to a non-MoE fit.
#[test]
fn expert_offload_auto_no_offload_when_everything_fits() {
    let e = moe_estimate(10, 100, 100); // 400 MiB/layer, 4 GiB total
    let snap = snapshot(&[24]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

    assert_eq!(packed.args.ngl, Some(10));
    assert_eq!(packed.expert_offload_bytes, 0);
    assert_eq!(packed.expert_offload_layers, 0);
    assert!(cpu_bytes(&packed) == 0);
    assert!(packed.args.override_tensor.is_empty());
}

/// `expert_offload = N` offloads exactly the N tail-most expert layers to
/// CPU even on a roomy card, via `--n-cpu-moe N` (not per-tensor `-ot`).
#[test]
fn expert_offload_layers_n_offloads_tail_layers() {
    let e = moe_estimate(10, 100, 100); // fits easily
    let snap = snapshot(&[24]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Layers(3)), &snap, &alloc).unwrap();

    assert_eq!(
        packed.args.ngl,
        Some(NGL_OFFLOAD_ALL),
        "attention stays on GPU; -ncmoe pulls the trailing experts back"
    );
    assert_eq!(
        packed.args.n_cpu_moe,
        Some(3),
        "offload the 3 tail expert layers"
    );
    assert_eq!(packed.expert_offload_layers, 3);
    // 3 layers × 3 experts × 100 MiB.
    assert_eq!(packed.expert_offload_bytes, 9 * 100 * MIB);
    assert!(
        packed.args.override_tensor.is_empty(),
        "no per-tensor -ot, got {:?}",
        packed.args.override_tensor
    );
}

/// A manual `expert_offload = N` too small to relieve the card pins the
/// remaining experts to their home GPU regardless of fit. That overflow is
/// rejected with `ManualExpertsDoNotFit` rather than silently
/// over-committing the GPU into a spawn-time OOM.
#[test]
fn expert_offload_manual_rejects_when_gpu_overflows() {
    // 10 layers, 100 MiB attn + 900 MiB experts each (10 GiB). Offloading
    // only the 2 tail layers leaves ~7 GiB of experts pinned to a 4 GiB card.
    let e = moe_estimate(10, 100, 300);
    let snap = snapshot(&[4]);
    let alloc = AllocationTable::new();
    let err = pack(&e, &moe_svc(OffloadMode::Layers(2)), &snap, &alloc)
        .expect_err("under-sized manual offload must not over-commit the GPU");
    let PackError::ManualExpertsDoNotFit {
        needed, available, ..
    } = err
    else {
        panic!("expected ManualExpertsDoNotFit, got {err:?}");
    };
    assert!(
        needed > available,
        "the retained experts must exceed the pooled GPU capacity, \
         got needed={needed} available={available}"
    );
    // The single allowed card is named, so the operator can cross-reference
    // the overflow against `GET /api/devices`.
    let shortfalls = err.shortfalls();
    assert_eq!(
        shortfalls.iter().map(|s| s.device).collect::<Vec<_>>(),
        vec![DeviceId::Gpu(0)],
        "got {shortfalls:?}"
    );
    // Every entry has to be a genuine shortfall. This is the pooled
    // failure path, where `requested` is a per-card share of an aggregate
    // ask — without the filter a card with room shows up in the
    // placement-failure list claiming it needs less than it holds.
    assert!(
        shortfalls
            .iter()
            .all(|s| s.available_bytes < s.requested_bytes),
        "got {shortfalls:?}"
    );
}

/// The pooled-overflow report must not name a card that can take its
/// share. Two lopsided GPUs (one nearly full, one roomy) whose combined
/// capacity still can't hold the retained experts: the roomy card
/// satisfies the even share, so only the full one is a reason for the
/// failure and only it belongs in the list.
#[test]
fn manual_expert_overflow_names_only_the_cards_that_came_up_short() {
    let e = moe_estimate(10, 100, 300);
    // gpu:0 is tiny, gpu:1 is roomy; neither combination fits the pinned
    // experts, but gpu:1 individually clears the even per-card share.
    let snap = snapshot(&[1, 7]);
    let alloc = AllocationTable::new();
    let err = pack(&e, &moe_svc(OffloadMode::Layers(2)), &snap, &alloc)
        .expect_err("under-sized manual offload must not over-commit the GPUs");

    let shortfalls = err.shortfalls();
    assert!(
        !shortfalls.is_empty(),
        "a capacity failure always names at least one device"
    );
    assert!(
        shortfalls
            .iter()
            .all(|s| s.available_bytes < s.requested_bytes),
        "no card that fits its share may be listed, got {shortfalls:?}"
    );
}

/// Auto offload spreads across both GPUs before touching the CPU: a model
/// that fits in the two cards' combined VRAM but not either alone lands
/// entirely on the GPUs. Nothing is offloaded, so no `--n-cpu-moe` and no
/// `-ot` — the runtime splits the layers across both cards itself.
#[test]
fn expert_offload_auto_prefers_second_gpu() {
    // 20 layers, 100 MiB attn + 900 MiB experts = ~20 GiB. Two 12 GiB
    // cards hold it together (24 GiB) but neither alone does, so the
    // experts must split across both.
    let e = moe_estimate(20, 100, 300);
    let snap = snapshot(&[12, 12]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

    assert_eq!(packed.args.ngl, Some(20));
    assert_eq!(
        packed.args.n_cpu_moe, None,
        "nothing offloaded → no --n-cpu-moe"
    );
    assert_eq!(
        cpu_bytes(&packed),
        0,
        "experts prefer the GPUs over the CPU"
    );
    assert_eq!(
        packed.expert_offload_bytes, 0,
        "CPU offload metric counts host bytes only"
    );
    assert!(
        packed.args.override_tensor.is_empty(),
        "the runtime owns the cross-GPU split; no synthesised -ot, got {:?}",
        packed.args.override_tensor
    );
    let g0 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(0))
        .copied()
        .unwrap_or(0);
    let g1 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(1))
        .copied()
        .unwrap_or(0);
    assert!(
        g0 > 0 && g1 > 0,
        "both cards carry weight (g0={g0} g1={g1})"
    );
}

/// Symmetric two-GPU balance (the deepseek4 shape): tiny non-expert weight
/// plus huge experts must spread evenly across both cards. First-fit used
/// to pile every layer — and thus every expert's home GPU — onto gpu:0,
/// overloading it into an `insufficient_capacity` error while gpu:1 sat idle.
#[test]
fn expert_offload_auto_balances_symmetric_gpus() {
    // 40 layers, 150 MiB attn + 3×700 MiB experts: ~6 GiB attention, ~84
    // GiB experts — far past 2×24 GiB, so the surplus spills to CPU, but
    // the GPU-resident half must be balanced across both cards.
    let e = moe_estimate(40, 150, 700);
    let snap = snapshot(&[24, 24]);
    let packed = pack(
        &e,
        &moe_svc(OffloadMode::Auto),
        &snap,
        &AllocationTable::new(),
    )
    .unwrap();

    let g0 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(0))
        .copied()
        .unwrap_or(0);
    let g1 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(1))
        .copied()
        .unwrap_or(0);
    assert!(
        g0 > 0 && g1 > 0,
        "both cards must hold weight (g0={g0} g1={g1})"
    );
    assert!(
        cpu_bytes(&packed) > 0,
        "the surplus experts must spill to CPU"
    );
    // Balanced within ~one expert tensor — neither card overloaded.
    let (hi, lo) = (g0.max(g1), g0.min(g1));
    assert!(
        hi - lo <= 1024 * MIB,
        "cards must be balanced within ~1 expert (g0={g0} g1={g1})"
    );
    // And each card must fit inside its 24 GiB.
    assert!(
        g0 <= 24 * GIB && g1 <= 24 * GIB,
        "must fit 24 GiB (g0={g0} g1={g1})"
    );
}

/// Regression for the live `deepseek-v4-flash` failure: the real estimate
/// (~96 GiB weights, 9848 MiB compute buffer, 6657 B/token KV over 131072
/// context, 43 all-MoE layers) must auto-fit on two 24 GiB cards. Before
/// the balance + one-layer-fudge fixes this reported
/// `insufficient_capacity: no fit on gpu:0`.
#[test]
fn deepseek4_like_auto_fits_two_24gib_cards() {
    let n_layers = 43u32;
    let nonexp = 140 * MIB; // ~6 GiB of attention across 43 layers
    let exp = 700 * MIB; // 3 × 700 MiB experts/layer → ~88 GiB experts
    let mut per_layer = Vec::new();
    let mut experts = Vec::new();
    for layer in 0..n_layers {
        per_layer.push(nonexp + 3 * exp);
        for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
            experts.push(ExpertTensor {
                layer,
                kind,
                bytes: exp,
            });
        }
    }
    let e = Estimate {
        weights_bytes: (nonexp + 3 * exp) * n_layers as u64 + 414 * MIB,
        kv_per_token: 6657,
        compute_buffer_mb: 9848,
        output_buffer_bytes: 0,
        mtp_bytes: 0,
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer: NonLayer {
            output_head_bytes: 414 * MIB,
            token_embd_bytes: 414 * MIB,
            other_bytes: 0,
        },
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: (0..n_layers).collect(),
        expert_tensors: Some(experts),
        context: 131072,
        architecture: SmolStr::new("deepseek4"),
    };
    // The real box has 125 GiB RAM for the ~60 GiB of CPU-side experts;
    // widen the default snapshot's host budget to match.
    let mut snap = snapshot(&[24, 24]);
    snap.cpu = Some(CpuSnapshot {
        total_bytes: 125 * GIB,
        available_bytes: 110 * GIB,
    });
    let packed = pack(
        &e,
        &moe_svc(OffloadMode::Auto),
        &snap,
        &AllocationTable::new(),
    )
    .expect("deepseek4 auto must fit two 24 GiB cards");
    let g0 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(0))
        .copied()
        .unwrap_or(0);
    let g1 = packed
        .allocation
        .bytes
        .get(&DeviceId::Gpu(1))
        .copied()
        .unwrap_or(0);
    // Both cards used, both within capacity, and balanced.
    assert!(g0 > 0 && g1 > 0 && cpu_bytes(&packed) > 0);
    assert!(
        g0 <= 24 * GIB && g1 <= 24 * GIB,
        "g0={g0} g1={g1} must fit 24 GiB"
    );
    assert!(
        g0.abs_diff(g1) <= 1500 * MIB,
        "cards balanced: g0={g0} g1={g1}"
    );
    // Roughly the empirical G8-G10: a meaningful chunk of experts on GPU.
    assert!(
        packed.expert_offload_layers > 0,
        "some experts spill to CPU"
    );
}

/// Offloading more experts than host RAM can hold (minus the CPU reserve)
/// is rejected with `CpuDoesNotFit` rather than silently over-committing.
#[test]
fn expert_offload_rejects_when_cpu_is_full() {
    let e = moe_estimate(10, 100, 900); // ~1 GiB attn, ~27 GiB experts
    // A 24 GiB card holds the attention plus most experts; the ~4 GiB
    // expert surplus must spill, but the host has only 2 GiB free.
    let mut snap = snapshot(&[24]);
    snap.cpu = Some(CpuSnapshot {
        total_bytes: 4 * GIB,
        available_bytes: 2 * GIB,
    });
    let alloc = AllocationTable::new();
    let err = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc)
        .expect_err("CPU offload must not exceed host RAM");
    assert!(
        matches!(err, PackError::CpuDoesNotFit { .. }),
        "expected CpuDoesNotFit, got {err:?}"
    );
}
