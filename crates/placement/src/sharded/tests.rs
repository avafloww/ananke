//! Tests for the tensor/row-split placement path in
//! [`crate::sharded`]: per-GPU share arithmetic, the
//! weighted `tensor_split` ratio, the slop accounting, and the MTP weight
//! split this path builds itself.

use std::collections::BTreeMap;

use ananke_config::placement::{PlacementPolicy, SplitMode};
use ananke_estimate::{Buffers, Estimate, Layout, Mtp, NonLayer};
use ananke_gguf::Architecture;

use super::*;
use crate::{
    AllocationTable,
    devices::DeviceId,
    entry::pack,
    test_support::{GIB, snapshot, svc, trivial_estimate},
};

/// Tensor split shards every layer across both GPUs in parallel: emits
/// `-ngl 999`, equal `--tensor-split 1,1`, `--split-mode tensor`, and
/// `--main-gpu 0`, with each GPU pledged roughly half the model rather
/// than first-fit filling GPU 0.
#[test]
fn tensor_split_shards_equally_across_gpus() {
    let e = trivial_estimate(20, 1024); // 20 layers × 1 GiB = 20 GiB
    let snap = snapshot(&[24, 24]);
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    let packed = pack(&e, &s, &snap, &alloc).unwrap();

    assert_eq!(packed.args.ngl, Some(999));
    assert_eq!(packed.args.split_mode, Some(SplitMode::Tensor));
    assert_eq!(packed.args.main_gpu, Some(0));
    assert_eq!(packed.args.tensor_split.as_deref(), Some(&[1u32, 1][..]));

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
    // With no non-layer tensors or MTP overhead (trivial estimate), the
    // two shards are exactly equal, and each holds ~half the 20 GiB.
    assert_eq!(g0, g1, "shards must be balanced; got {g0} vs {g1}");
    let half = 10u64 * 1024 * 1024 * 1024;
    assert!(g0 >= half, "each shard should carry ~half the weights");
}

/// Tensor split shards the output head and the MTP draft context across
/// every spanned GPU; only the vision projector (the non-layer "other"
/// bytes) rides the main GPU. Measured on Qwen 3.6 27B (`--split-mode
/// tensor`), enabling MTP added the same VRAM to *both* cards (≈1.4 GiB
/// each), and the main GPU's only premium was the non-sharded mmproj — so
/// the per-GPU difference must equal exactly `other_bytes`. Lumping them onto
/// `--main-gpu` instead over-pledges that card by the whole output head plus
/// the whole MTP context.
#[test]
fn tensor_split_shards_output_head_and_mtp_across_gpus() {
    let gib = 1024 * 1024 * 1024u64;
    let per_layer_bytes: Vec<u64> = (0..20).map(|_| gib).collect();
    let output_head = gib; // tensor-parallel sharded
    let other = gib / 2; // mmproj/vision — main GPU only
    let token_embd = gib / 2; // CPU
    let mtp_bytes = 3 * gib; // tensor-parallel sharded
    let weights_bytes = 20 * gib + output_head + other + token_embd;
    let e = Estimate {
        weights_bytes,
        kv_per_token: 0,
        layout: Layout {
            per_layer_bytes: Some(per_layer_bytes),
            non_layer: NonLayer {
                output_head_bytes: output_head,
                token_embd_bytes: token_embd,
                tied_head_bytes: 0,
                other_bytes: other,
            },
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 400,
            ..Buffers::default()
        },
        mtp: Mtp {
            bytes: mtp_bytes,
            ..Mtp::default()
        },
        ..Estimate::empty(Architecture::Qwen35, 4096)
    };
    let snap = snapshot(&[24, 24]);
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    let packed = pack(&e, &s, &snap, &alloc).unwrap();

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
    // If the output head and MTP rode the main GPU, the difference would be
    // `other + output_head + mtp`. Sharded, the only premium is the mmproj.
    assert_eq!(
        g0 - g1,
        other,
        "main GPU premium must be only the mmproj (got g0={g0}, g1={g1})"
    );
    // The last GPU must carry its half of the MTP draft context — proof it
    // is sharded, not lumped on the main GPU.
    assert!(
        g1 >= mtp_bytes / 2,
        "MTP must be sharded onto the last GPU (g1={g1}, mtp/2={})",
        mtp_bytes / 2
    );
    // Token embeddings ride the CPU, as on the layer path.
    let cpu = packed
        .allocation
        .bytes
        .get(&DeviceId::Cpu)
        .copied()
        .unwrap_or(0);
    assert_eq!(cpu, token_embd, "token embeddings must ride the CPU");
}

/// A tied-embedding model — no separate `output.weight` — keeps a second,
/// GPU-resident copy of its embedding table under a tensor split spanning more
/// than one card, because the head's matmul is sharded and a CPU-resident
/// weight cannot be. The CPU copy stays too.
///
/// Measured against the same model on one card, where nothing is sharded and
/// no copy appears: across three tied-embedding models, the two-way tensor
/// split holds one extra table's worth over the layer split, to within a few
/// MiB each time.
#[test]
fn a_tied_head_is_copied_onto_the_cards_only_when_sharded() {
    let gib = 1024 * 1024 * 1024u64;
    let token_embd = gib;
    let build = |output_head: u64| Estimate {
        weights_bytes: 20 * gib + output_head + token_embd,
        kv_per_token: 0,
        layout: Layout {
            per_layer_bytes: Some((0..20).map(|_| gib).collect()),
            non_layer: NonLayer {
                output_head_bytes: output_head,
                token_embd_bytes: token_embd,
                // `collect_non_layer` sets this to `token_embd.weight`'s size
                // exactly when the model ships no head of its own.
                tied_head_bytes: if output_head == 0 { token_embd } else { 0 },
                other_bytes: 0,
            },
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 400,
            ..Buffers::default()
        },
        ..Estimate::empty(Architecture::Gemma4, 4096)
    };
    let gpu_total = |cards: &[u64], estimate: &Estimate| -> u64 {
        let mut s = svc(PlacementPolicy::GpuOnly, None);
        s.split_mode = SplitMode::Tensor;
        pack(estimate, &s, &snapshot(cards), &AllocationTable::new())
            .expect("fit")
            .allocation
            .bytes
            .iter()
            .filter(|(id, _)| matches!(id, DeviceId::Gpu(_)))
            .map(|(_, b)| *b)
            .sum()
    };

    // Across two cards a tied model holds the table twice — once because it is
    // the head, once more because the sharded matmul needs its own copy —
    // against an untied model's single `output.weight` of the same size.
    let tied = build(0);
    let untied = build(gib);
    assert_eq!(
        gpu_total(&[24, 24], &tied) - gpu_total(&[24, 24], &untied),
        token_embd,
        "a tied head across two cards must cost one table more than an \
         equally-sized `output.weight`"
    );
    // One card shards nothing, so no copy appears — which is what the
    // single-GPU measurements show for every model in the set. Going from one
    // card to two adds the copy *and* a second compute buffer, since llama.cpp
    // builds the graph on each device; both are named so neither can hide the
    // other.
    let compute = 400 * 1024 * 1024;
    assert_eq!(
        gpu_total(&[24, 24], &tied) - gpu_total(&[48], &tied),
        token_embd + compute,
        "the copy must appear only once the split actually spans cards, \
         alongside the second card's own compute buffer"
    );
}

/// In a sharded split every spanned GPU must hold its shard — there is no
/// CPU spill. A GPU too small for its equal share fails the pack with
/// `ShardDoesNotFit`, naming the offending GPU.
#[test]
fn tensor_split_rejects_when_a_shard_overflows() {
    let e = trivial_estimate(40, 1024); // 40 GiB → 20 GiB per shard
    let snap = snapshot(&[24, 8]); // GPU 1 can't hold a 20 GiB shard
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    let err = pack(&e, &s, &snap, &alloc).unwrap_err();
    assert!(
        matches!(err, PackError::ShardDoesNotFit { gpu_index: 1, .. }),
        "expected ShardDoesNotFit on gpu:1, got {err:?}"
    );
}

/// With only one GPU available, tensor split is meaningless — fall back to
/// the ordinary single-GPU placement and emit no `--split-mode`/`--main-gpu`.
#[test]
fn tensor_split_with_one_gpu_falls_back_to_layer() {
    let e = trivial_estimate(4, 1024);
    let snap = snapshot(&[24]); // single GPU
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    let packed = pack(&e, &s, &snap, &alloc).unwrap();
    assert_eq!(packed.args.split_mode, None);
    assert_eq!(packed.args.main_gpu, None);
    assert_eq!(packed.args.ngl, Some(4), "single-GPU layer count, not 999");
}

/// Heterogeneous GPUs: a 2.6:1 weight ratio gives the smaller GPU a smaller
/// share so a model that would overflow under an equal split fits. The emitted
/// `--tensor-split` preserves the same ratio as integers, and the main GPU
/// (lowest id) absorbs the rounding remainder so the pledge sums are exact.
#[test]
fn tensor_split_weighted_shards_proportionally_and_fits_smaller_gpu() {
    let e = trivial_estimate(20, 1024); // 20 layers × 1 GiB = 20 GiB
    // 24 GB + 8 GB cards. Equal split would give each GPU a 10 GiB shard
    // plus compute/fudge, which does not fit the 8 GB card.
    let snap = snapshot(&[24, 8]);
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    s.tensor_split_weights = Some(vec![2.6f32, 1.0f32]);
    let packed = pack(&e, &s, &snap, &alloc).unwrap();

    assert_eq!(packed.args.ngl, Some(999));
    assert_eq!(packed.args.split_mode, Some(SplitMode::Tensor));
    assert_eq!(packed.args.main_gpu, Some(0));
    // 2.6:1 reduces to an integer ratio of 13:5.
    assert_eq!(
        packed.args.tensor_split.as_deref(),
        Some(&[13u32, 5][..]),
        "expected 2.6:1 to reduce to 13:5, got {:?}",
        packed.args.tensor_split
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
    // GPU 1 must be pledged less than the equal-split case (~10 GiB).
    assert!(
        g1 < 8 * GIB,
        "weighted split must give the smaller GPU a smaller share; got {g1}"
    );
    // The larger GPU carries the bulk of the model plus the rounding remainder.
    assert!(
        g0 > g1,
        "GPU 0 (weight 2.6) must be pledged more than GPU 1 (weight 1.0); got g0={g0} g1={g1}"
    );
}

#[test]
fn weighted_tensor_split_converts_floats_to_integer_ratio() {
    assert_eq!(weighted_tensor_split(&[2.6f32, 1.0f32]), vec![13, 5]);
    assert_eq!(weighted_tensor_split(&[1.0f32, 1.0f32]), vec![1, 1]);
    assert_eq!(weighted_tensor_split(&[3.0f32, 1.0f32]), vec![3, 1]);
    // Three-way weights reduce by their GCD.
    assert_eq!(
        weighted_tensor_split(&[2.0f32, 1.0f32, 1.0f32]),
        vec![2, 1, 1]
    );
}

/// Reversed-order weights must produce the reversed ratio, confirming the
/// function doesn't assume descending order.
#[test]
fn weighted_tensor_split_handles_reversed_order() {
    assert_eq!(weighted_tensor_split(&[1.0f32, 2.6f32]), vec![5, 13]);
}

/// `weighted_tensor_split` scales by `SCALE = 10_000`, so only 4 decimal
/// places are meaningful. `1.33333 × 10000 = 13333.3` rounds to `13333`;
/// the GCD of 13333 and 10000 is 1, so no reduction occurs. Pin this
/// exact output to document the precision limit.
#[test]
fn weighted_tensor_split_rounds_beyond_four_decimals() {
    assert_eq!(
        weighted_tensor_split(&[1.33333f32, 1.0f32]),
        vec![13333, 10000]
    );
}

/// `integer_shares` must produce pledge proportions consistent with the
/// emitted `--tensor-split` ratio. Weights `[2.6, 1.0]` reduce to
/// `[13, 5]` (ratio_sum=18). A total of 18 gives `[13, 5]` — each GPU's
/// pledge share equals its tensor-split value. A total of 19 gives
/// `[14, 5]` (floor of 13.72 and 5.28, remainder 1 to GPU 0), confirming
/// the proportions still track the ratio.
#[test]
fn integer_shares_match_tensor_split_ratio() {
    let ratio = vec![13u32, 5u32];
    let ratio_sum: u64 = 18;
    assert_eq!(integer_shares(18, &ratio, ratio_sum), vec![13, 5]);
    assert_eq!(integer_shares(19, &ratio, ratio_sum), vec![14, 5]);
    // Shares always sum to total.
    assert_eq!(
        integer_shares(19, &ratio, ratio_sum).iter().sum::<u64>(),
        19
    );
    assert_eq!(
        integer_shares(18, &ratio, ratio_sum).iter().sum::<u64>(),
        18
    );
}

/// A weight-count mismatch at pack time returns
/// `PackError::InvalidTensorSplitWeights`, not `WeightsDoNotFit`. This
/// covers the runtime count-mismatch path that slips through validation's
/// gap (when `gpu_allow` is unset).
#[test]
fn packer_rejects_wrong_weight_count() {
    let e = trivial_estimate(20, 1024);
    let snap = snapshot(&[24, 24]);
    let alloc = AllocationTable::new();
    let mut s = svc(PlacementPolicy::GpuOnly, None);
    s.split_mode = SplitMode::Tensor;
    // 3 weights but only 2 GPUs in the snapshot.
    s.tensor_split_weights = Some(vec![1.0f32, 1.0f32, 1.0f32]);
    // Clear placement_override so the packer's sharded path runs.
    s.placement_override.clear();
    let err = pack(&e, &s, &snap, &alloc).unwrap_err();
    assert!(
        matches!(
            err,
            PackError::InvalidTensorSplitWeights {
                expected: 2,
                got: 3
            }
        ),
        "expected InvalidTensorSplitWeights, got {err:?}"
    );
}

/// The sharded path builds its own MTP shares and never calls
/// `seed_mtp_overhead`, so it needs its own proof that a separate draft
/// model's weights land in the weight tally rather than the runtime one.
#[test]
fn a_sharded_separate_draft_models_weights_are_tallied_as_weights() {
    let draft_weights = 108 * 1024 * 1024;
    let compute = 300 * 1024 * 1024;
    let per_layer_bytes: Vec<u64> = (0..10).map(|_| 200 * 1024 * 1024).collect();
    let weights_bytes: u64 = per_layer_bytes.iter().sum();
    let e = Estimate {
        weights_bytes,
        kv_per_token: 0,
        layout: Layout {
            per_layer_bytes: Some(per_layer_bytes),
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 400,
            ..Buffers::default()
        },
        mtp: Mtp {
            bytes: draft_weights + compute,
            weight_bytes: draft_weights,
            ..Mtp::default()
        },
        ..Estimate::empty(Architecture::Gemma4, 4096)
    };
    let mut svc = svc(PlacementPolicy::GpuOnly, Some(vec![0, 1]));
    svc.placement_override = BTreeMap::new();
    svc.split_mode = ananke_config::placement::SplitMode::Tensor;
    let packed = pack(&e, &svc, &snapshot(&[24, 24]), &AllocationTable::new())
        .expect("sharded draft-MTP model must pack");

    assert_eq!(
        packed.rolling.gpu_weight_bytes,
        weights_bytes + draft_weights,
        "the draft's weights shard as weights, the compute half does not"
    );
}
