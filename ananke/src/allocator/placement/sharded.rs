//! Tensor/row-split placement: shards every layer across all spanned GPUs
//! in parallel, instead of assigning whole layers via first-fit.

use crate::{
    allocator::placement::{
        entry::ONE_LAYER_FUDGE_MULTIPLIER,
        packer::Packer,
        types::{PackError, ShardedPlan},
    },
    config::DeviceSlot,
};

impl<'a> Packer<'a> {
    /// Tensor/row split: shard the whole model across every spanned GPU in
    /// parallel rather than assigning whole layers. Each GPU pledges a
    /// proportional share of the layer weights, the KV cache, the output head,
    /// the MTP draft context, and the compute buffer, plus a proportional share
    /// of the one-layer fudge. The proportion is taken from
    /// `tensor_split_weights` when set, otherwise every GPU gets the historical
    /// equal `1/n` share. llama.cpp's tensor-parallel modes split those tensors
    /// across the spanned devices — empirically the main GPU carries no
    /// measurable output-head or MTP premium — so modelling them as a per-GPU
    /// share rather than a lump on `--main-gpu` keeps the pledge in line with
    /// the real footprint. Only the vision projector (the residual "other"
    /// weights, which llama.cpp keeps on the main device) and any weight bytes
    /// not in the per-layer breakdown ride the main GPU. Token embeddings ride
    /// the CPU, as on the layer path.
    ///
    /// There is no CPU spill: a share that overruns a spanned GPU's capacity
    /// is a hard [`PackError::ShardDoesNotFit`], since tensor parallelism ties
    /// every GPU's share to the same proportions and can't offload the
    /// remainder.
    pub(crate) fn distribute_sharded(&mut self) -> Result<(), PackError> {
        let mut gpus = self.allowed_gpus.clone();
        gpus.sort_unstable();
        let main = gpus[0];

        let n_layers = self.per_layer.len() as u64;
        let per_layer_sum: u64 = self.per_layer.iter().sum();
        let per_layer_avg = per_layer_sum / n_layers;
        let non_layer = &self.estimate.non_layer;
        // The vision projector ("other") stays on the main GPU; the output head
        // is sharded across all of them (see below).
        let main_only = non_layer.other_bytes;
        // `weights_bytes` covers per-layer + non-layer + mmproj/anything else;
        // the leftover (vision projector, etc.) rides the main GPU.
        let remainder = self.estimate.weights_bytes.saturating_sub(
            per_layer_sum + non_layer.output_head_bytes + main_only + non_layer.token_embd_bytes,
        );
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let compute_total = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let fudge_total = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + kv_total / n_layers);

        // Default weights give the historical equal split; explicit weights are
        // validated to be one-per-allowed-GPU in ascending id order.
        let weights = self
            .svc
            .tensor_split_weights
            .as_deref()
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0f32; gpus.len()]);
        if weights.len() != gpus.len() {
            return Err(PackError::InvalidTensorSplitWeights {
                expected: gpus.len(),
                got: weights.len(),
            });
        }

        // Derive pledge shares from the same integer ratio emitted to
        // `--tensor-split`, so no GPU is under-pledged relative to its actual
        // tensor-split share. The integer ratio is computed once here and
        // reused for both the pledge book and the argv.
        let tensor_split = weighted_tensor_split(&weights);
        let ratio_sum: u64 = tensor_split.iter().map(|&v| v as u64).sum();

        let weights_shares = integer_shares(per_layer_sum, &tensor_split, ratio_sum);
        let kv_shares = integer_shares(kv_total, &tensor_split, ratio_sum);
        let sharded_non_layer_total = non_layer.output_head_bytes + self.estimate.mtp_bytes;
        let sharded_non_layer_shares =
            integer_shares(sharded_non_layer_total, &tensor_split, ratio_sum);
        let compute_shares = integer_shares(compute_total, &tensor_split, ratio_sum);
        let fudge_shares = integer_shares(fudge_total, &tensor_split, ratio_sum);

        if non_layer.token_embd_bytes > 0 {
            *self.per_device.entry(DeviceSlot::Cpu).or_default() += non_layer.token_embd_bytes;
        }

        for (idx, &gpu) in gpus.iter().enumerate() {
            let mut bytes = weights_shares[idx]
                + kv_shares[idx]
                + sharded_non_layer_shares[idx]
                + compute_shares[idx]
                + fudge_shares[idx];
            if gpu == main {
                bytes += main_only + remainder;
            }
            if bytes > self.gpu_available(gpu) {
                return Err(PackError::ShardDoesNotFit {
                    gpu_index: gpu,
                    bytes,
                });
            }
            *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += bytes;
        }

        // The integer `tensor_split` ratio was computed above so the pledge
        // book and the argv share the same proportions. llama.cpp normalises
        // the tensor-split list, so the integer ratio is what matters, not the
        // absolute values.
        self.sharded = Some(ShardedPlan {
            mode: self.svc.split_mode,
            tensor_split,
        });
        Ok(())
    }
}

/// Split `total` into per-GPU shares proportional to the integer `ratio`.
/// Each entry is the floor of its exact share, and the main GPU (index 0, the
/// lowest-id GPU) absorbs the rounding remainder so the shares sum exactly to
/// `total`. Uses `u128` intermediates to avoid overflow for large totals ×
/// ratio. The `ratio` is the same integer vector emitted to `--tensor-split`,
/// so the pledge book tracks the actual tensor-split proportions.
fn integer_shares(total: u64, ratio: &[u32], ratio_sum: u64) -> Vec<u64> {
    if ratio_sum == 0 {
        return vec![0; ratio.len()];
    }
    let mut shares: Vec<u64> = ratio
        .iter()
        .map(|&v| (total as u128 * v as u128 / ratio_sum as u128) as u64)
        .collect();
    let allocated: u64 = shares.iter().sum();
    let remainder = total.saturating_sub(allocated);
    if !shares.is_empty() {
        shares[0] += remainder;
    }
    shares
}

/// Convert float `weights` into small integer ratios for `--tensor-split`.
///
/// llama.cpp accepts a comma-separated list of proportions and normalises by
/// the sum, so only the ratio matters. We scale the weights by a fixed factor,
/// round to integers, and reduce by the GCD so the emitted values stay small
/// and readable (e.g. `[2.6, 1.0]` becomes `[13, 5]`). If the reduced values do
/// not fit in `u32`, they are scaled down further while preserving the ratio as
/// closely as possible. This keeps the historical `vec![1, 1]` shape when the
/// operator does not override weights, and emits a matching integer ratio when
/// they do.
fn weighted_tensor_split(weights: &[f32]) -> Vec<u32> {
    const SCALE: f64 = 10_000.0;
    let scaled: Vec<u64> = weights
        .iter()
        .map(|&w| ((w as f64 * SCALE).round() as u64).max(1))
        .collect();
    let g = scaled.iter().fold(0u64, |a, &b| gcd_u64(a, b));
    let reduced: Vec<u64> = scaled
        .iter()
        .map(|&v| v.checked_div(g).unwrap_or(v))
        .collect();
    let max = reduced.iter().copied().max().unwrap_or(1);
    if max > u32::MAX as u64 {
        let factor = max / (u32::MAX as u64) + 1;
        reduced
            .iter()
            .map(|&v| (v / factor).max(1) as u32)
            .collect()
    } else {
        reduced.iter().map(|&v| v as u32).collect()
    }
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use super::*;
    use crate::{
        allocator::{
            AllocationTable,
            placement::{
                entry::pack,
                test_support::{GIB, snapshot, svc, trivial_estimate},
            },
        },
        config::{PlacementPolicy, SplitMode},
        devices::DeviceId,
        estimator::{Estimate, NonLayer},
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
    /// the per-GPU difference must equal exactly `other_bytes`. Regression
    /// against the original lump-on-`--main-gpu` accounting, which over-pledged
    /// the main GPU by the whole output head plus the whole MTP context.
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
            compute_buffer_mb: 400,
            output_buffer_bytes: 0,
            mtp_bytes,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer {
                output_head_bytes: output_head,
                token_embd_bytes: token_embd,
                other_bytes: other,
            },
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 4096,
            architecture: SmolStr::new("qwen35"),
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
}
