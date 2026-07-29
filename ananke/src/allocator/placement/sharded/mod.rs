//! Tensor/row-split placement: shards every layer across all spanned GPUs
//! in parallel, instead of assigning whole layers via first-fit.

#[cfg(test)]
mod tests;

use crate::{
    allocator::placement::{
        entry::ONE_LAYER_FUDGE_MULTIPLIER,
        packer::{Charge, Packer},
        types::{PackError, ShardedPlan},
    },
    config::{DeviceSlot, OffloadMode},
};

impl<'a> Packer<'a> {
    /// Determine how many expert layers to offload to CPU in the sharded
    /// path, charge them, and return the count and total bytes.
    ///
    /// Uses the same `OffloadMode` logic as `distribute_experts_ncmoe`:
    /// `Layers(n)` offloads exactly `n`, `Auto` offloads the trailing
    /// surplus that does not fit the combined GPU pool, `Off` offloads
    /// nothing. The offloaded expert bytes are charged to CPU and removed
    /// from the per-layer sum the sharded path distributes across GPUs.
    fn sharded_expert_offload(&mut self) -> (u32, u64) {
        let mut layers: Vec<u32> = self.expert_bytes_by_layer.keys().copied().collect();
        layers.sort_unstable();
        let total = layers.len() as u32;

        // `--n-cpu-moe N` offloads the expert tensors of the N tail-most
        // expert layers to a CPU-mapped buffer; the corresponding attention
        // tensors stay on GPU for every layer — confirmed by production
        // measurements on Qwen3.6-35B-A3B (n_cpu_moe=40, 41 expert layers):
        // physical GPU model = 2 × 1431 = 2862 MiB ≈ all attention (1837) +
        // 1 expert layer (568) + output head (242) + nextn tensors (~215).
        // `Layers(n)` honours the layer count; `Auto` offloads the trailing
        // surplus that does not fit the combined GPU pool.
        let n_cpu = match self.offload_mode {
            OffloadMode::Off => 0,
            OffloadMode::Layers(n) => n.min(total),
            OffloadMode::Auto => total,
        };
        let keep = total - n_cpu;

        let mut offloaded = 0u64;
        for (i, &l) in layers.iter().enumerate() {
            if (i as u32) >= keep {
                let b = self.expert_bytes_by_layer[&l];
                self.charge(DeviceSlot::Cpu, b, Charge::Weights);
                self.expert_offload_cpu_bytes += b;
                self.expert_offload_cpu_layers.insert(l);
                offloaded += b;
            }
        }
        (n_cpu, offloaded)
    }

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

        // When expert offload is enabled, move the offloaded expert layers'
        // bytes to CPU before sharding the rest. This matches llama.cpp's
        // behaviour: `--n-cpu-moe N` with `--split-mode tensor` shards the
        // non-expert tensors across GPUs while moving expert tensors to the
        // CUDA_Host buffer.
        let (n_cpu, offloaded_total_bytes) = if self.expert_aware {
            self.sharded_expert_offload()
        } else {
            (0, 0u64)
        };
        if n_cpu > 0 {
            self.n_cpu_moe = Some(n_cpu);
            self.fallback_on_gpu = true;
        }

        // `offloaded_total_bytes` includes both expert and non-expert (attention)
        // per-layer weights moved to CPU. Subtract from the per-layer sum so the
        // sharded distribution only covers GPU-resident layers.
        let per_layer_sum: u64 = self
            .per_layer
            .iter()
            .sum::<u64>()
            .saturating_sub(offloaded_total_bytes);
        let per_layer_avg = per_layer_sum / n_layers;
        let non_layer = &self.estimate.non_layer;
        // The vision projector ("other") stays on the main GPU; the output head
        // is sharded across all of them (see below).
        let main_only = non_layer.other_bytes;
        // `weights_bytes` covers per-layer + non-layer + mmproj/anything else;
        // the leftover (vision projector, etc.) rides the main GPU. Subtract
        // the offloaded bytes from both `per_layer_sum` and `weights_bytes` so
        // the remainder does not re-charge them to the main GPU.
        let gpu_weights = self
            .estimate
            .weights_bytes
            .saturating_sub(offloaded_total_bytes);
        let remainder = gpu_weights.saturating_sub(
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
        // The output head is model weight; the MTP draft context is a runtime
        // allocation. Sharded the same way, but tallied apart so the host-pool
        // observation can subtract the file-backed share (see
        // [`crate::allocator::placement::types::RollingInputs`]).
        let output_head_shares =
            integer_shares(non_layer.output_head_bytes, &tensor_split, ratio_sum);
        // A separate draft model's weights are read through its own mmap, so
        // they shard as weights; an embedded head's overhead is all runtime
        // allocation. Same split as `seed_mtp_overhead`, which this path
        // bypasses.
        let mtp_weight = self.estimate.mtp_weight_bytes.min(self.estimate.mtp_bytes);
        let mtp_weight_shares = integer_shares(mtp_weight, &tensor_split, ratio_sum);
        let mtp_runtime_shares = integer_shares(
            self.estimate.mtp_bytes.saturating_sub(mtp_weight),
            &tensor_split,
            ratio_sum,
        );
        let compute_shares = integer_shares(compute_total, &tensor_split, ratio_sum);
        let fudge_shares = integer_shares(fudge_total, &tensor_split, ratio_sum);

        if non_layer.token_embd_bytes > 0 {
            self.charge(DeviceSlot::Cpu, non_layer.token_embd_bytes, Charge::Weights);
        }
        // This path returns before the layer walk's step 4, so it charges the
        // host overhead itself rather than inheriting it.
        self.add_host_overhead();

        for (idx, &gpu) in gpus.iter().enumerate() {
            let mut weight_bytes =
                weights_shares[idx] + output_head_shares[idx] + mtp_weight_shares[idx];
            let runtime_bytes = kv_shares[idx] + mtp_runtime_shares[idx] + compute_shares[idx];
            if gpu == main {
                weight_bytes += main_only + remainder;
            }
            let slot = DeviceSlot::Gpu(gpu);
            // Charge first, then gate on what was actually charged: the three
            // kinds are scaled and tallied separately, so summing their real
            // costs is the only figure guaranteed to match the reservation.
            let bytes = self.charge(slot.clone(), weight_bytes, Charge::Weights)
                + self.charge(slot.clone(), runtime_bytes, Charge::Runtime)
                + self.charge(slot.clone(), fudge_shares[idx], Charge::Slop);
            let available = self.gpu_available(gpu);
            if bytes > available {
                return Err(PackError::ShardDoesNotFit {
                    gpu_index: gpu,
                    bytes,
                    available,
                });
            }
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
