//! Tensor/row-split placement: shards every layer across all spanned GPUs
//! in parallel, instead of assigning whole layers via first-fit.

#[cfg(test)]
mod tests;

use ananke_config::placement::{DeviceSlot, OffloadMode};

use crate::{
    entry::ONE_LAYER_FUDGE_MULTIPLIER,
    experts_ncmoe::Ncmoe,
    packer::{Charge, Packer},
    types::{PackError, ShardedPlan},
};

impl<'a> Packer<'a> {
    /// Determine which expert layers to offload to CPU in the sharded path,
    /// charge them, and return the `-ncmoe` argument and the total bytes.
    ///
    /// `Layers(n)` offloads what the runtime's own `-ncmoe n` would, `Auto`
    /// offloads every expert layer, and `Off` offloads none. The offloaded
    /// expert bytes are charged to CPU and removed from the per-layer sum the
    /// sharded path distributes across GPUs.
    fn sharded_expert_offload(&mut self) -> (u32, u64) {
        let mut layers: Vec<u32> = self.expert_bytes_by_layer.keys().copied().collect();
        layers.sort_unstable();

        // `--n-cpu-moe N` moves the expert tensors of blocks `[0, N)` — the
        // *leading* ones — to a CPU-mapped buffer, while every block's attention
        // tensors stay on GPU. The load log names each tensor it moves, and they
        // start at `blk.0` for Qwen3.6-35B-A3B (`--n-cpu-moe 40`: blocks 0-39)
        // and at `blk.1` for laguna (`--n-cpu-moe 39`, block 0 dense: blocks
        // 1-38) — so `N` bounds *blocks*, not expert layers.
        //
        // Which end matters whenever the quant gives later blocks wider
        // experts: the two ends of the range differ by enough that taking the
        // wrong one leaves the cards short. Confirmed against a production
        // cell, where the physical GPU model buffer comes to all attention plus
        // the retained block's experts, the output head, and the nextn
        // tensors.
        //
        // ik_llama takes the other end and counts expert layers rather than
        // blocks; [`crate::experts_ncmoe::Ncmoe`] holds
        // both conventions and the measurements behind them.
        //
        // `Layers(n)` honours the runtime's argument; `Auto` offloads every
        // expert layer, since this path has no per-card budget to fit a suffix
        // against.
        // ik picks a different end depending on how many devices the model spans,
        // and a trailing window wastes the MTP head's slots.
        let gpu_count = self.allowed_gpus.len();
        let head_layers = self.estimate.mtp_head_expert_layers;
        let plan = match self.offload_mode {
            OffloadMode::Off => {
                Ncmoe::for_runtime(self.placement, 0, &layers, gpu_count, head_layers)
            }
            OffloadMode::Layers(n) => {
                Ncmoe::for_runtime(self.placement, n, &layers, gpu_count, head_layers)
            }
            OffloadMode::Auto => Ncmoe::keeping(self.placement, 0, &layers),
        };

        let mut offloaded = 0u64;
        for &l in &layers {
            if plan.keeps(l, &layers) {
                continue;
            }
            let b = self.expert_bytes_by_layer[&l];
            self.charge(DeviceSlot::Cpu, b, Charge::Weights);
            self.expert_offload_cpu_bytes += b;
            self.expert_offload_cpu_layers.insert(l);
            offloaded += b;
        }
        (plan.flag(), offloaded)
    }

    /// Tensor/row split: shard the whole model across every spanned GPU in
    /// parallel rather than assigning whole layers. Each GPU pledges a
    /// proportional share of the layer weights, the KV cache, the output head,
    /// the MTP draft context, and the compute buffer, plus a proportional share
    /// of the one-layer fudge. The proportion is taken from
    /// `tensor_split_weights` when set, otherwise every GPU gets an equal
    /// `1/n` share. llama.cpp's tensor-parallel modes split those tensors
    /// across the spanned devices — empirically the main GPU carries no
    /// measurable output-head or MTP premium — so modelling them as a per-GPU
    /// share rather than a lump on `--main-gpu` keeps the pledge in line with
    /// the real footprint. Only the vision projector (the residual "other"
    /// weights, which llama.cpp keeps on the main device) and any weight bytes
    /// not in the per-layer breakdown ride the main GPU. Token embeddings ride
    /// the CPU as on the layer path, except that a tied-embedding model spanning
    /// more than one card also keeps a sharded GPU copy of them — see below.
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
        // Charged to every spanned GPU, not divided between them: llama.cpp
        // builds the same graph on each device under a tensor split rather than
        // splitting one, and its reported compute buffer reads identically on
        // one card and on two at every context measured. Dividing it under-
        // reserves every card by a factor of the GPU count — Qwen3.6-35B-A3B
        // needs 1332 MiB *per* card against 685 pledged across both. See
        // [`ananke_estimate::compute_model`].
        let compute_per_gpu = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let fudge_total = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + kv_total / n_layers);

        // Default weights give an equal split; explicit weights are
        // validated to be one-per-allowed-GPU in ascending id order.
        let weights = self
            .placement
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

        // A tied-embedding model has no separate output head, so the embedding
        // table *is* the head. One copy always — it is the output head, and the logits are computed
        // on the device (see [`Packer::seed_non_layer`]) — and a *second* once
        // the split actually spans cards, because the head's matmul is then
        // sharded and a CPU-resident weight cannot be. At one card a tensor
        // split allocates exactly what a layer split does for every model
        // measured, which is what says the second copy is caused by sharding
        // rather than by the split mode.
        //
        // Two cards, measured against the same model on one: gemma-4-31B-QAT
        // 17232 MiB against 16471 for a 756 MiB table, Qwen3-4B 3064 against
        // 2759 for 304, gemma-3-27B 16880 against 15773 for 1103. Every model
        // that ships its own `output.weight` holds the same under either split.
        let tied_copies = if gpus.len() > 1 { 2 } else { 1 };
        let tied_head_shares = integer_shares(
            non_layer.tied_head_bytes * tied_copies,
            &tensor_split,
            ratio_sum,
        );
        let weights_shares = integer_shares(per_layer_sum, &tensor_split, ratio_sum);
        let kv_shares = integer_shares(kv_total, &tensor_split, ratio_sum);
        // The output head is model weight; the MTP draft context is a runtime
        // allocation. Sharded the same way, but tallied apart so the host-pool
        // observation can subtract the file-backed share (see
        // [`crate::types::RollingInputs`]).
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
        let fudge_shares = integer_shares(fudge_total, &tensor_split, ratio_sum);

        if non_layer.token_embd_bytes > 0 {
            self.charge(DeviceSlot::Cpu, non_layer.token_embd_bytes, Charge::Weights);
        }
        // This path returns before the layer walk's step 4, so it charges the
        // host overhead itself rather than inheriting it.
        self.add_host_overhead();

        // The replicated tensors are already counted once inside the split pool,
        // so what each card still owes is the rest of a full copy. Spread evenly
        // across symmetric cards that is `replicated x (cards - 1) / cards` each,
        // which totals the one extra copy a two-way split was measured holding.
        let replicated_extra = self
            .estimate
            .tensor_split_replicated_bytes
            .saturating_mul(gpus.len().saturating_sub(1) as u64)
            / gpus.len().max(1) as u64;

        for (idx, &gpu) in gpus.iter().enumerate() {
            let mut weight_bytes = weights_shares[idx]
                + output_head_shares[idx]
                + mtp_weight_shares[idx]
                + tied_head_shares[idx]
                + replicated_extra;
            let runtime_bytes = kv_shares[idx] + mtp_runtime_shares[idx] + compute_per_gpu;
            let mut runtime_bytes = runtime_bytes;
            if gpu == main {
                weight_bytes += main_only + remainder;
                // The CLIP graph buffer rides the main GPU whatever the split:
                // llama.cpp names one device for it.
                runtime_bytes += self.estimate.mmproj_graph_bytes;
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
            mode: self.placement.split_mode,
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
/// closely as possible. This keeps the `vec![1, 1]` shape when the
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
