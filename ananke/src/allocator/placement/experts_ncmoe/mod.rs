//! Phase B of the expert-aware path: deciding how many leading expert
//! *layers* to offload to CPU as whole units via `--n-cpu-moe`, after
//! [`crate::allocator::placement::experts_nonexpert`] has pinned every
//! layer's non-expert weight to a GPU.

#[cfg(test)]
mod tests;

use crate::{
    allocator::placement::{
        entry::PackMode,
        packer::{Charge, Packer},
        types::PackError,
    },
    config::{DeviceSlot, OffloadMode},
};

impl<'a> Packer<'a> {
    /// Phase B: offload the leading surplus expert *layers* to CPU as whole
    /// units and record `--n-cpu-moe N`, letting the runtime split the
    /// GPU-resident experts across cards itself.
    ///
    /// This replaces per-tensor `-ot` placement. Scattering a layer's
    /// gate/up/down across CUDA0/CUDA1/CPU defeats the runtime's fused
    /// multi-threaded CPU MoE kernel (measured ~24× slower generation on
    /// ik_llama — the CPU experts fall back to a ~2-core path) and can exceed
    /// llama.cpp's `GGML_SCHED_MAX_SPLIT_INPUTS` graph-split limit, a hard
    /// abort at load. `--n-cpu-moe` keeps whole layers together, avoiding both.
    ///
    /// `--n-cpu-moe N` offloads the expert tensors of blocks `[0, N)` — the
    /// *leading* ones — so the retained set is a trailing suffix. The load log
    /// names every tensor it moves, and they start at `blk.1` for laguna
    /// (`--n-cpu-moe 39`, block 0 dense: blocks 1-38) and at `blk.0` for
    /// Qwen3.6-35B-A3B (`--n-cpu-moe 40`: blocks 0-39). Note that `N` counts
    /// *blocks*, not expert layers, so a model with leading dense blocks
    /// offloads fewer expert layers than `N`.
    ///
    /// `Auto` picks the smallest `N` that lets the trailing expert layers fit
    /// the combined GPU pool (what remains after non-expert weights + KV +
    /// compute headroom were reserved); `Layers(n)` uses `n` directly and fails
    /// with [`PackError::ManualExpertsDoNotFit`] if the retained experts still
    /// overflow. Whole layers spilled in Phase A already carry their experts in
    /// the CPU lump and are skipped here.
    pub(crate) fn distribute_experts_ncmoe(&mut self) -> Result<(), PackError> {
        let mut layers: Vec<u32> = self
            .expert_bytes_by_layer
            .keys()
            .copied()
            .filter(|l| !self.spilled_layers.contains(l))
            .collect();
        layers.sort_unstable();

        // Combined GPU budget for experts across all allowed cards; the runtime
        // balances the layer split, so account against the pool, not per-card.
        // This is a *corrected* budget (see `Packer::charge`), so every expert
        // byte weighed against it below is corrected too.
        let pool: u64 = self
            .allowed_gpus
            .iter()
            .map(|g| self.gpu_remaining.get(g).copied().unwrap_or(0))
            .sum();

        // The block index the offload stops below — the value that goes to
        // `--n-cpu-moe`, not a count of expert layers.
        let cutoff = match self.offload_mode {
            OffloadMode::Layers(n) => n,
            // The retained set is a trailing suffix, so the greedy walk runs
            // from the last block down and the cutoff is the lowest block it
            // could keep. Keeping nothing means offloading every expert layer,
            // which is one past the highest.
            OffloadMode::Auto => {
                let mut used = 0u64;
                let mut lowest_kept = layers.last().map(|l| l + 1).unwrap_or(0);
                for &l in layers.iter().rev() {
                    let cost = self.vram_cost(self.expert_bytes_by_layer[&l]);
                    if used.saturating_add(cost) > pool {
                        break;
                    }
                    used += cost;
                    lowest_kept = l;
                }
                lowest_kept
            }
            OffloadMode::Off => 0,
        };

        // Expert layers below the cutoff → CPU; the rest stay on GPU.
        let mut gpu_expert_bytes = 0u64;
        for &l in &layers {
            let b = self.expert_bytes_by_layer[&l];
            if l >= cutoff {
                gpu_expert_bytes += b;
            } else {
                self.charge(DeviceSlot::Cpu, b, Charge::Weights);
                self.expert_offload_cpu_bytes += b;
                self.expert_offload_cpu_layers.insert(l);
            }
        }
        let n_cpu = self.expert_offload_cpu_layers.len() as u32;
        // Corrected, to match `pool` and the per-card charges below.
        let gpu_expert_cost = self.vram_cost(gpu_expert_bytes);

        // A manual `Layers(n)` too small to relieve the cards overflows the
        // GPU pool; reject rather than silently over-committing (`Auto` chose
        // `keep` to fit, so it never trips this).
        // `Demand` is exempt for the same reason Phase A is: it asks what the
        // model needs on an unbounded host and spawns nothing, so a manual
        // offload count that overflows the *bounded* GPU pool must still
        // produce a total rather than no figure at all.
        if gpu_expert_cost > pool && !matches!(self.mode, PackMode::Demand) {
            // The retained experts are distributed evenly across the allowed
            // GPUs below, so the per-card ask is the even share — that is what
            // each GPU's remaining capacity is measured against.
            let per_gpu = gpu_expert_cost / self.allowed_gpus.len().max(1) as u64;
            return Err(PackError::ManualExpertsDoNotFit {
                needed: gpu_expert_cost,
                available: pool,
                shortfalls: self.gpu_shortfalls(per_gpu),
            });
        }

        // Total retained (GPU-resident) expert bytes. The runtime piles these
        // onto the last CUDA device, so `finish` biases `--tensor-split` to
        // give that card fewer non-expert layers to compensate.
        self.ncmoe_kept_expert_bytes = gpu_expert_bytes;

        // Distribute the retained experts evenly across the GPUs for the
        // reservation — the room-biased `--tensor-split` makes the runtime
        // reproduce this balanced target. The sub-`n_gpus`-byte remainder
        // rides on the first card.
        let n_gpus = self.allowed_gpus.len() as u64;
        if n_gpus > 0 && gpu_expert_bytes > 0 {
            let share = gpu_expert_bytes / n_gpus;
            let mut remainder = gpu_expert_bytes - share * n_gpus;
            for gpu in self.allowed_gpus.clone() {
                let add = share + std::mem::take(&mut remainder);
                let charged = self.charge(DeviceSlot::Gpu(gpu), add, Charge::Weights);
                let rem = self.gpu_remaining.entry(gpu).or_default();
                *rem = rem.saturating_sub(charged);
            }
        }

        // When nothing is offloaded (the whole model fits), keep the plain
        // layer-split shape — no `--n-cpu-moe 0`, and `ngl` stays the layer
        // count — so a fully-resident MoE looks identical to a non-MoE fit.
        if n_cpu > 0 {
            // The *cutoff*, not the count: `--n-cpu-moe` names a block bound,
            // and on a model with leading dense blocks the two differ.
            self.n_cpu_moe = Some(cutoff);
            // `-ngl 999` puts all layers on GPU; `-ncmoe` then pulls the
            // leading experts back to CPU and the runtime owns the cross-GPU
            // split.
            self.fallback_on_gpu = true;
        }
        Ok(())
    }
}
