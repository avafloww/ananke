//! Phase B of the expert-aware path: deciding how many trailing expert
//! *layers* to offload to CPU as whole units via `--n-cpu-moe`, after
//! [`crate::allocator::placement::experts_nonexpert`] has pinned every
//! layer's non-expert weight to a GPU.

#[cfg(test)]
mod tests;

use crate::{
    allocator::placement::{entry::PackMode, packer::Packer, types::PackError},
    config::{DeviceSlot, OffloadMode},
};

impl<'a> Packer<'a> {
    /// Phase B: offload the trailing surplus expert *layers* to CPU as whole
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
    /// `-ncmoe` offloads the *last* `N` MoE layers, so the retained set is
    /// always a leading prefix. `Auto` picks the smallest `N` that lets the
    /// leading expert layers fit the combined GPU pool (what remains after
    /// non-expert weights + KV + compute headroom were reserved); `Layers(n)`
    /// uses `n` directly and fails with [`PackError::ManualExpertsDoNotFit`]
    /// if the retained experts still overflow. Whole layers spilled in Phase A
    /// already carry their experts in the CPU lump and are skipped here.
    pub(crate) fn distribute_experts_ncmoe(&mut self) -> Result<(), PackError> {
        let mut layers: Vec<u32> = self
            .expert_bytes_by_layer
            .keys()
            .copied()
            .filter(|l| !self.spilled_layers.contains(l))
            .collect();
        layers.sort_unstable();
        let total = layers.len() as u32;

        // Combined GPU budget for experts across all allowed cards; the runtime
        // balances the layer split, so account against the pool, not per-card.
        let pool: u64 = self
            .allowed_gpus
            .iter()
            .map(|g| self.gpu_remaining.get(g).copied().unwrap_or(0))
            .sum();

        let n_cpu = match self.offload_mode {
            OffloadMode::Layers(n) => n.min(total),
            OffloadMode::Auto => {
                let mut used = 0u64;
                let mut keep = 0u32;
                for &l in &layers {
                    let b = self.expert_bytes_by_layer[&l];
                    if used.saturating_add(b) <= pool {
                        used += b;
                        keep += 1;
                    } else {
                        break;
                    }
                }
                total - keep
            }
            OffloadMode::Off => 0,
        };
        let keep = total - n_cpu;

        // Trailing `n_cpu` expert layers → CPU; leading `keep` stay on GPU.
        let mut gpu_expert_bytes = 0u64;
        for (i, &l) in layers.iter().enumerate() {
            let b = self.expert_bytes_by_layer[&l];
            if (i as u32) < keep {
                gpu_expert_bytes += b;
            } else {
                *self.per_device.entry(DeviceSlot::Cpu).or_default() += b;
                self.expert_offload_cpu_bytes += b;
                self.expert_offload_cpu_layers.insert(l);
            }
        }

        // A manual `Layers(n)` too small to relieve the cards overflows the
        // GPU pool; reject rather than silently over-committing (`Auto` chose
        // `keep` to fit, so it never trips this).
        // `Demand` is exempt for the same reason Phase A is: it asks what the
        // model needs on an unbounded host and spawns nothing, so a manual
        // offload count that overflows the *bounded* GPU pool must still
        // produce a total rather than no figure at all.
        if gpu_expert_bytes > pool && !matches!(self.mode, PackMode::Demand) {
            // The retained experts are distributed evenly across the allowed
            // GPUs below, so the per-card ask is the even share — that is what
            // each GPU's remaining capacity is measured against.
            let per_gpu = gpu_expert_bytes / self.allowed_gpus.len().max(1) as u64;
            return Err(PackError::ManualExpertsDoNotFit {
                needed: gpu_expert_bytes,
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
                *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += add;
                let rem = self.gpu_remaining.entry(gpu).or_default();
                *rem = rem.saturating_sub(add);
            }
        }

        // When nothing is offloaded (the whole model fits), keep the plain
        // layer-split shape — no `--n-cpu-moe 0`, and `ngl` stays the layer
        // count — so a fully-resident MoE looks identical to a non-MoE fit.
        if n_cpu > 0 {
            self.n_cpu_moe = Some(n_cpu);
            // `-ngl 999` puts all layers on GPU; `-ncmoe` then pulls the
            // trailing experts back to CPU and the runtime owns the cross-GPU
            // split.
            self.fallback_on_gpu = true;
        }
        Ok(())
    }
}
