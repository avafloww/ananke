//! Phase B of the expert-aware path: deciding how many leading expert
//! *layers* to offload to CPU as whole units via `--n-cpu-moe`, after
//! [`crate::experts_nonexpert`] has pinned every
//! layer's non-expert weight to a GPU.

#[cfg(test)]
mod tests;

use ananke_config::placement::{DeviceSlot, OffloadMode, PlacementInputs};

use crate::{
    entry::PackMode,
    packer::{Charge, Packer},
    types::PackError,
};

/// Which end of the expert layers `-ncmoe` moves to the host, and what its
/// argument counts. The two runtimes disagree on both, and each was pinned
/// exactly against a load log.
///
/// **Mainline** moves the expert tensors of blocks `[0, N)` — the leading ones,
/// with `N` bounding *blocks* rather than counting expert layers. Its log names
/// every tensor it moves: for laguna at `--n-cpu-moe 39` they run `blk.1` to
/// `blk.38` (block 0 being dense), summing to the 41496 MiB its host buffer
/// reports.
///
/// **ik_llama** depends on the device count, which its own source spells out
/// (`llama-load-tensors.cpp`): with `split_mode` attn or graph, or `ncmoe >=
/// n_layer`, or **fewer than two devices**, it overrides the leading blocks like
/// mainline. Only a layer split across two or more devices takes the other
/// branch, where it distributes the count per device in proportion to each one's
/// layer share and then walks *backwards* from the last block — "it is better to
/// go backwards to avoid issues when there are layers without MoE tensors" — so
/// the window is trailing.
///
/// The distinction is worth 1692 MiB on laguna, whose quant widens later blocks:
/// its leading 38 and trailing 39 differ by exactly that, which is what the
/// one-card cells were short by while this modelled every ik service as trailing.
///
/// A trailing window also swallows the MTP head. ik overrides those blocks and
/// then never loads them, so the slots are wasted: Qwen3.6-35B-A3B at `-ncmoe 40`
/// on two cards logs 22734 MiB of overrides and puts 22172 in its host buffer,
/// short by exactly block 40's 562 MiB. Verified to 0.00 MiB across five cells
/// spanning one and two cards and `-ncmoe` 20, 30, and 40.
pub(crate) enum Ncmoe {
    /// Mainline: offload blocks below this index.
    LeadingBlocksBelow(u32),
    /// ik_llama: offload this many trailing expert layers.
    TrailingLayers(u32),
}

impl Ncmoe {
    /// Is this service served by the fork?
    pub(crate) fn is_ik(placement: &PlacementInputs) -> bool {
        placement.ik_llama
    }

    /// The plan for an explicit `-ncmoe n` under this service's runtime and
    /// placement.
    ///
    /// `gpus` is how many devices the model spans and `mtp_head_layers` how many
    /// expert layers the MTP head accounted for — both of which change which
    /// blocks ik picks. See the type's own documentation.
    pub(crate) fn for_runtime(
        placement: &PlacementInputs,
        n: u32,
        layers: &[u32],
        gpus: usize,
        mtp_head_layers: u32,
    ) -> Self {
        if Self::is_ik(placement) && gpus >= 2 && n < layers.len() as u32 + mtp_head_layers {
            // The window is taken over the full block range, so the head blocks
            // it reaches are overridden and then never loaded.
            let real = n.saturating_sub(mtp_head_layers);
            Self::TrailingLayers(real.min(layers.len() as u32))
        } else {
            Self::LeadingBlocksBelow(n)
        }
    }

    /// The plan that retains `kept` expert layers, from whichever end this
    /// runtime keeps them.
    pub(crate) fn keeping(placement: &PlacementInputs, kept: u32, layers: &[u32]) -> Self {
        let total = layers.len() as u32;
        if Self::is_ik(placement) {
            Self::TrailingLayers(total.saturating_sub(kept))
        } else {
            // Mainline keeps a trailing suffix, so the bound is the lowest block
            // it retains — or one past the highest when it retains none.
            Self::LeadingBlocksBelow(match layers.len().checked_sub(kept as usize) {
                Some(i) if i < layers.len() => layers[i],
                _ => layers.last().map(|l| l + 1).unwrap_or(0),
            })
        }
    }

    pub(crate) fn keeps(&self, layer: u32, layers: &[u32]) -> bool {
        match *self {
            Self::LeadingBlocksBelow(bound) => layer >= bound,
            Self::TrailingLayers(n) => {
                let kept = layers.len().saturating_sub(n as usize);
                layers
                    .iter()
                    .position(|&l| l == layer)
                    .is_some_and(|i| i < kept)
            }
        }
    }

    /// The value to pass on the command line.
    pub(crate) fn flag(&self) -> u32 {
        match *self {
            Self::LeadingBlocksBelow(bound) => bound,
            Self::TrailingLayers(n) => n,
        }
    }
}

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

        // Which end this runtime takes, and how it counts (see [`Ncmoe`]). The
        // greedy `Auto` walk starts from whichever end stays on the GPU.
        let gpu_count = self.allowed_gpus.len();
        let head_layers = self.estimate.mtp_head_expert_layers;
        let plan = match self.offload_mode {
            OffloadMode::Layers(n) => {
                Ncmoe::for_runtime(self.placement, n, &layers, gpu_count, head_layers)
            }
            OffloadMode::Auto => {
                let mut used = 0u64;
                let mut kept = 0u32;
                let retained_from_the_top = !Ncmoe::is_ik(self.placement);
                let walk: Vec<u32> = if retained_from_the_top {
                    layers.iter().rev().copied().collect()
                } else {
                    layers.clone()
                };
                for &l in &walk {
                    let cost = self.vram_cost(self.expert_bytes_by_layer[&l]);
                    if used.saturating_add(cost) > pool {
                        break;
                    }
                    used += cost;
                    kept += 1;
                }
                Ncmoe::keeping(self.placement, kept, &layers)
            }
            OffloadMode::Off => {
                Ncmoe::for_runtime(self.placement, 0, &layers, gpu_count, head_layers)
            }
        };

        let mut gpu_expert_bytes = 0u64;
        for &l in &layers {
            let b = self.expert_bytes_by_layer[&l];
            if plan.keeps(l, &layers) {
                gpu_expert_bytes += b;
            } else {
                self.charge(DeviceSlot::Cpu, b, Charge::Weights);
                self.expert_offload_cpu_bytes += b;
                self.expert_offload_cpu_layers.insert(l);
            }
        }
        let n_cpu = self.expert_offload_cpu_layers.len() as u32;
        let flag = plan.flag();
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
            // Not the count of offloaded layers: mainline's argument bounds
            // blocks, so the two differ on a model with leading dense blocks.
            self.n_cpu_moe = Some(flag);
            // `-ngl 999` puts all layers on GPU; `-ncmoe` then pulls the
            // leading experts back to CPU and the runtime owns the cross-GPU
            // split.
            self.fallback_on_gpu = true;
        }
        Ok(())
    }
}
