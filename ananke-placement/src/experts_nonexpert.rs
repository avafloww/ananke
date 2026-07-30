//! Phase A of the expert-aware (MoE `--n-cpu-moe`) path: placing each
//! layer's non-expert weight and KV share on a GPU, ahead of the Phase B
//! expert-offload decision in [`crate::experts_ncmoe`].

use ananke_config::placement::DeviceSlot;

use crate::{
    entry::PackMode,
    packer::{Charge, Packer},
    types::PackError,
};

impl<'a> Packer<'a> {
    /// Phase A of the expert-aware path: place each layer's *non-expert* weight
    /// (attention, norms) plus its KV share on a GPU via first-fit, recording
    /// the layer's home GPU. A layer whose non-expert part doesn't fit any GPU
    /// spills whole (experts included) to CPU, exactly like the non-MoE hybrid
    /// path. Experts are left for
    /// [`crate::experts_ncmoe::Packer::distribute_experts_ncmoe`].
    pub(crate) fn place_nonexpert_layers(&mut self) -> Result<(), PackError> {
        self.initialise_gpu_remaining();
        let n_layers = self.per_layer.len() as u64;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let kv_per_layer = kv_total.checked_div(n_layers).unwrap_or(0);

        for (idx, full_bytes) in self.per_layer.clone().into_iter().enumerate() {
            if full_bytes == 0 {
                continue;
            }
            let idx = idx as u32;
            let exp_bytes = self.expert_bytes_by_layer.get(&idx).copied().unwrap_or(0);
            let nonexp = full_bytes.saturating_sub(exp_bytes);
            // Corrected: `gpu_remaining` is a corrected budget (see
            // `Packer::charge`), so the fit check has to be too.
            let layer_cost = self.vram_cost(nonexp.saturating_add(kv_per_layer));
            // Place on the GPU with the most remaining capacity, not the first
            // that fits. For a MoE whose non-expert weight is tiny (deepseek4's
            // is ~a few hundred MiB/layer), first-fit would pile every layer —
            // and therefore every layer's KV and its experts' "home" — onto
            // gpu:0, overloading it while gpu:1 sits idle. Balancing by most-
            // free keeps the two cards even (and is capacity-proportional on
            // asymmetric GPUs: the bigger card stays most-free longer).
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .filter(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= layer_cost)
                .max_by_key(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0));
            match placed {
                Some(gpu) => {
                    *self.gpu_remaining.entry(gpu).or_default() -= layer_cost;
                    self.charge(DeviceSlot::Gpu(gpu), nonexp, Charge::Weights);
                    *self.layers_per_gpu.entry(gpu).or_default() += 1;
                    self.layer_home.insert(idx, gpu);
                }
                // A whole-layer CPU spill is valid only for a non-MoE hybrid.
                // On the expert-aware (`--n-cpu-moe`) path the runtime keeps
                // every layer's non-expert weight + KV on GPU (`-ngl 999`) and
                // spills only experts, so a non-expert layer that fits no GPU
                // is *not* offloadable: fail here so admission evicts a resident
                // model and retries with real VRAM, rather than planning a
                // CPU-spill the `-ngl 999` child ignores and then OOMs on
                // (a live "child exited during starting" when loading laguna
                // on top of a resident gemma).
                // `Demand` is exempt from the expert-aware restriction: it
                // spawns nothing, so there is no `-ngl 999` child to OOM.
                // Without the exemption a MoE whose non-expert weight alone
                // overflows the bare cards yields no figure at all — the same
                // "no number for an unplaceable model" hole this reporting
                // work exists to close, just `null` instead of `0 B`.
                None if self.allow_cpu
                    && (!self.expert_aware || matches!(self.mode, PackMode::Demand)) =>
                {
                    self.charge(DeviceSlot::Cpu, full_bytes, Charge::Weights);
                    self.layers_on_cpu += 1;
                    self.spilled_layers.insert(idx);
                }
                None => {
                    return Err(PackError::LayerDoesNotFit {
                        layer_index: idx,
                        bytes: nonexp,
                        shortfalls: self.gpu_shortfalls(layer_cost),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ananke_config::placement::OffloadMode;

    use super::*;
    use crate::{
        AllocationTable,
        entry::pack,
        test_support::{moe_estimate, moe_svc, snapshot},
    };

    /// On the expert-aware path a layer's non-expert weight is GPU-only
    /// (`-ngl 999`), so when it doesn't fit the available VRAM the pack must
    /// fail (`LayerDoesNotFit`) — driving the supervisor to evict a resident
    /// model and retry — rather than spilling the whole layer to CPU and
    /// reporting a fit that the `-ngl 999` child then OOMs on.
    #[test]
    fn expert_offload_nonexpert_gpu_only_fails_instead_of_cpu_spilling() {
        // ~2 GiB of attention across 20 layers; a resident model has left only
        // ~1 GiB free on the single card, so the attention cannot all fit.
        let e = moe_estimate(20, 100, 700);
        let snap = snapshot(&[1]); // 24 GiB card with only 1 GiB free
        let alloc = AllocationTable::new();
        let err = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc)
            .expect_err("non-expert weight is GPU-only; a tight card must fail, not CPU-spill");
        assert!(
            matches!(err, PackError::LayerDoesNotFit { .. }),
            "expected LayerDoesNotFit (→ evict + retry), got {err:?}"
        );
    }

    /// Per-service `gpu_headroom_mb` shrinks usable VRAM, so a model that packs
    /// with no headroom offloads strictly more once headroom is reserved.
    #[test]
    fn expert_offload_headroom_forces_more_offload() {
        let e = moe_estimate(12, 100, 300);
        let snap = snapshot(&[6]);
        let alloc = AllocationTable::new();

        let tight = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();
        let mut svc = moe_svc(OffloadMode::Auto);
        svc.gpu_headroom_mb = 2048;
        let with_headroom = pack(&e, &svc, &snap, &alloc).unwrap();

        assert!(
            with_headroom.expert_offload_bytes >= tight.expert_offload_bytes,
            "more reserved headroom must not offload less: {} < {}",
            with_headroom.expert_offload_bytes,
            tight.expert_offload_bytes
        );
        assert!(with_headroom.expert_offload_bytes > tight.expert_offload_bytes);
    }
}
