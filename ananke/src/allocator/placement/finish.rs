//! Materialising the finished `Packed` result: deriving `-ngl`,
//! `--tensor-split`, `--split-mode`/`--main-gpu`, and `-ot` from the packer's
//! accumulated per-device state.

use crate::{
    allocator::placement::{
        entry::{NGL_CPU_ONLY, NGL_OFFLOAD_ALL},
        packer::Packer,
        types::{CommandArgs, Packed},
    },
    config::DeviceSlot,
    devices::{Allocation, DeviceId},
};

impl<'a> Packer<'a> {
    /// Step 6: materialise the final `Packed` — derive -ngl, --tensor-split,
    /// -ot, and convert the per_device map into an `Allocation`.
    pub(crate) fn finish(self) -> Packed {
        // Sharded (tensor/row) split: every layer is offloaded and divided
        // across all spanned GPUs by the configured proportions (equal by
        // default, or weighted via tensor_split_weights), so emit `-ngl 999`,
        // the plan's `--tensor-split` ratio, the `--split-mode`, and
        // `--main-gpu` (visible index 0 — the lowest-id GPU, which cuda_env
        // renders first).
        let (ngl, tensor_split, split_mode, main_gpu) = if let Some(plan) = &self.sharded {
            (
                Some(NGL_OFFLOAD_ALL),
                Some(plan.tensor_split.clone()),
                Some(plan.mode),
                Some(0),
            )
        } else {
            let total_on_gpus: u32 = self.layers_per_gpu.values().sum();
            let ngl = if self.allowed_gpus.is_empty() {
                Some(NGL_CPU_ONLY)
            } else if self.fallback_on_gpu {
                Some(NGL_OFFLOAD_ALL)
            } else {
                Some(total_on_gpus)
            };

            // Ratios in CUDA_VISIBLE_DEVICES-remapped order: must be in
            // ascending GPU-id order to match CUDA device numbering, regardless
            // of the placement sort order.
            let tensor_split = if self.allowed_gpus.len() > 1 && total_on_gpus > 0 {
                let mut gpus_by_id = self.allowed_gpus.clone();
                gpus_by_id.sort_unstable();
                if self.n_cpu_moe.is_some() {
                    // `--n-cpu-moe`: the runtime distributes the non-expert
                    // layers + KV by `--tensor-split` but piles the *retained*
                    // experts onto the last CUDA device, and the head device
                    // carries the output logits buffer. A naive even split then
                    // overflows the last card (a live glm-dsa OOM: 14.6 GiB on
                    // CUDA1 vs 9.5 on CUDA0). Bias the split by each card's
                    // *room* for distributable layers — `available` minus its
                    // fixed load (compute buffer everywhere; logits + output
                    // head on the first card; the retained experts on the last)
                    // — so the distributable fills the leftover room evenly and
                    // both cards land at the same total. MiB counts act as
                    // proportions; llama normalises.
                    let compute_bytes = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
                    // Same head as the reservation (lowest id = CUDA visible 0 =
                    // runtime main_gpu), and the same capped logits term, so the
                    // split and the pledge book agree on which card is the head.
                    let head = self.head_gpu();
                    let logits = self.head_logits_bytes();
                    let last = gpus_by_id.last().copied();
                    // Whether the retained experts *clump* on the last card
                    // (needing a bias) or distribute evenly (no bias). Measured:
                    // with many retained layers the runtime spreads them evenly
                    // (laguna, 23/47 kept → balanced at 50/50), but with few it
                    // pins roughly half to the last card (glm-dsa, 8/80 kept →
                    // 14.6/9.5 GiB at 50/50). Gate on a low kept fraction (<1/5).
                    let total_exp_layers = self.expert_bytes_by_layer.len() as u32;
                    let kept_layers = total_exp_layers.saturating_sub(self.n_cpu_moe.unwrap_or(0));
                    let experts_clump =
                        total_exp_layers > 0 && kept_layers.saturating_mul(5) < total_exp_layers;
                    Some(
                        gpus_by_id
                            .iter()
                            .map(|g| {
                                let mut fixed = compute_bytes;
                                if head == Some(*g) {
                                    fixed += logits + self.estimate.non_layer.output_head_bytes;
                                }
                                if last == Some(*g) && experts_clump {
                                    // Only when experts clump: count half the
                                    // retained experts as fixed here (the runtime
                                    // moves the other half with the split), which
                                    // lands the bias near the empirical balance
                                    // (glm ~58/42) without over-correcting.
                                    fixed += self.ncmoe_kept_expert_bytes / 2;
                                }
                                let room = self.gpu_available(*g).saturating_sub(fixed);
                                ((room / (1024 * 1024)) as u32).max(1)
                            })
                            .collect(),
                    )
                } else {
                    Some(
                        gpus_by_id
                            .iter()
                            .map(|g| self.layers_per_gpu.get(g).copied().unwrap_or(0))
                            .collect(),
                    )
                }
            } else {
                None
            };
            (ngl, tensor_split, None, None)
        };

        // Operator-declared `-ot` rules pass straight through; the packer no
        // longer synthesises expert-offload rules (whole-layer offload rides
        // on `--n-cpu-moe`, emitted above).
        let override_tensor = self
            .svc
            .llama_cpp()
            .map(|lc| lc.override_tensor.clone())
            .unwrap_or_default();

        let expert_offload_bytes = self.expert_offload_cpu_bytes;
        let expert_offload_layers = self.expert_offload_cpu_layers.len() as u32;

        let allocation = Allocation {
            bytes: self
                .per_device
                .into_iter()
                .map(|(slot, bytes)| {
                    let id = match slot {
                        DeviceSlot::Cpu => DeviceId::Cpu,
                        DeviceSlot::Gpu(n) => DeviceId::Gpu(n),
                    };
                    (id, bytes)
                })
                .collect(),
        };

        Packed {
            allocation,
            args: CommandArgs {
                ngl,
                tensor_split,
                override_tensor,
                split_mode,
                main_gpu,
                n_cpu_moe: self.n_cpu_moe,
            },
            expert_offload_bytes,
            expert_offload_layers,
        }
    }
}
