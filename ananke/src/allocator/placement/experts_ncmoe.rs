//! Phase B of the expert-aware path: deciding how many trailing expert
//! *layers* to offload to CPU as whole units via `--n-cpu-moe`, after
//! [`crate::allocator::placement::experts_nonexpert`] has pinned every
//! layer's non-expert weight to a GPU.

use crate::{
    allocator::placement::{packer::Packer, types::PackError},
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
        if gpu_expert_bytes > pool {
            return Err(PackError::ManualExpertsDoNotFit {
                gpu_index: self.allowed_gpus.first().copied().unwrap_or(0),
                bytes: gpu_expert_bytes.saturating_sub(pool),
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

#[cfg(test)]
mod tests {
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
        assert!(
            matches!(err, PackError::ManualExpertsDoNotFit { gpu_index: 0, .. }),
            "expected ManualExpertsDoNotFit on gpu:0, got {err:?}"
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
    /// overloading it into an `insufficient_vram` error while gpu:1 sat idle.
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
    /// `insufficient_vram: no fit on gpu:0`.
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
}
