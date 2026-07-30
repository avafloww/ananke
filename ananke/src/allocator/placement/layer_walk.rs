//! The default (non-expert-aware, non-sharded) placement path: a first-fit
//! walk that assigns whole layers to the most-free-first GPU list, then
//! reserves KV, the compute buffer, and the one-layer fudge on top.

use crate::{
    allocator::placement::{
        packer::{Charge, Packer},
        types::PackError,
    },
    config::DeviceSlot,
};

impl<'a> Packer<'a> {
    /// Step 2: first-fit layer walker. Pre-reserves per-GPU headroom for
    /// steps 3-5 so we don't fill to the brim and then overflow. Returns
    /// `PackError` if a layer's bytes don't fit on any allowed GPU and CPU
    /// spill is disabled.
    ///
    /// KV cost is folded into the per-layer fit check (`layer_bytes +
    /// kv_per_layer`) so KV headroom accumulates alongside layers rather than
    /// being validated in a separate post-walk pass. This lets large
    /// long-context models span GPUs correctly while keeping small models on
    /// the single least-busy GPU.
    pub(crate) fn walk_layers(&mut self) -> Result<(), PackError> {
        self.initialise_gpu_remaining();

        let n_layers = self.per_layer.len() as u64;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let kv_per_layer = kv_total.checked_div(n_layers).unwrap_or(0);

        for (idx, bytes) in self.per_layer.clone().into_iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            // Corrected, since `gpu_remaining` is a corrected budget: the
            // headroom pre-reservation and every other charge are scaled, so a
            // raw layer cost would fit more layers than the reservation can pay
            // for.
            let layer_cost = self.vram_cost(bytes.saturating_add(kv_per_layer));
            // First-fit on the sorted (most-free-first) GPU list: fills the
            // least-busy GPU before spilling to the next. Small models stay on
            // one GPU; models that genuinely span multiple GPUs still pack
            // correctly because the KV cost is folded into the fit check.
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .find(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= layer_cost);
            match placed {
                Some(gpu) => {
                    *self.gpu_remaining.entry(gpu).or_default() -= layer_cost;
                    self.charge(DeviceSlot::Gpu(gpu), bytes, Charge::Weights);
                    *self.layers_per_gpu.entry(gpu).or_default() += 1;
                }
                None if self.allow_cpu => {
                    self.charge(DeviceSlot::Cpu, bytes, Charge::Weights);
                    self.layers_on_cpu += 1;
                }
                None => {
                    return Err(PackError::LayerDoesNotFit {
                        layer_index: idx as u32,
                        bytes,
                        shortfalls: self.gpu_shortfalls(layer_cost),
                    });
                }
            }
        }
        Ok(())
    }

    /// Fallback for architectures (Mamba, unknown) that didn't supply a
    /// per-layer breakdown: place the entire weights bundle on the first GPU
    /// with room, or spill to CPU.
    pub(crate) fn place_fallback_weights(&mut self) -> Result<(), PackError> {
        if !self.per_layer.is_empty() || self.estimate.weights_bytes == 0 {
            return Ok(());
        }
        let bytes = self.estimate.weights_bytes;
        let cost = self.vram_cost(bytes);
        for gpu in self.allowed_gpus.clone() {
            let rem = self.gpu_remaining.entry(gpu).or_default();
            if *rem >= cost {
                *rem -= cost;
                self.charge(DeviceSlot::Gpu(gpu), bytes, Charge::Weights);
                self.fallback_on_gpu = true;
                return Ok(());
            }
        }
        if self.allow_cpu {
            self.charge(DeviceSlot::Cpu, bytes, Charge::Weights);
            Ok(())
        } else {
            Err(PackError::WeightsDoNotFit {
                shortfalls: self.gpu_shortfalls(cost),
            })
        }
    }

    /// Step 3: add KV bytes to GPUs proportional to layers placed, or to CPU
    /// for layers that spilled.
    pub(crate) fn add_kv_bytes(&mut self) {
        let n_layers = self.per_layer.len() as u32;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        if n_layers == 0 || kv_total == 0 {
            return;
        }
        for gpu in self.allowed_gpus.clone() {
            let share = self.layers_per_gpu.get(&gpu).copied().unwrap_or(0);
            if share > 0 {
                let bytes = kv_total * share as u64 / n_layers as u64;
                self.charge(DeviceSlot::Gpu(gpu), bytes, Charge::Runtime);
            }
        }
        if self.layers_on_cpu > 0 {
            let bytes = kv_total * self.layers_on_cpu as u64 / n_layers as u64;
            self.charge(DeviceSlot::Cpu, bytes, Charge::Runtime);
        }
    }

    /// Step 4: compute buffer per active *GPU*, plus the host-side overhead
    /// on the CPU slot.
    ///
    /// The two are different quantities and were once the same number: the
    /// CPU slot used to be charged `compute_buffer_mb`, which is calibrated
    /// against `nvidia-smi` VRAM readings and has never been measured on a
    /// host backend. What the host actually holds is the pinned graph arena
    /// and the prompt cache, which scale differently and exist even when every
    /// layer is on a GPU — see [`crate::estimator::host_buffer`].
    pub(crate) fn add_compute_buffer(&mut self) {
        let compute_bytes = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        // The output logits buffer is materialised only on the head GPU.
        // `compute_buffer_mb` is calibrated against that head GPU, so it already
        // includes the logits term; every *other* GPU's real compute buffer is
        // smaller by that amount. Trim it off the secondaries so their
        // reservation reflects reality and the freed VRAM fills with expert
        // weight instead. CPU and the head GPU keep the full term. Must use the
        // same head + trim as [`Packer::initialise_gpu_remaining`] and the
        // tensor-split. See [`crate::estimator::Estimate::output_buffer_bytes`].
        let head_gpu = self.head_gpu();
        let logits = self.head_logits_bytes();
        let active_gpus: Vec<DeviceSlot> = self
            .per_device
            .keys()
            .filter(|s| matches!(s, DeviceSlot::Gpu(_)))
            .cloned()
            .collect();
        for slot in active_gpus {
            let mut add = compute_bytes;
            if let DeviceSlot::Gpu(id) = slot {
                if head_gpu != Some(id) {
                    add = add.saturating_sub(logits);
                } else {
                    // The vision projector's CLIP graph buffer rides one
                    // device, and llama.cpp names it: the same head GPU that
                    // holds the projector's weights.
                    add = add.saturating_add(self.estimate.mmproj_graph_bytes);
                }
            }
            self.charge(slot, add, Charge::Runtime);
        }
        self.add_host_overhead();
    }

    /// Charge the `Cpu` slot the host-side overhead. Unconditional, because it
    /// is held whatever the placement — a fully GPU-offloaded model still pins
    /// a KQ mask on the host and still runs a CUDA runtime there.
    ///
    /// The prompt cache is charged separately as slop: it is a *cap* the cache
    /// grows into rather than an allocation made at load, so reserving it is
    /// right but predicting it is not. Counted as a prediction it would make
    /// every host observation read as a large over-reservation — an 8 GiB
    /// default against tens of MiB actually held — and pin the correction to
    /// its clamp floor.
    pub(crate) fn add_host_overhead(&mut self) {
        if self.estimate.host_overhead_bytes > 0 {
            self.charge(
                DeviceSlot::Cpu,
                self.estimate.host_overhead_bytes,
                Charge::Runtime,
            );
        }
        if self.estimate.host_cache_bytes > 0 {
            self.charge(
                DeviceSlot::Cpu,
                self.estimate.host_cache_bytes,
                Charge::Slop,
            );
        }
        if self.estimate.host_slot_bytes > 0 {
            self.charge(DeviceSlot::Cpu, self.estimate.host_slot_bytes, Charge::Slop);
        }
    }

    /// Step 5: one-layer fudge for tensor-split slop.
    pub(crate) fn add_one_layer_fudge(&mut self) {
        let n_layers = self.per_layer.len() as u32;
        if n_layers == 0 || self.per_layer.is_empty() {
            return;
        }
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let per_layer_avg = self.effective_layer_avg();
        let per_layer_kv = kv_total / n_layers as u64;
        let fudge_each = crate::allocator::placement::entry::ONE_LAYER_FUDGE_MULTIPLIER
            * (per_layer_avg + per_layer_kv);
        let slots: Vec<DeviceSlot> = self.per_device.keys().cloned().collect();
        for slot in slots {
            match slot {
                DeviceSlot::Gpu(_) => {
                    self.charge(slot, fudge_each, Charge::Slop);
                }
                DeviceSlot::Cpu if self.layers_on_cpu > 0 => {
                    self.charge(slot, fudge_each, Charge::Slop);
                }
                _ => {}
            }
        }
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
                entry::{pack, pack_optimistic},
                test_support::{snapshot, svc, trivial_estimate},
            },
        },
        config::PlacementPolicy,
        devices::DeviceId,
        estimator::{Estimate, NonLayer},
    };

    /// The `-ot` synthesiser collapses a tail of fully-offloaded layers into a
    /// single grouped rule with layer and kind alternations.
    #[test]
    fn single_gpu_fits() {
        let e = trivial_estimate(10, 100); // 10 layers × 100 MiB = 1 GiB
        let snap = snapshot(&[8]); // 8 GB free
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(10));
        assert!(packed.args.tensor_split.is_none());
    }

    /// llama.cpp names one device for the projector's CLIP graph buffer, so it
    /// is charged once and on the head GPU — not once per card the way the
    /// compute buffer is. Charging it per card would over-reserve a two-card
    /// span by 248 MiB.
    #[test]
    fn the_mmproj_graph_buffer_rides_the_head_gpu_alone() {
        let mut e = trivial_estimate(20, 200);
        let graph = 248 * 1024 * 1024;
        e.mmproj_graph_bytes = graph;
        let plain = {
            let mut without = e.clone();
            without.mmproj_graph_bytes = 0;
            pack(
                &without,
                &svc(PlacementPolicy::GpuOnly, Some(vec![0, 1])),
                &snapshot(&[12, 12]),
                &AllocationTable::new(),
            )
            .expect("fit")
        };
        let packed = pack(
            &e,
            &svc(PlacementPolicy::GpuOnly, Some(vec![0, 1])),
            &snapshot(&[12, 12]),
            &AllocationTable::new(),
        )
        .expect("fit");
        let total = |p: &crate::allocator::placement::Packed| -> u64 {
            p.allocation
                .bytes
                .iter()
                .filter(|(id, _)| matches!(id, DeviceId::Gpu(_)))
                .map(|(_, b)| *b)
                .sum()
        };
        assert_eq!(
            total(&packed) - total(&plain),
            graph,
            "exactly one graph buffer across both cards"
        );
    }

    #[test]
    fn multi_gpu_split_ratios_match_layer_counts() {
        // 20 layers, 1 GiB each; 2 GPUs with 12 GB free each.
        let e = trivial_estimate(20, 1024); // 20 GiB
        let snap = snapshot(&[12, 12]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split.iter().sum::<u32>(), packed.args.ngl.unwrap());
    }

    #[test]
    fn hybrid_spills_to_cpu() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[0]); // GPU full
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::Hybrid, None), &snap, &alloc).unwrap();
        // All layers should have spilled to CPU.
        assert!(packed.allocation.bytes.contains_key(&DeviceId::Cpu));
    }

    #[test]
    fn cpu_only_emits_ngl_zero_and_no_split() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[8]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::CpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(0));
        assert!(packed.args.tensor_split.is_none());
    }

    /// Small models (fewer bytes than a single GPU's headroom-adjusted
    /// capacity) should land entirely on one GPU rather than being splayed
    /// across devices. With two equal-free GPUs and two layers that both fit
    /// on GPU 0, first-fit should produce [2, 0], not [1, 1].
    #[test]
    fn first_fit_packs_small_model_onto_one_gpu() {
        let e = trivial_estimate(2, 1024); // 2 layers, 1 GiB each
        let snap = snapshot(&[20, 20]); // 20 GB free on each of two GPUs
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        // Both layers fit on GPU 0 — no reason to split across GPUs.
        assert_eq!(packed.args.ngl, Some(2));
        // tensor_split is emitted but GPU 1 carries zero layers.
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.iter().sum::<u32>(), 2);
        assert_eq!(
            split[0], 2,
            "all layers should be on GPU 0 (first in sorted order for equal capacity)"
        );
        assert_eq!(split[1], 0);
    }

    /// When GPU 0 already has reservations from other services, GPU 1 has
    /// more pledge-book headroom and is sorted first. A new model that fits
    /// on one GPU goes entirely to GPU 1, regardless of how nvml_free
    /// compares between the two devices.
    #[test]
    fn first_fit_targets_least_pledged_gpu() {
        let e = trivial_estimate(4, 1024); // 4 × 1 GiB layers
        // Free bytes differ from the pledge picture — the sort must ignore them.
        let snap = snapshot(&[10, 16]);
        // Another service has 6 GiB committed on GPU 0; GPU 1 is unencumbered.
        let mut reserved = AllocationTable::new();
        let mut other = BTreeMap::new();
        other.insert(DeviceSlot::Gpu(0), 6 * 1024u64); // MB
        reserved.insert(SmolStr::new("other"), other);
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &reserved).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.iter().sum::<u32>(), 4);
        // GPU 1 (more pledge headroom) is sorted first; all layers land there.
        assert!(
            split[1] >= split[0],
            "first-fit should place all layers on GPU 1 (less pledged); got {split:?}"
        );
    }

    /// `pack_optimistic` ignores nvml's view of free bytes and trusts the
    /// pledge book (`total - reserved`) alone. Used on the retry-after-
    /// eviction path: the victims have been drained from `reserved`, but
    /// nvml still reports their realized usage until the drain actually
    /// lands. `pack` would reject the placement here; `pack_optimistic`
    /// should succeed.
    #[test]
    fn pack_optimistic_ignores_stale_nvml_free() {
        let e = trivial_estimate(4, 1024); // 4 GiB of layers
        let snap = snapshot(&[0]); // nvml says 0 free, but total = 24 GB
        let alloc = AllocationTable::new();

        // Conservative pack: `min(0, 24-0) = 0` per GPU, no spill allowed.
        let err = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc);
        assert!(
            err.is_err(),
            "pack must reject placement when nvml reports 0 free and spill is off"
        );

        // Optimistic pack: trust `total - reserved = 24 GB`, layers fit.
        let packed = pack_optimistic(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("pack_optimistic must succeed when the pledge book allows it");
        assert_eq!(packed.args.ngl, Some(4));
    }

    /// Regression for the 256K-context Gemma 4 31B repro: folding KV cost
    /// into the per-layer fit check during the walk (rather than a post-walk
    /// validation step) must still allow this model to pack across two GPUs.
    /// The model (60 layers × 296 MB ≈ 17.8 GB weights + 11.85 GB KV) does
    /// not fit on a single 24 GB GPU once KV is included, so the walk spills
    /// part of it to the second GPU.
    #[test]
    fn long_context_moe_packs_across_two_gpus() {
        // Gemma 4 31B numbers from the live failure log: 60 layers at ~296 MB
        // avg (≈17.8 GB total), kv_per_token = 45220, 256 K context, compute
        // buffer 3792 MB.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 296 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 45220,
            compute_buffer_mb: 3792,
            output_buffer_bytes: 0,
            mtp_bytes: 0,
            mtp_weight_bytes: 0,
            mmproj_graph_bytes: 0,
            mtp_head_expert_layers: 0,
            tensor_split_replicated_bytes: 0,
            host_overhead_bytes: 0,
            host_cache_bytes: 0,
            host_slot_bytes: 0,
            host_checkpoint_bytes: 0,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 262_144,
            architecture: SmolStr::new("gemma4"),
        };
        // 2×24 GB 3090s, fully free, empty pledge book.
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("Gemma 4 31B at 256K context must pack on 2×24 GB");
        let split = packed.args.tensor_split.as_ref().expect("two-GPU split");
        // Total layers must add up.
        assert_eq!(split.iter().sum::<u32>(), 60);
        // KV cost folds layers onto the first GPU until full; the remainder
        // spills to the second. Both must carry layers.
        assert!(
            split[0] > 0 && split[1] > 0,
            "both GPUs must carry layers for this model; got {split:?}"
        );
    }

    /// A model whose KV-inclusive layer cost overflows even a single GPU must
    /// be rejected. With KV folded into the per-layer walk cost, the walker
    /// detects the overflow directly (LayerDoesNotFit) rather than via a
    /// separate post-walk validation step.
    #[test]
    fn long_context_overflows_single_gpu() {
        // 60 layers × 200 MB = 12 GB weights. kv_total at 128K tokens with
        // kv_per_token = 120 KB ≈ 15 GB. Per-layer cost = 200 + 261 MB ≈
        // 461 MB. Only 47 layers fit on a 24 GB GPU once compute buffer (2 GB)
        // and fudge are reserved; layer 48 overflows.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 200 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 120_000,
            compute_buffer_mb: 2048,
            output_buffer_bytes: 0,
            mtp_bytes: 0,
            mtp_weight_bytes: 0,
            mmproj_graph_bytes: 0,
            mtp_head_expert_layers: 0,
            tensor_split_replicated_bytes: 0,
            host_overhead_bytes: 0,
            host_cache_bytes: 0,
            host_slot_bytes: 0,
            host_checkpoint_bytes: 0,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 131_072,
            architecture: SmolStr::new("qwen3"),
        };
        // Single 24 GB GPU, no spill allowed.
        let snap = snapshot(&[24]);
        let alloc = AllocationTable::new();
        let err = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect_err("KV-inclusive layer cost must overflow the single GPU");
        assert!(
            matches!(err, PackError::LayerDoesNotFit { .. }),
            "expected LayerDoesNotFit when KV pushes layer cost past GPU capacity, got {err:?}"
        );
    }
}
