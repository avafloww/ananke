//! Reserving the MTP / NextN draft-context overhead ahead of the layer walk.

use ananke_config::placement::DeviceSlot;

use crate::packer::{Charge, Packer};

impl<'a> Packer<'a> {
    /// Reserve the MTP / NextN draft-context overhead (its KV cache plus
    /// compute buffer) as a single lump on the *last* allowed GPU. At
    /// runtime llama.cpp attaches this second context to the GPU that hosts
    /// the MTP head — the model's trailing layer — which the first-fit
    /// walker places on the GPU it fills last (the least-free in the sort,
    /// i.e. the spill target that keeps the most leftover room). Pinning the
    /// lump there both matches where it physically lands and avoids piling
    /// it onto the most-free GPU that the walker is simultaneously filling
    /// to the brim (which would overflow that card by the MTP size). Seeded
    /// *before* the layer walk so the walker reserves room for it. Zero when
    /// MTP is off or the model carries no MTP head.
    pub(crate) fn seed_mtp_overhead(&mut self) {
        if self.estimate.mtp_bytes == 0 {
            return;
        }
        let target = match self.allowed_gpus.last() {
            Some(last_gpu) => DeviceSlot::Gpu(*last_gpu),
            None => DeviceSlot::Cpu,
        };
        // A separate draft model's weights are read through its own mmap, so
        // they belong in the weight tally the host observation subtracts; an
        // embedded head's overhead is all runtime allocation.
        let weights = self.estimate.mtp_weight_bytes.min(self.estimate.mtp_bytes);
        self.charge(target.clone(), weights, Charge::Weights);
        self.charge(
            target,
            self.estimate.mtp_bytes.saturating_sub(weights),
            Charge::Runtime,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ananke_config::placement::PlacementPolicy;
    use ananke_estimate::{Estimate, NonLayer};
    use smol_str::SmolStr;

    use crate::{
        AllocationTable,
        devices::DeviceId,
        entry::pack,
        test_support::{snapshot, svc},
    };

    /// Regression for the live "insufficient_capacity on gpu:0" failure: the MTP
    /// draft-context lump must ride the *last* GPU (the spill target the
    /// trailing MTP head lands on), not pile onto the most-free GPU the
    /// first-fit walker is already filling to the brim. A model that spans
    /// both 24 GB cards plus a 3 GiB MTP lump must pack without overflowing
    /// GPU 0.
    #[test]
    fn mtp_overhead_rides_last_gpu_without_overflowing_first() {
        // 40 layers × 700 MiB ≈ 27.3 GiB of weights — does not fit one card,
        // so the walker spills onto GPU 1.
        let per_layer_bytes: Vec<u64> = (0..40).map(|_| 700 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let mtp_bytes = 3 * 1024 * 1024 * 1024;
        let e = Estimate {
            weights_bytes,
            kv_per_token: 0,
            compute_buffer_mb: 1000,
            output_buffer_bytes: 0,
            mtp_bytes,
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
            context: 4096,
            architecture: SmolStr::new("qwen35"),
        };
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("MTP model spanning two cards must pack");
        let gpu0 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(0))
            .copied()
            .unwrap_or(0);
        let gpu1 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(1))
            .copied()
            .unwrap_or(0);
        let cap = 24u64 * 1024 * 1024 * 1024;
        assert!(
            gpu0 <= cap,
            "GPU 0 must not be over-pledged: {gpu0} > {cap}"
        );
        assert!(
            gpu1 >= mtp_bytes,
            "the MTP lump must ride the last GPU (gpu1={gpu1}, mtp={mtp_bytes})"
        );
    }

    /// A separate draft model (`-md`) contributes real GGUF tensors, read
    /// through its own mmap and therefore resident in the process's file RSS.
    /// They belong in `gpu_weight_bytes` rather than in the runtime tally,
    /// which is what keeps the weight and runtime totals meaning what they say.
    #[test]
    fn a_separate_draft_models_weights_are_tallied_as_weights() {
        let per_layer_bytes: Vec<u64> = (0..10).map(|_| 200 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let draft_weights = 108 * 1024 * 1024;
        let compute = 300 * 1024 * 1024;
        let e = Estimate {
            weights_bytes,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            output_buffer_bytes: 0,
            mtp_bytes: draft_weights + compute,
            mtp_weight_bytes: draft_weights,
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
            context: 4096,
            architecture: SmolStr::new("gemma4"),
        };
        let packed = pack(
            &e,
            &svc(PlacementPolicy::GpuOnly, Some(vec![0])),
            &snapshot(&[24]),
            &AllocationTable::new(),
        )
        .expect("draft-MTP model must pack");

        assert_eq!(
            packed.rolling.gpu_weight_bytes,
            weights_bytes + draft_weights,
            "the draft's weights belong in the weight tally"
        );
        // The compute half stays out of it, but both halves are still reserved.
        let reserved: u64 = packed.allocation.bytes.values().copied().sum::<u64>();
        assert!(reserved >= weights_bytes + draft_weights + compute);
    }
}
