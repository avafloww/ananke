//! Reserving the MTP / NextN draft-context overhead ahead of the layer walk.

use crate::{allocator::placement::packer::Packer, config::DeviceSlot};

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
        *self.per_device.entry(target).or_default() += self.estimate.mtp_bytes;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use crate::{
        allocator::{
            AllocationTable,
            placement::{
                entry::pack,
                test_support::{snapshot, svc},
            },
        },
        config::PlacementPolicy,
        devices::DeviceId,
        estimator::{Estimate, NonLayer},
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
}
