//! Applying a service's learned rolling corrections to a placement, and
//! recording what the next correction will need from it.
//!
//! Every byte the packer attributes to a device flows through
//! [`Packer::charge`], which scales it by the destination pool's factor. That
//! is what makes a learned correction reach the placement at all: the pack
//! paths that matter place from `per_layer_bytes` and `expert_tensors`, so
//! scaling the `weights_bytes` scalar (as the correction once did) moved only
//! the tensor-split remainder and the fallback-architecture path — the layer
//! walk never read it.

use std::collections::BTreeMap;

use ananke_config::placement::DeviceSlot;

use crate::{packer::Packer, types::RollingInputs};

/// What a charged byte count represents. The distinction drives two separate
/// tallies: which bytes a host-RSS reading double-counts for GPU-resident
/// tensors ([`Self::Weights`]), and which bytes are a prediction of usage at
/// all ([`Self::Slop`] is not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Charge {
    /// Model tensors read from the GGUF: layer weights, the output head, token
    /// embeddings, expert tensors, a separate draft model's weights.
    Weights,
    /// Memory the runtime allocates rather than reads: KV cache, compute
    /// buffers, the MTP draft context.
    Runtime,
    /// Deliberate padding that is reserved but never expected to be used —
    /// today the one-layer tensor-split fudge.
    ///
    /// Excluded from the rolling bases. The correction's ratio is
    /// observed-over-*predicted*, and slop is not a prediction: counting it
    /// would make an accurate estimate read as an over-reservation by the slop
    /// fraction, and the next placement would then subtract exactly the
    /// headroom that was added on purpose. (See
    /// [`crate::packer::Packer::initialise_gpu_remaining`]
    /// for what that headroom is protecting against.)
    Slop,
}

impl<'a> Packer<'a> {
    /// Add `raw_bytes` to `slot`'s reservation, scaled by that pool's rolling
    /// correction, and return the amount actually charged.
    ///
    /// Every byte the packer attributes to a device goes through here, so that
    /// a learned correction reaches the whole reservation — not just the
    /// `weights_bytes` scalar, which the layer-aware paths never read. The
    /// correction is derived from an observed total over a predicted total, so
    /// applying it uniformly to every component of that total is what makes the
    /// next observation's ratio comparable to this one's.
    ///
    /// A fit check must compare a *corrected* cost against `gpu_remaining`;
    /// use [`Self::vram_cost`] to compute it before
    /// deciding a destination, then charge the same raw bytes here.
    pub(crate) fn charge(&mut self, slot: DeviceSlot, raw_bytes: u64, kind: Charge) -> u64 {
        let scaled = self.corrections.scale(&slot, raw_bytes);
        *self.per_device.entry(slot.clone()).or_default() += scaled;
        match kind {
            Charge::Weights => {
                *self.raw_per_device.entry(slot.clone()).or_default() += raw_bytes;
                *self.raw_weight_per_device.entry(slot).or_default() += raw_bytes;
            }
            Charge::Runtime => {
                *self.raw_per_device.entry(slot).or_default() += raw_bytes;
            }
            Charge::Slop => {}
        }
        scaled
    }

    /// The corrected cost of `raw_bytes` on any GPU. Uniform across cards —
    /// the correction is per pool, not per device — so a first-fit search can
    /// compute it once before choosing which card to charge.
    pub(crate) fn vram_cost(&self, raw_bytes: u64) -> u64 {
        self.corrections.scale(&DeviceSlot::Gpu(0), raw_bytes)
    }

    /// The uncorrected bases and GPU-resident weight total the rolling
    /// correction needs to turn this placement's next observation into a
    /// like-for-like ratio.
    pub(crate) fn rolling_inputs(&self) -> RollingInputs {
        let gpu_sum = |m: &BTreeMap<DeviceSlot, u64>| -> u64 {
            m.iter()
                .filter(|(slot, _)| matches!(slot, DeviceSlot::Gpu(_)))
                .map(|(_, b)| *b)
                .sum()
        };
        RollingInputs {
            uncorrected_vram_bytes: gpu_sum(&self.raw_per_device),
            uncorrected_host_bytes: self
                .raw_per_device
                .get(&DeviceSlot::Cpu)
                .copied()
                .unwrap_or(0),
            gpu_weight_bytes: gpu_sum(&self.raw_weight_per_device),
            cpu_weight_bytes: self
                .raw_weight_per_device
                .get(&DeviceSlot::Cpu)
                .copied()
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use ananke_config::placement::{OffloadMode, PlacementPolicy, SplitMode};

    use crate::{
        AllocationTable, Corrections,
        entry::{pack, pack_corrected},
        test_support::{
            GIB, MIB, cpu_bytes, moe_estimate, moe_svc, snapshot, svc, trivial_estimate,
        },
    };

    fn gpu_total(p: &crate::Packed) -> u64 {
        p.allocation
            .bytes
            .iter()
            .filter(|(id, _)| matches!(id, crate::devices::DeviceId::Gpu(_)))
            .map(|(_, b)| *b)
            .sum()
    }

    /// The point of the whole exercise: a VRAM correction above 1.0 has to
    /// change the placement. It reserves more per layer, so fewer layers fit
    /// the card and the surplus spills to the host.
    ///
    /// Scaling `Estimate::weights_bytes` — what the correction used to do —
    /// could not produce this, because the layer walk places from
    /// `per_layer_bytes` and never reads that scalar.
    #[test]
    fn a_vram_correction_moves_the_layer_walk() {
        // 40 layers × 500 MiB = ~19.5 GiB of weight against one 24 GiB card:
        // fits uncorrected, must not fit whole once inflated by 40%.
        let est = trivial_estimate(40, 500);
        let s = {
            let mut s = svc(PlacementPolicy::Hybrid, Some(vec![0]));
            s.placement_override = Default::default();
            s
        };
        let snap = snapshot(&[24]);
        let table = AllocationTable::new();

        let neutral = pack(&est, &s, &snap, &table).expect("uncorrected fit");
        assert_eq!(cpu_bytes(&neutral), 0, "uncorrected, every layer is on GPU");

        let corrected = pack_corrected(
            &est,
            &s,
            &snap,
            &table,
            Corrections {
                vram: 1.4,
                host: 1.0,
            },
            false,
        )
        .expect("corrected pack still resolves by spilling");
        assert!(
            cpu_bytes(&corrected) > 0,
            "a 1.4× VRAM correction must push layers off the card"
        );
        assert!(
            gpu_total(&corrected) > gpu_total(&neutral),
            "the layers that stayed must each be reserved at the corrected size \
             ({} vs {})",
            gpu_total(&corrected),
            gpu_total(&neutral)
        );
    }

    /// The host correction scales the CPU side and nothing else — the pools are
    /// independent, so a hybrid MoE that over-runs its host estimate must not
    /// have its GPU reservation touched.
    #[test]
    fn a_host_correction_scales_only_the_cpu_side() {
        // Experts far exceed one card, so the packer offloads a large CPU lump.
        let est = moe_estimate(32, 100, 400);
        let s = moe_svc(OffloadMode::Auto);
        let snap = snapshot(&[24]);
        let table = AllocationTable::new();

        let neutral = pack(&est, &s, &snap, &table).expect("neutral pack");
        assert!(cpu_bytes(&neutral) > GIB, "the fixture must spill experts");

        let corrected = pack_corrected(
            &est,
            &s,
            &snap,
            &table,
            Corrections {
                vram: 1.0,
                host: 1.25,
            },
            false,
        )
        .expect("host correction cannot break the GPU fit");
        assert_eq!(
            gpu_total(&corrected),
            gpu_total(&neutral),
            "a host correction must leave the GPU reservation alone"
        );
        // Not exactly 1.25× the neutral figure: which experts spill is decided
        // against the (unchanged) GPU pool, so the same bytes land on the host
        // and are then charged at the corrected rate.
        assert!(
            cpu_bytes(&corrected) > cpu_bytes(&neutral),
            "the host lump must be reserved at the corrected size"
        );
    }

    /// The bases handed to the next rolling update are the *uncorrected*
    /// predictions, so each sample estimates observed-over-predicted instead of
    /// integrating against a base the last correction already moved.
    ///
    /// Isolated to the host pool: a VRAM correction changes *which* experts fit
    /// the cards, so the host base legitimately moves with it — it describes the
    /// placement that was actually made. A host correction changes no placement
    /// decision (the GPU pool it is measured against is untouched), so the base
    /// must come out byte-identical while the reservation itself grows.
    #[test]
    fn rolling_inputs_are_uncorrected() {
        let est = moe_estimate(32, 100, 400);
        let s = moe_svc(OffloadMode::Auto);
        let snap = snapshot(&[24]);
        let table = AllocationTable::new();

        let neutral = pack(&est, &s, &snap, &table).expect("neutral pack");
        let corrected = pack_corrected(
            &est,
            &s,
            &snap,
            &table,
            Corrections {
                vram: 1.0,
                host: 1.25,
            },
            false,
        )
        .expect("corrected pack");

        assert_eq!(
            neutral.rolling.uncorrected_host_bytes, corrected.rolling.uncorrected_host_bytes,
            "the host base must not move with the host correction"
        );
        assert_eq!(
            neutral.rolling.uncorrected_vram_bytes, corrected.rolling.uncorrected_vram_bytes,
            "nor the VRAM base"
        );
        assert!(
            cpu_bytes(&corrected) > cpu_bytes(&neutral),
            "…even though the reservation itself did"
        );
        assert_eq!(
            cpu_bytes(&neutral),
            neutral.rolling.uncorrected_host_bytes,
            "an uncorrected pack's reservation and base are the same number"
        );
    }

    /// `gpu_weight_bytes` counts model tensors on GPUs and nothing else: it is
    /// subtracted from a measured RSS peak, so a KV or compute-buffer byte in
    /// there would silently deflate the host observation.
    #[test]
    fn gpu_weight_bytes_excludes_runtime_allocations() {
        let mut est = trivial_estimate(20, 200);
        est.kv_per_token = 64 * 1024;
        est.compute_buffer_mb = 400;
        let s = {
            let mut s = svc(PlacementPolicy::Hybrid, Some(vec![0]));
            s.placement_override = Default::default();
            s
        };
        let packed = pack(&est, &s, &snapshot(&[24]), &AllocationTable::new()).expect("fit");

        let gpu_reserved = gpu_total(&packed);
        let weights = packed.rolling.gpu_weight_bytes;
        assert_eq!(
            weights,
            20 * 200 * MIB,
            "every layer's weight, and only the weight"
        );
        assert!(
            gpu_reserved > weights,
            "the reservation must also carry KV + compute buffer ({gpu_reserved} vs {weights})"
        );
    }

    /// The slop the packer reserves on purpose must not reach the rolling
    /// base. It is never used, so counting it would make an accurate estimate
    /// read as an over-reservation by the slop fraction — and the next
    /// placement would then subtract exactly the headroom that
    /// `initialise_gpu_remaining` exists to hold.
    #[test]
    fn deliberate_slop_is_reserved_but_not_predicted() {
        let est = trivial_estimate(20, 200);
        let s = {
            let mut s = svc(PlacementPolicy::Hybrid, Some(vec![0]));
            s.placement_override = Default::default();
            s
        };
        let packed = pack(&est, &s, &snapshot(&[24]), &AllocationTable::new()).expect("fit");

        // Pinned absolutely, not as a difference: a term charged twice would
        // grow the reservation and the base together and leave any relative
        // assertion satisfied.
        let weights = 20 * 200 * MIB;
        let compute = est.compute_buffer_mb as u64 * MIB;
        let fudge = 200 * MIB;
        assert_eq!(
            packed.rolling.uncorrected_vram_bytes,
            weights + compute,
            "the prediction is the weights and the compute buffer"
        );
        assert_eq!(
            gpu_total(&packed),
            weights + compute + fudge,
            "the reservation is that plus one layer of slop"
        );
    }

    /// The sharded path builds its per-GPU share from five separately-rounded
    /// terms, so it is the easiest place to charge one of them twice or leave
    /// one out of the base. Pin the relationship: reservation minus prediction
    /// is exactly the fudge, on every spanned card.
    #[test]
    fn a_sharded_reservation_carries_exactly_one_fudge_share() {
        let est = trivial_estimate(20, 200);
        let mut s = svc(PlacementPolicy::GpuOnly, Some(vec![0, 1]));
        s.placement_override = Default::default();
        s.split_mode = SplitMode::Tensor;
        let packed =
            pack(&est, &s, &snapshot(&[24, 24]), &AllocationTable::new()).expect("sharded fit");

        // Absolute, for the reason given in the layer-path test above: the
        // five terms are rounded and charged separately, and a difference
        // assertion cannot tell a doubled term from a correct one.
        //
        // The compute buffer appears once *per card*, unlike the weights and
        // the KV: llama.cpp builds the same graph on every device under a
        // tensor split rather than dividing one between them, so a two-card
        // span pays it twice. That is the whole point of
        // [`ananke_estimate::compute_buffer::tensor_split_per_device`] being a
        // per-device figure.
        let weights = 20 * 200 * MIB;
        let compute = est.compute_buffer_mb as u64 * MIB * 2;
        let fudge = 200 * MIB;
        assert_eq!(
            packed.rolling.uncorrected_vram_bytes,
            weights + compute,
            "the prediction is the weights and both cards' compute buffers, \
             summed back across the shards"
        );
        assert_eq!(
            gpu_total(&packed),
            weights + compute + fudge,
            "the reservation is that plus one layer of slop"
        );
    }

    /// The two pack paths must agree on the host side. They once did not: the
    /// layer walk charged the `Cpu` slot a full GPU-calibrated compute buffer
    /// and the sharded path charged it nothing, so the same model on the same
    /// cards produced host slots differing by ~3.8 GiB — which is what made a
    /// multiplicative host ratio meaningless. Both now charge the modelled
    /// host overhead, and nothing else non-weight.
    #[test]
    fn both_pack_paths_charge_the_same_host_overhead() {
        let mut est = trivial_estimate(20, 200);
        est.non_layer.token_embd_bytes = GIB;
        let s = svc(PlacementPolicy::GpuOnly, Some(vec![0, 1]));
        let mut s = s;
        s.placement_override = Default::default();
        let snap = snapshot(&[24, 24]);

        let layer_split = pack(&est, &s, &snap, &AllocationTable::new()).expect("layer fit");
        s.split_mode = ananke_config::placement::SplitMode::Tensor;
        let sharded = pack(&est, &s, &snap, &AllocationTable::new()).expect("sharded fit");

        // Both hold exactly the token embeddings on the host …
        assert_eq!(layer_split.rolling.cpu_weight_bytes, GIB);
        assert_eq!(sharded.rolling.cpu_weight_bytes, GIB);
        // … and both charge the same non-weight host bytes on top.
        assert_eq!(
            layer_split.rolling.uncorrected_host_bytes, sharded.rolling.uncorrected_host_bytes,
            "the two paths must not disagree about the host slot"
        );
        assert_eq!(
            layer_split.rolling.uncorrected_host_bytes - GIB,
            est.host_overhead_bytes,
            "the only non-weight host term is the modelled overhead"
        );
    }

    /// A hybrid MoE — the shape issue #34 is about — puts its offloaded
    /// experts in `cpu_weight_bytes`, so it clears the host pool's learning
    /// floor and its `Cpu` slot is dominated by real host-resident weight.
    #[test]
    fn a_hybrid_moe_reports_its_offloaded_experts_as_host_weight() {
        let est = moe_estimate(48, 100, 400);
        let s = moe_svc(OffloadMode::Auto);
        let packed = pack(&est, &s, &snapshot(&[24]), &AllocationTable::new()).expect("hybrid fit");

        // `moe_estimate` carries no token embeddings, so the offloaded
        // experts are the whole of the host-resident weight; with embeddings
        // the invariant is `token_embd + offloaded experts`.
        assert_eq!(est.non_layer.token_embd_bytes, 0);
        assert_eq!(
            packed.rolling.cpu_weight_bytes, packed.expert_offload_bytes,
            "the host weight is exactly what was offloaded"
        );
        assert!(
            packed.rolling.cpu_weight_bytes * 10 > packed.rolling.uncorrected_host_bytes * 9,
            "offloaded experts must dominate the CPU slot ({} of {})",
            packed.rolling.cpu_weight_bytes,
            packed.rolling.uncorrected_host_bytes
        );
    }
}
