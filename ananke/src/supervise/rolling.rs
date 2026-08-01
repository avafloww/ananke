//! Folding a finished run's observed memory back into the service's rolling
//! estimator corrections.
//!
//! One correction per memory pool, each learned from its own observation over
//! its own base — see [`crate::tracking::rolling`] for why a single mean over
//! the all-device total cannot be divided into a like-for-like ratio. This
//! module owns the two halves of that: capturing what a committed placement
//! predicted ([`RollingBase`]), and turning the peaks measured during the run
//! into samples ([`RunLoop::record_rolling_observation`]).
//!
//! Most of the care here is in deciding when *not* to record. A ratio whose
//! two sides measure different things does not announce itself — the clamp
//! turns it into a plausible-looking number — so every case where the
//! measurement cannot support a ratio is screened out explicitly rather than
//! left to average away.

use tracing::debug;

use crate::{
    allocator::placement::RollingInputs, supervise::RunLoop, tracking::rolling::MemoryClass,
};

impl RunLoop {
    /// Stash what the drain-time rolling update will need from the placement
    /// just committed. Called from both reservation-commit sites, right before
    /// the allocation table insert.
    ///
    /// Reads the inputs off `packed_for_spawn`, which the estimator path has
    /// just filled in. A `placement_override` service leaves it `None` and a
    /// command service fills it with zeroed inputs; both mean "not an
    /// estimate", and both end up skipping the update.
    pub(crate) fn capture_rolling_base(&mut self) {
        let inputs = self
            .packed_for_spawn
            .as_ref()
            .map(|p| p.rolling)
            .unwrap_or_default();
        self.rolling_base = RollingBase::new(inputs);
    }

    /// Fold this run's observed peaks into the service's two rolling
    /// corrections, one per memory pool.
    ///
    /// Each pool's ratio compares like with like — the pool's observed peak
    /// over the *uncorrected* bytes this run's placement predicted for it.
    /// Both are needed: a VRAM peak over an all-device base reads a hybrid
    /// MoE's accurate estimate as a 5× over-prediction (most of its footprint
    /// is host-resident experts), and a ratio taken against a corrected base
    /// would measure the last correction rather than the estimator.
    ///
    /// The decision of what to record is [`RollingBase::samples_from`]; this
    /// is the I/O around it.
    pub(crate) fn record_rolling_observation(&mut self) {
        let name = &self.init.identity.name;
        let base = self.rolling_base;
        let peak_vram = self.deps.observation.read_peak_vram(name);
        let peak_owned = self.deps.observation.read_peak_rss_owned(name);
        let peak_file = self.deps.observation.read_peak_rss_file(name);
        debug!(
            service = %name,
            peak_rss_file = peak_file,
            cpu_weight_bytes = base.inputs.cpu_weight_bytes,
            weights_anonymous = base.weights_are_anonymous(peak_file),
            "host-pool: classified host-resident weights as mapped or anonymous"
        );
        let samples = base.samples_from(peak_vram, peak_owned, peak_file);

        if samples.run_was_measurable {
            // The run survived to a clean drain, so an earlier OOM bump has
            // its answer: the larger reservation worked, and holding it would
            // over-pledge for the rest of the daemon's life. Before the
            // update, so this run's observation folds into the pool the bump
            // was hiding rather than into the bump.
            self.deps.rolling.clear_oom_bump(name);
        }

        match samples.vram {
            Some(peak) => self.deps.rolling.update(
                name,
                MemoryClass::Vram,
                peak,
                base.inputs.uncorrected_vram_bytes,
            ),
            None => debug!(
                service = %name,
                became_ready = base.became_ready,
                peak_vram,
                "no vram rolling sample from this run"
            ),
        }

        match samples.host {
            Some(peak) => {
                self.deps
                    .rolling
                    .update(name, MemoryClass::Host, peak, base.host_bytes())
            }
            None => debug!(
                service = %name,
                became_ready = base.became_ready,
                peak_owned,
                peak_file,
                host_base = base.host_bytes(),
                "no host rolling sample from this run"
            ),
        }
    }
}

/// What one run's observations are worth to each pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RollingSamples {
    /// The VRAM numerator to record, if any.
    pub(crate) vram: Option<u64>,
    /// The host numerator to record, if any.
    pub(crate) host: Option<u64>,
    /// Whether the run got far enough to be measured at all. Distinct from
    /// "produced a sample": a GPU-only service is measurable and still yields
    /// no host sample.
    pub(crate) run_was_measurable: bool,
}

/// What [`RunLoop::record_rolling_observation`] needs from the placement that
/// produced the current run's reservation, captured when the reservation is
/// committed so a mid-run config reload can't change it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RollingBase {
    pub(crate) inputs: RollingInputs,
    /// Whether the run passed its health check, i.e. finished loading.
    ///
    /// Memory sampled before that point is a snapshot of a partial load, and
    /// the observation tables keep monotonic peaks, so a run that ended during
    /// loading leaves a peak that is real but *incomplete* — 8 GiB of a 21 GiB
    /// model. Feeding that to `update` is a ratio of 0.38, which the clamp
    /// turns into a plausible-looking 0.8 and the packer then spends fitting a
    /// quarter more model onto a card than holds. Only a run that became ready
    /// can have been measured whole.
    pub(crate) became_ready: bool,
}

impl RollingBase {
    /// Capture the rolling inputs of a committed placement.
    pub(crate) fn new(inputs: RollingInputs) -> Self {
        Self {
            inputs,
            became_ready: false,
        }
    }

    /// The host-pool numerator: the observed peak of *anonymous* host memory,
    /// or `None` when there was no measurement.
    ///
    /// Anonymous rather than total, because total resident memory is not a
    /// measure of what this process holds. llama.cpp maps the GGUF with
    /// `MAP_POPULATE` and, after loading, unmaps only the fragments outside
    /// the span of host-resident tensors — so a hybrid model leaves nearly the
    /// whole file resident as clean, reclaimable file pages, GPU-destined
    /// weights included. `RssAnon + RssShmem` is what the runtime actually allocated:
    /// the pinned graph arena, the prompt cache, the CPU-side KV cache, the
    /// heap.
    ///
    /// The matching denominator is [`Self::host_bytes`], which drops the
    /// weight term for the same reason.
    pub(crate) fn host_peak(&self, owned: u64, file: u64) -> Option<u64> {
        let numerator = if self.weights_are_anonymous(file) {
            owned.saturating_sub(self.inputs.cpu_weight_bytes)
        } else {
            owned
        };
        Some(numerator).filter(|b| *b > 0)
    }

    /// Whether this run read its host-resident weights into anonymous memory
    /// rather than mapping them, judged by whether they showed up in the
    /// mapped counter.
    ///
    /// This is measured rather than inferred from configuration because no
    /// flag reliably says. Mainline llama.cpp maps (logging `CPU_Mapped model
    /// buffer size`), and its mapped RSS was the host weight total plus a
    /// couple of hundred MiB of shared libraries in every configuration tried.
    /// ik_llama, on the same model and the same flags, logged `CPU buffer
    /// size` and put 56 GiB of weights in anonymous memory with a mapped RSS
    /// of 72 MiB. Trusting `mmap`/`-rtr` reads such a run's weights into the
    /// numerator while leaving them out of the denominator — a 56 GiB
    /// mismatch the clamp turns into a 50% over-reservation.
    ///
    /// The halfway threshold is deliberately loose: the two regimes differ by
    /// orders of magnitude, not by a few percent.
    fn weights_are_anonymous(&self, file: u64) -> bool {
        self.inputs.cpu_weight_bytes > 0 && file < self.inputs.cpu_weight_bytes / 2
    }

    /// Decide what this run's observations are worth to each pool.
    ///
    /// Pure, because every interesting case here is a case where the honest
    /// answer is "record nothing" — and a ratio whose two sides measure
    /// different things does not announce itself, so those cases have to be
    /// enumerable and testable rather than inferred from a log.
    ///
    /// A run that never became ready yields nothing at all: memory sampled
    /// during loading is a peak of a *partial* model, and the observation
    /// tables keep monotonic peaks, so 8 GiB of a 21 GiB model reads as a real
    /// 0.38 over-prediction that the clamp launders into a plausible 0.8.
    ///
    /// A zero peak is likewise the absence of a measurement rather than an
    /// observation of zero — the snapshotter drops ticks that attribute
    /// nothing.
    pub(crate) fn samples_from(
        &self,
        peak_vram_bytes: u64,
        peak_owned_bytes: u64,
        peak_file_bytes: u64,
    ) -> RollingSamples {
        if !self.became_ready {
            return RollingSamples {
                vram: None,
                host: None,
                run_was_measurable: false,
            };
        }
        RollingSamples {
            vram: Some(peak_vram_bytes).filter(|p| *p > 0),
            host: self.host_peak(peak_owned_bytes, peak_file_bytes),
            run_was_measurable: true,
        }
    }

    /// Mark the run as fully loaded. See [`Self::became_ready`].
    pub(crate) fn run_became_ready(&mut self) {
        self.became_ready = true;
    }

    /// The host-pool denominator: the part of the `Cpu` slot that corresponds
    /// to anonymous memory, as the estimator predicted it.
    ///
    /// Weights leave both sides of the ratio. Where they land in `/proc`
    /// depends on the runtime — mapped by mainline llama.cpp, anonymous by
    /// ik_llama — so [`Self::host_peak`] removes them from the numerator when
    /// the measurement says they are there, and this removes them from the
    /// denominator unconditionally.
    ///
    /// It also puts the correction where the uncertainty is. Host-resident
    /// weight is exact arithmetic over the GGUF tensor table; there is nothing
    /// to learn about it. What the daemon models rather than reads — the graph
    /// arena, the process baseline, the CPU KV share — is precisely what is
    /// left in this denominator.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.inputs
            .uncorrected_host_bytes
            .saturating_sub(self.inputs.cpu_weight_bytes)
    }
}
#[cfg(test)]
mod tests {
    use ananke_config::units::MIB;
    use smol_str::SmolStr;

    use super::*;
    use crate::tracking::rolling::{MIN_TRUSTED_SAMPLES, MemoryClass, RollingTable};

    const GIB: u64 = 1024 * 1024 * 1024;

    /// `host` is the whole `Cpu` slot; `cpu_weight` is the host-resident model
    /// weight within it, which leaves both sides of the host ratio.
    fn base(vram: u64, host: u64, cpu_weight: u64) -> RollingBase {
        RollingBase {
            inputs: RollingInputs {
                uncorrected_vram_bytes: vram,
                uncorrected_host_bytes: host,
                // Not exercised by these tests: the host pool never reads the
                // GPU-side weight total.
                gpu_weight_bytes: 0,
                cpu_weight_bytes: cpu_weight,
            },
            became_ready: true,
        }
    }

    /// The `deepseek-v4-flash` shape: ~20 GiB of VRAM and ~105 GiB of experts
    /// in host RAM. Each pool's peak is divided by that pool's own base, so an
    /// accurate estimate converges to 1.0 on both — where a single mean over
    /// the all-device total reads the VRAM peak as a 5x over-prediction and
    /// pins to the 0.8 clamp floor.
    #[test]
    fn hybrid_converges_to_neutral_in_both_pools() {
        // 105 GiB of host bytes, of which 96 GiB is mapped expert weight and
        // the remaining 9 GiB is the prompt cache, the graph arena, and the
        // CPU's KV share.
        let b = base(20 * GIB, 105 * GIB, 96 * GIB);
        assert_eq!(b.host_bytes(), 9 * GIB, "the weight term leaves the base");

        let table = RollingTable::new();
        let svc = SmolStr::new("hybrid-moe");
        for _ in 0..MIN_TRUSTED_SAMPLES {
            table.update(
                &svc,
                MemoryClass::Vram,
                20 * GIB,
                b.inputs.uncorrected_vram_bytes,
            );
            table.update(
                &svc,
                MemoryClass::Host,
                b.host_peak(9 * GIB, 96 * GIB).expect("a peak was observed"),
                b.host_bytes(),
            );
        }
        let c = table.get(&svc).corrections();
        assert_eq!(c.vram, 1.0);
        assert_eq!(c.host, 1.0);
    }

    /// A `cpu-only` service has no GPU slot, so the VRAM pool never learns —
    /// but the host pool must, at any size. There is no absolute weight floor:
    /// one set high enough to matter would exclude every model under 27 GiB,
    /// and with a denominator that is a genuine prediction of anonymous memory
    /// there is nothing for a floor to protect against, so a small model learns
    /// like a large one.
    #[test]
    fn a_small_cpu_only_service_still_learns_the_host_pool() {
        // An 11 GiB model — a 13B at Q6_K — with 9 GiB of anonymous runtime
        // memory on top.
        let b = base(0, 20 * GIB, 11 * GIB);
        assert_eq!(b.host_bytes(), 9 * GIB);

        let table = RollingTable::new();
        let svc = SmolStr::new("cpu-only");
        for _ in 0..MIN_TRUSTED_SAMPLES {
            table.update(&svc, MemoryClass::Vram, 0, b.inputs.uncorrected_vram_bytes);
            table.update(
                &svc,
                MemoryClass::Host,
                b.host_peak(10 * GIB, 11 * GIB)
                    .expect("a peak was observed"),
                b.host_bytes(),
            );
        }
        let rc = table.get(&svc);
        assert_eq!(rc.vram.samples, 0, "no GPU slot, nothing to learn");
        assert_eq!(rc.host.samples, MIN_TRUSTED_SAMPLES);
        assert!(
            rc.corrections().host > 1.0,
            "10 GiB observed against a 9 GiB prediction is an under-estimate"
        );
    }

    /// Where the weights land in `/proc` is a property of the *runtime*, not
    /// of any flag ananke sets, so the numerator is decided by measurement.
    ///
    /// The two rows are the *same model with the same flags* on the two
    /// runtimes ananke supports — Qwen3.6-35B-A3B under `--n-cpu-moe 40`,
    /// one GPU, no `--no-mmap` on either. Mainline mapped its 24 GiB of
    /// offloaded experts (`CPU_Mapped model buffer size`), reporting 465 MiB
    /// owned against 24.9 GiB mapped; ik_llama allocated them anonymously
    /// (`CPU buffer size`), reporting 23.8 GiB owned against 164 MiB mapped.
    ///
    /// The difference is specific to the expert-offload path: at `-ngl 0` the
    /// same ik build *does* map, and honours `--no-mmap` when given it. So no
    /// configuration ananke can read tells the two apart — only the
    /// measurement does.
    ///
    /// Getting it wrong is not a rounding error. Treating ik's run as mapped
    /// divides 23.8 GiB by a base of a few hundred MiB, which clamps to 1.5
    /// and over-reserves a host slot tens of GiB wide by half.
    #[test]
    fn the_numerator_follows_the_measurement_not_the_flags() {
        // mainline: experts mapped, so `owned` is already weights-free.
        let mapped = base(4 * GIB, 25 * GIB, 24 * GIB);
        assert_eq!(
            mapped.host_peak(465 * MIB, 24 * GIB),
            Some(465 * MIB),
            "mapped weights are not in the owned figure; nothing to remove"
        );

        // ik_llama: same model, same flags, experts anonymous.
        let anon = base(4 * GIB, 25 * GIB, 24 * GIB);
        assert_eq!(
            anon.host_peak(24 * GIB + 465 * MIB, 164 * MIB),
            Some(465 * MIB),
            "anonymous weights inflate the owned figure and must be removed"
        );

        // Either way the same host prediction is what gets divided into.
        assert_eq!(mapped.host_bytes(), GIB);
        assert_eq!(anon.host_bytes(), GIB);
    }

    /// A GPU-resident service holds no host *weight*, so its whole `Cpu` slot
    /// is anonymous and is its own denominator, so it learns like any other
    /// service. A slot made of token embeddings plus a borrowed GPU compute
    /// buffer would be a pair no host measurement can match, and the two pack
    /// paths would disagree about it by the whole buffer.
    #[test]
    fn a_gpu_resident_service_learns_from_its_whole_host_slot() {
        let b = RollingBase {
            inputs: RollingInputs {
                uncorrected_vram_bytes: 21 * GIB,
                // The prompt cache and the pinned graph arena; token
                // embeddings are mapped, so they are weight, not anonymous.
                uncorrected_host_bytes: 9 * GIB,
                gpu_weight_bytes: 16 * GIB,
                cpu_weight_bytes: GIB,
            },
            became_ready: true,
        };
        assert_eq!(b.host_bytes(), 8 * GIB, "the mapped embeddings drop out");
        assert_eq!(b.host_peak(8 * GIB, GIB), Some(8 * GIB));
    }

    /// An operator-declared reservation (`placement_override`, or a command
    /// service's `min_mb`) is not an estimate, so both pools stay untouched.
    #[test]
    fn a_reservation_without_an_estimate_teaches_nothing() {
        let b = RollingBase::default();
        let table = RollingTable::new();
        let svc = SmolStr::new("comfyui");
        table.update(
            &svc,
            MemoryClass::Vram,
            8 * GIB,
            b.inputs.uncorrected_vram_bytes,
        );
        table.update(&svc, MemoryClass::Host, 12 * GIB, b.host_bytes());
        let rc = table.get(&svc);
        assert_eq!(rc.vram.samples, 0);
        assert_eq!(rc.host.samples, 0);
    }
}
