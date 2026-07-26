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
    allocator::placement::RollingInputs, config::validate::ServiceConfig, supervise::RunLoop,
    tracking::rolling::MemoryClass,
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
        self.rolling_base = RollingBase::new(inputs, &self.current_svc());
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
        let peak_rss = self.deps.observation.read_peak_rss(name);
        let samples = base.samples_from(peak_vram, peak_rss);

        if samples.run_was_measurable {
            // The run survived to a clean drain, so an earlier OOM bump has
            // its answer: the larger reservation worked, and holding it would
            // over-pledge for the rest of the daemon's life.
            self.deps.rolling.clear_synthetic(name);
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
                peak_rss,
                gpu_weight_bytes = base.inputs.gpu_weight_bytes,
                cpu_weight_bytes = base.inputs.cpu_weight_bytes,
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

/// The per-process host footprint no reservation models: llama-server itself,
/// the CUDA runtime's host-side allocations, pinned staging buffers, the HTTP
/// stack. Roughly fixed, and an assumption rather than a measurement — it is
/// named here so the learning gate below states a tolerance instead of a
/// magic number.
const ASSUMED_PROCESS_HOST_FOOTPRINT_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Minimum host-resident model weight before the host pool will learn from a
/// run, bounding the *numerator's* unmodelled term. At nine times the assumed
/// footprint the ratio it can distort is capped at ~11%, matching the bound
/// [`HOST_LEARNING_MIN_WEIGHT_PERCENT`] puts on the denominator's term — the
/// two gates share one tolerance rather than each guessing separately.
const HOST_LEARNING_MIN_CPU_WEIGHT_BYTES: u64 = 9 * ASSUMED_PROCESS_HOST_FOOTPRINT_BYTES;

/// Minimum share of the `Cpu` slot that must be model weight, in percent,
/// bounding the *denominator's* unmodelled term at the same ~11% as
/// [`HOST_LEARNING_MIN_CPU_WEIGHT_BYTES`] bounds the numerator's.
///
/// The slot's only other component of consequence is the compute buffer,
/// charged to the CPU backend at the size calibrated for a GPU — 3792 MiB for
/// a calibrated architecture, 400 MiB for an uncalibrated one, against a real
/// CPU-side buffer that is neither. An absolute weight floor doesn't bound
/// that: it is a *relative* error, and at 19 GiB of expert weight it is worth
/// −7% while at 97 GiB it is worth −2%. Requiring the weight to dominate caps
/// it, and caps it in the same place regardless of model size.
const HOST_LEARNING_MIN_WEIGHT_PERCENT: u64 = 90;

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
    /// Whether the child runs with the GGUF mmap'd (llama.cpp's default).
    /// Decides whether the GPU-resident weight bytes have to come off the
    /// observed RSS peak — see [`Self::host_peak`].
    pub(crate) mmap: bool,
}

impl RollingBase {
    /// Capture the rolling inputs of a committed placement.
    pub(crate) fn new(inputs: RollingInputs, svc: &ServiceConfig) -> Self {
        Self {
            inputs,
            became_ready: false,
            // llama.cpp mmaps unless told otherwise; a command service has no
            // model mapping to worry about and reports zero weight bytes
            // anyway, so the flag is immaterial there.
            mmap: svc
                .llama_cpp()
                .map(|lc| lc.mmap != Some(false))
                .unwrap_or(false),
        }
    }

    /// The host-pool numerator derived from an observed RSS peak, or `None`
    /// when the measurement can't support one.
    ///
    /// # The weight floor
    ///
    /// A process has a host footprint no reservation models: llama-server
    /// itself, the CUDA runtime's host-side allocations, pinned staging
    /// buffers, the HTTP stack. It is roughly fixed, so it is noise against a
    /// hybrid MoE's ~100 GiB of offloaded experts and larger than the entire
    /// `Cpu` slot of a GPU-resident service — whose slot is token embeddings
    /// plus the CPU-side compute buffer, a couple of GiB at most. Learning a
    /// *multiplicative* factor from the latter would read a fixed overhead as a
    /// proportional error and pin the mean to a clamp: measured at ~0.5 on the
    /// layer path (which charges the CPU a full compute buffer) and ~2.4 on the
    /// sharded path (which charges it none), for the same model on the same
    /// cards. That is issue #34's failure — a ratio whose two sides measure
    /// different things, hidden by the clamp — with the pools swapped.
    ///
    /// So the host pool learns only where both unmodelled terms are minor
    /// relative to the weight the ratio is really about — see
    /// [`Self::host_pool_is_measurable`]. Otherwise there is no sample at all:
    /// not learning is strictly better than learning the wrong number, and a
    /// wrong host factor here is quieter than the bug it replaced, since a
    /// plausible 0.93 carries no clamp artefact to signal that it is an
    /// accounting error rather than a measurement.
    ///
    /// With the GGUF mmap'd, llama.cpp reads GPU-destined tensors *through the
    /// mapping*, so their pages count against `VmRSS` even though the tensors
    /// live in VRAM at runtime. Left alone, that inflates the host peak by
    /// nearly the whole GPU-resident weight total. The bytes to remove are
    /// known exactly — the packer chose which tensors go where, and their sizes
    /// come from the GGUF tensor table, not from the modelled terms (KV,
    /// compute buffers) that carry the estimator's uncertainty.
    ///
    /// Under `--no-mmap` there is nothing to subtract: GPU tensors are staged
    /// through a buffer that is freed after upload, so `VmRSS` is already
    /// host-resident bytes only.
    ///
    /// `None` when the subtraction would saturate. A peak at or below the
    /// GPU-resident weight total means the sampler never caught the process
    /// with its host pages resident (a short run, or reclaim under memory
    /// pressure). Recording it would hand `update` a ratio near zero, which
    /// pins the mean to its `0.8` clamp floor — the exact silent failure this
    /// accounting exists to avoid.
    pub(crate) fn host_peak(&self, peak_rss_bytes: u64) -> Option<u64> {
        if !self.host_pool_is_measurable() {
            return None;
        }
        let subtract = if self.mmap {
            self.inputs.gpu_weight_bytes
        } else {
            0
        };
        peak_rss_bytes.checked_sub(subtract).filter(|b| *b > 0)
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
    /// nothing — and the host side additionally needs the placement to be one
    /// the host pool can measure at all ([`Self::host_pool_is_measurable`]).
    pub(crate) fn samples_from(&self, peak_vram_bytes: u64, peak_rss_bytes: u64) -> RollingSamples {
        if !self.became_ready {
            return RollingSamples {
                vram: None,
                host: None,
                run_was_measurable: false,
            };
        }
        RollingSamples {
            vram: Some(peak_vram_bytes).filter(|p| *p > 0),
            host: self.host_peak(peak_rss_bytes),
            run_was_measurable: true,
        }
    }

    /// Mark the run as fully loaded. See [`Self::became_ready`].
    pub(crate) fn run_became_ready(&mut self) {
        self.became_ready = true;
    }

    /// Whether this placement's host side can support a meaningful ratio at
    /// all: enough host-resident model weight, and weight enough of the `Cpu`
    /// slot, that neither unmodelled term decides the outcome. See
    /// [`HOST_LEARNING_MIN_CPU_WEIGHT_BYTES`] and
    /// [`HOST_LEARNING_MIN_WEIGHT_PERCENT`].
    pub(crate) fn host_pool_is_measurable(&self) -> bool {
        self.inputs.cpu_weight_bytes >= HOST_LEARNING_MIN_CPU_WEIGHT_BYTES
            && self.inputs.cpu_weight_bytes * 100
                >= self.inputs.uncorrected_host_bytes * HOST_LEARNING_MIN_WEIGHT_PERCENT
    }

    /// The host-pool denominator: the `Cpu` slot as the estimator predicted it.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.inputs.uncorrected_host_bytes
    }
}
#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use super::*;
    use crate::tracking::rolling::{MIN_TRUSTED_SAMPLES, MemoryClass, RollingTable};

    const GIB: u64 = 1024 * 1024 * 1024;

    fn base(vram: u64, host: u64, gpu_weight: u64, mmap: bool) -> RollingBase {
        // A hybrid's `Cpu` slot is offloaded expert weight plus the CPU-side
        // compute buffer; 4% of a slot this size is the shape that clears both
        // learning gates.
        let cpu_weight = host - host / 25;
        RollingBase {
            inputs: RollingInputs {
                uncorrected_vram_bytes: vram,
                uncorrected_host_bytes: host,
                gpu_weight_bytes: gpu_weight,
                cpu_weight_bytes: cpu_weight,
            },
            mmap,
            became_ready: true,
        }
    }

    /// The `deepseek-v4-flash` shape: ~20 GiB of VRAM and ~105 GiB of experts
    /// in host RAM. Each pool's peak is divided by that pool's own base, so an
    /// accurate estimate converges to 1.0 on both — where a single mean over
    /// the all-device total read the VRAM peak as a 5× over-prediction and
    /// pinned to the 0.8 clamp floor.
    #[test]
    fn hybrid_converges_to_neutral_in_both_pools() {
        let b = base(20 * GIB, 105 * GIB, 18 * GIB, true);
        // With the mapping in play, the observed RSS peak carries the
        // GPU-resident weights as well as the host-resident experts.
        let observed_rss = 105 * GIB + 18 * GIB;

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
                b.host_peak(observed_rss)
                    .expect("peak exceeds the subtraction"),
                b.host_bytes(),
            );
        }
        let c = table.get(&svc).corrections();
        assert_eq!(c.vram, 1.0);
        assert_eq!(c.host, 1.0);
    }

    /// A `cpu-only` service has no GPU slot, so the VRAM pool never learns —
    /// but the host pool must, which is the whole point of splitting them.
    /// Nothing is subtracted from its RSS peak either: it holds no GPU weights.
    #[test]
    fn cpu_only_learns_the_host_pool_only() {
        let b = base(0, 60 * GIB, 0, true);
        assert!(b.inputs.cpu_weight_bytes >= HOST_LEARNING_MIN_CPU_WEIGHT_BYTES);
        assert_eq!(b.host_peak(66 * GIB), Some(66 * GIB));

        let table = RollingTable::new();
        let svc = SmolStr::new("cpu-only");
        for _ in 0..MIN_TRUSTED_SAMPLES {
            table.update(&svc, MemoryClass::Vram, 0, b.inputs.uncorrected_vram_bytes);
            table.update(&svc, MemoryClass::Host, 66 * GIB, b.host_bytes());
        }
        let rc = table.get(&svc);
        assert_eq!(rc.vram.samples, 0);
        assert_eq!(rc.corrections().vram, 1.0);
        assert_eq!(rc.host.samples, MIN_TRUSTED_SAMPLES);
        assert!(rc.corrections().host > 1.0);
    }

    /// `--no-mmap` stages GPU tensors through a buffer it frees, so `VmRSS` is
    /// already host-only and nothing may be subtracted. Subtracting anyway
    /// would read a correct host reservation as a large over-prediction.
    #[test]
    fn no_mmap_subtracts_nothing() {
        let mapped = base(20 * GIB, 105 * GIB, 18 * GIB, true);
        let unmapped = base(20 * GIB, 105 * GIB, 18 * GIB, false);
        assert_eq!(mapped.host_peak(120 * GIB), Some(102 * GIB));
        assert_eq!(unmapped.host_peak(120 * GIB), Some(120 * GIB));
    }

    /// An RSS peak at or below the GPU-resident weight total means the sampler
    /// never caught the host pages resident. Recording the saturated
    /// difference would feed `update` a ratio near zero and pin the mean to
    /// the 0.8 floor, so the sample is dropped instead.
    #[test]
    fn saturated_subtraction_yields_no_sample() {
        let b = base(20 * GIB, 105 * GIB, 18 * GIB, true);
        assert_eq!(b.host_peak(18 * GIB), None);
        assert_eq!(b.host_peak(2 * GIB), None);
        assert_eq!(b.host_peak(0), None);
    }

    /// A GPU-resident service's `Cpu` slot is token embeddings plus the
    /// CPU-side compute buffer — a couple of GiB against a per-process host
    /// footprint (llama-server, the CUDA runtime, pinned buffers) that no
    /// reservation models. A multiplicative ratio there measures the
    /// unmodelled term, so the host pool must not learn from it at all.
    #[test]
    fn a_gpu_resident_service_teaches_the_host_pool_nothing() {
        let b = RollingBase {
            inputs: RollingInputs {
                uncorrected_vram_bytes: 21 * GIB,
                // token_embd + the CPU-side compute buffer.
                uncorrected_host_bytes: 5 * GIB,
                gpu_weight_bytes: 16 * GIB,
                cpu_weight_bytes: GIB,
            },
            mmap: true,
            became_ready: true,
        };
        // A peak that would otherwise read as a 0.5× over-reservation.
        assert_eq!(b.host_peak(18 * GIB), None);
        // …and one that would read as a 2.4× under-estimate on the sharded
        // path, whose `Cpu` slot carries no compute buffer at all.
        let sharded = RollingBase {
            inputs: RollingInputs {
                uncorrected_host_bytes: GIB,
                ..b.inputs
            },
            mmap: true,
            became_ready: true,
        };
        assert_eq!(sharded.host_peak(18 * GIB), None);
    }

    /// A run that never finished loading yields nothing to either pool. Its
    /// peaks are real but partial — the observation tables are monotonic, so a
    /// model evicted mid-load leaves a fraction of its footprint recorded — and
    /// a fractional peak over a whole-model base is exactly the ratio the
    /// clamp launders into a plausible 0.8.
    #[test]
    fn a_run_that_never_became_ready_yields_no_samples() {
        let mut b = base(21 * GIB, 60 * GIB, 18 * GIB, true);
        b.became_ready = false;
        // Peaks consistent with a load interrupted a third of the way in.
        let s = b.samples_from(7 * GIB, 20 * GIB);
        assert_eq!(s.vram, None);
        assert_eq!(s.host, None);
        assert!(!s.run_was_measurable);

        // The same peaks from a run that did become ready are recorded — the
        // guard is the readiness, not the values.
        b.became_ready = true;
        let s = b.samples_from(7 * GIB, 20 * GIB);
        assert_eq!(s.vram, Some(7 * GIB));
        assert!(s.host.is_some());
        assert!(s.run_was_measurable);
    }

    /// A zero peak is the absence of a measurement, not a measurement of zero.
    #[test]
    fn a_zero_peak_is_not_a_sample() {
        let b = base(21 * GIB, 60 * GIB, 0, true);
        let s = b.samples_from(0, 0);
        assert_eq!(s.vram, None);
        assert_eq!(s.host, None);
        // …but the run was still measurable, so an OOM bump is still resolved.
        assert!(s.run_was_measurable);
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
