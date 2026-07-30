//! Per-service rolling correction, tracked separately per memory pool.
//!
//! A service's footprint spans two pools that fail independently: VRAM on the
//! GPUs, and host RAM. A hybrid MoE keeps most of its bytes in host RAM, a
//! GPU-only service keeps none there, and the estimator's error in one pool
//! says nothing about its error in the other. One mean covering both cannot be
//! divided into a like-for-like ratio — a VRAM-only peak over an all-device
//! base reads an accurate estimate as a large over-prediction — so each pool
//! carries its own [`ClassCorrection`], fed by its own observation.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;
use tracing::{info, warn};

use crate::daemon::events::EventBus;

/// Number of observed samples a service must accumulate before its rolling
/// mean is trusted to scale placement. Below this, a single early sample —
/// which can be skewed by a cold-cache first load or a one-off measurement
/// artifact — would otherwise swing the pledge by up to ±50%, which has
/// over-pledged a shard past a GPU's capacity and blocked re-placement.
pub const MIN_TRUSTED_SAMPLES: u32 = 3;

/// The pool a correction applies to. Each is learned from its own observation
/// against its own reservation base; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryClass {
    /// GPU VRAM: the sum of the reservation's `Gpu(_)` slots, observed as the
    /// attributed NVML peak.
    Vram,
    /// Host RAM: the reservation's `Cpu` slot, observed as the process-tree
    /// RSS peak with the GPU-resident share of the model mapping removed.
    Host,
}

impl MemoryClass {
    /// Stable identifier used in log fields and on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vram => "vram",
            Self::Host => "host",
        }
    }
}

/// One pool's learned correction.
#[derive(Debug, Clone, Copy)]
pub struct ClassCorrection {
    pub mean: f64,
    pub samples: u32,
    /// Count of consecutive samples with |mean-1.0| > 0.3.
    pub drift_samples: u32,
    /// Set when this pool's state came from an OOM-retry bump rather than an
    /// observation. Such a pool is trusted immediately (that is the point of
    /// the bump) but is *not* evidence, so the first run that survives to a
    /// clean drain wipes it — see [`RollingTable::clear_synthetic`].
    pub synthetic: bool,
}

impl Default for ClassCorrection {
    fn default() -> Self {
        Self {
            mean: 1.0,
            samples: 0,
            drift_samples: 0,
            synthetic: false,
        }
    }
}

impl ClassCorrection {
    /// The factor to actually apply to a placement.
    ///
    /// Returns a neutral `1.0` until at least [`MIN_TRUSTED_SAMPLES`] real
    /// observations have accumulated, so that a single noisy early sample
    /// cannot push a shard past a device's capacity. An OOM-retry bump
    /// promotes `samples` past the gate deliberately (see
    /// [`RollingTable::bump_for_oom_retry`]) so the corrective nudge is
    /// applied immediately on the retry path.
    pub fn effective(&self) -> f64 {
        if self.samples >= MIN_TRUSTED_SAMPLES {
            self.mean
        } else {
            1.0
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RollingCorrection {
    pub vram: ClassCorrection,
    pub host: ClassCorrection,
}

impl RollingCorrection {
    /// The gated factors, ready to hand to the packer.
    pub fn corrections(&self) -> Corrections {
        Corrections {
            vram: self.vram.effective(),
            host: self.host.effective(),
        }
    }

    /// Borrow one pool's correction.
    pub fn class(&self, class: MemoryClass) -> &ClassCorrection {
        match class {
            MemoryClass::Vram => &self.vram,
            MemoryClass::Host => &self.host,
        }
    }

    fn class_mut(&mut self, class: MemoryClass) -> &mut ClassCorrection {
        match class {
            MemoryClass::Vram => &mut self.vram,
            MemoryClass::Host => &mut self.host,
        }
    }
}

pub use ananke_placement::Corrections;

#[derive(Clone, Default)]
pub struct RollingTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, RollingCorrection>>>,
    events: Option<EventBus>,
}

impl RollingTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a `RollingTable` that publishes [`ananke_api::events::Event::EstimatorDrift`]
    /// whenever an update moves a rolling mean by more than 5%.
    pub fn with_events(events: EventBus) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BTreeMap::new())),
            events: Some(events),
        }
    }

    pub fn get(&self, name: &SmolStr) -> RollingCorrection {
        self.inner.read().get(name).copied().unwrap_or_default()
    }

    /// Inject a synthetic 1.4× correction into both pools after an OOM kill,
    /// to force the next placement to reserve more memory before retrying.
    ///
    /// The trigger is a SIGKILL shortly after spawn — the kernel OOM killer or
    /// a cgroup limit, so a host-RAM verdict — but the signal doesn't say which
    /// pool the daemon mis-modelled to get there: an under-reserved GPU side
    /// spills more to the host than planned and surfaces as exactly this kill.
    /// Both pools are nudged, including one the daemon has declined to *learn*
    /// from. A bump is not learning: it is a one-shot response to a hard
    /// failure, and a service whose host pool is unmeasurable is still a
    /// service the kernel just killed for using too much host memory.
    ///
    /// What makes that safe is [`Self::clear_synthetic`]: the bump is marked as
    /// not-an-observation and the first run that survives to a clean drain
    /// removes it, so a pool that never records a real sample cannot hold the
    /// synthetic factor forever.
    pub fn bump_for_oom_retry(&self, name: &SmolStr) {
        // A ratio of 1.4 signals the estimator was 40% short, which is the
        // maximum useful nudge before triggering the drift warning path.
        for class in [MemoryClass::Vram, MemoryClass::Host] {
            self.update(name, class, 140, 100);
        }
        // The OOM bump must take effect on the immediate retry, so promote the
        // sample count past the trust gate even if this is the service's first
        // observation. Without this, `effective()` would ignore the bump until
        // two more real samples landed — defeating the retry nudge.
        let mut guard = self.inner.write();
        if let Some(entry) = guard.get_mut(name) {
            for class in [MemoryClass::Vram, MemoryClass::Host] {
                let c = entry.class_mut(class);
                c.samples = c.samples.max(MIN_TRUSTED_SAMPLES);
                c.synthetic = true;
            }
        }
    }

    /// Drop any pool whose state came from an OOM bump rather than an
    /// observation, returning it to neutral.
    ///
    /// Called when a run reaches a clean drain, which is the evidence the bump
    /// was waiting for: the larger reservation worked. Keeping it past that
    /// point would pledge 40% more memory than the service needs for the rest
    /// of the daemon's life — and for a pool that records no real samples,
    /// nothing would ever dilute it.
    pub fn clear_synthetic(&self, name: &SmolStr) {
        let mut guard = self.inner.write();
        let Some(entry) = guard.get_mut(name) else {
            return;
        };
        for class in [MemoryClass::Vram, MemoryClass::Host] {
            let c = entry.class_mut(class);
            if c.synthetic {
                *c = ClassCorrection::default();
            }
        }
    }

    /// Fold one observation of `class` into that pool's rolling mean.
    ///
    /// Both arguments must measure the same pool: the VRAM peak against the
    /// GPU-slot reservation total, the host peak against the `Cpu` slot.
    /// Mixing them (a VRAM peak over an all-device base) makes an accurate
    /// estimate read as a large over-prediction and pins the mean to the `0.8`
    /// clamp floor. `base_estimate_bytes` is the *uncorrected* reservation, so
    /// each sample independently estimates observed-over-predicted rather than
    /// integrating against a base the last correction already moved.
    ///
    /// A zero base means the service holds nothing in this pool — a GPU-only
    /// service has no `Cpu` slot, a `cpu-only` one no GPU slots — and the
    /// sample is skipped rather than treated as a ratio of zero.
    pub fn update(
        &self,
        name: &SmolStr,
        class: MemoryClass,
        observed_peak_bytes: u64,
        base_estimate_bytes: u64,
    ) {
        if base_estimate_bytes == 0 {
            return;
        }
        let ratio = observed_peak_bytes as f64 / base_estimate_bytes as f64;
        let mut guard = self.inner.write();
        let entry = guard.entry(name.clone()).or_default().class_mut(class);
        let prev_mean = entry.mean;
        let n = entry.samples as f64 + 1.0;
        let new_mean = (entry.mean * (n - 1.0) + ratio) / n;
        entry.mean = new_mean.clamp(0.8, 1.5);
        entry.samples = entry.samples.saturating_add(1);

        if (entry.mean - 1.0).abs() > 0.3 {
            entry.drift_samples = entry.drift_samples.saturating_add(1);
            if entry.drift_samples >= 5 {
                warn!(
                    service = %name,
                    class = class.as_str(),
                    mean = entry.mean,
                    "estimator_drift: rolling mean has been >0.3 away from 1.0 for 5+ runs"
                );
            }
        } else {
            entry.drift_samples = 0;
        }

        if entry.mean > 1.2 {
            warn!(
                service = %name,
                class = class.as_str(),
                mean = entry.mean,
                "rolling correction: under-estimation"
            );
        } else if entry.mean < 0.85 {
            warn!(
                service = %name,
                class = class.as_str(),
                mean = entry.mean,
                "rolling correction: over-reservation"
            );
        } else {
            info!(
                service = %name,
                class = class.as_str(),
                mean = entry.mean,
                sample = entry.samples,
                "rolling correction updated"
            );
        }

        // Capture values needed for event publishing before releasing the lock.
        let final_mean = entry.mean;
        // > 5% shift in the rolling mean warrants an EstimatorDrift event.
        let significant_shift =
            prev_mean == 0.0 || ((final_mean - prev_mean) / prev_mean).abs() > 0.05;
        drop(guard);

        if significant_shift && let Some(events) = &self.events {
            events.publish(ananke_api::events::Event::EstimatorDrift {
                service: name.clone(),
                class: class.as_str().to_string(),
                rolling_mean: final_mean as f32,
                at_ms: crate::tracking::now_unix_ms(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeviceSlot;

    #[test]
    fn mean_converges_to_observed_ratio() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        for _ in 0..5 {
            // observed=120, base=100, ratio=1.2
            t.update(&svc, MemoryClass::Vram, 120, 100);
        }
        let rc = t.get(&svc);
        assert!((rc.vram.mean - 1.2).abs() < 0.05);
    }

    /// The pools are independent: learning about VRAM must not move the host
    /// mean, since a hybrid's two sides carry unrelated errors.
    #[test]
    fn classes_are_independent() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        for _ in 0..MIN_TRUSTED_SAMPLES {
            t.update(&svc, MemoryClass::Vram, 120, 100);
        }
        let rc = t.get(&svc);
        assert!(rc.vram.effective() > 1.0);
        assert_eq!(rc.host.samples, 0);
        assert_eq!(rc.host.effective(), 1.0);
        assert_eq!(rc.corrections().host, 1.0);
    }

    #[test]
    fn mean_clamps_high() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, MemoryClass::Host, 1000, 100); // ratio = 10
        assert_eq!(t.get(&svc).host.mean, 1.5);
    }

    #[test]
    fn mean_clamps_low() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, MemoryClass::Host, 10, 100); // ratio = 0.1
        assert_eq!(t.get(&svc).host.mean, 0.8);
    }

    #[test]
    fn zero_base_is_noop() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, MemoryClass::Vram, 100, 0);
        assert_eq!(t.get(&svc).vram.samples, 0);
    }

    #[test]
    fn effective_ignores_under_min_samples() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        // A single skewed observation moves the raw mean but must not be
        // trusted to scale placement yet.
        t.update(&svc, MemoryClass::Vram, 150, 100); // ratio = 1.5
        let rc = t.get(&svc);
        assert!(rc.vram.mean > 1.2);
        assert_eq!(rc.vram.effective(), 1.0);

        // Once enough samples accumulate, the gate opens and the real mean
        // applies.
        t.update(&svc, MemoryClass::Vram, 150, 100);
        t.update(&svc, MemoryClass::Vram, 150, 100);
        let rc = t.get(&svc);
        assert_eq!(rc.vram.samples, MIN_TRUSTED_SAMPLES);
        assert_eq!(rc.vram.effective(), rc.vram.mean);
    }

    #[test]
    fn oom_bump_bypasses_the_gate_in_both_pools() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        // No prior samples: the bump must take effect immediately on retry.
        t.bump_for_oom_retry(&svc);
        let rc = t.get(&svc);
        for c in [rc.vram, rc.host] {
            assert!(c.samples >= MIN_TRUSTED_SAMPLES);
            assert!(c.effective() > 1.0);
            assert_eq!(c.effective(), c.mean);
        }
    }

    /// The bump is a response to a failure, not evidence, so the first run
    /// that survives to a clean drain removes it. Without this the synthetic
    /// 1.4 would pledge 40% extra for the daemon's remaining life — and in a
    /// pool that records no real samples (a service whose host side the daemon
    /// declines to learn from), nothing would ever dilute it.
    #[test]
    fn a_clean_run_clears_the_oom_bump() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.bump_for_oom_retry(&svc);
        assert!(t.get(&svc).host.effective() > 1.0);

        t.clear_synthetic(&svc);
        let rc = t.get(&svc);
        for c in [rc.vram, rc.host] {
            assert_eq!(c.effective(), 1.0);
            assert_eq!(c.samples, 0);
            assert!(!c.synthetic);
        }
    }

    /// Clearing must not touch a pool that learned honestly.
    #[test]
    fn clearing_spares_real_samples() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        for _ in 0..MIN_TRUSTED_SAMPLES {
            t.update(&svc, MemoryClass::Vram, 130, 100);
        }
        let learned = t.get(&svc).vram.mean;
        t.clear_synthetic(&svc);
        assert_eq!(t.get(&svc).vram.mean, learned);
        assert_eq!(t.get(&svc).vram.samples, MIN_TRUSTED_SAMPLES);
    }

    /// A correction above 1.0 must round up: truncating a scaled byte count is
    /// an under-reservation, which is the failure the correction exists to
    /// prevent.
    #[test]
    fn scale_rounds_up_and_short_circuits_neutral() {
        let c = Corrections {
            vram: 1.1,
            host: 0.9,
        };
        assert_eq!(c.scale(&DeviceSlot::Gpu(0), 101), 112); // 111.1 → 112
        assert_eq!(c.scale(&DeviceSlot::Cpu, 101), 91); // 90.9 → 91
        assert_eq!(Corrections::NEUTRAL.scale(&DeviceSlot::Gpu(0), 101), 101);
    }
}
