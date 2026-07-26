//! Linux-only: per-service observed memory. Reads NVML for GPU VRAM and
//! `/proc/{pid}/status` for CPU VmRSS.
//!
//! Two shapes of the same signal live here, because the daemon's consumers
//! genuinely want different things from it:
//!
//! - **Peaks** — monotonic high-water marks over the current run. The
//!   operator-facing footprint (`/api/services`, the event stream) and the
//!   rolling estimator correction both want "how big did this ever get".
//! - **Current** — the latest sample. The balloon resolver wants this: its
//!   own rolling window turns it into a *decaying* recent peak, which a
//!   monotonic input can never be, and its over-ceiling watchdog has to see
//!   a breach subside or it latches one spike into a kill/respawn loop.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;

/// Observed memory per service, across the current run.
#[derive(Clone, Default)]
pub struct ObservationTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, ObservedState>>>,
}

#[derive(Debug, Clone, Default)]
struct ObservedState {
    /// High-water mark of `vram_bytes + rss_bytes`. Surfaced to the
    /// `/api/services` endpoint and event stream as the operator-facing
    /// "observed footprint" of the service.
    peak_bytes: u64,
    /// High-water mark of GPU VRAM bytes alone, attributed across every
    /// pid in the per-service pid set. Tracked separately because the
    /// dynamic-allocation pledge models one component at a time —
    /// pledging combined
    /// VRAM+RSS would inflate the pledge with the python interpreter's
    /// RSS and falsely trip the over-commit check (regression: an SDXL
    /// inference's 8 GB VRAM + 12 GB RSS used to pledge as 20 GB on the
    /// GPU and trigger a self-eviction that wasn't justified).
    peak_vram_bytes: u64,
    /// High-water mark of host RSS alone, the mirror of `peak_vram_bytes`.
    /// The rolling correction's host-pool numerator derives from this; see
    /// [`ObservationTable::read_peak_rss`] for why it can't be used raw.
    peak_rss_bytes: u64,
    /// Latest GPU VRAM sample. Retained alongside the peak because the
    /// balloon resolver has to be able to watch usage come *down*.
    current_vram_bytes: u64,
    /// Latest host RSS sample. The mirror image of `current_vram_bytes`:
    /// a cpu-only dynamic service's reservation lands on the CPU device,
    /// so its pledge and its ceiling are both measured against RSS.
    /// Reading VRAM there would always see zero and the ceiling would
    /// never trip, no matter how far the service overran.
    current_rss_bytes: u64,
    pids: Vec<u32>,
    /// Cgroup v2 path under which this service's actual workload pids
    /// live, when the operator declared one in `[service.tracking]`.
    /// Set once at supervisor spawn alongside `register`; the snapshotter
    /// reads it to widen the per-service pid set with cgroup-resident
    /// pids that aren't descendants of the registered child.
    cgroup_parent: Option<SmolStr>,
}

impl ObservationTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, service: &SmolStr, pid: u32) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        if !entry.pids.contains(&pid) {
            entry.pids.push(pid);
        }
    }

    /// Record (or clear) the cgroup parent path for a service. The
    /// snapshotter consults this when sampling so containerised pids that
    /// aren't process-tree descendants of the registered child still get
    /// attributed correctly.
    pub fn set_cgroup_parent(&self, service: &SmolStr, parent: Option<SmolStr>) {
        let mut guard = self.inner.write();
        guard.entry(service.clone()).or_default().cgroup_parent = parent;
    }

    /// Record one observation of the service's combined footprint
    /// (`vram + rss`) and each component. The peaks update monotonically;
    /// the current readings are replaced outright. `vram_bytes` may be zero
    /// on a CPU-only service.
    pub fn record_sample(&self, service: &SmolStr, vram_bytes: u64, rss_bytes: u64) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        let total = vram_bytes.saturating_add(rss_bytes);
        if total > entry.peak_bytes {
            entry.peak_bytes = total;
        }
        if vram_bytes > entry.peak_vram_bytes {
            entry.peak_vram_bytes = vram_bytes;
        }
        if rss_bytes > entry.peak_rss_bytes {
            entry.peak_rss_bytes = rss_bytes;
        }
        entry.current_vram_bytes = vram_bytes;
        entry.current_rss_bytes = rss_bytes;
    }

    /// Combined `vram + rss` peak. The frontend / `/api/services`
    /// surfaces this as the operator-facing observed footprint.
    pub fn read_peak(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_bytes)
            .unwrap_or(0)
    }

    /// VRAM-only peak. The rolling estimator correction folds this in at
    /// drain time against a GPU-slots-only reservation base, and it is
    /// deliberately VRAM-only rather than the combined `read_peak` — see the
    /// comment on `ObservedState::peak_vram_bytes`.
    pub fn read_peak_vram(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_vram_bytes)
            .unwrap_or(0)
    }

    /// Host-RSS-only peak, as `VmRSS` summed over the service's pid set.
    ///
    /// **Not comparable to the `Cpu` slot of a reservation on its own.**
    /// llama.cpp mmaps the GGUF by default and reads GPU-destined tensors
    /// through that mapping, so those pages count against this process's file
    /// RSS even though they live in VRAM at runtime. The rolling correction
    /// subtracts the packer's GPU-resident weight bytes before using this as a
    /// host-pool numerator; the daemon's `RollingBase::host_peak` does that
    /// before dividing, and only for a service holding enough host-resident
    /// weight for the ratio to mean anything.
    pub fn read_peak_rss(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_rss_bytes)
            .unwrap_or(0)
    }

    /// Latest VRAM-only sample. Zero when the service has never been
    /// sampled, which reads as "not using the GPU" at every call site.
    pub fn read_current_vram(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.current_vram_bytes)
            .unwrap_or(0)
    }

    /// Latest host-RSS-only sample. The mirror of [`Self::read_current_vram`]
    /// for a cpu-pinned service, whose reservation is host RAM.
    pub fn read_current_rss(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.current_rss_bytes)
            .unwrap_or(0)
    }

    pub fn pids(&self, service: &SmolStr) -> Vec<u32> {
        self.inner
            .read()
            .get(service)
            .map(|s| s.pids.clone())
            .unwrap_or_default()
    }

    pub fn cgroup_parent(&self, service: &SmolStr) -> Option<SmolStr> {
        self.inner.read().get(service)?.cgroup_parent.clone()
    }

    pub fn clear(&self, service: &SmolStr) {
        self.inner.write().remove(service);
    }
}

/// Thin rename of [`crate::system::ProcFs::vm_rss`] so the snapshotter
/// can keep calling `read_vm_rss(proc, pid)` and not have to know which
/// `/proc` file actually backs it. Returns `None` when the pid has
/// exited or the status entry isn't populated yet.
pub fn read_vm_rss(proc: &dyn crate::system::ProcFs, pid: u32) -> Option<u64> {
    proc.vm_rss(pid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::InMemoryProcFs;

    #[test]
    fn read_vm_rss_goes_through_procfs() {
        let proc = InMemoryProcFs::new();
        proc.set_vm_rss(4242, 5120 * 1024);
        assert_eq!(read_vm_rss(&proc, 4242), Some(5120 * 1024));
    }

    #[test]
    fn read_vm_rss_none_when_pid_missing() {
        let proc = InMemoryProcFs::new();
        assert_eq!(read_vm_rss(&proc, 9999), None);
    }

    #[test]
    fn peak_is_monotonic() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        // Combined peak walks up: vram+rss = 100, 50, 200.
        t.record_sample(&svc, 60, 40);
        t.record_sample(&svc, 30, 20);
        t.record_sample(&svc, 120, 80);
        assert_eq!(t.read_peak(&svc), 200);
        // VRAM-only peak tracks separately (60 → 30 doesn't lower it).
        assert_eq!(t.read_peak_vram(&svc), 120);
    }

    /// The current readings follow the sample in both directions, which is
    /// the whole point of retaining them next to the peaks: the balloon
    /// resolver cannot see a service settle back down from a peak.
    #[test]
    fn current_follows_the_latest_sample_downwards() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.record_sample(&svc, 120, 80);
        assert_eq!(t.read_current_vram(&svc), 120);
        assert_eq!(t.read_current_rss(&svc), 80);
        t.record_sample(&svc, 30, 20);
        assert_eq!(t.read_current_vram(&svc), 30);
        assert_eq!(t.read_current_rss(&svc), 20);
        // …while the peaks are untouched by the fall.
        assert_eq!(t.read_peak_vram(&svc), 120);
        assert_eq!(t.read_peak(&svc), 200);
    }

    /// An unsampled service reads as zero rather than panicking or
    /// inheriting another service's numbers.
    #[test]
    fn current_is_zero_before_the_first_sample() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("never-started");
        assert_eq!(t.read_current_vram(&svc), 0);
        assert_eq!(t.read_current_rss(&svc), 0);
    }

    /// Observed combined peak and VRAM-only peak don't interfere: a tick
    /// with high RSS but low VRAM doesn't lift the VRAM peak (so the
    /// dynamic pledge stays modest), and a tick with high VRAM but low
    /// RSS lifts the VRAM peak even when the combined peak doesn't move.
    #[test]
    fn vram_and_combined_peaks_track_independently() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        // Tick 1: 4 GB VRAM + 6 GB RSS. Combined 10 GB, VRAM 4 GB.
        t.record_sample(&svc, 4 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak(&svc), 10 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak_vram(&svc), 4 * 1024 * 1024 * 1024);
        // Tick 2: 8 GB VRAM + 1 GB RSS. Combined 9 GB (won't move), VRAM 8 GB.
        t.record_sample(&svc, 8 * 1024 * 1024 * 1024, 1024 * 1024 * 1024);
        assert_eq!(t.read_peak(&svc), 10 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak_vram(&svc), 8 * 1024 * 1024 * 1024);
    }

    /// A cpu-only service reports no VRAM at all, so the balloon resolver has
    /// to read RSS for it — reading VRAM would see zero however far the
    /// service overran its `max_reserve_gb` ceiling.
    #[test]
    fn rss_tracks_a_service_with_no_vram() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("cpu-only");
        t.record_sample(&svc, 0, 6 * 1024 * 1024 * 1024);
        t.record_sample(&svc, 0, 2 * 1024 * 1024 * 1024);
        assert_eq!(t.read_current_vram(&svc), 0);
        assert_eq!(
            t.read_current_rss(&svc),
            2 * 1024 * 1024 * 1024,
            "RSS is the only signal a cpu-only service produces, and it has \
             to follow the workload back down"
        );
        // The combined peak still remembers the high-water mark for the
        // operator-facing footprint.
        assert_eq!(t.read_peak(&svc), 6 * 1024 * 1024 * 1024);
    }

    #[test]
    fn clear_resets() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.record_sample(&svc, 50, 50);
        t.clear(&svc);
        assert_eq!(t.read_peak(&svc), 0);
        assert_eq!(t.read_peak_vram(&svc), 0);
        assert_eq!(t.read_current_vram(&svc), 0);
        assert_eq!(t.read_current_rss(&svc), 0);
    }
}
