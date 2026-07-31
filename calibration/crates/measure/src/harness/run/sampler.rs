//! Sample a process's resident memory on a fixed cadence, keeping peaks.
//!
//! ananke does not read `/proc` once; its snapshotter samples every two seconds
//! and keeps *monotonic peaks*, which is what the rolling correction later
//! divides by. A single snapshot therefore measures a different quantity than the
//! daemon does, and misses anything transient — the pinned staging ring during a
//! `--no-mmap` load, or growth part-way through a request. Matching the cadence
//! makes a measurement here directly comparable to what the daemon will observe,
//! and keeping the whole trace makes growth visible rather than merely suspected.
//!
//! The aggregation is pure ([`Sampler`]); only [`SamplerThread`] touches the
//! outside world, and its cadence is deliberately real time rather than the
//! injected clock — two seconds is the figure that makes this comparable to the
//! daemon, so it is not a knob.

use std::{
    collections::BTreeMap,
    sync::{Arc, mpsc},
    time::Duration,
};

use parking_lot::Mutex;

use crate::{
    harness::sys::Deps,
    record::{GpuUsage, Rss, RssSnapshot, Sample},
};

pub(crate) const INTERVAL: Duration = Duration::from_secs(2);

/// The trace and the running peaks. Nothing here reads the outside world, so the
/// peak-keeping and the growth arithmetic are testable on their own.
#[derive(Debug, Default)]
pub(crate) struct Sampler {
    trace: Vec<Sample>,
    peak: Readings,
    /// The first reading of each counter, which growth is measured from.
    ///
    /// Per counter rather than "the first sample", because a card the driver
    /// only reports part-way through the run has its own first reading, and
    /// measuring its growth from zero would report its whole allocation as
    /// growth.
    first: Readings,
}

impl Sampler {
    pub(crate) fn observe(
        &mut self,
        t_seconds: f64,
        at_utc: String,
        rss: RssSnapshot,
        gpu: BTreeMap<u32, u64>,
    ) {
        let sample = Sample {
            // A tenth of a second, as every recorded trace spells it.
            t_seconds: (t_seconds * 10.0).round() / 10.0,
            at_utc,
            rss,
            gpu: GpuUsage {
                total_mib: (!gpu.is_empty()).then(|| gpu.values().sum()),
                used_mib: gpu,
            },
        };
        self.peak.raise_to(&sample);
        self.first.record_first(&sample);
        self.trace.push(sample);
    }

    pub(crate) fn samples(&self) -> usize {
        self.trace.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.trace.is_empty()
    }

    pub(crate) fn trace(&self) -> &[Sample] {
        &self.trace
    }

    /// Peak, then the growth since the first sample: what accumulated after
    /// startup, which is what separates "allocated on first use" from "still
    /// climbing when we stopped looking".
    ///
    /// The final reading, the sample count, and the load time are the caller's
    /// to fill in — see [`crate::harness::run::assemble::rss_summary`], which
    /// owns the whole block. Everything a *sample* can say is here.
    pub(crate) fn summary(&self) -> Rss {
        let peak = self.peak.rss.unwrap_or_default();
        let first = self.first.rss.unwrap_or_default();
        let growth = |peak: u64, first: u64| whole(peak.saturating_sub(first));
        Rss {
            rss_total_kb: whole(peak.rss_total_kb),
            rss_anon_kb: whole(peak.rss_anon_kb),
            rss_file_kb: whole(peak.rss_file_kb),
            rss_shmem_kb: whole(peak.rss_shmem_kb),
            gpu_used_mib: self.peak.gpu.total_mib,
            per_card: self.peak.gpu.used_mib.clone(),
            growth_rss_total_kb: growth(peak.rss_total_kb, first.rss_total_kb),
            growth_rss_anon_kb: growth(peak.rss_anon_kb, first.rss_anon_kb),
            growth_rss_file_kb: growth(peak.rss_file_kb, first.rss_file_kb),
            growth_rss_shmem_kb: growth(peak.rss_shmem_kb, first.rss_shmem_kb),
            ..Rss::default()
        }
    }
}

/// One reading of every counter a peak is kept per.
///
/// Every field is optional-until-seen rather than zero-by-default, because zero
/// is a legitimate reading: a cell with no CUDA holds no pinned memory all run,
/// and "never measured" has to stay distinguishable from "measured as nothing".
#[derive(Debug, Default)]
struct Readings {
    rss: Option<RssSnapshot>,
    gpu: GpuUsage,
}

impl Readings {
    fn raise_to(&mut self, sample: &Sample) {
        let rss = self.rss.get_or_insert(sample.rss);
        rss.rss_total_kb = rss.rss_total_kb.max(sample.rss.rss_total_kb);
        rss.rss_anon_kb = rss.rss_anon_kb.max(sample.rss.rss_anon_kb);
        rss.rss_file_kb = rss.rss_file_kb.max(sample.rss.rss_file_kb);
        rss.rss_shmem_kb = rss.rss_shmem_kb.max(sample.rss.rss_shmem_kb);
        if let Some(total) = sample.gpu.total_mib {
            self.gpu.total_mib = Some(self.gpu.total_mib.unwrap_or(total).max(total));
        }
        for (card, mib) in &sample.gpu.used_mib {
            let peak = self.gpu.used_mib.entry(*card).or_insert(*mib);
            *peak = (*peak).max(*mib);
        }
    }

    /// Keep whatever each counter first read, and nothing after it.
    fn record_first(&mut self, sample: &Sample) {
        self.rss.get_or_insert(sample.rss);
        if self.gpu.total_mib.is_none() {
            self.gpu.total_mib = sample.gpu.total_mib;
        }
        for (card, mib) in &sample.gpu.used_mib {
            self.gpu.used_mib.entry(*card).or_insert(*mib);
        }
    }
}

/// A counter as the record spells it: signed, because the growth beside it is a
/// difference.
fn whole(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Take one reading of a pid through the injected `/proc` and driver.
///
/// Returns `false` once the process is gone: the trace so far still stands, and
/// continuing would append zeros that read as a memory cliff.
pub(crate) fn observe_once(deps: &Deps, sampler: &mut Sampler, pid: u32) -> bool {
    let Some(rss) = deps.procfs.status(pid) else {
        return false;
    };
    sampler.observe(
        deps.clock.elapsed().as_secs_f64(),
        deps.clock.now_utc(),
        rss,
        deps.gpu.per_process_mib(pid),
    );
    true
}

/// The production edge: a thread sampling from spawn until it is dropped.
///
/// Sampling starts at spawn, not at readiness, because the load itself is where
/// the transients are — the pinned staging ring on a `--no-mmap` load is gone by
/// the time the server answers `/health`.
pub(crate) struct SamplerThread {
    sampler: Arc<Mutex<Sampler>>,
    /// Dropping the sender is the stop signal, so a panic on the driving side
    /// cannot leave the thread sampling a pid that has been reused.
    stop: mpsc::Sender<()>,
    handle: std::thread::JoinHandle<()>,
}

impl SamplerThread {
    pub(crate) fn start(deps: &Deps, pid: u32) -> Self {
        let sampler = Arc::new(Mutex::new(Sampler::default()));
        let (stop, stopped) = mpsc::channel();
        let worker = sampler.clone();
        let deps = deps.clone();
        let handle = std::thread::spawn(move || {
            loop {
                if !observe_once(&deps, &mut worker.lock(), pid) {
                    return;
                }
                // A real timeout, not the injected clock: the two-second cadence
                // is what makes these figures comparable to the daemon's.
                match stopped.recv_timeout(INTERVAL) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    _ => return,
                }
            }
        });
        Self {
            sampler,
            stop,
            handle,
        }
    }

    pub(crate) fn stop(self) -> Sampler {
        let Self {
            sampler,
            stop,
            handle,
        } = self;
        drop(stop);
        let _ = handle.join();
        std::mem::take(&mut sampler.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(anon: u64, shmem: u64) -> RssSnapshot {
        RssSnapshot {
            rss_total_kb: anon + shmem,
            rss_anon_kb: anon,
            rss_file_kb: 0,
            rss_shmem_kb: shmem,
        }
    }

    #[test]
    fn the_peak_is_kept_and_growth_is_measured_from_the_first_sample() {
        let mut sampler = Sampler::default();
        sampler.observe(0.0, "t0".to_owned(), snapshot(1_000, 0), BTreeMap::new());
        sampler.observe(2.0, "t1".to_owned(), snapshot(9_000, 0), BTreeMap::new());
        sampler.observe(4.04, "t2".to_owned(), snapshot(4_000, 0), BTreeMap::new());

        let summary = sampler.summary();
        assert_eq!(sampler.samples(), 3);
        assert_eq!(summary.rss_anon_kb, 9_000);
        assert_eq!(summary.growth_rss_anon_kb, 8_000);
        // A counter that was zero throughout reads as zero rather than as
        // something that was never measured.
        assert_eq!(summary.rss_shmem_kb, 0);
        assert_eq!(summary.growth_rss_shmem_kb, 0);
        assert_eq!(sampler.trace()[2].t_seconds, 4.0);
    }

    #[test]
    fn the_gpu_total_is_absent_without_a_driver_reading_and_summed_with_one() {
        let mut sampler = Sampler::default();
        sampler.observe(0.0, "t0".to_owned(), snapshot(1, 0), BTreeMap::new());
        assert!(sampler.trace()[0].gpu.total_mib.is_none());
        assert!(sampler.summary().gpu_used_mib.is_none());

        sampler.observe(
            2.0,
            "t1".to_owned(),
            snapshot(1, 0),
            BTreeMap::from([(0, 12_000), (1, 11_000)]),
        );
        assert_eq!(sampler.trace()[1].gpu.total_mib, Some(23_000));
        let summary = sampler.summary();
        assert_eq!(summary.gpu_used_mib, Some(23_000));
        assert_eq!(summary.per_card.get(&1), Some(&11_000));
    }

    #[test]
    fn a_process_that_disappeared_ends_the_trace_rather_than_flattening_it() {
        let fakes = crate::harness::sys::Fakes::new(
            crate::harness::sys::FakeSpawner::new(),
            crate::harness::sys::FakeProcFs::new().with_status(snapshot(1_000, 20)),
            crate::harness::sys::FakeGpu::new(),
            crate::harness::sys::FakeHttp::new(),
        );
        let deps = fakes.deps();
        let mut sampler = Sampler::default();
        assert!(observe_once(&deps, &mut sampler, 1));
        fakes.procfs.forget_process();
        assert!(!observe_once(&deps, &mut sampler, 1));
        assert_eq!(sampler.samples(), 1);
    }
}
