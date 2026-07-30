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
    record::{GpuUsage, Metric, RssSnapshot, Sample},
};

pub(crate) const INTERVAL: Duration = Duration::from_secs(2);

/// The trace and the running peaks. Nothing here reads the outside world, so the
/// peak-keeping and the growth arithmetic are testable on their own.
#[derive(Debug, Default)]
pub(crate) struct Sampler {
    trace: Vec<Sample>,
    peak: BTreeMap<String, i64>,
    /// The first sample's metrics, which growth is measured from.
    first: BTreeMap<String, i64>,
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
            gpu_used_mib: (!gpu.is_empty()).then(|| gpu.values().sum()),
            gpu_per_device: GpuUsage { used_mib: gpu },
        };
        for (key, value) in metrics(&sample) {
            // `or_insert` before the comparison: a metric that is legitimately
            // zero for the whole run — `RssShmem` with no CUDA, so no pinned
            // allocations — never exceeds the default and would otherwise be
            // absent entirely rather than present as zero.
            let peak = self.peak.entry(key.clone()).or_insert(value);
            *peak = (*peak).max(value);
            self.first.entry(key).or_insert(value);
        }
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
    pub(crate) fn summary(&self) -> BTreeMap<String, Metric> {
        let mut summary: BTreeMap<String, Metric> = self
            .peak
            .iter()
            .map(|(key, value)| (key.clone(), Metric::Whole(*value)))
            .collect();
        for (key, first) in &self.first {
            let peak = self.peak.get(key).copied().unwrap_or(*first);
            summary.insert(format!("growth_{key}"), Metric::Whole(peak - first));
        }
        summary
    }
}

/// One sample flattened into the metrics a peak is kept per. The GPU total is
/// absent rather than zero when the driver reported nothing, because a CPU-only
/// cell has no VRAM reading to peak.
fn metrics(sample: &Sample) -> Vec<(String, i64)> {
    let mut metrics = vec![
        ("rss_total_kb", sample.rss.rss_total_kb),
        ("rss_anon_kb", sample.rss.rss_anon_kb),
        ("rss_file_kb", sample.rss.rss_file_kb),
        ("rss_shmem_kb", sample.rss.rss_shmem_kb),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), i64::try_from(value).unwrap_or(i64::MAX)))
    .collect::<Vec<_>>();
    if let Some(total) = sample.gpu_used_mib {
        metrics.push((
            "gpu_used_mib".to_owned(),
            i64::try_from(total).unwrap_or(i64::MAX),
        ));
    }
    for (index, mib) in &sample.gpu_per_device.used_mib {
        metrics.push((
            format!("gpu{index}_used_mib"),
            i64::try_from(*mib).unwrap_or(i64::MAX),
        ));
    }
    metrics
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
        assert_eq!(summary["rss_anon_kb"], Metric::Whole(9_000));
        assert_eq!(summary["growth_rss_anon_kb"], Metric::Whole(8_000));
        // A metric that was zero throughout is present as zero rather than
        // absent, which is the difference between "no pinned memory" and "not
        // measured".
        assert_eq!(summary["rss_shmem_kb"], Metric::Whole(0));
        assert_eq!(summary["growth_rss_shmem_kb"], Metric::Whole(0));
        assert_eq!(sampler.trace()[2].t_seconds, 4.0);
    }

    #[test]
    fn the_gpu_total_is_absent_without_a_driver_reading_and_summed_with_one() {
        let mut sampler = Sampler::default();
        sampler.observe(0.0, "t0".to_owned(), snapshot(1, 0), BTreeMap::new());
        assert!(sampler.trace()[0].gpu_used_mib.is_none());
        assert!(!sampler.summary().contains_key("gpu_used_mib"));

        sampler.observe(
            2.0,
            "t1".to_owned(),
            snapshot(1, 0),
            BTreeMap::from([(0, 12_000), (1, 11_000)]),
        );
        assert_eq!(sampler.trace()[1].gpu_used_mib, Some(23_000));
        let summary = sampler.summary();
        assert_eq!(summary["gpu_used_mib"], Metric::Whole(23_000));
        assert_eq!(summary["gpu1_used_mib"], Metric::Whole(11_000));
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
