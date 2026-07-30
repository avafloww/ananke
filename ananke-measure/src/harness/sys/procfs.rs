//! Linux-only: reads `/proc/<pid>/status` and `/proc/meminfo`.
//!
//! The per-process read takes the same three figures ananke's own `ProcFs` does,
//! so a measurement here is directly comparable to what the daemon will observe
//! in production. Which figures those are is load-bearing: `RssAnon + RssShmem`
//! is what the process owns (`cudaMallocHost` is accounted as *shmem*, so the
//! pinned graph arena lands there and not in `RssAnon`), while `RssFile` is the
//! mapped GGUF and is the discriminator for whether host weights are anonymous.

use crate::record::RssSnapshot;

pub trait ProcFs: Send + Sync {
    /// `None` once the process is gone — a sampler that treated that as zero
    /// would record a cliff instead of the end of a trace.
    fn status(&self, pid: u32) -> Option<RssSnapshot>;
    /// Host memory the kernel says is available, in GiB. What the pre-flight fit
    /// gate weighs a model against.
    fn mem_available_gib(&self) -> f64;
    /// Swap in use, in GiB. The watchdog's subject: it is the difference from a
    /// baseline that matters, not the absolute figure.
    fn swap_used_gib(&self) -> f64;
    /// Whatever `/proc/cpuinfo` calls the CPU, and the host's total memory.
    fn cpu_model(&self) -> Option<String>;
    fn mem_total_gib(&self) -> f64;
}

pub struct LocalProcFs;

impl ProcFs for LocalProcFs {
    fn status(&self, pid: u32) -> Option<RssSnapshot> {
        let text = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        let mut snapshot = RssSnapshot::default();
        for line in text.lines() {
            let Some((key, rest)) = line.split_once(':') else {
                continue;
            };
            let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|kb| kb.parse().ok())
            else {
                continue;
            };
            match key {
                "VmRSS" => snapshot.rss_total_kb = kb,
                "RssAnon" => snapshot.rss_anon_kb = kb,
                "RssFile" => snapshot.rss_file_kb = kb,
                "RssShmem" => snapshot.rss_shmem_kb = kb,
                _ => {}
            }
        }
        Some(snapshot)
    }

    fn mem_available_gib(&self) -> f64 {
        meminfo_gib(&["MemAvailable"])
    }

    fn swap_used_gib(&self) -> f64 {
        let total = meminfo_gib(&["SwapTotal"]);
        let free = meminfo_gib(&["SwapFree"]);
        total - free
    }

    fn cpu_model(&self) -> Option<String> {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        text.lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, value)| value.trim().to_owned())
    }

    fn mem_total_gib(&self) -> f64 {
        meminfo_gib(&["MemTotal"])
    }
}

/// One `/proc/meminfo` field, converted from its kB to GiB. A field the kernel
/// does not carry reads as zero, which is the conservative direction for both
/// callers: no memory available, and no swap in use to have grown from.
fn meminfo_gib(keys: &[&str]) -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0.0;
    };
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(key, _)| keys.contains(key))
        .filter_map(|(_, rest)| rest.split_whitespace().next())
        .filter_map(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / (1024.0 * 1024.0))
        .next()
        .unwrap_or_default()
}

/// A `/proc` that is whatever a test says it is, including a pid that has gone
/// away and a swap figure that climbs on every read.
#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
pub struct FakeProcFs {
    inner: parking_lot::Mutex<FakeProcFsState>,
}

#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
struct FakeProcFsState {
    status: Option<RssSnapshot>,
    available_gib: f64,
    swap_gib: f64,
    /// Added to `swap_used_gib` on every read, so a watchdog test can watch it
    /// cross a limit without a clock or a thread.
    swap_growth_gib: f64,
}

#[cfg(any(test, feature = "test-fakes"))]
impl FakeProcFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_status(self, status: RssSnapshot) -> Self {
        self.inner.lock().status = Some(status);
        self
    }

    pub fn with_available_gib(self, gib: f64) -> Self {
        self.inner.lock().available_gib = gib;
        self
    }

    pub fn with_swap_growth_gib(self, per_read: f64) -> Self {
        self.inner.lock().swap_growth_gib = per_read;
        self
    }

    /// Report the pid as gone, the way a crashed server does mid-trace.
    pub fn forget_process(&self) {
        self.inner.lock().status = None;
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl ProcFs for FakeProcFs {
    fn status(&self, _pid: u32) -> Option<RssSnapshot> {
        self.inner.lock().status
    }

    fn mem_available_gib(&self) -> f64 {
        self.inner.lock().available_gib
    }

    fn swap_used_gib(&self) -> f64 {
        let mut state = self.inner.lock();
        state.swap_gib += state.swap_growth_gib;
        state.swap_gib
    }

    fn cpu_model(&self) -> Option<String> {
        Some("Fake CPU".to_owned())
    }

    fn mem_total_gib(&self) -> f64 {
        self.inner.lock().available_gib
    }
}
