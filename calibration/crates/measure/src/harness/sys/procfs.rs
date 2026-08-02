//! Linux-only: reads `/proc/<pid>/status` and `/proc/meminfo`.
//!
//! The per-process read takes the same three figures ananke's own `ProcFs` does,
//! so a measurement here is directly comparable to what the daemon will observe
//! in production. Which figures those are is load-bearing: `RssAnon + RssShmem`
//! is what the process owns (`cudaMallocHost` is accounted as *shmem*, so the
//! pinned graph arena lands there and not in `RssAnon`), while `RssFile` is the
//! mapped GGUF and is the discriminator for whether host weights are anonymous.

use std::collections::BTreeMap;

use crate::record::RssSnapshot;

pub trait ProcFs: Send + Sync {
    /// Anonymous resident bytes per mapping, from `/proc/<pid>/smaps`, keyed by the
    /// mapping's name. A total says a process holds more than the model predicts;
    /// this says whether that is the heap, a library arena, or a pinned CUDA
    /// allocation. `None` when the pid has exited.
    fn smaps_anon(&self, pid: u32) -> Option<BTreeMap<String, u64>>;

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
    fn smaps_anon(&self, pid: u32) -> Option<BTreeMap<String, u64>> {
        Some(parse_smaps_anon(
            &std::fs::read_to_string(format!("/proc/{pid}/smaps")).ok()?,
        ))
    }

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
/// Sum `Anonymous:` over each mapping in an `smaps` dump, keyed by mapping name.
///
/// A mapping header is `addr perms offset dev inode [path]`; the path is absent for
/// anonymous memory, which is where the heap and most of what is being hunted lives,
/// so those are grouped under `[anon]` rather than dropped.
fn parse_smaps_anon(content: &str) -> BTreeMap<String, u64> {
    let mut by: BTreeMap<String, u64> = BTreeMap::new();
    let mut current = "[anon]".to_string();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Anonymous:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0);
            *by.entry(current.clone()).or_default() += kb * 1024;
        } else if is_mapping_header(line) {
            current = mapping_name(line);
        }
    }
    by
}

/// The pathname from a mapping header, or `[anon]` where there is none.
///
/// Taken as everything after the fifth field rather than as the sixth, because the
/// path is the rest of the line and may contain spaces — model directories routinely
/// do. Splitting on whitespace keys `/models/My Model.gguf` under `/models/My`, which
/// attributes its bytes to a label that matches nothing.
fn mapping_name(line: &str) -> String {
    // addr, perms, offset, dev, inode — then the remainder is the path.
    let mut rest = line.trim_start();
    for _ in 0..5 {
        let Some(end) = rest.find(char::is_whitespace) else {
            return "[anon]".to_string();
        };
        rest = rest[end..].trim_start();
    }
    if rest.is_empty() {
        "[anon]".to_string()
    } else {
        rest.to_string()
    }
}

/// Whether a line opens a new mapping, as against continuing one's fields.
///
/// The header's first field is a hex address range; a field line is `Key: value`.
fn is_mapping_header(line: &str) -> bool {
    line.split_whitespace()
        .next()
        .is_some_and(|first| first.contains('-') && !first.ends_with(':'))
}

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
#[cfg(test)]
mod smaps_tests {
    use super::parse_smaps_anon;

    /// A real dump, trimmed: a named mapping, the heap, a bare anonymous one, and a
    /// path with a space in it. Model directories routinely have spaces, and the
    /// mapping the probe most wants to name is the one most likely to carry one.
    const SAMPLE: &str = "7f0000000000-7f0000200000 r-xp 00000000 fd:00 1234    /usr/lib/libcuda.so
Anonymous:            64 kB
VmFlags: rd ex mr mw me
7f0000200000-7f0000400000 rw-p 00000000 00:00 0       [heap]
Anonymous:          2048 kB
7f0000400000-7f0000600000 rw-p 00000000 00:00 0
Anonymous:           512 kB
7f0000600000-7f0000800000 r--p 00000000 fd:00 5678    /models/My Model.gguf
Anonymous:           128 kB
";

    #[test]
    fn sums_anonymous_bytes_per_mapping() {
        let by = parse_smaps_anon(SAMPLE);
        assert_eq!(by.get("/usr/lib/libcuda.so"), Some(&(64 * 1024)));
        assert_eq!(by.get("[heap]"), Some(&(2048 * 1024)));
    }

    /// A mapping with no pathname is anonymous memory, which is where the heap and
    /// most of what the probe hunts actually lives. Dropping it would lose the term.
    #[test]
    fn a_mapping_without_a_path_is_grouped_as_anonymous() {
        assert_eq!(parse_smaps_anon(SAMPLE).get("[anon]"), Some(&(512 * 1024)));
    }

    /// The pathname is the rest of the line, not the sixth whitespace field.
    /// Splitting on whitespace keys this under `/models/My`, which names nothing and
    /// silently misattributes the mapping's bytes.
    #[test]
    fn a_path_containing_a_space_keeps_all_of_it() {
        let by = parse_smaps_anon(SAMPLE);
        assert_eq!(by.get("/models/My Model.gguf"), Some(&(128 * 1024)));
        assert!(!by.contains_key("/models/My"), "the path was truncated");
    }

    /// `VmFlags:` and the other per-mapping fields are not mapping headers, and a
    /// dump that ends mid-mapping yields what it has rather than panicking.
    #[test]
    fn field_lines_do_not_open_a_mapping() {
        let by = parse_smaps_anon(SAMPLE);
        assert!(!by.keys().any(|k| k.starts_with("rd ")), "{by:?}");
        assert!(parse_smaps_anon("7f00-7f01 rw-p 0 00:00 0").is_empty());
    }
}

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
    smaps: BTreeMap<String, u64>,
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

    /// Preload the per-mapping anonymous breakdown a probe will read.
    pub fn with_smaps(self, smaps: BTreeMap<String, u64>) -> Self {
        self.inner.lock().smaps = smaps;
        self
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
    fn smaps_anon(&self, _pid: u32) -> Option<BTreeMap<String, u64>> {
        Some(self.inner.lock().smaps.clone())
    }

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
