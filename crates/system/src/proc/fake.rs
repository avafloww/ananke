//! In-memory `/proc` test double: pre-populate the values a test run
//! needs to see, keyed by pid, instead of writing synthesised kernel
//! text.

use std::{collections::BTreeMap, io};

use parking_lot::RwLock;

use crate::proc::{Meminfo, ProcFs, Rss};

/// Test impl keyed on pid. Callers pre-populate the values a test run
/// needs to see, including a single shared `meminfo`; reads that don't
/// match a registered value return the "exited" signal (`None` /
/// `NotFound` io error).
#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default, Clone)]
pub struct InMemoryProcFs {
    inner: std::sync::Arc<RwLock<InMemoryProcFsState>>,
}

#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
struct InMemoryProcFsState {
    meminfo: Option<Meminfo>,
    rss: BTreeMap<u32, Rss>,
    comm: BTreeMap<u32, String>,
    cmdline: BTreeMap<i32, String>,
    parent: BTreeMap<u32, u32>,
    cgroup: BTreeMap<u32, String>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl InMemoryProcFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_meminfo(&self, m: Meminfo) {
        self.inner.write().meminfo = Some(m);
    }

    /// Preload a process's resident memory as entirely owned (anonymous),
    /// which is the shape of a service holding no mapped model.
    pub fn set_vm_rss(&self, pid: u32, bytes: u64) {
        self.inner.write().rss.insert(
            pid,
            Rss {
                total: bytes,
                owned: bytes,
                file: 0,
            },
        );
    }

    /// Preload the full breakdown, for tests that care which part is which.
    pub fn set_rss(&self, pid: u32, rss: Rss) {
        self.inner.write().rss.insert(pid, rss);
    }

    pub fn set_comm(&self, pid: u32, comm: impl Into<String>) {
        self.inner.write().comm.insert(pid, comm.into());
    }

    /// Preload the `/proc/<pid>/cmdline` value the orphan reconciler
    /// would otherwise read off disk. Pass the command as a single
    /// space-separated string.
    pub fn set_cmdline(&self, pid: i32, cmdline: impl Into<String>) {
        self.inner.write().cmdline.insert(pid, cmdline.into());
    }

    /// Preload `/proc/<pid>/stat`'s parent-pid field. Used by the
    /// snapshotter's descendants walk; tests express the process tree by
    /// stating each child's parent.
    pub fn set_parent(&self, child: u32, parent: u32) {
        self.inner.write().parent.insert(child, parent);
    }

    /// Preload `/proc/<pid>/cgroup`'s v2 path (the value after `0::`).
    /// Tests use this to model containerised pids whose cgroup sits
    /// under a service's declared `cgroup_parent`.
    pub fn set_cgroup(&self, pid: u32, path: impl Into<String>) {
        self.inner.write().cgroup.insert(pid, path.into());
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl ProcFs for InMemoryProcFs {
    fn meminfo(&self) -> io::Result<Meminfo> {
        self.inner
            .read()
            .meminfo
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "meminfo not preloaded"))
    }

    fn rss(&self, pid: u32) -> Option<Rss> {
        self.inner.read().rss.get(&pid).copied()
    }

    fn comm(&self, pid: u32) -> Option<String> {
        self.inner.read().comm.get(&pid).cloned()
    }

    fn cmdline(&self, pid: i32) -> Option<String> {
        self.inner.read().cmdline.get(&pid).cloned()
    }

    fn parent_pid(&self, pid: u32) -> Option<u32> {
        self.inner.read().parent.get(&pid).copied()
    }

    fn all_pids(&self) -> Vec<u32> {
        let s = self.inner.read();
        let mut out: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        out.extend(s.rss.keys());
        out.extend(s.comm.keys());
        out.extend(s.parent.keys());
        out.extend(s.parent.values());
        out.extend(s.cgroup.keys());
        out.into_iter().collect()
    }

    fn cgroup_path(&self, pid: u32) -> Option<String> {
        self.inner.read().cgroup.get(&pid).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_round_trips_preloaded_values() {
        let proc = InMemoryProcFs::new();
        proc.set_meminfo(Meminfo {
            total_bytes: 1024,
            available_bytes: 512,
        });
        proc.set_vm_rss(4242, 8192);
        proc.set_comm(4242, "llama-server");
        proc.set_cmdline(4242, "llama-server -m model.gguf");

        assert_eq!(proc.meminfo().unwrap().available_bytes, 512);
        assert_eq!(proc.rss(4242).map(|r| r.total), Some(8192));
        assert_eq!(proc.comm(4242), Some("llama-server".into()));
        assert_eq!(
            proc.cmdline(4242).as_deref(),
            Some("llama-server -m model.gguf")
        );

        // Pids that weren't preloaded look exited.
        assert_eq!(proc.rss(9999), None);
        assert_eq!(proc.comm(9999), None);
        assert_eq!(proc.cmdline(9999), None);
    }
}
