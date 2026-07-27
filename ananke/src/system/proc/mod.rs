//! Linux-only: `/proc` abstraction.
//!
//! `/proc` isn't really a filesystem — every read is a synthesised view
//! of kernel state — so routing it through `Fs` made every test stage
//! synthetic text, only to have the consumer parse it back into a
//! semantic value. This module models the daemon's actual reads as
//! typed trait methods. The production impl reads `/proc` directly; the
//! test impl takes pre-parsed values keyed by pid.
//!
//! Every `/proc` read the daemon performs should go through here. Adding
//! a new read should add a method rather than reaching for `std::fs`
//! directly.
//!
//! Linux-only: `LocalProcFs` assumes `/proc` exists and follows the
//! kernel conventions (NUL-separated cmdline, "VmRSS:" key in status,
//! etc.). Non-Linux hosts would need a different impl.
//!
//! The real `/proc` reader and its text parsers live in [`local`]; the
//! in-memory test double lives in [`fake`].

mod local;

#[cfg(any(test, feature = "test-fakes"))]
mod fake;

use std::io;

#[cfg(any(test, feature = "test-fakes"))]
pub use fake::InMemoryProcFs;
pub use local::LocalProcFs;

/// A process's resident memory, split by what each part can be compared
/// against. Read in one pass so the three figures describe the same instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rss {
    /// `VmRSS` — everything resident.
    pub total: u64,
    /// `RssAnon + RssShmem`: memory the process allocated. Both halves are
    /// needed — the heap and the KV cache are anonymous, but `cudaMallocHost`,
    /// where llama.cpp's pinned graph arena lives, is accounted as shmem.
    pub owned: u64,
    /// `RssFile`: pages backed by a mapping. For a llama.cpp child this is the
    /// model's host-resident weights — measured at the host weight total plus
    /// ~150-230 MiB of shared libraries, across every configuration tried,
    /// including hybrids where most of the file lives on a GPU.
    pub file: u64,
}

/// Parsed `/proc/meminfo` values the scheduler cares about.
#[derive(Debug, Clone, Copy)]
pub struct Meminfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// Semantic `/proc` reader. Production: [`LocalProcFs`]. Tests:
/// [`InMemoryProcFs`], which accepts pre-parsed values instead of
/// synthesised text.
///
/// Cgroup methods assume cgroup v2 unified hierarchy (modern systemd /
/// NixOS). Hosts still on v1 will see `cgroup_path` return `None` for
/// every pid; pledge attribution then falls back to the descendants-only
/// view.
pub trait ProcFs: Send + Sync {
    /// Parse `/proc/meminfo`, returning MemTotal + MemAvailable in bytes.
    /// `MemAvailable` is preferred over `MemFree` so page cache reclaim
    /// doesn't bias the scheduler.
    fn meminfo(&self) -> io::Result<Meminfo>;

    /// The `/proc/<pid>/status` resident-memory breakdown. `None` when the
    /// pid has exited or the entry isn't fully populated yet.
    fn rss(&self, pid: u32) -> Option<Rss>;

    /// `/proc/<pid>/comm` as the raw command name (trimmed). `None` when
    /// the pid has exited.
    fn comm(&self, pid: u32) -> Option<String>;

    /// `/proc/<pid>/cmdline` with NULs replaced by spaces. `None` when
    /// the pid has exited. Used by orphan reconciliation to verify that
    /// a recorded pid still belongs to the recorded service.
    fn cmdline(&self, pid: i32) -> Option<String>;

    /// Parent pid from `/proc/<pid>/stat` (field 4). `None` when the pid
    /// has exited or stat parsing fails.
    ///
    /// Field 2 (`comm`) is parenthesised and may itself contain spaces,
    /// parens, or other punctuation, so a naive whitespace split is wrong.
    /// The parser here scans for the **last** `)` before splitting the
    /// remainder.
    fn parent_pid(&self, pid: u32) -> Option<u32>;

    /// All numeric pid entries currently visible under `/proc`. Used by
    /// the snapshotter to build a parent map once per tick.
    fn all_pids(&self) -> Vec<u32>;

    /// Cgroup v2 path from `/proc/<pid>/cgroup` (the value after `0::`).
    /// `None` when the pid has exited, the entry doesn't exist, or the
    /// host is on cgroup v1 (no `0::` line).
    fn cgroup_path(&self, pid: u32) -> Option<String>;
}

/// Transitive descendants of `root` via `proc.parent_pid()` walks. Includes
/// the root itself. Cheap — bounded by the number of currently-running
/// pids and the depth of the parent chain (typically ≤ 10).
///
/// For workloads that fork children (wrapper scripts, multi-process
/// servers), this captures every child whose VRAM/RSS the snapshotter
/// should attribute to the registered root pid. Containerised workloads
/// are NOT covered — the container is reparented out of the daemon's
/// process tree; cgroup-based attribution closes that gap separately.
pub fn descendants(proc: &dyn ProcFs, root: u32) -> Vec<u32> {
    descendants_from_map(&parent_map(proc), root)
}

/// Build a `pid → parent_pid` map from a single `/proc` walk. The
/// snapshotter calls this once per tick and reuses the map for every
/// service's descendant computation.
pub fn parent_map(proc: &dyn ProcFs) -> std::collections::BTreeMap<u32, u32> {
    let mut map = std::collections::BTreeMap::new();
    for pid in proc.all_pids() {
        if let Some(ppid) = proc.parent_pid(pid) {
            map.insert(pid, ppid);
        }
    }
    map
}

/// Pure-data version of [`descendants`] that consumes a pre-built parent
/// map. Splitting the walk from the `/proc` read lets callers reuse the
/// map across many roots without paying for repeated directory scans.
pub fn descendants_from_map(parents: &std::collections::BTreeMap<u32, u32>, root: u32) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut out = vec![root];
    let mut seen: BTreeSet<u32> = [root].into_iter().collect();
    let mut frontier: Vec<u32> = vec![root];
    while let Some(parent) = frontier.pop() {
        for (&child, &ppid) in parents {
            if ppid == parent && seen.insert(child) {
                out.push(child);
                frontier.push(child);
            }
        }
    }
    out
}

/// Pids whose cgroup v2 path equals `parent` or sits anywhere inside its
/// subtree. The match is exact-or-followed-by-`/` so `foo.slice` doesn't
/// match a sibling `foo.slice-evil.scope`.
pub fn pids_in_cgroup_subtree(proc: &dyn ProcFs, parent: &str) -> Vec<u32> {
    proc.all_pids()
        .into_iter()
        .filter(|pid| match proc.cgroup_path(*pid) {
            Some(cg) => cg == parent || cg.starts_with(&format!("{parent}/")),
            None => false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// Descendants walk: parent map `2→1, 3→1, 4→2, 5→4` rooted at 1
    /// must include every transitive child plus the root itself.
    #[test]
    fn descendants_walks_full_subtree() {
        let mut parents = BTreeMap::new();
        parents.insert(2u32, 1u32);
        parents.insert(3u32, 1u32);
        parents.insert(4u32, 2u32);
        parents.insert(5u32, 4u32);
        parents.insert(99u32, 50u32); // unrelated tree
        let mut out = descendants_from_map(&parents, 1);
        out.sort();
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
    }

    /// A root with no children returns just itself.
    #[test]
    fn descendants_of_leaf_is_singleton() {
        let parents: BTreeMap<u32, u32> = BTreeMap::new();
        assert_eq!(descendants_from_map(&parents, 42), vec![42]);
    }

    /// `pids_in_cgroup_subtree` must accept exact match and descendant
    /// paths, but reject sibling cgroups whose name starts with the same
    /// prefix (`foo.slice` vs. `foo.slice-evil.scope`).
    #[test]
    fn cgroup_subtree_matches_exact_and_children_only() {
        let proc = InMemoryProcFs::new();
        proc.set_cgroup(10, "/system.slice/ananke-comfyui.slice");
        proc.set_cgroup(11, "/system.slice/ananke-comfyui.slice/docker-abc.scope");
        proc.set_cgroup(12, "/system.slice/ananke-comfyui.slice-evil.scope");
        proc.set_cgroup(13, "/system.slice/other.scope");
        let mut pids = pids_in_cgroup_subtree(&proc, "/system.slice/ananke-comfyui.slice");
        pids.sort();
        assert_eq!(pids, vec![10, 11]);
    }

    #[test]
    fn cgroup_subtree_returns_empty_when_no_match() {
        let proc = InMemoryProcFs::new();
        proc.set_cgroup(1, "/system.slice/other.scope");
        let pids = pids_in_cgroup_subtree(&proc, "/system.slice/missing.slice");
        assert!(pids.is_empty());
    }
}
