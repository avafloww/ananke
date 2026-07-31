//! One row of `measurements.ndjson`, top to bottom.
//!
//! Field order is the format. Serde emits a struct's fields in declaration
//! order, so the declaration order here is the canonical column order of the
//! dataset and changing it changes every line's bytes.

use serde::{Deserialize, Serialize};

use crate::parsed::Parsed;
pub use crate::record::{
    factors::{Factors, Runtime},
    hardware::{Cpu, Gpu, Hardware, Provenance},
    memory::{Checkpoint, GpuUsage, Rss, RssSnapshot, Sample},
};

mod factors;
mod hardware;
mod memory;

/// Bumped whenever a record's shape changes in a way an analysis must notice.
///
/// 1: the original flat CSV-era rows. 2: nested NDJSON with hardware and
/// traces. 3: generic per-device breakdown (tensor-split included), per-process
/// GPU memory, model identity, first-occurrence metadata, retained log tails.
pub const SCHEMA: u32 = 3;

/// One measured cell.
///
/// `deny_unknown_fields`: the row's top level is closed. Every key a writer has
/// ever emitted is named below, and a key that is not is a schema drift the
/// round-trip test should fail on rather than silently drop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    /// The cell's stable identity, so a rerun skips what has already been
    /// measured.
    pub cell: String,
    pub status: Status,
    /// Facts that make a stale row identifiable later: when, on what box, and
    /// against which binary.
    pub provenance: Provenance,
    pub hardware: Hardware,
    pub factors: Factors,
    #[serde(default)]
    pub parsed: Parsed,
    /// Peak resident memory, with the final reading and the growth since
    /// startup alongside it.
    #[serde(default)]
    pub rss: Rss,
    /// The end of a failed run's log, so a bad record says why it is bad.
    pub log_tail: String,
    /// The archived log's file name, which is what makes a record
    /// re-parseable rather than merely re-readable.
    pub log: String,
    /// The full time series, not just its summary: growth is a shape, and a
    /// peak alone cannot distinguish "allocated on first use" from "still
    /// climbing when we stopped looking".
    pub trace: Vec<Sample>,
    /// Memory against tokens, which a time series alone cannot give.
    pub checkpoints: Vec<Checkpoint>,
    /// Set when the `parsed` block was rebuilt from the archived log, so an
    /// analysis can tell which rows carry the newer fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reparsed: Option<bool>,
}

/// Why a record holds what it holds. Every reader takes `Ok` and skips the
/// rest, so a status it does not understand is a status it correctly ignores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Ok,
    /// Another server still held the port, so nothing was measured.
    PortBusy,
    FailedToLoad,
    Timeout,
    SkippedInsufficientMemory,
    HarnessError,
    /// A runtime upgrade invalidated the row; it keeps its data and its
    /// archived log, and only the status changes.
    StaleRuntime,
}

impl Record {
    /// Host memory the process *owns*, in bytes.
    ///
    /// `RssAnon + RssShmem`, not `VmRSS`: `cudaMallocHost` is accounted as
    /// shmem, and `RssFile` is the mapped GGUF, which llama.cpp populates and
    /// then leaves resident as clean reclaimable pages.
    pub fn owned_bytes(&self) -> i64 {
        (self.rss.rss_anon_kb + self.rss.rss_shmem_kb) * 1024
    }

    /// The same figure in whole MiB, truncated.
    pub fn owned_mib(&self) -> i64 {
        (self.rss.rss_anon_kb + self.rss.rss_shmem_kb) / 1024
    }
}
