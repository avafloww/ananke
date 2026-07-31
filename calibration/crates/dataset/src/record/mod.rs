//! One row of `measurements.ndjson`, top to bottom.
//!
//! Field order is the format. Serde emits a struct's fields in declaration
//! order, so the declaration order here is the canonical column order of the
//! dataset and changing it changes every line's bytes.

use serde::{Deserialize, Serialize};

use crate::parsed::Parsed;
pub use crate::record::{
    factors::{Factors, FlashAttn, KvType, Runtime},
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
/// `deny_unknown_fields`: the row's top level is closed, so a key no writer has
/// emitted is a schema drift the round-trip test fails on rather than silently
/// drops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    /// The cell's stable identity, so a rerun skips what has already been
    /// measured.
    pub cell: String,
    pub status: Status,
    pub provenance: Provenance,
    pub hardware: Hardware,
    pub factors: Factors,
    #[serde(default)]
    pub parsed: Parsed,
    #[serde(default)]
    pub rss: Rss,
    /// The end of a failed run's log, so a bad record says why it is bad.
    pub log_tail: String,
    /// The archived log's file name, which is what makes a record re-parseable
    /// rather than merely re-readable.
    pub log: String,
    /// The full time series, not just its summary: a peak alone cannot
    /// distinguish "allocated on first use" from "still climbing when we
    /// stopped looking".
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
    /// The cell's identity, which every row carries: [`Record::cell`] has no
    /// serde default, so a row written before the schema carried one fails to
    /// parse rather than arriving without an identity to group it by.
    pub fn cell_id(&self) -> &str {
        &self.cell
    }

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

    /// One card's driver reading, keyed by the *physical* id: the sampler
    /// records `gpu{id}_used_mib` while the loader's breakdown rows are in
    /// visible order, so a cell pinned to GPU 1 has its usage under
    /// `gpu1_used_mib` and its breakdown row under `CUDA0`. A zero reads as
    /// absent — the driver reports one for a card the process never touched.
    pub fn gpu_card_used_mib(&self, card: u32) -> Option<u64> {
        self.rss.per_card.get(&card).copied().filter(|mib| *mib > 0)
    }

    /// How many cards the driver reported usage on. [`Rss::gpu_used_mib`] is
    /// the process total and is deliberately excluded.
    pub fn cards_measured(&self) -> usize {
        self.rss.per_card.values().filter(|mib| **mib > 0).count()
    }
}
