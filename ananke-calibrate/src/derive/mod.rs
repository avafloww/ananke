//! Deriving the estimator's tuned constants from the measurement dataset.
//!
//! One function per constant, each a reduction over the record list, plus
//! `emit` to write `ananke-tuning/tuning.json` and `emit_check` to verify the
//! committed one still matches the data.
//!
//! Two rules run through all of it, and both were learned by getting them wrong.
//!
//! **A constant quoted without its spread invites more confidence than the data
//! supports.** So a deriver that reduces by median goes through
//! [`stats::consensus`], which refuses when its cells disagree rather than
//! averaging a real difference away, and one that reduces by maximum goes through
//! [`stats::check_no_outlier_dominates`], which refuses when a single cell decides
//! the value outright. Neither is defensive: between them they caught ten wrong
//! conclusions in one campaign.
//!
//! **A pairing key must pin every factor that could differ.** `bool(n_cpu_moe)`
//! instead of the count paired a `--n-cpu-moe 20` cell with a `--n-cpu-moe 40` one
//! and spread a rate across 13935% of its median; a key missing `ngl` matched cells
//! that had placed different weights on the GPU and reported 23.1 bytes an entry
//! against a true 3.7. Where a key is built here, it names everything.
//!
//! One ordering dependency is real and is modelled as an argument rather than as
//! shared state: [`baseline::baseline_offset`] subtracts the per-architecture rates
//! [`pinned::no_flash_attn_rates`] produces, and without them it silently folds a
//! per-token arena term into a flat baseline. Passing it as an argument rather
//! than through shared state means the order cannot be got wrong at compile time.

use std::collections::BTreeMap;

pub mod arena;
pub mod baseline;
pub mod dataset;
pub mod emit;
pub mod error;
pub mod graph;
pub mod mtp;
pub mod ordered;
pub mod pinned;
pub mod recurrent;
pub mod shape;
pub mod stats;
pub mod tuning;
pub mod vram;

/// One derived constant: the value that ships, and what it rests on.
///
/// The evidence is not decoration. `emit` writes it into `tuning.json` beside the
/// value, so a reader can see how many cells and which architectures a number came
/// from without re-running the campaign — and `emit --check` compares it, so a
/// deriver whose value happens to land right while its inputs changed still fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scalar {
    pub value: i64,
    pub evidence: String,
}

/// A per-architecture rate table.
///
/// Several terms genuinely differ by architecture — ik's MoE rate by a third, the
/// quantised-cache rate by a factor of forty — and one value would either
/// under-reserve the worst or over-reserve the rest. A table says so instead of
/// picking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub by_arch: BTreeMap<String, i64>,
    pub evidence: String,
}

impl Table {
    /// The largest rate, which is what an unmeasured architecture inherits: it
    /// over-reserves rather than OOMs.
    pub fn worst(&self) -> i64 {
        self.by_arch.values().copied().max().unwrap_or(0)
    }
}

/// A table of *observations* rather than rates, keyed by architecture and then by
/// the configuration each was taken at.
///
/// Recorded, not fitted. These are the measurements behind a value that has to be
/// held by hand, written down where they cannot go stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedTable {
    pub by_arch: BTreeMap<String, BTreeMap<String, i64>>,
    pub evidence: String,
}
