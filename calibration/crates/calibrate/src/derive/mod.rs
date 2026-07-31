//! Deriving the estimator's tuned constants from the measurement dataset.
//!
//! One function per constant, each a reduction over the record list, plus
//! `emit` to write `crates/tuning/tuning.json` and `emit_check` to verify the
//! committed one still matches the data.
//!
//! Two rules run through all of it, and both were learned by getting them wrong.
//!
//! **A constant quoted without its spread invites more confidence than the data
//! supports.** So a deriver that reduces by median goes through
//! [`stats::consensus`], which refuses when its cells disagree rather than
//! averaging a real difference away. Not defensive: it caught ten wrong conclusions
//! in one campaign.
//!
//! [`stats::check_no_outlier_dominates`] is the same idea for a maximum, refusing
//! when a single cell decides the value outright — but it guards only
//! [`baseline::headroom`] today, not every deriver that reduces by maximum. The
//! others take a bare `max` and would not notice the contaminated pair the guard was
//! written for. Wiring it in is worth doing and is not a no-op: several of those
//! constants are set by their worst cell by design, so each needs its tolerance
//! chosen rather than inherited.
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
pub mod keys;
pub mod mtp;
pub mod ordered;
pub mod pair;
pub mod pinned;
pub mod recurrent;
pub mod shape;
pub mod stats;
pub mod tuning;
pub mod units;
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

/// A rate table, keyed at one of the vocabularies in [`keys`].
///
/// Several terms genuinely differ by architecture — ik's MoE rate by a third, the
/// quantised-cache rate by a factor of forty — and one value would either
/// under-reserve the worst or over-reserve the rest. A table says so instead of
/// picking.
///
/// `K` is the whole point of the type. Four tables here are keyed four ways and a
/// lookup that misses takes [`Self::worst`] rather than raising, so a mismatched
/// vocabulary is silently the wrong rate; naming it in the type makes that a
/// compile error. The *serialised* field stays `by_arch`, which is the committed
/// document's spelling for every one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table<K> {
    pub by_key: BTreeMap<K, i64>,
    pub evidence: String,
}

impl<K> Table<K> {
    /// The largest rate, which is what an unmeasured key inherits: it
    /// over-reserves rather than OOMs.
    pub fn worst(&self) -> i64 {
        self.by_key.values().copied().max().unwrap_or(0)
    }
}

/// A table of *observations* rather than rates, keyed by architecture and then by
/// the configuration each was taken at.
///
/// Recorded, not fitted. These are the measurements behind a value that has to be
/// held by hand, written down where they cannot go stale. Nothing looks a row up,
/// so its architecture keys stay plain strings rather than [`keys::ArchKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedTable {
    pub by_arch: BTreeMap<String, BTreeMap<String, i64>>,
    pub evidence: String,
}
