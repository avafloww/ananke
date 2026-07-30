//! Report where the dataset is too thin to have measured what it claims.
//!
//! A term measured at a single point in some axis looks flat in that axis. That is
//! not a hypothetical: it produced four wrong constants in this campaign.
//!
//! - The flash-attention cost read as an inconsistent baseline shift, because
//!   seventeen of its nineteen cells sat at one context and one batch. Swept, it is
//!   a clean per-token rate.
//! - The shared-cache window mask was three copies, from a sweep taken entirely at
//!   ubatch 512 where a 1024-token window spans two batches. At 2048 it is two.
//! - A separate MTP draft's compute was context-independent, from cells at two
//!   contexts that happened to agree.
//! - The MTP overhead's slot dependence was confounded with context, because every
//!   one-slot pair sat at one context and the only four-slot pair at another.
//!
//! Each was found by accident, late, after the constant had been in use. This turns
//! the audit into something that runs: for every regime the estimator models, how
//! many distinct points exist along the axes that regime's rule depends on. A regime
//! measured at one point is reported whether or not anybody currently suspects it.
//!
//! `--check` is what CI runs. It fails on a regime that both feeds a constant and
//! has one point in an axis, which is the configuration that has been wrong every
//! time so far.

use std::collections::BTreeSet;

use crate::record::Record;

/// An axis a regime's rule can depend on.
///
/// An enum rather than a string key, so a regime naming an axis the record does not
/// carry is a compile error. Indexing the factor set by name instead would report a
/// typo'd axis as "one distinct point" — indistinguishable from the failure this is
/// meant to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Context,
    Ubatch,
    Gpus,
    Parallel,
    Concurrency,
    PromptTokens,
}

impl Axis {
    /// This cell's coordinate along the axis, as a comparable token.
    fn value(self, record: &Record) -> String {
        let f = &record.factors;
        match self {
            Axis::Context => f.ctx.to_string(),
            Axis::Ubatch => f.ubatch.unwrap_or(512).to_string(),
            Axis::Gpus => f.gpus.clone(),
            Axis::Parallel => f.parallel.unwrap_or(1).to_string(),
            Axis::Concurrency => f.concurrency.unwrap_or(1).to_string(),
            Axis::PromptTokens => f.probe_prompt_tokens.unwrap_or(4).to_string(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Axis::Context => "ctx",
            Axis::Ubatch => "ubatch",
            Axis::Gpus => "gpus",
            Axis::Parallel => "parallel",
            Axis::Concurrency => "concurrency",
            Axis::PromptTokens => "probe_prompt_tokens",
        }
    }
}

/// A configuration the estimator models, and what its rule rests on.
pub struct Regime {
    pub name: &'static str,
    /// Which cells are in this regime.
    pub select: fn(&Record) -> bool,
    /// The axes the rule depends on. A rule that scales with the batch cannot be
    /// checked by cells at one batch, however many there are, which is why this is
    /// per-axis rather than a cell count.
    pub axes: &'static [Axis],
    /// What is fitted from it, for the report.
    pub constant: &'static str,
}

/// Every regime the estimator models.
pub const REGIMES: &[Regime] = &[
    Regime {
        name: "flash attention off",
        select: |r| r.factors.flash_attn.as_deref() != Some("on"),
        axes: &[Axis::Context, Axis::Ubatch, Axis::Gpus],
        constant: "no_flash_attn_rates",
    },
    Regime {
        name: "quantised KV",
        select: |r| r.factors.kv_type.as_deref() != Some("f16"),
        axes: &[Axis::Context, Axis::Ubatch],
        constant: "quantised_cache_rates, quantised KV compute",
    },
    Regime {
        name: "tensor split",
        select: |r| r.factors.split.as_deref() == Some("tensor"),
        axes: &[Axis::Context, Axis::Ubatch, Axis::Gpus],
        constant: "tensor_split_baseline",
    },
    Regime {
        name: "shared KV cache",
        select: |r| r.factors.kv_unified && r.factors.parallel.unwrap_or(1) > 1,
        axes: &[Axis::Context, Axis::Ubatch],
        constant: "the window-mask count",
    },
    Regime {
        name: "multiple slots",
        select: |r| r.factors.parallel.unwrap_or(1) > 1,
        axes: &[Axis::Context, Axis::Ubatch, Axis::Parallel],
        constant: "mask streams, MTP KV",
    },
    Regime {
        name: "concurrent requests",
        select: |r| r.factors.concurrency.unwrap_or(1) > 1,
        axes: &[Axis::Concurrency, Axis::Context],
        constant: "per_slot_host_bytes",
    },
    Regime {
        name: "checkpointed prompt",
        select: |r| r.factors.probe_prompt_tokens.unwrap_or(4) >= 8192,
        axes: &[Axis::Context, Axis::Gpus],
        constant: "checkpoint_headroom_bytes",
    },
    Regime {
        name: "MTP",
        select: |r| r.factors.spec_type.is_some(),
        axes: &[Axis::Context, Axis::Parallel],
        constant: "MTP draft compute, MTP host bytes",
    },
    Regime {
        name: "ik_llama",
        select: |r| r.factors.runtime == "ik",
        axes: &[Axis::Context, Axis::Ubatch, Axis::Gpus],
        constant: "ik_moe_rates, baseline @ik",
    },
    Regime {
        name: "hybrid",
        select: |r| r.factors.n_cpu_moe.is_some(),
        axes: &[Axis::Context, Axis::Ubatch],
        constant: "expert offload placement",
    },
    Regime {
        name: "single card",
        // A cell with no GPUs at all counts as one card here, matching the rule the
        // mask-copy constant was fitted under. Four cells are CPU-only and arguably
        // belong in neither bucket, but changing which side they fall on would move a
        // constant rather than only its audit.
        select: |r| r.gpu_ids().len().max(1) == 1,
        axes: &[Axis::Context, Axis::Ubatch],
        constant: "the mask-copy rule at one copy",
    },
    Regime {
        name: "no mmap",
        select: |r| r.factors.no_mmap,
        axes: &[Axis::Context, Axis::Ubatch],
        constant: "host_peak's RssFile discriminator",
    },
];

/// One regime's coverage: how many cells, and the thinnest axis it rests on.
#[derive(Debug, Clone)]
pub struct Coverage {
    pub name: &'static str,
    pub constant: &'static str,
    pub cells: usize,
    pub points: usize,
    pub thinnest: &'static str,
}

impl Coverage {
    /// Whether this regime feeds a constant from a single point in an axis.
    pub fn is_thin(&self) -> bool {
        self.cells > 0 && self.points < 2
    }
}

/// Audit every regime against the dataset.
///
/// Only cells that produced an arena reading count: a cell that failed to load has
/// no measurement in any axis, and counting it would make a thin regime look covered.
///
/// A *zero* reading counts as no reading, not as a measurement of zero. Fourteen ok
/// cells report `arena_mib: 0`, which means the parse never found the buffer line
/// rather than that the arena was empty — llama.cpp always has one. Treating them as
/// measured inflated three regimes' cell counts.
pub fn audit(records: &[Record]) -> Vec<Coverage> {
    let measured: Vec<&Record> = records
        .iter()
        .filter(|r| r.status == "ok" && r.parsed.arena_mib.is_some_and(|v| v > 0.0))
        .collect();

    let mut out: Vec<Coverage> = REGIMES
        .iter()
        .map(|regime| {
            let group: Vec<&&Record> = measured.iter().filter(|r| (regime.select)(r)).collect();
            if group.is_empty() {
                return Coverage {
                    name: regime.name,
                    constant: regime.constant,
                    cells: 0,
                    points: 0,
                    thinnest: "never measured",
                };
            }
            let (thinnest, points) = regime
                .axes
                .iter()
                .map(|axis| {
                    let distinct: BTreeSet<String> = group.iter().map(|r| axis.value(r)).collect();
                    (*axis, distinct.len())
                })
                .min_by_key(|(_, count)| *count)
                .expect("every regime names at least one axis");
            Coverage {
                name: regime.name,
                constant: regime.constant,
                cells: group.len(),
                points,
                thinnest: thinnest.name(),
            }
        })
        .collect();
    out.sort_by_key(|c| c.name);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::read_ndjson;

    const MEASUREMENTS: &str = "../scripts/calibration/data/measurements.ndjson";

    /// Every modelled regime varies in the axes its rule depends on.
    ///
    /// The same invariant CI enforces, kept here so `cargo test` catches a regime
    /// added without the sweep to support it — the failure this whole module exists
    /// to make visible.
    #[test]
    fn no_modelled_regime_rests_on_a_single_point() {
        let text = std::fs::read_to_string(MEASUREMENTS).expect("the dataset is readable");
        let records = read_ndjson(&text).expect("the dataset parses");
        let thin: Vec<&str> = audit(&records)
            .iter()
            .filter(|c| c.is_thin())
            .map(|c| c.name)
            .collect();
        assert!(
            thin.is_empty(),
            "these regimes have one distinct point in an axis their rule depends on: {thin:?}"
        );
    }

    /// Every regime matches at least one cell.
    ///
    /// A predicate that matches nothing reports "never measured", which reads as a
    /// gap in the data when it may equally be a typo in the predicate. The four
    /// coarsened-key bugs this campaign found were all of that shape.
    #[test]
    fn every_regime_selects_something() {
        let text = std::fs::read_to_string(MEASUREMENTS).expect("the dataset is readable");
        let records = read_ndjson(&text).expect("the dataset parses");
        let empty: Vec<&str> = audit(&records)
            .iter()
            .filter(|c| c.cells == 0)
            .map(|c| c.name)
            .collect();
        assert!(empty.is_empty(), "these regimes matched no cell: {empty:?}");
    }
}
