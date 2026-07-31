//! Reading the measurement dataset, and the checks that decide whether it can
//! be fitted at all.

use std::collections::BTreeMap;

use ananke_measure::record::Status;
use jiff::Timestamp;

use crate::{
    derive::error::{DeriveError, Result},
    record::{Record, read_ndjson},
};

/// Every completed measurement in an NDJSON dataset.
///
/// A row that did not reach a serving state carries no figure to fit, and a
/// `stale-runtime` row was recorded against a binary that is no longer
/// installed, so only `ok` survives.
pub fn load(text: &str) -> Result<Vec<Record>> {
    let rows = read_ndjson(text).map_err(DeriveError::malformed)?;
    Ok(rows
        .into_iter()
        .filter(|record| record.status == Status::Ok)
        .collect())
}

/// When a record was taken.
pub fn measured_at(record: &Record) -> Timestamp {
    instant(&record.provenance.measured_at_utc)
}

/// An ISO-8601 stamp as an instant, the epoch if it cannot be read.
///
/// Every consumer goes through this rather than comparing the strings. The two
/// agree only while every stamp in the file is the same fixed width at the same
/// offset, which is a property of today's dataset and not of the format.
pub fn instant(stamp: &str) -> Timestamp {
    stamp.parse().unwrap_or(Timestamp::UNIX_EPOCH)
}

/// Drop rows superseded by a later measurement of the same cell.
///
/// A cell id hashes the factors, so two rows sharing one describe the same
/// configuration — and when they were taken under different runtime builds, the
/// older one describes a program that is no longer installed. Fitting across
/// both fits two programs at once.
///
/// Keeping the newest is the rule rather than averaging because the quantity is
/// deterministic within a build: repeats taken back to back reproduce to the
/// megabyte, so a disagreement is a change in the runtime, not noise. GLM-5.2's
/// production cell reads 38708 MiB on one build and 34978 MiB on the next.
///
/// This is the only statement of the rule; the compute-model fit calls it rather
/// than carrying a second copy, which would be free to order the stamps
/// differently and select a different row.
pub fn latest_per_cell(rows: &[Record]) -> Vec<Record> {
    let mut newest: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, record) in rows.iter().enumerate() {
        let cell = record.cell_id();
        match newest.get(cell) {
            // Strictly later, so the first of two rows sharing a timestamp wins.
            Some(&held) if measured_at(record) <= measured_at(&rows[held]) => {}
            _ => {
                newest.insert(cell, index);
            }
        }
    }
    let keep: std::collections::BTreeSet<usize> = newest.into_values().collect();
    // Order preserved from the input, so an analysis that walks the dataset sees
    // it in measurement order rather than hash order.
    rows.iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, r)| r.clone())
        .collect()
}

/// How much of the dataset predates the runtime that is installed now.
///
/// Not a failure — re-measuring 500 cells costs days of GPU time, and most terms
/// do not move between builds: qwen35moe and laguna read identically across the
/// upgrade that shrank glm-dsa's compute buffer by a third. But "most" is doing
/// real work in that sentence, so the split is reported rather than assumed, per
/// runtime, with the newest build named.
pub fn report_stale_builds(rows: &[Record]) -> Vec<String> {
    let mut by_runtime: BTreeMap<&str, BTreeMap<&str, Vec<Timestamp>>> = BTreeMap::new();
    for record in rows {
        if record.status != Status::Ok {
            continue;
        }
        by_runtime
            .entry(record.factors.runtime.name())
            .or_default()
            .entry(&record.provenance.runtime_sha256)
            .or_default()
            .push(measured_at(record));
    }
    let mut out = Vec::new();
    for (runtime, builds) in &by_runtime {
        if builds.len() < 2 {
            continue;
        }
        let newest = builds
            .iter()
            .max_by_key(|(_, times)| latest(times))
            .map(|(build, _)| *build)
            .expect("the map is non-empty");
        let current = builds[newest].len();
        let stale: usize = builds
            .iter()
            .filter(|(b, _)| **b != newest)
            .map(|(_, v)| v.len())
            .sum();
        out.push(format!(
            "{runtime}: {stale} of {} cells predate the newest build {newest}; their \
             terms are calibrated against a program that is no longer installed",
            stale + current,
        ));
    }
    out
}

/// Refuse a dataset whose runtime builds disagree about the same cell.
///
/// A cell id hashes the *factors*, and the runtime binary is not one of them —
/// so upgrading llama.cpp leaves every existing row describing a program that is
/// no longer installed, with nothing to say so. That is not hypothetical:
/// ik_llama's DSA compute buffer shrank by a third between two builds, and
/// GLM-5.2 measured 11524 MiB of it on one and 7794 on the next at byte-identical
/// settings. The stale row is what made its measured grid look like it
/// accelerated with context, which is what made every candidate shape fail to fit
/// it.
///
/// So: where the same configuration was measured under two binaries, the two
/// readings must agree. They usually do, and where they do not, the older row has
/// to go rather than be averaged in.
///
/// `runtime_sha256` is what makes this checkable, and is the reason to keep
/// recording it on every row.
pub fn check_runtime_builds(rows: &[Record], tolerance: f64) -> Result<()> {
    let mut by_cell: BTreeMap<&str, BTreeMap<&str, u64>> = BTreeMap::new();
    for record in rows {
        let Some(used) = record.rss.gpu_used_mib.filter(|v| *v != 0) else {
            continue;
        };
        if record.status != Status::Ok {
            continue;
        }
        by_cell
            .entry(record.cell_id())
            .or_default()
            .insert(&record.provenance.runtime_sha256, used);
    }
    let mut disagreed = Vec::new();
    for (cell, readings) in &by_cell {
        if readings.len() < 2 {
            continue;
        }
        let low = *readings.values().min().expect("non-empty");
        let high = *readings.values().max().expect("non-empty");
        if low != 0 && (high - low) as f64 / low as f64 > tolerance {
            disagreed.push(format!(
                "{cell} reads {low}-{high} MiB across {} builds",
                readings.len()
            ));
        }
    }
    if disagreed.is_empty() {
        return Ok(());
    }
    disagreed.sort();
    Err(DeriveError::disagreement(format!(
        "the same configuration measures differently under different runtime \
         builds: {}. A constant fitted across both is fitted to two programs. \
         Re-measure under the installed build and drop the older rows.",
        disagreed.join("; ")
    )))
}

/// The default tolerance `check_runtime_builds` allows between two builds.
pub const BUILD_TOLERANCE: f64 = 0.10;

fn latest(times: &[Timestamp]) -> Timestamp {
    times.iter().copied().max().unwrap_or(Timestamp::UNIX_EPOCH)
}
