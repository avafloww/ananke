//! The two dataset helpers the fit needs.
//!
//! They live here rather than in a general dataset module because the fit is
//! their only consumer.

use std::collections::{HashMap, HashSet};

use crate::record::Record;

/// Drop rows superseded by a later measurement of the same cell.
///
/// A cell id hashes the factors, so two rows sharing one describe the same
/// configuration — and when they were taken under different runtime builds, the
/// older one describes a program that is no longer installed. Fitting across both
/// fits two programs at once.
///
/// Keeping the newest is the rule rather than averaging because the quantity is
/// deterministic within a build: repeats taken back to back reproduce to the
/// megabyte, so a disagreement is a change in the runtime, not noise. GLM-5.2's
/// production cell reads 38708 MiB on one build and 34978 on the next.
pub fn latest_per_cell(rows: &[Record]) -> Vec<&Record> {
    let mut newest: HashMap<&str, usize> = HashMap::new();
    for (index, record) in rows.iter().enumerate() {
        let Some(cell) = record.cell.as_deref() else {
            continue;
        };
        let supersedes = match newest.get(cell) {
            None => true,
            Some(&current) => {
                record.provenance.measured_at_utc > rows[current].provenance.measured_at_utc
            }
        };
        if supersedes {
            newest.insert(cell, index);
        }
    }
    // Order preserved from the input, so an analysis that walks the dataset sees
    // it in measurement order rather than hash order.
    let keep: HashSet<usize> = newest.into_values().collect();
    rows.iter()
        .enumerate()
        .filter(|(index, _)| keep.contains(index))
        .map(|(_, record)| record)
        .collect()
}

/// Per-device `compute + unaccounted` for a cell with no breakdown table.
///
/// ik_llama does not print llama.cpp's memory breakdown, so the column the fit
/// wants does not exist. It is still recoverable: the driver's total for the
/// process, less the weights it loaded onto each card and less the context it
/// allocated there, over the number of cards. Everything left is the graph plus
/// whatever the driver holds that the runtime never named — which is exactly what
/// the table's two columns sum to.
///
/// Reproduces the production GLM-5.2 cell: 38708 MiB from the driver against
/// 14064 of weights and 11904 of cache leaves 6370 per card, and the runtime's own
/// compute-buffer lines account for 6260 of that.
pub fn table_less_compute(record: &Record) -> Option<f64> {
    let total = record.gpu_used_mib().filter(|v| *v != 0)? as f64;
    let devices = record
        .factors
        .gpus
        .split(',')
        .filter(|g| !g.is_empty())
        .count()
        .max(1) as f64;
    let mut weights = 0.0;
    let mut kv = 0.0;
    for context in &record.parsed.contexts {
        for (name, buffers) in &context.buffers {
            if !name.starts_with("CUDA") || name.ends_with("Host") {
                continue;
            }
            weights += buffers.get("model").copied().unwrap_or(0.0);
            kv += buffers.get("kv").copied().unwrap_or(0.0)
                + buffers.get("rs").copied().unwrap_or(0.0);
        }
    }
    if weights == 0.0 {
        return None;
    }
    let remainder = total - weights - kv;
    // A negative remainder means the buffer lines and the driver disagree about
    // what is on the cards — a partial offload the parse cannot attribute — and is
    // not a compute-buffer measurement.
    (remainder > 0.0).then_some(remainder / devices)
}
