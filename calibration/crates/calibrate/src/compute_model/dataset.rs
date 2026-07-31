//! The one dataset helper the fit needs and nothing else does.
//!
//! Superseding a cell by a later measurement of it is *not* here: that rule is
//! [`crate::derive::dataset::latest_per_cell`], which the fit calls. A second copy
//! here would be free to order the stamps as strings, which agrees with the
//! timestamps only while every one of them is the same fixed width at the same
//! offset.

use ananke_dataset::BufferRole;

use crate::record::Record;

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
    let total = record.rss.gpu_used_mib.filter(|v| *v != 0)? as f64;
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
            weights += buffers.get(&BufferRole::Model).copied().unwrap_or(0.0);
            kv += buffers.get(&BufferRole::Kv).copied().unwrap_or(0.0)
                + buffers.get(&BufferRole::Rs).copied().unwrap_or(0.0);
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
