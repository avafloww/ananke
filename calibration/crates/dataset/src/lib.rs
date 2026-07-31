//! The one schema `calibration/data/measurements.ndjson` is written and read
//! with.
//!
//! The dataset had three independent descriptions of itself: the harness's
//! write-only `Record`, the calibration tools' read-only one, and a
//! hand-extracting `serde_json::Value` walk in the estimator's integration
//! test. They disagreed — the two `Parsed` blocks differed by seven fields —
//! and neither could round-trip, so a maintenance pass over the dataset had to
//! splice raw bytes rather than re-serialise a row.
//!
//! This crate is the union of the two, taking the better-typed side of each
//! disagreement, and it derives both halves of serde on every type. That makes
//! the round-trip checkable, and [`tests/roundtrip.rs`] checks it against every
//! committed row: a field this schema fails to model is a test failure rather
//! than a silently dropped column.
//!
//! [`json::to_dataset_json`] is part of the format, not of any one tool: the
//! dataset's spacing and escaping are what a cell's identity is hashed over.

pub mod json;
pub mod parsed;
pub mod record;

pub use crate::{
    json::to_dataset_json,
    parsed::{BufferRole, Context, DeviceRow, HostBreakdown, KvPool, Mapped, Parsed, RsPool},
    record::{
        Checkpoint, Cpu, Factors, FlashAttn, Gpu, GpuUsage, Hardware, KvType, Provenance, Record,
        Rss, RssSnapshot, Runtime, SCHEMA, Sample, Status,
    },
};

/// Read an NDJSON measurement file, skipping blank lines.
///
/// Errors name the line, because a dataset of six hundred rows is not something
/// a bare serde message locates.
///
/// A blank `cell` is rejected here rather than tolerated downstream. Every reader
/// groups rows by that identity — superseding a stale measurement, attributing a
/// cell to a question — and a row without one either disappears from the group or
/// joins a bucket of everything else that has none. Checking once at the boundary
/// is what lets [`Record::cell_id`] be total.
pub fn read_ndjson(text: &str) -> Result<Vec<Record>, String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            let record: Record = serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", index + 1))?;
            if record.cell.is_empty() {
                return Err(format!(
                    "line {}: the row carries no cell id, so nothing can group it \
                     with the other measurements of its configuration",
                    index + 1
                ));
            }
            Ok(record)
        })
        .collect()
}
