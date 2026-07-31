//! Linux-only: run a llama-server, watch what it costs, and write the row.
//!
//! The other half of this crate. [`crate::parse`] reads a log; this drives the
//! process that produces one — it spawns the server with a cell's flags, waits for
//! it to load, sends probe requests, samples `/proc` and the driver on the same
//! two-second cadence ananke's own snapshotter uses, and appends one NDJSON
//! record. It also holds the two maintenance passes over an existing dataset:
//! re-deriving `parsed` from the archived logs, and retiring rows a runtime
//! upgrade invalidated.
//!
//! Cells are keyed by a hash of their factors and a cell already present in the
//! output is skipped, so a long campaign survives interruption and can be extended
//! later without redoing work. That key is the most delicate thing here: see
//! [`cell::cell_id`] for what belongs in it and what a missing field costs.
//!
//! The outside world is behind the traits in [`sys`], with an in-memory
//! implementation of each, so every decision the harness makes is testable without
//! a process, a driver, a socket, a file, or a wall-clock wait. The one thing still
//! reached for directly is the probing in [`host`] that shells out for the box's
//! identity, which decides nothing.
//!
//! That includes the shell in [`run`]. It orders the phases and does not decide
//! much, but the order *is* the contract a driver commits against — a cell is
//! reported once, after its row is on disk — and "thin" is not a reason to leave a
//! contract unchecked.

pub mod cli;

/// The dataset's own writer, which lives with the schema because the spacing and
/// escaping it produces *are* the format — a cell's identity is hashed over those
/// bytes. Everything the harness writes goes through it.
pub(crate) use ananke_dataset::to_dataset_json;

/// The seam a driver needs: measure a plan, and name a cell the way the dataset
/// does. `ananke-calibrate`'s campaign generates the plan and commits the rows as
/// they land, and both halves have to agree with the harness on what a cell *is* —
/// so the identity is shared rather than restated. Everything else here stays
/// private.
pub use crate::harness::{
    cell::cell_id,
    error::Error,
    run::{Completed, Options, Summary, measure_cells, measure_cells_with},
};

mod cell;
mod dataset;
mod error;
mod host;
mod maintain;
mod run;

/// Diagnostics for what a once-per-configuration sample cannot separate: whether a
/// term is allocated once or accumulates with use.
pub mod probe;

/// The trait seams themselves, because [`measure_cells_with`] takes them: a driver
/// substitutes the world to check the contract it depends on. The in-memory
/// implementations inside stay behind `test-fakes` — the seams are API, the fakes
/// are a test aid.
pub mod sys;
