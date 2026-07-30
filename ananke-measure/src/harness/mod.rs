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
//! a process, a driver, a socket, or a wall-clock wait. What is deliberately *not*
//! tested that way is the shell in [`run`] that orders the phases and the probing
//! in [`host`] that shells out for the box's identity; both are thin, and neither
//! decides anything.

pub mod cli;

mod cell;
mod dataset;
mod error;
mod host;
mod json;
mod maintain;
mod run;

/// Public only under `test-fakes`, so a consumer outside this crate can drive a
/// measurement against the in-memory implementations; private otherwise, because
/// the trait seams are an implementation detail of the harness rather than an API.
#[cfg(feature = "test-fakes")]
pub mod sys;
#[cfg(not(feature = "test-fakes"))]
mod sys;
