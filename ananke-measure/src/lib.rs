//! Read a llama-server load log into the decomposition ananke's memory
//! constants are fitted against.
//!
//! Every memory constant in the estimator should be traceable to a row the
//! calibration campaign produced, and a row records a *decomposition* rather
//! than a total, because the decomposition is what the model is built from:
//!
//! - `arena` — the graph allocator's buffer, as the loader logs it. Pinned
//!   (`CUDA_Host`) whenever a GPU is present, plain `CPU` otherwise.
//! - `pinned` — `RssShmem - arena`. `cudaMallocHost` is accounted as shmem, so
//!   this is the pinned memory that is *not* the graph arena.
//! - `baseline` — `RssAnon - CPU KV - (host weights, when anonymous)`. What the
//!   process holds beyond anything the model already explains.
//!
//! This crate is the reading half: it turns a captured log into [`Parsed`] and
//! describes the NDJSON record that carries it. Running a server, sampling
//! `/proc`, and polling the driver are deliberately not here — the crate stays
//! a pure library with no process spawner in it, and it links neither the
//! estimator nor the packer so that measurement and estimation cannot drift
//! into each other.

pub mod parse;
pub mod record;

pub use crate::{
    parse::{Parsed, parse_log},
    record::{Record, SCHEMA},
};
