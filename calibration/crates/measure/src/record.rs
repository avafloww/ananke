//! The NDJSON record a measured cell is written as.
//!
//! Every type here belongs to [`ananke_dataset`], the one schema the dataset is
//! written and read with. This module is a re-export, so that
//! `crate::record::Status` and its siblings resolve locally, and so the harness's
//! own docs can name the row it writes without reaching across crates at every
//! use.
//!
//! This is the schema only. What fills `trace`, `checkpoints`, and `rss` is
//! [`crate::harness`], which samples them; the types here describe the row
//! rather than producing it, so a reader of the dataset needs none of the
//! harness.

pub use ananke_dataset::{
    Checkpoint, Cpu, DEFAULT_PROBE_PROMPT_TOKENS, FULLY_OFFLOADED, Factors, FlashAttn, Gpu,
    GpuUsage, Hardware, KvType, Provenance, Record, Rss, RssSnapshot, Runtime, SCHEMA, Sample,
    Status,
};
