//! The measurement record as it appears in `measurements.ndjson`.
//!
//! Every type here belongs to [`ananke_dataset`], the one schema the dataset is
//! written and read with. A second, tolerant reader on the calibration side —
//! only the fields the derivers want, everything else an `Option` or a `Value`
//! map — would build this campaign's signature failure mode into it: a factor
//! the harness starts varying that the reader never learns about, so a term is
//! fitted across cells that differ in a way the fit cannot see. Four wrong
//! constants came from exactly that. `tests/factors.rs` enumerates the
//! deliberate omissions instead, which is a decision somebody has to write down
//! rather than one a missing field makes silently.
//!
//! The module is a re-export so `crate::record::Record` keeps resolving, and so
//! the derivers' docs can name the row they read without reaching across crates
//! at every use.

pub use ananke_dataset::{
    Context, DeviceRow, Factors, FlashAttn, Gpu, Hardware, KvPool, KvType, Parsed, Provenance,
    Record, RsPool, Rss, Runtime, Status, read_ndjson,
};
