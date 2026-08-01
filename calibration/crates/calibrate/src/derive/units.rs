//! The one spelling of a mebibyte, re-exported.
//!
//! The measurements, the log lines they are parsed from, and `tuning.json`'s own
//! `$comment`s are all written in MiB; the constants ship in bytes. Every
//! derivation crosses that boundary at least once. The definitions live in
//! `ananke-config` because the estimator and the daemon cross it too.

pub use ananke_config::units::{MIB, MIB_F64, MIB_I64};
