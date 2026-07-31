//! The one spelling of a mebibyte.
//!
//! The measurements, the log lines they are parsed from, and this document's own
//! `$comment`s are all written in MiB; the constants ship in bytes. Every
//! derivation crosses that boundary at least once, so the conversion has one name
//! rather than an inline `1048576` at each of eighteen sites.

/// A mebibyte.
pub const MIB: u64 = 1024 * 1024;

/// [`MIB`] where the surrounding arithmetic is signed, which a derived constant's
/// value is.
pub const MIB_I64: i64 = MIB as i64;

/// [`MIB`] where the surrounding arithmetic is floating point, which every
/// residual and rate is.
pub const MIB_F64: f64 = MIB as f64;
