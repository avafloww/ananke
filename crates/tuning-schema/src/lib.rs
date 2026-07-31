//! The type of `tuning.json`.
//!
//! The document is read by four independent places — `ananke-tuning`'s build
//! script, the derivers, the emitter, and the compute-model fitter.
//! [`Document`] is the one declaration they share, so a section is spelled once
//! rather than once per reader, and a key that moves is a compile error rather
//! than a find by whichever reader happens to notice.
//!
//! Two constraints hold this crate in place, and both are load-bearing:
//!
//! - **Field order is the document's key order.** `serde_json` serialises a
//!   struct in declaration order, and the committed file's maps are sorted. So
//!   every struct here declares its fields alphabetically, and every map is a
//!   [`std::collections::BTreeMap`]. A field moved out of order rewrites the
//!   file.
//! - **It is a leaf.** A build script cannot depend on the crate it builds, so
//!   this cannot live in `ananke-tuning`, and it takes nothing but `serde`.

pub mod compute_model;
mod document;
mod tables;

pub use crate::{
    document::{Constant, ConstantValue, Document, DoesNotFit, Kind},
    tables::{RateTable, RateTableName, SlotScaling, TableLessObservations},
};
