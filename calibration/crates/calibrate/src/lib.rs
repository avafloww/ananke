//! Planning a calibration campaign, and deriving the estimator's constants
//! from what it measures.
//!
//! Three groups of binaries, and the split is the point. `plan` and `campaign`
//! decide what to measure and drive [`ananke_measure`] over it; `fit` and
//! `emit` turn the resulting dataset into `crates/tuning/tuning.json`;
//! `coverage`, `validate`, `scoreboard`, `crossval`, and `estimates` say
//! whether the result is worth shipping. Only the middle pair writes anything.
//!
//! Two rules run through the whole crate.
//! A derivation that pairs two cells must pin **every** factor that could
//! differ between them, and a constant reduced from a group of cells must
//! refuse when the group disagrees rather than averaging a real difference
//! away. See [`derive`] for how each is enforced.
//!
//! `calibration/README.md` is the workflow: how to add a model, run the
//! campaign, refit, and read the result.

pub mod campaign;
pub mod compute_model;
pub mod coverage;
pub mod crossval;
pub mod derive;
pub mod models;
pub mod plan;
pub mod record;
pub mod validate;
