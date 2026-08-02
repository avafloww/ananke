//! Measured tuning constants, generated from `tuning.json` at build time.
//!
//! The JSON at the crate root is the source of truth. It is produced by
//! `ananke-calibrate`'s `fit` and `emit` binaries from the measurement dataset
//! (`calibration/README.md` is the workflow), and
//! `build.rs` turns it into the constants below — so a value the estimator uses cannot
//! drift from the data that justifies it without that showing up as a diff in a
//! generated file rather than as an unexplained edit.
//!
//! Each constant carries its evidence in its own doc comment: how many models
//! and cells it rests on, and where the measurement is weak. Several are chosen
//! for *reachability* rather than closeness — the rolling correction clamps to
//! `[0.8, 1.5]` of observation, so a constant that puts a model outside that
//! band cannot be recovered by any amount of observation, and a "closer" value
//! that does so is strictly worse.
//!
//! To change one, re-run the campaign and regenerate the JSON. Editing the
//! generated constants directly is not possible, which is the point.
//!
//! This is a crate of its own rather than a module of `ananke` so that
//! regenerating the constants does not rebuild the web UI: `tuning.json` is a
//! `rerun-if-changed` input, and `ananke`'s build script also runs the
//! frontend's `npm run build`.

include!(concat!(env!("OUT_DIR"), "/tuning_constants.rs"));
