//! Derive the estimator's constants from the measurements, or verify them.
//!
//! With `--check` it is the CI gate: a constant that no longer follows from the
//! dataset fails the build rather than drifting quietly.
//!
//! `compute_model` is written by the fitter (`ananke_calibrate::compute_model`),
//! not here. `--check` compares the whole document including that section, so the
//! two halves are verified together; only the *deriving* is split.

use std::{path::Path, process::ExitCode};

use ananke_calibrate::{
    derive::emit::{emit_check, emit_write},
    record::read_ndjson,
};
use ananke_measure::record::Status;

const MEASUREMENTS: &str = "calibration/data/measurements.ndjson";
const TUNING: &str = "crates/tuning/tuning.json";

fn main() -> ExitCode {
    let check = std::env::args().any(|a| a == "--check");

    let measurements = match std::fs::read_to_string(MEASUREMENTS) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };
    let records = match read_ndjson(&measurements) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parsing {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };
    let tuning = match std::fs::read_to_string(TUNING) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading {TUNING}: {e}");
            return ExitCode::from(2);
        }
    };

    // Only the cells that loaded carry usable readings; the rest are counted by
    // the derivers themselves, which report what they skipped.
    let ok: Vec<_> = records
        .iter()
        .filter(|r| r.status == Status::Ok)
        .cloned()
        .collect();

    let result = if check {
        emit_check(&ok, &tuning)
    } else {
        emit_write(&ok, &tuning)
    };
    let emitted = match result {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    for note in &emitted.notes {
        println!("  note: {note}");
    }
    if check {
        // The count after de-duplication, which is what the derivers actually
        // fitted against — `ok.len()` counts superseded rows the emit dropped.
        println!(
            "tuning.json matches the dataset ({} measurements)",
            emitted.measurements
        );
        return ExitCode::SUCCESS;
    }

    let rendered = match serde_json::to_string_pretty(&emitted.document) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("serialising the document: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = std::fs::write(Path::new(TUNING), rendered + "\n") {
        eprintln!("writing {TUNING}: {e}");
        return ExitCode::from(2);
    }
    println!("wrote {TUNING} from {} measurements", emitted.measurements);
    for line in &emitted.changed {
        println!("  changed: {line}");
    }
    for line in &emitted.failed {
        println!("  failed: {line}");
    }
    ExitCode::SUCCESS
}
