//! Refit the compute model from the measurements and write it into `tuning.json`.
//!
//! The one section `emit` does not touch. It carries the committed `compute_model`
//! through unchanged, which is what lets the derivers be checked on their own — so
//! without this binary the fit would be reachable only from a test, and a fresh
//! dataset could not produce a fresh model.
//!
//! ```sh
//! cargo run -p ananke-calibrate --bin fit            # refit and write
//! cargo run -p ananke-calibrate --bin fit -- --check # does the committed fit follow?
//! ```
//!
//! Run it *after* `emit`. The dependency is one-way: the design rows are
//! normalised by `MAINLINE_LAYER_SPLIT_MASK_COPIES`, which `emit` derives, and no
//! deriver reads this section back. So the pair is ordered, not iterated.

use std::{path::Path, process::ExitCode};

use ananke_calibrate::{
    compute_model::{collect, dataset::latest_per_cell, document_section},
    derive::tuning::Tuning,
    record::read_ndjson,
};
use ananke_measure::record::Status;

const MEASUREMENTS: &str = "calibration/data/measurements.ndjson";
const TUNING: &str = "crates/tuning/tuning.json";
/// The constant every design row is normalised by; see the read below.
const MASK_COPIES: &str = "MAINLINE_LAYER_SPLIT_MASK_COPIES";

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
    let tuning_text = match std::fs::read_to_string(TUNING) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading {TUNING}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut document: serde_json::Value = match serde_json::from_str(&tuning_text) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("parsing {TUNING}: {e}");
            return ExitCode::from(2);
        }
    };

    // Only the cells that loaded carry usable readings, and only the newest row
    // for each cell — the same reduction `emit` applies, so the two halves of the
    // document describe one dataset.
    let ok: Vec<_> = records
        .into_iter()
        .filter(|r| r.status == Status::Ok)
        .collect::<Vec<_>>();
    let rows = latest_per_cell(&ok);
    // Read from the document rather than the compiled constant of the same name.
    // `emit` writes that constant into this file, and a compiled copy is one build
    // behind it — which would put a `cargo build` in the middle of the pipeline and
    // make the result depend on whether anyone remembered to run it.
    //
    // And it is read as required rather than assumed: the value normalises every
    // design row, so a missing one does not perturb the fit, it moves every
    // coefficient in it.
    let copies = match Tuning::parse(&tuning_text) {
        Ok(tuning) => match tuning.constant(MASK_COPIES).map(u32::try_from) {
            Some(Ok(copies)) => copies,
            Some(Err(e)) => {
                eprintln!("{TUNING}'s {MASK_COPIES} is not a mask count: {e}");
                return ExitCode::from(2);
            }
            None => {
                eprintln!(
                    "{TUNING} declares no {MASK_COPIES}; run `cargo run -p \
                     ananke-calibrate --bin emit` to derive it before fitting"
                );
                return ExitCode::from(2);
            }
        },
        Err(e) => {
            eprintln!("parsing {TUNING}: {e}");
            return ExitCode::from(2);
        }
    };
    let groups = collect(&rows, copies, false);

    let (section, notes) = match document_section(&groups) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("fitting the compute model: {e}");
            return ExitCode::from(1);
        }
    };
    for note in &notes {
        println!("  note: {note}");
    }

    if check {
        if document.get("compute_model") == Some(&section) {
            println!("compute_model matches the dataset ({} rows)", rows.len());
            return ExitCode::SUCCESS;
        }
        eprintln!(
            "compute_model no longer follows from the dataset; run `cargo run -p \
             ananke-calibrate --bin fit` to refit it"
        );
        return ExitCode::from(1);
    }

    let unchanged = document.get("compute_model") == Some(&section);
    document["compute_model"] = section;
    let rendered = match serde_json::to_string_pretty(&document) {
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
    println!(
        "wrote {TUNING}'s compute_model from {} rows{}",
        rows.len(),
        if unchanged { " (unchanged)" } else { "" }
    );
    ExitCode::SUCCESS
}
