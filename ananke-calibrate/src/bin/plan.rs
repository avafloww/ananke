//! Print the measurement plan for one question, or the whole campaign.
//!
//! See [`ananke_calibrate::plan`] for why the sweeps are generated rather than
//! hand-written, and [`ananke_calibrate::coverage`] for the audit they answer.
//!
//! ```text
//! cargo run -p ananke-calibrate --bin plan -- curves > curves.json
//! cargo run -p ananke-measure --bin measure -- --plan curves.json --out data/measurements.ndjson
//! ```

use std::process::ExitCode;

use ananke_calibrate::plan::{all_cells, cells_for, library::Library, question_names, to_json};

fn main() -> ExitCode {
    let Some(question) = std::env::args().nth(1) else {
        eprintln!("usage: plan <question>\n\nquestions: {}", questions());
        return ExitCode::from(2);
    };

    let lib = Library::from_env();
    let cells = if question == "all" {
        all_cells(&lib)
    } else {
        match cells_for(&question, &lib) {
            Some(cells) => cells,
            None => {
                eprintln!("{question} is not a question\n\nquestions: {}", questions());
                return ExitCode::from(2);
            }
        }
    };

    println!("{}", to_json(&cells));
    ExitCode::SUCCESS
}

fn questions() -> String {
    let mut names = question_names();
    names.push("all");
    names.join(", ")
}
