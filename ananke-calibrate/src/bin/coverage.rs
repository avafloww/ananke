//! Report where the dataset is too thin to have measured what it claims.
//!
//! See [`ananke_calibrate::coverage`] for why this exists and which four
//! constants it was written after.

use std::process::ExitCode;

use ananke_calibrate::{coverage::audit, record::read_ndjson};

const MEASUREMENTS: &str = "scripts/calibration/data/measurements.ndjson";

fn main() -> ExitCode {
    let check = std::env::args().any(|a| a == "--check");
    let text = match std::fs::read_to_string(MEASUREMENTS) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };
    let records = match read_ndjson(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parsing {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };

    let rows = audit(&records);
    println!(
        "{:22}{:>7}{:>26}   constant",
        "regime", "cells", "thinnest axis"
    );
    let mut thin = Vec::new();
    for row in &rows {
        let mark = if row.is_thin() { "  <-- one point" } else { "" };
        println!(
            "{:22}{:>7}{:>26}   {}{mark}",
            row.name,
            row.cells,
            format!("{} ({})", row.thinnest, row.points),
            row.constant
        );
        if row.is_thin() {
            thin.push(row);
        }
    }

    if !check {
        return ExitCode::SUCCESS;
    }
    if thin.is_empty() {
        println!("\nevery modelled regime varies in the axes its rule depends on");
        return ExitCode::SUCCESS;
    }
    println!("\nregimes measured at a single point in an axis their rule depends on:");
    for row in &thin {
        println!(
            "  {}: one distinct {}, and {} is fitted from it",
            row.name, row.thinnest, row.constant
        );
    }
    println!(
        "\nA rule that is wrong in that axis is invisible at one point. Add a \
         second before trusting the constant."
    );
    ExitCode::from(1)
}
