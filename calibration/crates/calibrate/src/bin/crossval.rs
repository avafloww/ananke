//! Report leave-one-model-out cross-validation over the derived constants.
//!
//! The campaign's analysis protocol asks for accuracy this way rather than from the
//! `holdout` question, whose models are in the fitting set. See
//! [`ananke_calibrate::crossval`] for what a fold is and what it cannot answer.
//!
//! ```sh
//! cargo run -p ananke-calibrate --bin crossval
//! cargo run -p ananke-calibrate --bin crossval -- --constant MMPROJ_GRAPH_BYTES
//! cargo run -p ananke-calibrate --bin crossval -- --check --tolerance 10
//! ```

use std::process::ExitCode;

use ananke_calibrate::{
    crossval::{ConstantReport, cross_validate},
    derive::tuning::Tuning,
    record::{Record, read_ndjson},
};

const MEASUREMENTS: &str = "calibration/data/measurements.ndjson";
const TUNING: &str = "crates/tuning/tuning.json";

fn main() -> ExitCode {
    let mut tolerance = 10.0_f64;
    let mut check = false;
    let mut constant: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tolerance" => tolerance = args.next().and_then(|v| v.parse().ok()).unwrap_or(10.0),
            "--check" => check = true,
            "--constant" => constant = args.next(),
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let (rows, tuning) = match load() {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("crossval: {error}");
            return ExitCode::from(2);
        }
    };

    let mut reports = cross_validate(&rows, &tuning);
    if let Some(wanted) = constant.as_deref() {
        reports.retain(|r| r.constant == wanted);
        if reports.is_empty() {
            eprintln!("crossval: no constant named {wanted}");
            return ExitCode::from(2);
        }
    }

    for report in &reports {
        print(report);
    }
    summarise(&reports, tolerance, check)
}

fn load() -> Result<(Vec<Record>, Tuning), String> {
    let text = std::fs::read_to_string(MEASUREMENTS).map_err(|e| format!("{MEASUREMENTS}: {e}"))?;
    let rows: Vec<Record> = read_ndjson(&text)
        .map_err(|e| format!("{MEASUREMENTS}: {e}"))?
        .into_iter()
        .filter(|r| r.status == "ok")
        .collect();
    let tuning =
        Tuning::parse(&std::fs::read_to_string(TUNING).map_err(|e| format!("{TUNING}: {e}"))?)
            .map_err(|e| format!("{TUNING}: {e}"))?;
    Ok((rows, tuning))
}

fn print(report: &ConstantReport) {
    let full = match report.full.value() {
        Some(value) => value.to_string(),
        None => "not derived".to_string(),
    };
    println!("\n{}  (full fit: {full})", report.constant);

    let evaluated = report.evaluated();
    if evaluated.is_empty() {
        // Not a failure. Most constants rest on a handful of models, and a model
        // that contributes no cell to one produces two identical fits and no fold.
        println!("  no model can be held out and re-predicted");
    }
    for fold in evaluated {
        let (Some(without), Some(alone), Some(error)) = (
            fold.without.value(),
            fold.alone.value(),
            fold.generalisation(),
        ) else {
            continue;
        };
        println!(
            "  {:<44} others say {:>14}  it says {:>14}  {:+.1}%",
            fold.model, without, alone, error
        );
    }

    let load_bearing = report.load_bearing();
    if !load_bearing.is_empty() {
        println!(
            "  rests on {}: without it the constant cannot be fitted at all",
            load_bearing.join(", ")
        );
    }
}

/// The worst fold per constant: the model, the relative error, and the absolute
/// difference in the constant's own units.
///
/// The absolute figure is there because the relative one invites a reading it does
/// not support. `MMPROJ_GRAPH_BYTES` generalises at +77%, which sounds alarming and
/// is 108 MiB — around a quarter of a percent of the estimate it lands in. A
/// constant's error is not its estimate's error, and a summary that printed only
/// percentages would suggest otherwise.
type Worst<'a> = (&'a str, &'a str, f64, i64);

fn summarise(reports: &[ConstantReport], tolerance: f64, check: bool) -> ExitCode {
    let mut worst: Vec<Worst<'_>> = Vec::new();
    let mut single_support = Vec::new();
    let mut unevaluable = Vec::new();
    for report in reports {
        match report.worst() {
            Some((fold, error)) => {
                let delta = fold.without.value().unwrap_or(0) - fold.alone.value().unwrap_or(0);
                worst.push((report.constant, fold.model.as_str(), error, delta));
            }
            None => unevaluable.push(report.constant),
        }
        if !report.load_bearing().is_empty() {
            single_support.push(report.constant);
        }
    }
    worst.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));

    println!("\n{} constant(s) cross-validated", reports.len());
    if !unevaluable.is_empty() {
        println!(
            "{} not cross-validatable — no model can be held out and re-predicted: {}",
            unevaluable.len(),
            unevaluable.join(", ")
        );
    }
    if !single_support.is_empty() {
        println!(
            "{} rest on a single model, so they are model-specific fits rather than \
             architecture constants: {}",
            single_support.len(),
            single_support.join(", ")
        );
    }

    let over: Vec<_> = worst
        .iter()
        .filter(|(_, _, e, _)| e.abs() > tolerance)
        .collect();
    for (constant, model, error, delta) in &worst {
        println!(
            "  worst fold  {constant:<40} {model:<44} {error:+.1}%  ({delta:+} in its own units)"
        );
    }
    if worst.is_empty() {
        println!("no fold could be evaluated, so there is no generalisation figure");
        return ExitCode::SUCCESS;
    }
    println!(
        "\nworst generalisation error: {:+.1}% (tolerance {tolerance}%)",
        worst[0].2
    );
    if check && !over.is_empty() {
        eprintln!(
            "{} constant(s) generalise worse than {tolerance}%",
            over.len()
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
