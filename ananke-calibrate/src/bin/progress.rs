//! Report how far the campaign has got, and whether it is still moving.
//!
//! Replaces `scripts/calibration/progress.py`, which had stopped reporting
//! anything: it looked for `data/<phase>.ndjson` files the campaign no longer
//! writes, so every row read `0/?`. See [`ananke_calibrate::campaign::progress`]
//! for what replaced the filename globbing.
//!
//! ```sh
//! cargo run -p ananke-calibrate --bin progress
//! cargo run -p ananke-calibrate --bin progress -- --watch
//! ```

use std::{process::ExitCode, time::Duration};

use ananke_calibrate::{
    campaign::progress::{Report, idle_minutes, report},
    plan::Library,
    record::Record,
};
use clap::Parser;

const MEASUREMENTS: &str = "scripts/calibration/data/measurements.ndjson";
/// How long without a record before a campaign is presumed finished or stuck. The
/// slowest single cell in the campaign — a 205 GiB `--no-mmap` load — takes well
/// under this.
const IDLE_MINUTES: i64 = 45;
const WATCH_EVERY: Duration = Duration::from_secs(30);

#[derive(Debug, Parser)]
#[command(
    name = "progress",
    about = "Report the calibration campaign's progress"
)]
struct Args {
    /// Refresh every thirty seconds.
    #[arg(long)]
    watch: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let lib = Library::from_env();
    loop {
        if args.watch {
            // Clear and home, so a watched report stays in one screenful.
            print!("\x1b[2J\x1b[H");
        }
        match once(&lib) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("progress: {e}");
                return ExitCode::from(2);
            }
        }
        if !args.watch {
            return ExitCode::SUCCESS;
        }
        std::thread::sleep(WATCH_EVERY);
    }
}

fn once(lib: &Library) -> Result<(), String> {
    let text = std::fs::read_to_string(MEASUREMENTS).map_err(|e| format!("{MEASUREMENTS}: {e}"))?;
    let mut records = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        // A torn final line is expected against a live campaign: the harness may be
        // mid-append. Everything before it still counts.
        match serde_json::from_str::<Record>(line) {
            Ok(record) => records.push(record),
            Err(_) => continue,
        }
    }
    print(&report(&records, lib));
    Ok(())
}

fn print(report: &Report) {
    println!("{:<18} {:>12}  outstanding", "question", "measured");
    for question in &report.questions {
        let issues = if question.issues.is_empty() {
            "-".to_string()
        } else {
            question
                .issues
                .iter()
                .map(|(status, n)| format!("{n}x {status}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "{:<18} {:>12}  {issues}",
            question.name,
            format!("{}/{}", question.measured, question.planned)
        );
    }

    let percent = if report.planned == 0 {
        0.0
    } else {
        100.0 * report.measured as f64 / report.planned as f64
    };
    println!(
        "\ntotal {}/{} ({percent:.0}%) over distinct cells",
        report.measured, report.planned
    );
    if report.unplanned > 0 {
        println!(
            "{} measured cell(s) no question currently plans",
            report.unplanned
        );
    }
    match report.last_record.as_deref() {
        None => println!("no records yet"),
        Some(when) => match idle_minutes(when) {
            None => println!("last record {when}"),
            Some(minutes) => {
                let state = if minutes < IDLE_MINUTES {
                    "running"
                } else {
                    "stalled or finished"
                };
                println!("last record {minutes} min ago — {state}");
            }
        },
    }
}
