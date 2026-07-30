//! Run the calibration campaign end to end, committing as it goes.
//!
//! The driver owns the loop: it generates the schedule, hands each cell to the
//! harness, and commits the dataset at every cell boundary.
//!
//! ```sh
//! cargo run -p ananke-calibrate --bin campaign               # every cell, cheapest order
//! cargo run -p ananke-calibrate --bin campaign -- --dry-run  # print the schedule and stop
//! cargo run -p ananke-calibrate --bin campaign -- --only laguna
//! ```

use std::{path::PathBuf, process::ExitCode, time::Duration};

use ananke_calibrate::{
    campaign::{self, Campaign, HEADROOM_GIB, git::LocalGit},
    plan::Library,
};
use clap::Parser;

const DATA: &str = "scripts/calibration/data";

#[derive(Debug, Parser)]
#[command(name = "campaign", about = "Run the calibration campaign end to end")]
struct Args {
    /// Substring filter on the cell label.
    #[arg(long)]
    only: Option<String>,
    /// Seconds to wait for a model to load; a 200 GiB `--no-mmap` load takes
    /// minutes.
    #[arg(long, default_value_t = 2400)]
    load_timeout: u64,
    /// Print the schedule and stop.
    #[arg(long)]
    dry_run: bool,
    /// Host memory to leave free.
    #[arg(long, default_value_t = HEADROOM_GIB)]
    headroom_gib: f64,
    /// Swap growth that ends the run.
    #[arg(long, default_value_t = 4.0)]
    swap_limit_gib: f64,
    #[arg(long, default_value_t = 18099)]
    port: u16,
    #[arg(long, default_value = "/tmp/ananke-calibration")]
    log_dir: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let data = PathBuf::from(DATA);
    let campaign = Campaign {
        out: data.join("measurements.ndjson"),
        plan: data.join("plan.json"),
        // The dataset, the schedule that produced it, and the logs that make it
        // re-parseable. Scoped deliberately: see `campaign::git`.
        data_paths: vec![
            data.join("measurements.ndjson"),
            data.join("plan.json"),
            data.join("logs"),
        ],
        only: args.only,
        load_timeout: Duration::from_secs(args.load_timeout),
        headroom_gib: args.headroom_gib,
        log_dir: args.log_dir,
        archive_dir: data.join("logs"),
        port: args.port,
        swap_limit_gib: args.swap_limit_gib,
    };

    let lib = Library::from_env();
    let cells = campaign::schedule(&campaign, &lib);

    if args.dry_run {
        println!("{} cells", cells.len());
        for cell in &cells {
            println!("  {}", cell.label);
        }
        return ExitCode::SUCCESS;
    }
    if cells.is_empty() {
        eprintln!("campaign: no cell matches the filter");
        return ExitCode::FAILURE;
    }

    // The schedule is written before anything runs, so a reader of a campaign in
    // flight can see what it intends and not only what it has done. A filtered run
    // writes it outside the committed data; see `campaign::schedule_path`.
    let plan_path = campaign::schedule_path(&campaign);
    if let Some(parent) = plan_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("creating {}: {e}", parent.display());
        return ExitCode::from(2);
    }
    if let Err(e) = std::fs::write(&plan_path, ananke_calibrate::plan::to_json(&cells)) {
        eprintln!("writing {}: {e}", plan_path.display());
        return ExitCode::from(2);
    }

    println!("{} cells planned", cells.len());
    match campaign::run(&campaign, &cells, &LocalGit::at(".")) {
        Ok(summary) => {
            println!(
                "\ncampaign finished: {} measured, {} skipped, {} without a measurement",
                summary.measured, summary.skipped, summary.failed
            );
            // A swap abort stopped the campaign part-way; the operator has to see
            // that in the exit status rather than in the scrollback.
            if summary.aborted_on_swap.is_some() {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("campaign: {e}");
            ExitCode::from(1)
        }
    }
}
