//! The `measure` binary's command line.
//!
//! One binary rather than the Python's two. `measure_one.py` existed because the
//! pre-flight fit gate weighs the *model file*, which is the wrong quantity for
//! heavy expert offload — GLM-5.2's file is 205 GiB and its process peaks at 187 —
//! so it ran the same `measure()` with the gate removed and a swap watchdog added.
//! Here the gate is a `--force` away and the watchdog is always on: it costs
//! nothing when nothing is paging, and it tripped twice on GLM during the campaign.

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use clap::Parser;

use crate::harness::{
    cell, dataset,
    error::Error,
    maintain,
    run::{self, Options},
    sys::Deps,
};

/// The port the campaign has always used. Deliberately not 8080: a cell must not
/// silently measure a server the operator started for something else.
const DEFAULT_PORT: u16 = 18099;
/// How far a re-measurement has to move before it counts as the runtime having
/// changed under us, rather than as run-to-run noise.
const STALE_TOLERANCE: f64 = 0.02;

#[derive(Debug, Parser)]
#[command(
    name = "measure",
    about = "Measure a llama-server configuration's host-memory decomposition"
)]
struct Args {
    /// The NDJSON the campaign accumulates. Read for what is already measured,
    /// appended to, and — for the two maintenance passes — rewritten.
    #[arg(long)]
    out: PathBuf,
    /// A JSON list of cells to run, each object a factor set.
    #[arg(long)]
    plan: Option<PathBuf>,
    /// Mark rows a runtime upgrade invalidated, proven by a re-measured cell, as
    /// `stale-runtime` so every reader skips them.
    #[arg(long)]
    retire_stale_builds: bool,
    /// Re-derive every record's `parsed` block from its archived log instead of
    /// measuring anything.
    #[arg(long)]
    reparse: bool,
    #[arg(long, default_value = "/tmp/ananke-calibration")]
    log_dir: PathBuf,
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,
    /// Where to keep the gzipped load logs; they are what makes a record
    /// re-parseable later.
    #[arg(long, default_value = "scripts/calibration/data/logs")]
    archive_dir: PathBuf,
    /// Host memory to leave free; a cell needing more than the remainder is skipped
    /// rather than risking the machine.
    #[arg(long, default_value_t = 30.0)]
    headroom_gib: f64,
    /// Seconds to wait for a model to load; a 200 GiB `--no-mmap` load takes
    /// minutes.
    #[arg(long, default_value_t = 1800)]
    load_timeout: u64,
    /// Swap growth that ends the run. The margin on a heavy hybrid is a gigabyte or
    /// two, so overcommitting is a real possibility rather than a formality — and a
    /// box that is paging is not measuring anything.
    #[arg(long, default_value_t = 4.0)]
    swap_limit_gib: f64,
    /// Measure a cell the pre-flight fit gate would refuse. The gate weighs the
    /// model file against available memory, which over-charges an
    /// expert-offloaded cell whose GPU-resident share never touches host RAM.
    #[arg(long)]
    force: bool,
    /// Measure a cell that already has a record. The cell id hashes the factors and
    /// the runtime is not one of them, so an upgraded binary needs this to be
    /// re-measured at all.
    #[arg(long)]
    remeasure: bool,
}

pub fn main() -> ExitCode {
    match dispatch(Args::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("measure: failed to {error}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: Args) -> Result<ExitCode, Error> {
    if args.retire_stale_builds {
        return retire(&args.out);
    }
    if args.reparse {
        return reparse(&args.out, &args.archive_dir);
    }
    let Some(plan) = args.plan.as_deref() else {
        eprintln!("measure: --plan is required unless --reparse or --retire-stale-builds is given");
        return Ok(ExitCode::FAILURE);
    };
    let cells = cell::load_plan(&std::fs::read_to_string(plan).map_err(Error::io)?)?;
    let options = Options {
        out: args.out,
        log_dir: args.log_dir,
        archive_dir: Some(args.archive_dir),
        port: args.port,
        load_timeout: Duration::from_secs(args.load_timeout),
        headroom_gib: args.headroom_gib,
        swap_limit_gib: args.swap_limit_gib,
        force: args.force,
        remeasure: args.remeasure,
    };
    // No observer: the standalone binary measures one plan and reports at the end.
    // `campaign` is the driver that wants per-cell notice.
    let summary = run::run_cells(&Deps::local(), &cells, &options, &mut |_, _| {})?;
    println!(
        "{} measured, {} skipped, {} without a measurement",
        summary.measured, summary.skipped, summary.failed
    );
    // A swap abort is a failure of the box, not of the plan, and the operator has
    // to see it in the exit status: the campaign stopped part-way.
    Ok(if summary.aborted_on_swap.is_some() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

fn reparse(out: &Path, archive_dir: &Path) -> Result<ExitCode, Error> {
    let lines = dataset::read_lines(out)?;
    let (rewritten, report) = maintain::reparse(&lines, &|log| {
        dataset::read_archived_log(&archive_dir.join(log))
    });
    if rewritten == lines {
        println!(
            "{} records reproduce exactly, {} without an archived log; nothing to rewrite",
            report.unchanged, report.skipped
        );
        return Ok(ExitCode::SUCCESS);
    }
    dataset::write_lines(out, &rewritten)?;
    println!(
        "reparsed {} records ({} unchanged, {} without an archived log)",
        report.rewritten, report.unchanged, report.skipped
    );
    Ok(ExitCode::SUCCESS)
}

fn retire(out: &Path) -> Result<ExitCode, Error> {
    let lines = dataset::read_lines(out)?;
    let (rewritten, report) = maintain::retire_stale_builds(&lines, STALE_TOLERANCE);
    for (arch, build) in &report.builds {
        println!("retired {arch} rows measured under {build}");
    }
    if rewritten != lines {
        dataset::write_lines(out, &rewritten)?;
    }
    println!("{} row(s) marked stale-runtime", report.retired);
    Ok(ExitCode::SUCCESS)
}
