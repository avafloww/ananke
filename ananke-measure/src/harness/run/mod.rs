//! Measure one cell, and then a plan of them.
//!
//! This is the imperative shell. Every decision it makes is delegated to a pure
//! or faked piece — readiness ([`readiness`]), the swap limit ([`watchdog`]), the
//! peaks ([`sampler`]), the record ([`assemble`]) — and what is left here is the
//! order of the phases and the I/O between them.
//!
//! Two orderings in it are load-bearing and both cost real time to learn:
//!
//! - the port is waited on *before* the spawn, because a leftover server wins the
//!   bind and every later cell then measures that same process;
//! - the log is parsed *after* the stop, because llama.cpp prints its
//!   memory-breakdown table while tearing the context down, so parsing a
//!   still-running server's log silently loses every per-device figure.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) use crate::harness::run::assemble::Outcome;
use crate::{
    harness::{
        cell, dataset,
        error::Error,
        host,
        run::{
            assemble::{record, rss_summary, tail},
            child::{spawn_server, stop_child},
            exercise::exercise,
            readiness::{Readiness, wait_for_port, wait_for_ready},
            sampler::SamplerThread,
            watchdog::SwapWatchdog,
        },
        sys::Deps,
    },
    parse_log,
    record::{Factors, Runtime, Status},
};

mod assemble;
mod bench;
mod child;
mod exercise;
mod readiness;
mod sampler;
mod watchdog;

/// Long enough for the previous cell's `TIME_WAIT` to clear.
const PORT_WAIT: Duration = Duration::from_secs(180);
/// Let a served process settle before the final reading: the last request's
/// allocations are not all visible the instant it returns.
const SETTLE: Duration = Duration::from_secs(3);
const TAIL_LINES: usize = 40;

pub(crate) struct Options {
    pub(crate) out: PathBuf,
    pub(crate) log_dir: PathBuf,
    /// Where the gzipped load logs go; they are what makes a record re-parseable
    /// later. Absent means the logs are left in the log directory only.
    pub(crate) archive_dir: Option<PathBuf>,
    pub(crate) port: u16,
    pub(crate) load_timeout: Duration,
    pub(crate) headroom_gib: f64,
    pub(crate) swap_limit_gib: f64,
    /// Measure a cell the pre-flight gate would refuse. See [`host::fits`] for why
    /// the gate is both right and too strict.
    pub(crate) force: bool,
    /// Measure a cell that already has a record. A cell id hashes the factors, so
    /// an unchanged configuration is normally skipped — but the runtime is not one
    /// of the factors, and when it changes under you the old record describes a
    /// different program.
    pub(crate) remeasure: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Summary {
    pub(crate) measured: usize,
    pub(crate) skipped: usize,
    pub(crate) failed: usize,
    /// How far past its baseline swap had grown when the watchdog stopped the run.
    pub(crate) aborted_on_swap: Option<f64>,
}

pub(crate) fn run_cells(
    deps: &Deps,
    cells: &[Factors],
    options: &Options,
) -> Result<Summary, Error> {
    let done = if options.remeasure {
        Default::default()
    } else {
        dataset::already_measured(&options.out)?
    };
    let mut summary = Summary::default();
    for (index, factors) in cells.iter().enumerate() {
        let prefix = format!("[{}/{}]", index + 1, cells.len());
        let id = cell::cell_id(factors);
        if done.contains(&id) {
            println!("{prefix} skip {} ({id})", factors.label);
            summary.skipped += 1;
            continue;
        }
        if !options.force && !host::fits(deps, factors, options.headroom_gib) {
            println!(
                "{prefix} skip {}: needs {:.0} GiB + {:.0} headroom, {:.0} available",
                factors.label,
                host::model_gib(factors),
                options.headroom_gib,
                deps.procfs.mem_available_gib()
            );
            let skipped = record(
                id,
                factors.clone(),
                host::provenance(deps, "true", Some(factors)),
                host::hardware(deps),
                Outcome::failed(Status::SkippedInsufficientMemory),
            );
            dataset::append(&options.out, &skipped)?;
            summary.skipped += 1;
            continue;
        }

        print!("{prefix} {} ... ", factors.label);
        flush();
        match measure(deps, factors, &id, options) {
            Run::Aborted(grown) => {
                // Not recorded: the process was stopped part-way, so whatever it
                // held is not a measurement of anything.
                println!(
                    "aborted — swap grew {grown:.1} GiB past the baseline before the box thrashed"
                );
                summary.aborted_on_swap = Some(grown);
                return Ok(summary);
            }
            Run::Measured(measurement) => {
                let status = measurement.status;
                println!("{}", describe(&measurement));
                dataset::append(&options.out, &measurement)?;
                if status == Status::Ok {
                    summary.measured += 1;
                } else {
                    summary.failed += 1;
                }
            }
        }
    }
    Ok(summary)
}

enum Run {
    /// Boxed: a record carries a whole trace, and the other variant is a float.
    Measured(Box<crate::record::Record>),
    /// The swap watchdog stopped the run; nothing is recorded.
    Aborted(f64),
}

fn measure(deps: &Deps, factors: &Factors, id: &str, options: &Options) -> Run {
    let binary = binary_for(factors.runtime);
    let provenance = host::provenance(deps, &binary, Some(factors));
    let hardware = host::hardware(deps);
    let finish = |outcome: Outcome| {
        Run::Measured(Box::new(record(
            id.to_owned(),
            factors.clone(),
            provenance.clone(),
            hardware.clone(),
            outcome,
        )))
    };

    let log_path = options.log_dir.join(format!("{id}-{}.log", factors.label));
    if !wait_for_port(deps, options.port, PORT_WAIT) {
        return finish(Outcome::failed(Status::PortBusy));
    }
    let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), options.swap_limit_gib);
    let mut child = match spawn_server(deps, factors, &binary, &log_path, options.port) {
        Ok(child) => child,
        Err(error) => {
            let mut outcome = Outcome::failed(Status::HarnessError);
            outcome.log_tail = format!("failed to spawn {binary}: {error}");
            return finish(outcome);
        }
    };

    // Sampling starts at the spawn, not at readiness: the load itself is where the
    // transients are — the pinned staging ring on a `--no-mmap` load is gone by the
    // time the server answers /health.
    let spawned_at = deps.clock.elapsed();
    let sampler = SamplerThread::start(deps, child.pid());
    let readiness = wait_for_ready(
        deps,
        child.as_mut(),
        options.port,
        spawned_at,
        options.load_timeout,
        &mut watchdog,
    );

    let load_seconds = match readiness {
        Readiness::Loaded { load_seconds } => load_seconds,
        Readiness::Swapping(grown) => {
            stop_child(deps, child.as_mut());
            drop(sampler.stop());
            return Run::Aborted(grown);
        }
        Readiness::Exited(_) | Readiness::TimedOut => {
            stop_child(deps, child.as_mut());
            drop(sampler.stop());
            let status = if matches!(readiness, Readiness::TimedOut) {
                Status::Timeout
            } else {
                Status::FailedToLoad
            };
            let mut outcome = Outcome::failed(status);
            outcome.log_tail = tail(&child.log(), TAIL_LINES);
            outcome.log = archive(&log_path, options);
            return finish(outcome);
        }
    };

    let checkpoints = exercise(deps, factors, options.port, child.pid(), &mut watchdog);
    deps.clock.sleep(SETTLE);
    let final_reading = deps.procfs.status(child.pid()).unwrap_or_default();
    let sampler = sampler.stop();
    stop_child(deps, child.as_mut());
    if let Some(grown) = watchdog.tripped() {
        return Run::Aborted(grown);
    }

    // After the stop, deliberately: the memory-breakdown table is printed on the
    // way out.
    let parsed = parse_log(&child.log());
    finish(Outcome {
        status: Status::Ok,
        parsed,
        rss: rss_summary(&sampler, final_reading, load_seconds),
        log_tail: String::new(),
        log: archive(&log_path, options),
        trace: sampler.trace().to_vec(),
        checkpoints,
    })
}

fn archive(log_path: &Path, options: &Options) -> String {
    options
        .archive_dir
        .as_deref()
        .map(|archive_dir| dataset::archive_log(log_path, archive_dir))
        .unwrap_or_default()
}

/// Which binary a cell's fork means. Overridable because the two are built
/// separately and a contributor's paths are not ours.
fn binary_for(runtime: Runtime) -> String {
    let (variable, default) = match runtime {
        Runtime::Mainline => ("MAINLINE_BIN", "llama-server"),
        Runtime::Ik => ("IK_BIN", "ik-llama-server"),
    };
    std::env::var(variable).unwrap_or_else(|_| default.to_owned())
}

/// The one-line progress report: the decomposition, in the terms the constants are
/// fitted in.
fn describe(measurement: &crate::record::Record) -> String {
    if measurement.status != Status::Ok {
        // The status as it will be written, so the console and the row agree.
        return serde_json::to_value(measurement.status)
            .ok()
            .and_then(|status| status.as_str().map(str::to_owned))
            .unwrap_or_else(|| "?".to_owned());
    }
    let mib = |key: &str| match measurement.rss.get(key) {
        Some(crate::record::Metric::Whole(kb)) => kb / 1024,
        Some(crate::record::Metric::Fractional(kb)) => (kb / 1024.0) as i64,
        None => 0,
    };
    format!(
        "arena={:.2} anon={} shmem={} file={} MiB",
        measurement.parsed.buffers[crate::parse::BufferKind::Arena].last,
        mib("rss_anon_kb"),
        mib("rss_shmem_kb"),
        mib("rss_file_kb")
    )
}

fn flush() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
