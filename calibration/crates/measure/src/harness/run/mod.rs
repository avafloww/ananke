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

pub struct Options {
    pub out: PathBuf,
    pub log_dir: PathBuf,
    /// Where the gzipped load logs go; they are what makes a record re-parseable
    /// later. Absent means the logs are left in the log directory only.
    pub archive_dir: Option<PathBuf>,
    pub port: u16,
    pub load_timeout: Duration,
    pub headroom_gib: f64,
    pub swap_limit_gib: f64,
    /// Measure a cell the pre-flight gate would refuse. See [`host::fits`] for why
    /// the gate is both right and too strict.
    pub force: bool,
    /// Measure a cell that already has a record. A cell id hashes the factors, so
    /// an unchanged configuration is normally skipped — but the runtime is not one
    /// of the factors, and when it changes under you the old record describes a
    /// different program.
    pub remeasure: bool,
}

#[derive(Debug, Default)]
pub struct Summary {
    pub measured: usize,
    pub skipped: usize,
    pub failed: usize,
    /// How far past its baseline swap had grown when the watchdog stopped the run.
    pub aborted_on_swap: Option<f64>,
}

/// What became of one cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completed {
    /// A record was written and it loaded.
    Measured,
    /// Already present, or refused by the pre-flight gate.
    Skipped,
    /// A record was written saying why it did not load.
    Failed,
}

/// Measure a plan against the real machine, reporting each cell as it finishes.
///
/// The observer is called once per cell, after the row is on disk and before the
/// next server spawns. That boundary is the point of it: a driver that commits the
/// dataset as the campaign runs must not do so while a record is being appended,
/// and this is the only moment at which there is no half-written line. Committing
/// on a timer from another process — the arrangement this replaces — has no such
/// guarantee.
pub fn measure_cells(
    cells: &[Factors],
    options: &Options,
    observer: &mut dyn FnMut(usize, Completed),
) -> Result<Summary, Error> {
    measure_cells_with(&Deps::local(), cells, options, observer)
}

/// As [`measure_cells`], against a given set of outside-world implementations.
///
/// Public because the observer contract above is a contract *with a driver*, and a
/// driver cannot check that it holds — that a cell is reported exactly once, after
/// its row is on disk — without substituting the world. `ananke-calibrate`'s
/// campaign is that driver, and until this existed the only thing standing behind
/// the contract was a reading of the loop.
pub fn measure_cells_with(
    deps: &Deps,
    cells: &[Factors],
    options: &Options,
    observer: &mut dyn FnMut(usize, Completed),
) -> Result<Summary, Error> {
    let done = if options.remeasure {
        Default::default()
    } else {
        dataset::already_measured(deps.files.as_ref(), &options.out)?
    };
    let mut summary = Summary::default();
    for (index, factors) in cells.iter().enumerate() {
        let prefix = format!("[{}/{}]", index + 1, cells.len());
        let id = cell::cell_id(factors);
        if done.contains(&id) {
            println!("{prefix} skip {} ({id})", factors.label);
            summary.skipped += 1;
            observer(index, Completed::Skipped);
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
            dataset::append(deps.files.as_ref(), &options.out, &skipped)?;
            summary.skipped += 1;
            observer(index, Completed::Skipped);
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
                dataset::append(deps.files.as_ref(), &options.out, &measurement)?;
                if status == Status::Ok {
                    summary.measured += 1;
                    observer(index, Completed::Measured);
                } else {
                    summary.failed += 1;
                    observer(index, Completed::Failed);
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
            outcome.log = archive(deps, &log_path, options);
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
        log: archive(deps, &log_path, options),
        trace: sampler.trace().to_vec(),
        checkpoints,
    })
}

fn archive(deps: &Deps, log_path: &Path, options: &Options) -> String {
    options
        .archive_dir
        .as_deref()
        .map(|archive_dir| dataset::archive_log(deps.files.as_ref(), log_path, archive_dir))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        harness::sys::{FakeFiles, FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes},
        record::Runtime,
    };

    const OUT: &str = "data/measurements.ndjson";

    /// Every cell is reported exactly once, in order, and always after its row is
    /// on disk.
    ///
    /// This is the contract `ananke-calibrate`'s campaign commits against: it
    /// commits from inside the observer, so a cell reported early would commit a
    /// half-written line and a cell reported twice would put an empty commit in the
    /// history. Reported *late* — after the next spawn — would be worse still,
    /// because the commit would then race a running server's log.
    ///
    /// The rows are counted at each notification rather than at the end, which is
    /// what makes this an ordering assertion instead of a tally.
    #[test]
    fn each_cell_is_reported_once_and_after_its_row_lands() {
        let fakes = fakes(FakeProcFs::new().with_available_gib(1.0));
        let deps = fakes.deps();
        let cells = [cell("a"), cell("b"), cell("c")];

        let mut seen = Vec::new();
        let summary = measure_cells_with(&deps, &cells, &options(), &mut |index, completed| {
            seen.push((index, completed, fakes.files.lines(OUT).len()));
        })
        .expect("the fakes do not fail");

        // Refused by the gate: 1 GiB available against a 30 GiB headroom. Each
        // refusal still writes a row saying so.
        assert_eq!(
            seen,
            [
                (0, Completed::Skipped, 1),
                (1, Completed::Skipped, 2),
                (2, Completed::Skipped, 3),
            ]
        );
        assert_eq!(summary.skipped, 3);
        assert_eq!(summary.measured, 0);
    }

    /// A cell that already has a row is reported without being measured again.
    ///
    /// The skip happens before the gate, so this path touches nothing at all — it
    /// is the one a resumed campaign spends most of its time in.
    #[test]
    fn an_already_measured_cell_is_reported_and_not_rerun() {
        let measured = cell("a");
        let existing = format!(
            r#"{{"cell":"{}","status":"ok","factors":{{}}}}"#,
            cell::cell_id(&measured)
        );
        let fakes = fakes(FakeProcFs::new().with_available_gib(1.0))
            .with_files(FakeFiles::new().with_file(OUT, format!("{existing}\n")));
        let deps = fakes.deps();

        let mut seen = Vec::new();
        let summary = measure_cells_with(
            &deps,
            &[measured, cell("b")],
            &options(),
            &mut |index, completed| seen.push((index, completed)),
        )
        .expect("the fakes do not fail");

        assert_eq!(
            seen,
            [(0, Completed::Skipped), (1, Completed::Skipped)],
            "both are skipped, for different reasons"
        );
        assert_eq!(
            fakes.files.lines(OUT).len(),
            2,
            "the already-measured cell appends nothing; the gate-refused one appends its row"
        );
        assert_eq!(summary.skipped, 2);
    }

    /// A swap abort reports nothing for the cell it stopped on, or for any after.
    ///
    /// Nothing is recorded for an aborted cell — the process was stopped part-way,
    /// so what it held is not a measurement — and the observer follows the record.
    /// A driver counting notifications must therefore never see a total that
    /// outruns the dataset, which is what would let it commit a claim to more rows
    /// than exist.
    #[test]
    fn an_aborted_run_reports_nothing_for_the_cell_it_stopped_on() {
        // Available memory clears the gate; swap then grows past the limit on every
        // reading, which is what the watchdog stops on.
        // The server has to still be loading for the watchdog to get a reading: it
        // samples while waiting for /health, and a server that answers on the first
        // poll is past the point where swap could stop it.
        let fakes = Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new()
                .with_available_gib(1024.0)
                .with_swap_growth_gib(8.0),
            FakeGpu::new(),
            FakeHttp::new().loading_for(4),
        );
        let deps = fakes.deps();

        let mut seen = Vec::new();
        let summary = measure_cells_with(
            &deps,
            &[cell("a"), cell("b")],
            &options(),
            &mut |index, completed| seen.push((index, completed)),
        )
        .expect("an abort is a summary, not an error");

        assert!(
            seen.is_empty(),
            "nothing was measured, so nothing is reported"
        );
        assert!(summary.aborted_on_swap.is_some(), "the watchdog stopped it");
        assert_eq!(
            fakes.files.lines(OUT).len(),
            0,
            "and no row was written for the cell it stopped on"
        );
    }

    /// `--force` skips the gate, and a cell that loads is reported as measured.
    #[test]
    fn a_cell_that_loads_is_reported_as_measured() {
        let fakes = fakes(FakeProcFs::new().with_available_gib(1.0));
        let deps = fakes.deps();
        let options = Options {
            force: true,
            ..options()
        };

        let mut seen = Vec::new();
        let summary = measure_cells_with(&deps, &[cell("a")], &options, &mut |index, completed| {
            seen.push((index, completed, fakes.files.lines(OUT).len()));
        })
        .expect("the fakes do not fail");

        assert_eq!(seen, [(0, Completed::Measured, 1)]);
        assert_eq!(summary.measured, 1);
    }

    /// The fake world, with a given `/proc`. The spawner, the driver, and the HTTP
    /// surface are at their defaults: a server that starts, answers `/health`, and
    /// exits when told.
    ///
    /// `host::provenance` still shells out for the binary's version string — the
    /// module doc carves that out — so a measured cell runs `<binary> --version`
    /// and takes `?` when there is no such binary. It decides nothing here.
    fn fakes(procfs: FakeProcFs) -> Fakes {
        Fakes::new(FakeSpawner::new(), procfs, FakeGpu::new(), FakeHttp::new())
    }

    fn options() -> Options {
        Options {
            out: PathBuf::from(OUT),
            log_dir: PathBuf::from("logs"),
            // No archiving: the log path never exists under the fakes, so archiving
            // would only be a test of `archive_log`'s tolerance of that.
            archive_dir: None,
            port: 18099,
            load_timeout: Duration::from_secs(1800),
            headroom_gib: 30.0,
            swap_limit_gib: 4.0,
            force: false,
            remeasure: false,
        }
    }

    fn cell(label: &str) -> Factors {
        Factors {
            label: label.to_owned(),
            model: format!("/models/{label}.gguf"),
            runtime: Runtime::Mainline,
            ctx: 32768,
            ..Default::default()
        }
    }
}
