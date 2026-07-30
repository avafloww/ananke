//! Run the calibration campaign end to end, committing as it goes.
//!
//! Designed to be left unattended. The schedule is one flat list rather than a
//! series of passes per question: the questions are separate, but running the
//! schedule once per question would reload every model once per question, and a
//! reload of the 205 GiB production quant costs more than all of its measurements
//! put together. [`crate::plan::all_cells`] is what orders it.
//!
//! Work is committed as it completes rather than at the end, so an interrupted
//! campaign leaves committed measurements behind and a resumed one skips them —
//! the harness keys cells by their factors, so resuming is just running again.
//!
//! The driver owns the loop and the harness reports each cell as it finishes, so
//! a commit happens only at a cell boundary, where there is no half-written line
//! by construction. A committer polling the dataset from beside the harness would
//! be simpler, and is the arrangement this replaces, but it can land a commit
//! while a record is being appended. Measurement still knows nothing about version
//! control: the harness has no idea git exists, and this module is the only place
//! the two meet.

use std::{path::PathBuf, time::Duration};

use ananke_measure::{
    harness::{Completed, Options, Summary, measure_cells_with, sys::Deps},
    record::Factors,
};

use crate::{
    campaign::git::{Outcome, Vcs, commit_data},
    plan::{Library, all_cells},
};

pub mod git;
pub mod progress;

/// Host memory to leave free. The largest models exist to nearly fill the machine
/// and the harness is the only thing running, so a wide margin buys nothing and
/// silently skips the cells that matter most.
pub const HEADROOM_GIB: f64 = 12.0;

/// How often to commit. Frequent enough that an interruption loses minutes rather
/// than hours; rare enough that the history stays readable.
pub const COMMIT_EVERY: Duration = Duration::from_secs(900);

/// What the campaign was asked to do.
pub struct Campaign {
    /// The NDJSON the campaign accumulates.
    pub out: PathBuf,
    /// Where the schedule is written, so a reader can see what was intended.
    pub plan: PathBuf,
    /// The paths a data commit is scoped to.
    pub data_paths: Vec<PathBuf>,
    /// Substring filter on the cell label.
    pub only: Option<String>,
    pub load_timeout: Duration,
    pub headroom_gib: f64,
    pub log_dir: PathBuf,
    pub archive_dir: PathBuf,
    pub port: u16,
    pub swap_limit_gib: f64,
}

/// The cells this campaign will attempt, in the order it will attempt them.
pub fn schedule(campaign: &Campaign, lib: &Library) -> Vec<Factors> {
    let mut cells = all_cells(lib);
    if let Some(only) = campaign.only.as_deref() {
        cells.retain(|cell| cell.label.contains(only));
    }
    cells
}

/// Where this run's schedule is written.
///
/// The tracked `plan.json` is the *whole* campaign's schedule, and it is one of the
/// paths a data commit is scoped to. A filtered run therefore writes its schedule to
/// the log directory instead: `--only laguna` writing the tracked file would replace
/// four hundred cells with thirty and then commit the truncation under a message
/// about measurements, leaving no sign in the history that the plan had been
/// narrowed rather than regenerated.
pub fn schedule_path(campaign: &Campaign) -> PathBuf {
    match campaign.only {
        None => campaign.plan.clone(),
        Some(_) => campaign.log_dir.join("plan.json"),
    }
}

/// Measure the schedule, committing the dataset as it fills.
pub fn run(campaign: &Campaign, cells: &[Factors], vcs: &dyn Vcs) -> Result<Summary, String> {
    run_with(campaign, cells, vcs, &Deps::local())
}

/// As [`run`], against a given set of outside-world implementations.
///
/// The commit cadence is the thing worth testing here and it only exists in
/// relation to the harness's per-cell notifications, so testing the two apart
/// leaves the seam between them — which is where a torn line would come from —
/// checked by nobody.
pub fn run_with(
    campaign: &Campaign,
    cells: &[Factors],
    vcs: &dyn Vcs,
    deps: &Deps,
) -> Result<Summary, String> {
    let options = Options {
        out: campaign.out.clone(),
        log_dir: campaign.log_dir.clone(),
        archive_dir: Some(campaign.archive_dir.clone()),
        port: campaign.port,
        load_timeout: campaign.load_timeout,
        headroom_gib: campaign.headroom_gib,
        swap_limit_gib: campaign.swap_limit_gib,
        force: false,
        remeasure: false,
    };

    let mut ledger = Ledger::new(vcs, &campaign.data_paths, COMMIT_EVERY, cells.len());
    let result = measure_cells_with(deps, cells, &options, &mut |_, completed| {
        ledger.cell(completed)
    });

    // The final commit happens on the error path too. The rows the run did measure
    // are on disk either way, but leaving them uncommitted after an I/O failure is
    // the one exit at which this module's promise — an interrupted campaign leaves
    // committed measurements behind — would not hold.
    let standing = match &result {
        Ok(summary) if summary.aborted_on_swap.is_some() => Standing::StoppedOnSwap,
        Ok(_) => Standing::Complete,
        Err(_) => Standing::Interrupted,
    };
    ledger.commit(standing);
    result.map_err(|e| e.to_string())
}

/// The campaign's bookkeeping for one cell: count it, and commit if it is time.
///
/// Separated from the closure so the cadence and the commit wiring can be driven by
/// a test. The alternative is to test them through [`run`], which needs a machine
/// with GPUs on it.
pub struct Ledger<'a> {
    vcs: &'a dyn Vcs,
    paths: &'a [PathBuf],
    cadence: Cadence,
    planned: usize,
    measured: usize,
}

impl<'a> Ledger<'a> {
    pub fn new(vcs: &'a dyn Vcs, paths: &'a [PathBuf], every: Duration, planned: usize) -> Self {
        Self {
            vcs,
            paths,
            cadence: Cadence::new(every),
            planned,
            measured: 0,
        }
    }

    /// Record what became of a cell, committing if the cadence is due.
    pub fn cell(&mut self, completed: Completed) {
        if completed == Completed::Measured {
            self.measured += 1;
        }
        if self.cadence.due() {
            self.commit(Standing::InFlight);
        }
    }

    /// Commit what has been measured so far.
    pub fn commit(&self, standing: Standing) {
        report(commit_data(
            self.vcs,
            self.paths,
            &progress_message(self.measured, self.planned, standing),
        ));
    }

    pub fn measured(&self) -> usize {
        self.measured
    }
}

/// Whether enough time has passed to commit again.
///
/// Asked at cell boundaries rather than on a timer, so the interval is a floor on
/// the gap between commits and not a promise about when one happens: a single cell
/// can take longer than the whole interval, and the commit then waits for it.
pub struct Cadence {
    every: Duration,
    last: std::time::Instant,
}

impl Cadence {
    pub fn new(every: Duration) -> Self {
        Self {
            every,
            last: std::time::Instant::now(),
        }
    }

    pub fn due(&mut self) -> bool {
        if self.last.elapsed() < self.every {
            return false;
        }
        self.last = std::time::Instant::now();
        true
    }
}

/// How a run stood when a commit was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// Measuring; more cells to come.
    InFlight,
    /// The schedule was worked through to its end.
    Complete,
    /// The swap watchdog stopped the run part-way. The rest of the schedule was
    /// never attempted.
    StoppedOnSwap,
    /// The harness returned an error, so the run ended where it was.
    Interrupted,
}

/// The message a data commit carries.
///
/// It names the generator and the harness so a reader of the history knows what
/// produced the rows, and points at the records' own hardware block for where they
/// were taken — the machine is not in the message because it is in the data.
///
/// Only a run that reached the end of its schedule says so. A swap abort returns
/// the same `Ok(summary)` as a finished run — the harness has recorded everything it
/// is going to — and calling that "campaign complete" tells a later reader of the
/// history that a schedule was exhausted when most of it was never attempted.
pub fn progress_message(measured: usize, planned: usize, standing: Standing) -> String {
    let (suffix, body) = match standing {
        Standing::InFlight => (
            String::new(),
            "Generated by ananke-calibrate's plan and measured by ananke-measure's \
             harness on this machine; see the README for what each cell answers and \
             each record's hardware block for where it was taken."
                .to_string(),
        ),
        Standing::Complete => (
            ", campaign complete".to_string(),
            format!(
                "{measured} of {planned} planned cells measured. Generated by \
                 ananke-calibrate's plan and measured by ananke-measure's harness."
            ),
        ),
        Standing::StoppedOnSwap => (
            ", stopped on swap".to_string(),
            format!(
                "{measured} of {planned} planned cells measured before the swap \
                 watchdog stopped the run; the rest of the schedule was not \
                 attempted. Generated by ananke-calibrate's plan and measured by \
                 ananke-measure's harness."
            ),
        ),
        Standing::Interrupted => (
            ", interrupted".to_string(),
            format!(
                "{measured} of {planned} planned cells measured before the harness \
                 failed; the rest of the schedule was not attempted. Generated by \
                 ananke-calibrate's plan and measured by ananke-measure's harness."
            ),
        ),
    };
    format!("data(calibration): {measured} measurements{suffix}\n\n{body}")
}

fn report(outcome: Outcome) {
    match outcome {
        Outcome::Committed => {}
        Outcome::NothingToDo => {}
        Outcome::Failed(error) => {
            println!("    COMMIT FAILED (the data is still on disk): {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use ananke_measure::harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes};

    use super::*;
    use crate::campaign::git::FakeGit;

    /// The interim message says how much is measured; the final one says how much
    /// of the plan that was.
    #[test]
    fn the_final_message_names_the_plan() {
        let interim = progress_message(12, 400, Standing::InFlight);
        assert!(interim.starts_with("data(calibration): 12 measurements\n"));
        assert!(!interim.contains("400"));

        let final_message = progress_message(398, 400, Standing::Complete);
        assert!(final_message.contains("campaign complete"));
        assert!(final_message.contains("398 of 400 planned cells measured"));
    }

    /// Only a run that reached the end of its schedule claims to have.
    ///
    /// A swap abort returns the same `Ok(summary)` as a finished run, so the
    /// distinction has to be drawn by the caller. Getting it wrong writes "campaign
    /// complete" into the history for a run that measured 30 of 400 cells and then
    /// stopped, and the history is what a later reader has.
    #[test]
    fn a_stopped_run_does_not_claim_completion() {
        for standing in [Standing::StoppedOnSwap, Standing::Interrupted] {
            let message = progress_message(30, 400, standing);
            assert!(
                !message.contains("campaign complete"),
                "{standing:?}: {message}"
            );
            assert!(message.contains("was not attempted"), "{standing:?}");
            assert!(message.contains("30 of 400"), "{standing:?}");
        }
    }

    /// The cadence commits during a run, and the standing distinguishes the commits.
    #[test]
    fn the_ledger_commits_on_cadence_and_at_the_end() {
        let git = FakeGit::dirty();
        let paths = vec![PathBuf::from("data/measurements.ndjson")];
        let mut ledger = Ledger::new(&git, &paths, Duration::ZERO, 3);
        for completed in [Completed::Measured, Completed::Skipped, Completed::Failed] {
            git.touch();
            ledger.cell(completed);
        }
        git.touch();
        ledger.commit(Standing::Complete);

        assert_eq!(ledger.measured(), 1, "only a measurement counts as one");
        let commits = git.commits();
        assert_eq!(commits.len(), 4, "three cadence commits and the final one");
        assert!(commits[0].starts_with("data(calibration): 1 measurements\n"));
        assert!(commits[3].contains("campaign complete"));
        assert_eq!(git.committed_paths(), vec![paths; 4]);
    }

    /// A long interval means no commit until the campaign ends.
    ///
    /// The cadence is asked at cell boundaries, so a schedule shorter than the
    /// interval commits exactly once — at the end — rather than not at all.
    #[test]
    fn a_short_run_still_commits_once() {
        let git = FakeGit::dirty();
        let paths = vec![PathBuf::from("data/measurements.ndjson")];
        let mut ledger = Ledger::new(&git, &paths, Duration::from_secs(900), 2);
        ledger.cell(Completed::Measured);
        ledger.cell(Completed::Measured);
        assert!(git.commits().is_empty(), "the interval has not elapsed");
        ledger.commit(Standing::Complete);
        assert_eq!(git.commits().len(), 1);
    }

    /// The driver commits what the harness measured, and says how it ended.
    ///
    /// This is the seam the whole module exists to hold: the harness notifies at a
    /// cell boundary, and the driver commits from inside that notification. Neither
    /// half means anything alone — a cadence with nothing to count, a notification
    /// nobody acts on — so the test runs them together against the harness's
    /// in-memory world. Splitting the driver across two processes would put this
    /// seam beyond the reach of any test.
    #[test]
    fn a_run_commits_at_cell_boundaries_and_reports_how_it_ended() {
        let git = FakeGit::dirty();
        let fakes = Fakes::new(
            FakeSpawner::new(),
            // Clears the pre-flight gate, so every cell is actually measured.
            FakeProcFs::new().with_available_gib(1024.0),
            FakeGpu::new(),
            FakeHttp::new(),
        );
        let campaign = Campaign {
            headroom_gib: 1.0,
            ..fixture()
        };
        let cells = [measurable("a"), measurable("b")];

        let summary = run_with(&campaign, &cells, &git, &fakes.deps()).expect("the fakes hold");

        assert_eq!(summary.measured, 2);
        let rows = fakes.files.lines(&campaign.out);
        assert_eq!(rows.len(), 2, "one row per cell");

        // One commit, at the end: the interval has not elapsed during a run this
        // short, and the final commit is unconditional.
        let commits = git.commits();
        assert_eq!(commits.len(), 1, "{commits:?}");
        assert!(commits[0].contains("2 measurements, campaign complete"));
        assert!(commits[0].contains("2 of 2 planned cells measured"));
        assert_eq!(git.committed_paths(), vec![campaign.data_paths.clone()]);
    }

    /// A run the swap watchdog stops commits what it got, and does not claim the
    /// schedule was finished.
    ///
    /// The regression this pins: `measure_cells_with` returns `Ok` on an abort, so
    /// the driver saw the same value a completed run returns.
    #[test]
    fn a_swap_abort_commits_what_it_measured_without_claiming_completion() {
        let git = FakeGit::dirty();
        let fakes = Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new()
                .with_available_gib(1024.0)
                .with_swap_growth_gib(8.0),
            FakeGpu::new(),
            // Still loading, so the watchdog gets a reading before /health answers.
            FakeHttp::new().loading_for(4),
        );
        let campaign = Campaign {
            headroom_gib: 1.0,
            ..fixture()
        };

        let summary = run_with(
            &campaign,
            &[measurable("a"), measurable("b")],
            &git,
            &fakes.deps(),
        )
        .expect("an abort is a summary, not an error");

        assert!(summary.aborted_on_swap.is_some());
        let commits = git.commits();
        assert_eq!(commits.len(), 1, "the final commit still happens");
        assert!(!commits[0].contains("campaign complete"), "{}", commits[0]);
        assert!(commits[0].contains("stopped on swap"), "{}", commits[0]);
        assert!(commits[0].contains("0 of 2"), "{}", commits[0]);
    }

    /// A cell the harness can measure under the fakes.
    fn measurable(label: &str) -> Factors {
        Factors {
            label: label.to_owned(),
            model: format!("/models/{label}.gguf"),
            ctx: 32768,
            ..Default::default()
        }
    }

    /// A filtered run does not write the tracked schedule.
    ///
    /// `plan.json` is committed with the data, so a `--only` run that wrote it would
    /// commit a truncated plan under a message about measurements.
    #[test]
    fn a_filtered_run_writes_its_schedule_elsewhere() {
        let full = fixture();
        assert_eq!(schedule_path(&full), full.plan);

        let filtered = Campaign {
            only: Some("laguna".to_string()),
            ..fixture()
        };
        assert_ne!(schedule_path(&filtered), filtered.plan);
        assert!(schedule_path(&filtered).starts_with(&filtered.log_dir));
    }

    /// A commit is not due the instant the campaign starts.
    ///
    /// The first cell of a long campaign can finish in seconds, and committing then
    /// would put an empty-to-one-row commit at the head of every run.
    #[test]
    fn the_first_cell_does_not_trigger_a_commit() {
        let mut cadence = Cadence::new(Duration::from_secs(900));
        assert!(!cadence.due());
    }

    /// A zero interval is always due, which is what a test driving the loop wants.
    #[test]
    fn a_zero_interval_is_always_due() {
        let mut cadence = Cadence::new(Duration::ZERO);
        assert!(cadence.due());
        assert!(cadence.due());
    }

    /// The filter selects by label substring, and selects nothing on no match
    /// rather than falling back to everything.
    #[test]
    fn the_filter_narrows_the_schedule() {
        let lib = Library::rooted("/fake/llm");
        let all = schedule(
            &Campaign {
                only: None,
                ..fixture()
            },
            &lib,
        );
        let laguna = schedule(
            &Campaign {
                only: Some("laguna".to_string()),
                ..fixture()
            },
            &lib,
        );
        let nothing = schedule(
            &Campaign {
                only: Some("no-such-model".to_string()),
                ..fixture()
            },
            &lib,
        );
        assert!(!all.is_empty());
        assert!(laguna.len() < all.len());
        assert!(laguna.iter().all(|c| c.label.contains("laguna")));
        assert!(nothing.is_empty());
    }

    fn fixture() -> Campaign {
        Campaign {
            out: PathBuf::from("data/measurements.ndjson"),
            plan: PathBuf::from("data/plan.json"),
            data_paths: vec![PathBuf::from("data")],
            only: None,
            load_timeout: Duration::from_secs(2400),
            headroom_gib: HEADROOM_GIB,
            log_dir: PathBuf::from("/tmp/ananke-calibration"),
            archive_dir: PathBuf::from("data/logs"),
            port: 18099,
            swap_limit_gib: 4.0,
        }
    }
}
