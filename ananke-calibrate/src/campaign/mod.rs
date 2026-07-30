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
//! One structural difference from the Python this replaces. There, `measure.py`
//! owned the loop and `campaign.py` polled every thirty seconds to commit
//! alongside it, in a second process; its own comment called that a compromise to
//! avoid coupling measurement to version control. It also meant the commit could
//! land while a record was being appended — `progress.py` carried a comment about
//! expecting torn lines. Here the driver owns the loop and the harness reports each
//! cell as it finishes, so a commit happens only at a cell boundary, where there is
//! no half-written line by construction. The coupling the Python avoided is still
//! avoided: the harness knows nothing about git, and this module is the only place
//! the two meet.

use std::{path::PathBuf, time::Duration};

use ananke_measure::{
    harness::{Completed, Options, Summary, measure_cells},
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

    let mut cadence = Cadence::new(COMMIT_EVERY);
    let mut measured = 0usize;
    let planned = cells.len();
    let summary = measure_cells(cells, &options, &mut |_, completed| {
        if completed == Completed::Measured {
            measured += 1;
        }
        if cadence.due() {
            report(commit_data(
                vcs,
                &campaign.data_paths,
                &progress_message(measured, planned, false),
            ));
        }
    })
    .map_err(|e| e.to_string())?;

    report(commit_data(
        vcs,
        &campaign.data_paths,
        &progress_message(summary.measured, planned, true),
    ));
    Ok(summary)
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

/// The message a data commit carries.
///
/// It names the generator and the harness so a reader of the history knows what
/// produced the rows, and points at the records' own hardware block for where they
/// were taken — the machine is not in the message because it is in the data.
pub fn progress_message(measured: usize, planned: usize, complete: bool) -> String {
    let headline = if complete {
        format!("data(calibration): {measured} measurements, campaign complete")
    } else {
        format!("data(calibration): {measured} measurements")
    };
    let body = if complete {
        format!(
            "{measured} of {planned} planned cells measured. Generated by \
             ananke-calibrate's plan and measured by ananke-measure's harness."
        )
    } else {
        "Generated by ananke-calibrate's plan and measured by ananke-measure's \
         harness on this machine; see the README for what each cell answers and \
         each record's hardware block for where it was taken."
            .to_string()
    };
    format!("{headline}\n\n{body}")
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
    use super::*;

    /// The interim message says how much is measured; the final one says how much
    /// of the plan that was.
    #[test]
    fn the_final_message_names_the_plan() {
        let interim = progress_message(12, 400, false);
        assert!(interim.starts_with("data(calibration): 12 measurements\n"));
        assert!(!interim.contains("400"));

        let final_message = progress_message(398, 400, true);
        assert!(final_message.contains("campaign complete"));
        assert!(final_message.contains("398 of 400 planned cells measured"));
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
