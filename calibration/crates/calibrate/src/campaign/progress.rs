//! How far the campaign has got, and whether it is still moving.
//!
//! Safe to run against a live campaign: it reads the dataset and stats the model
//! files to order the schedule, and touches neither the GPUs nor the running server
//! nor the plan on disk.
//!
//! Reporting per-question progress by globbing for `data/<phase>.ndjson` is the
//! arrangement this replaces. The campaign left that layout behind when it
//! consolidated to one `measurements.ndjson`, and the phase names it defaulted to
//! had stopped being questions, so every row printed `0/?` against a dataset of
//! 643 records and had done for some time.
//!
//! Progress here is keyed on **cell identity** instead: each question is asked what
//! cells it wants, each cell is named the way the harness names it, and the dataset
//! is looked up by that name. Nothing depends on a filename, and a question that is
//! renamed or that resweeps its cells reports correctly without anybody remembering
//! to rename a file.
//!
//! A cell can belong to more than one question — that is the whole reason the
//! campaign runs one merged schedule — so the per-question counts deliberately sum
//! to more than the dataset holds. The total is over distinct cells.

use std::collections::{BTreeMap, BTreeSet};

use ananke_measure::{harness::cell_id, record::Status};

use crate::{
    plan::{Library, QUESTIONS, all_cells},
    record::Record,
};

/// One question's standing.
#[derive(Debug, Clone)]
pub struct QuestionProgress {
    pub name: &'static str,
    /// Cells the question wants.
    pub planned: usize,
    /// Of those, cells that produced a measurement.
    pub measured: usize,
    /// Of those, cells recorded with some other status, commonest first.
    pub issues: Vec<(String, usize)>,
}

/// The campaign as a whole.
#[derive(Debug, Clone)]
pub struct Report {
    pub questions: Vec<QuestionProgress>,
    /// Distinct planned cells, and how many of them are measured.
    pub planned: usize,
    pub measured: usize,
    /// The most recent record's timestamp, as it appears in the data.
    pub last_record: Option<String>,
    /// Successfully measured cells no question currently plans. Not a fault in
    /// itself — a question can be retired, or a cell measured by hand — but it is
    /// also what a sweep that has quietly stopped generating its cells looks like.
    ///
    /// Counts only cells that reached `ok`, matching `measured`: a cell that failed
    /// to load and that nobody plans is not a measurement anybody is missing.
    pub unplanned: usize,
}

/// Build the report.
///
/// Rows from before the schema carried a cell id are ignored rather than guessed
/// at: their identity cannot be recomputed, and attributing them to a question by
/// label would credit a question with a cell it may not have asked for.
pub fn report(records: &[Record], lib: &Library) -> Report {
    let mut status_by_cell: BTreeMap<&str, Status> = BTreeMap::new();
    for record in records {
        let Some(cell) = record.cell_id() else {
            continue;
        };
        // The best status any row for this cell reached. A cell retried after a
        // load failure has two rows, and it is measured.
        let entry = status_by_cell.entry(cell).or_insert(record.status);
        if record.status == Status::Ok {
            *entry = Status::Ok;
        }
    }

    let questions = QUESTIONS
        .iter()
        .map(|(name, sweep)| {
            let ids: BTreeSet<String> = sweep(lib).iter().map(cell_id).collect();
            let mut measured = 0;
            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
            for id in &ids {
                match status_by_cell.get(id.as_str()) {
                    Some(Status::Ok) => measured += 1,
                    Some(other) => *counts.entry(status_label(*other)).or_default() += 1,
                    None => {}
                }
            }
            let mut issues: Vec<(String, usize)> = counts
                .into_iter()
                .map(|(status, n)| (status.to_string(), n))
                .collect();
            issues.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            QuestionProgress {
                name,
                planned: ids.len(),
                measured,
                issues,
            }
        })
        .collect();

    let planned_ids: BTreeSet<String> = all_cells(lib).iter().map(cell_id).collect();
    let measured = planned_ids
        .iter()
        .filter(|id| status_by_cell.get(id.as_str()) == Some(&Status::Ok))
        .count();
    let unplanned = status_by_cell
        .iter()
        .filter(|(id, status)| **status == Status::Ok && !planned_ids.contains(**id))
        .count();

    Report {
        questions,
        planned: planned_ids.len(),
        measured,
        last_record: records
            .iter()
            .map(|r| r.provenance.measured_at_utc.as_str())
            .filter(|when| !when.is_empty())
            .max()
            .map(str::to_string),
        unplanned,
    }
}

/// The wire form of a non-`Ok` status, for grouping and display.
///
/// Written out rather than derived from [`Status`]'s `Serialize` impl: the report
/// only ever needs the label, and matching exhaustively means a status this module
/// forgets to name is a compile error rather than a silently blank issue row.
fn status_label(status: Status) -> &'static str {
    match status {
        Status::Ok => "ok",
        Status::PortBusy => "port-busy",
        Status::FailedToLoad => "failed-to-load",
        Status::Timeout => "timeout",
        Status::SkippedInsufficientMemory => "skipped-insufficient-memory",
        Status::HarnessError => "harness-error",
        Status::StaleRuntime => "stale-runtime",
    }
}

/// Minutes between two ISO-8601 stamps, or `None` if either cannot be read or the
/// second precedes the first.
///
/// Asked for the *total* in minutes rather than a span's minutes component: for
/// timestamps `jiff` balances a span no higher than seconds, so a three-day gap
/// arrives as ~259200 seconds with a minutes component of zero. Reading that
/// component reports every idle campaign as "0 min ago — running", which is the one
/// answer this function exists to rule out.
///
/// A record stamped in the future gets no age for the same reason. Clamping the
/// negative gap to zero would answer "0 min ago — running" forever after a clock
/// stepped backwards, which is the failure this function is here to prevent, only
/// arrived at from the other side.
pub fn minutes_between(earlier: &str, later: &str) -> Option<i64> {
    let earlier: jiff::Timestamp = earlier.parse().ok()?;
    let later: jiff::Timestamp = later.parse().ok()?;
    let minutes = later.since(earlier).ok()?.total(jiff::Unit::Minute).ok()?;
    (minutes >= 0.0).then_some(minutes as i64)
}

/// Minutes since a record's timestamp.
pub fn idle_minutes(stamp: &str) -> Option<i64> {
    minutes_between(stamp, &jiff::Timestamp::now().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Factors, Hardware, Parsed, Provenance, Rss, read_ndjson};

    const MEASUREMENTS: &str = "../../data/measurements.ndjson";

    fn dataset() -> Vec<Record> {
        let text = std::fs::read_to_string(MEASUREMENTS).expect("the dataset is readable");
        read_ndjson(&text).expect("the dataset parses")
    }

    /// The report finds the real campaign's progress, and it is not zero.
    ///
    /// This is the assertion the filename-globbing arrangement would have failed:
    /// it reported `0/?` for every question against this same dataset, and nothing
    /// noticed because nothing checked.
    #[test]
    fn the_real_dataset_reports_real_progress() {
        let report = report(&dataset(), &Library::from_env());
        assert!(report.planned > 0, "the plan wants cells");
        assert!(
            report.measured > 100,
            "hundreds of planned cells are measured, got {}",
            report.measured
        );
        assert!(
            report.questions.iter().filter(|q| q.measured > 0).count() > 5,
            "most questions have progress"
        );
        assert!(report.last_record.is_some(), "the dataset is timestamped");
    }

    /// The dataset really does contain retried cells, so the merge below is load-
    /// bearing against it rather than a defence against a case that never occurs.
    #[test]
    fn the_dataset_contains_retries() {
        let records = dataset();
        let distinct: BTreeSet<&str> = records.iter().filter_map(|r| r.cell_id()).collect();
        let with_id = records.iter().filter(|r| r.cell_id().is_some()).count();
        assert!(
            with_id > distinct.len(),
            "{with_id} identified rows over {} distinct cells — no cell was retried, so \
             `a_retried_cell_is_measured_once` is testing a case the data does not have",
            distinct.len()
        );
    }

    /// A cell that failed and was retried is measured, and counted once.
    ///
    /// Both halves can fail: taking the *first* row's status leaves a recovered cell
    /// reported as outstanding forever, and counting rows rather than cells lets a
    /// question report more measured than it planned.
    #[test]
    fn a_retried_cell_is_measured_once() {
        let records = synthetic(&[
            ("cell-a", Status::FailedToLoad),
            ("cell-a", Status::Ok),
            ("cell-a", Status::Ok),
        ]);
        let report = report(&records, &Library::from_env());
        assert_eq!(report.unplanned, 1, "one cell, whatever its row count");
        assert_eq!(report.measured, 0, "and it is not one the plan asked for");
    }

    /// A cell that never loaded is not counted as an unplanned measurement.
    ///
    /// `unplanned` is reported to the operator as "measured cell(s) no question
    /// currently plans"; counting failures there inflates it — 91 against a true 78
    /// on the real dataset — and the inflated figure is what the identity check
    /// below keys on.
    #[test]
    fn an_unplanned_failure_is_not_an_unplanned_measurement() {
        let report = report(
            &synthetic(&[("cell-a", Status::FailedToLoad), ("cell-b", Status::Ok)]),
            &Library::from_env(),
        );
        assert_eq!(report.unplanned, 1);
    }

    /// A row from before the schema carried a cell id is ignored, not guessed at.
    #[test]
    fn an_unidentified_row_is_not_counted() {
        let report = report(
            &synthetic(&[("", Status::Ok), ("cell-a", Status::Ok)]),
            &Library::from_env(),
        );
        assert_eq!(report.unplanned, 1);
    }

    /// Records with the given cell ids and statuses. An empty id means a row from
    /// before the schema carried one.
    ///
    /// Built as values rather than parsed from JSON: what is under test is how the
    /// report merges rows, and a row spelled out in text would be a second, laxer
    /// statement of the schema beside the one `ananke_dataset` already makes.
    fn synthetic(rows: &[(&str, Status)]) -> Vec<Record> {
        rows.iter()
            .map(|(cell, status)| Record {
                schema: ananke_dataset::SCHEMA,
                cell: (*cell).to_owned(),
                status: *status,
                provenance: Provenance::default(),
                hardware: Hardware::default(),
                factors: Factors {
                    model: "m".to_owned(),
                    ctx: 4096,
                    ..Factors::default()
                },
                parsed: Parsed::default(),
                rss: Rss::default(),
                log_tail: String::new(),
                log: String::new(),
                trace: Vec::new(),
                checkpoints: Vec::new(),
                reparsed: None,
            })
            .collect()
    }

    /// A gap of days reads as thousands of minutes, not as zero.
    ///
    /// `jiff` balances a timestamp span no higher than seconds, so asking for the
    /// minutes *component* of a three-day gap answers zero — and a report built on
    /// that calls every long-finished campaign "running".
    #[test]
    fn a_long_gap_is_counted_in_full() {
        let then = "2026-07-27T06:11:17+00:00";
        let now = "2026-07-30T06:11:17+00:00";
        assert_eq!(minutes_between(then, now), Some(3 * 24 * 60));

        // And the short case the component-reading version got right by accident.
        assert_eq!(
            minutes_between("2026-07-30T06:11:17+00:00", "2026-07-30T06:41:17+00:00"),
            Some(30)
        );
    }

    /// A stamp that is not a timestamp is absent, not zero.
    #[test]
    fn an_unreadable_stamp_has_no_age() {
        assert_eq!(
            minutes_between("not a date", "2026-07-30T06:11:17+00:00"),
            None
        );
    }

    /// A record stamped in the future has no age either.
    ///
    /// Clamping it to zero — which the first version did — reports "0 min ago,
    /// running" for as long as the skew lasts, which is the same wrong answer the
    /// span-component bug gave, reached from the other direction.
    #[test]
    fn a_record_from_the_future_has_no_age() {
        assert_eq!(
            minutes_between("2026-07-30T06:11:17+00:00", "2026-07-29T06:11:17+00:00"),
            None
        );
    }

    /// The plan and the dataset agree on how cells are named.
    ///
    /// If they did not, every question would read `0/N`. The check is that the
    /// overlap is substantial, because a stale-but-plausible identity function
    /// produces a small one.
    #[test]
    fn the_plan_and_the_dataset_name_cells_the_same_way() {
        let report = report(&dataset(), &Library::from_env());
        assert!(
            report.measured * 2 > report.unplanned,
            "only {} of the dataset's cells are planned against {} that are not — \
             the plan and the harness disagree about cell identity",
            report.measured,
            report.unplanned
        );
    }
}
