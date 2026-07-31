//! Both maintenance passes, held against the checked-in campaign.
//!
//! The dataset is the oracle for this half of the harness, and the two passes are
//! testable against it in opposite directions.
//!
//! `--reparse` has to be a **no-op**. `tests/archived_logs.rs` already proves the
//! Rust parser reproduces the `parsed` block of every one of the 604 recorded
//! measurements, so re-deriving them can only produce what is already there —
//! which makes "the file comes back byte for byte" a real assertion rather than a
//! tautology: it also catches a pass that rewrote a line it had no reason to.
//!
//! `--retire-stale-builds` has to be **reproducible**. The dataset carries the
//! campaign's own retirement, so undoing it and running the rule again must select
//! the same rows — and because a status is spliced rather than re-serialised, the
//! round trip lands on the original bytes.
//!
//! Neither test writes to the dataset. Both read it, and the second edits a copy in
//! memory.

use std::path::{Path, PathBuf};

use crate::harness::{
    dataset,
    json::splice_member,
    maintain::{reparse, retire_stale_builds},
    sys::LocalFiles,
};

/// The campaign's own figure: the rows it retired when the build last changed.
const RETIRED_ROWS: usize = 14;

#[test]
fn reparsing_the_campaign_leaves_every_record_byte_identical() {
    let lines = campaign();
    let logs = data_dir().join("logs");
    let (out, report) = reparse(&lines, &|log| {
        dataset::read_archived_log(&LocalFiles, &logs.join(log))
    });

    // Named per line rather than asserted in bulk, because a single disagreement
    // is a finding about the parser and the log it came from.
    for (index, (before, after)) in lines.iter().zip(&out).enumerate() {
        assert_eq!(before, after, "record {index} was rewritten");
    }
    assert_eq!(report.rewritten, 0);
    // The dataset is checked in, so its size is a fact the test can assert on: a
    // pass that skipped everything would otherwise pass.
    assert!(
        report.unchanged > 500,
        "only {} records were re-derivable, so the no-op above proves little",
        report.unchanged
    );
    assert_eq!(report.unchanged + report.skipped, lines.len());
}

#[test]
fn retiring_reselects_exactly_the_rows_the_campaign_retired() {
    let lines = campaign();
    // Put the dataset back as it stood before the campaign's own retirement pass:
    // the rule reads `status == "ok"` rows, so a row already marked takes no part in
    // the comparison that marked it.
    let revived: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.contains(r#""status": "stale-runtime""#) {
                splice_member(line, "status", "\"ok\"").expect("status is a member")
            } else {
                line.clone()
            }
        })
        .collect();
    assert_eq!(
        revived
            .iter()
            .zip(&lines)
            .filter(|(revived, original)| revived != original)
            .count(),
        RETIRED_ROWS,
        "the revival should touch exactly the rows the campaign retired"
    );

    let (out, report) = retire_stale_builds(&revived, 0.02);
    assert_eq!(report.retired, RETIRED_ROWS);
    for (index, (expected, got)) in lines.iter().zip(&out).enumerate() {
        assert_eq!(expected, got, "record {index} came back differently");
    }
    // One architecture on one build, which is the finding the rule encodes: the
    // fork was not the unit that changed.
    assert_eq!(report.builds.len(), 1, "{:?}", report.builds);
}

fn campaign() -> Vec<String> {
    dataset::read_lines(&LocalFiles, &data_dir().join("measurements.ndjson"))
        .expect("the campaign's measurements are checked in")
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
