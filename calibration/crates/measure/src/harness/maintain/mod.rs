//! The two maintenance passes over an existing dataset: re-derive, and retire.
//!
//! Both are pure functions over the file's lines, with the log reader injected, so
//! either can be held against the checked-in dataset in a test without writing to
//! it. Both work on the typed row: a line is parsed into a [`Record`], the one
//! field the pass is about is set, and the record is written back out. The dataset
//! is canonical, so re-serialising a row it did not change reproduces the original
//! bytes — which is what keeps a one-field edit a one-field diff.
//!
//! Both still return the untouched line for a record they decide against, so a
//! pass over a file some other writer produced cannot reformat rows it had no
//! reason to visit.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    harness::to_dataset_json,
    parse_log,
    record::{Record, Status},
};

/// Rebuild every record's `parsed` block from its archived log.
///
/// The logs are kept precisely so that a question the parser could not answer when
/// a cell ran can still be answered later. Re-running the campaign to add a field
/// would cost days of GPU time and would measure a different llama.cpp build;
/// re-reading the logs costs seconds and changes nothing about what was observed.
///
/// A record whose log is missing keeps the `parsed` block it already has, with
/// `reparsed` left absent so an analysis can tell which rows carry the newer
/// fields. So does a record whose block the current parser reproduces exactly:
/// there is nothing to re-derive, and rewriting the line would claim a change that
/// did not happen. Rewriting every line unconditionally would make an idempotent
/// pass indistinguishable from a substantive one.
pub(crate) fn reparse(
    lines: &[String],
    read_log: &dyn Fn(&str) -> Option<String>,
) -> (Vec<String>, ReparseReport) {
    let mut report = ReparseReport::default();
    let mut out = Vec::with_capacity(lines.len());
    for line in lines {
        out.push(match reparse_line(line, read_log) {
            Reparsed::Rewritten(line) => {
                report.rewritten += 1;
                line
            }
            Reparsed::Unchanged => {
                report.unchanged += 1;
                line.clone()
            }
            Reparsed::Skipped => {
                report.skipped += 1;
                line.clone()
            }
        });
    }
    (out, report)
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ReparseReport {
    /// Records whose `parsed` block the current parser reads differently.
    pub(crate) rewritten: usize,
    /// Records the parser reproduces exactly.
    pub(crate) unchanged: usize,
    /// Records with no archived log to re-read, or no measurement in them.
    pub(crate) skipped: usize,
}

enum Reparsed {
    Rewritten(String),
    Unchanged,
    Skipped,
}

fn reparse_line(line: &str, read_log: &dyn Fn(&str) -> Option<String>) -> Reparsed {
    let Ok(mut record) = serde_json::from_str::<Record>(line) else {
        return Reparsed::Skipped;
    };
    // A cell that never loaded logged nothing to parse, so its `parsed` block is
    // the schema's zeroes and its log holds only the failure. Re-deriving it would
    // read the failure as a measurement that found nothing.
    if record.status != Status::Ok || record.log.is_empty() {
        return Reparsed::Skipped;
    }
    let Some(text) = read_log(&record.log) else {
        return Reparsed::Skipped;
    };
    let parsed = parse_log(&text);
    if parsed == record.parsed {
        return Reparsed::Unchanged;
    }
    record.parsed = parsed;
    record.reparsed = Some(true);
    Reparsed::Rewritten(to_dataset_json(&record))
}

/// Mark rows a runtime upgrade invalidated, so every reader skips them.
///
/// A cell id hashes the *factors*, and the runtime binary is not one of them.
/// Upgrade llama.cpp and every existing row keeps describing a program that is no
/// longer installed, with nothing to say so — and a constant fitted across the two
/// is fitted to two programs.
///
/// The evidence needed is one re-measurement per architecture. From this dataset's
/// upgrade: six of seven production cells reproduce to the megabyte, and the
/// seventh, GLM-5.2, moves 9.6% because ik's sparse-attention compute buffer shrank
/// by a third. laguna on the same fork is unchanged, so the fork is not the unit
/// that changed — `glm-dsa` is.
///
/// So an (architecture, build) pair is retired only where a cell measured under
/// both it and a later build disagrees. Being older is not evidence: three ik
/// builds appear in this dataset and only one is shown to differ, and retiring on
/// age alone would discard a whole batch-and-context sweep because a
/// `nixos-rebuild` landed between running it and checking it.
///
/// Retired rows keep their data and their archived log; only `status` changes, to
/// `stale-runtime`. That is deliberately the same gate every consumer already
/// applies — `ananke-calibrate`'s derivers and reports, and the estimator's
/// integration tests, all take `status == "ok"` — so none of them needs to learn
/// about builds, and one that forgets is looking at a status it understands rather
/// than being silently wrong.
pub(crate) fn retire_stale_builds(lines: &[String], tolerance: f64) -> (Vec<String>, RetireReport) {
    let records: Vec<Option<Record>> = lines
        .iter()
        .map(|line| serde_json::from_str(line).ok())
        .collect();
    let stale = stale_builds(&records, tolerance);

    let mut report = RetireReport {
        builds: stale.iter().cloned().collect(),
        retired: 0,
    };
    let mut out = Vec::with_capacity(lines.len());
    for (line, record) in lines.iter().zip(&records) {
        match record {
            Some(record) if record.status == Status::Ok && stale.contains(&identity(record)) => {
                report.retired += 1;
                let mut retired = record.clone();
                retired.status = Status::StaleRuntime;
                out.push(to_dataset_json(&retired));
            }
            _ => out.push(line.clone()),
        }
    }
    (out, report)
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RetireReport {
    /// The builds a later build disagreed with, per architecture.
    pub(crate) builds: Vec<BuildIdentity>,
    pub(crate) retired: usize,
}

/// The pair the rule is stated over: what ran, and which build ran it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BuildIdentity {
    pub(crate) arch: String,
    pub(crate) build: String,
}

/// Which (architecture, build) pairs a later build has been shown to disagree
/// with, on a cell measured under both.
fn stale_builds(records: &[Option<Record>], tolerance: f64) -> BTreeSet<BuildIdentity> {
    // Per architecture and cell: what each build read, and when it read it.
    let mut readings: BTreeMap<ArchCell, BTreeMap<String, Reading>> = BTreeMap::new();
    for record in records.iter().flatten() {
        if record.status != Status::Ok {
            continue;
        }
        // The driver's per-process total is the comparison: it is the one figure
        // every cell has on both forks, ik_llama printing no breakdown table at
        // all. A row without it cannot take part.
        let Some(used) = record.rss.gpu_used_mib.filter(|mib| *mib > 0) else {
            continue;
        };
        let id = identity(record);
        readings
            .entry(ArchCell {
                arch: id.arch,
                cell: record.cell.clone(),
            })
            .or_default()
            .insert(
                id.build,
                Reading {
                    used_mib: used as f64,
                    measured_at: record.provenance.measured_at_utc.clone(),
                },
            );
    }

    let mut stale = BTreeSet::new();
    for (key, seen) in readings {
        for (build, reading) in &seen {
            for (other_build, other) in &seen {
                if build == other_build {
                    continue;
                }
                // Older *and* different: a build that disagrees with one measured
                // before it is the newer program, and it is the older row that is
                // stale.
                let apart = (other.used_mib - reading.used_mib).abs() / reading.used_mib;
                if apart > tolerance && reading.measured_at < other.measured_at {
                    stale.insert(BuildIdentity {
                        arch: key.arch.clone(),
                        build: build.clone(),
                    });
                }
            }
        }
    }
    stale
}

/// One architecture's readings of one cell are what the rule compares.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ArchCell {
    arch: String,
    cell: String,
}

/// What one build read for a cell, and when.
#[derive(Debug, Clone)]
struct Reading {
    used_mib: f64,
    measured_at: String,
}

/// What ran, and which build ran it.
fn identity(record: &Record) -> BuildIdentity {
    BuildIdentity {
        arch: record.parsed.arch.clone(),
        build: record.provenance.runtime_sha256.clone(),
    }
}

#[cfg(test)]
mod oracle;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Parsed,
        record::{Factors, Hardware, Provenance, Rss, SCHEMA},
    };

    /// A row carrying only what the retirement rule reads.
    fn row(cell: &str, arch: &str, build: &str, when: &str, gpu: u64) -> String {
        let mut record = record();
        record.cell = cell.to_owned();
        record.parsed.arch = arch.to_owned();
        record.provenance.runtime_sha256 = build.to_owned();
        record.provenance.measured_at_utc = when.to_owned();
        record.rss.gpu_used_mib = Some(gpu);
        to_dataset_json(&record)
    }

    /// An otherwise empty `ok` record, since [`Record`] has no default: a status
    /// is never absent from a row, so there is no sensible one to pick.
    fn record() -> Record {
        Record {
            schema: SCHEMA,
            cell: String::new(),
            status: Status::Ok,
            provenance: Provenance::default(),
            hardware: Hardware::default(),
            factors: Factors::default(),
            parsed: Parsed::default(),
            rss: Rss::default(),
            log_tail: String::new(),
            log: String::new(),
            trace: Vec::new(),
            checkpoints: Vec::new(),
            reparsed: None,
        }
    }

    /// The whole point of the rule: an architecture whose figure moved is retired
    /// on the older build, and one that reproduces is left alone even though its
    /// rows are just as old.
    #[test]
    fn only_a_build_a_later_one_disagrees_with_is_retired() {
        let lines = vec![
            row(
                "glm",
                "glm-dsa",
                "oldbuild",
                "2026-07-01T00:00:00+00:00",
                44_000,
            ),
            row(
                "glm",
                "glm-dsa",
                "newbuild",
                "2026-07-20T00:00:00+00:00",
                40_000,
            ),
            row(
                "lag",
                "laguna",
                "oldbuild",
                "2026-07-01T00:00:00+00:00",
                22_000,
            ),
            row(
                "lag",
                "laguna",
                "newbuild",
                "2026-07-20T00:00:00+00:00",
                22_010,
            ),
        ];
        let (out, report) = retire_stale_builds(&lines, 0.02);
        assert_eq!(
            report.builds,
            vec![BuildIdentity {
                arch: "glm-dsa".to_owned(),
                build: "oldbuild".to_owned(),
            }]
        );
        assert_eq!(report.retired, 1);
        assert!(
            out[0].contains(r#""status": "stale-runtime""#),
            "{}",
            out[0]
        );
        for line in &out[1..] {
            assert!(line.contains(r#""status": "ok""#), "{line}");
        }
    }

    /// Being older is not evidence. A build that no re-measurement contradicts
    /// keeps its rows, or a `nixos-rebuild` between running a sweep and checking it
    /// would discard the sweep.
    #[test]
    fn an_unchallenged_build_keeps_its_rows() {
        let lines = vec![
            row(
                "a",
                "qwen35moe",
                "oldbuild",
                "2026-07-01T00:00:00+00:00",
                22_000,
            ),
            row(
                "b",
                "qwen35moe",
                "newbuild",
                "2026-07-20T00:00:00+00:00",
                44_000,
            ),
        ];
        let (out, report) = retire_stale_builds(&lines, 0.02);
        assert_eq!(report.retired, 0, "different cells are not comparable");
        assert_eq!(out, lines);
    }

    #[test]
    fn a_record_with_no_archived_log_keeps_the_parsed_block_it_has() {
        let no_log = {
            let mut record = record();
            record.parsed.arch = "x".to_owned();
            record
        };
        let never_loaded = Record {
            status: Status::FailedToLoad,
            log: "a.log.gz".to_owned(),
            ..record()
        };
        let lines = vec![to_dataset_json(&no_log), to_dataset_json(&never_loaded)];
        let (out, report) = reparse(&lines, &|_| Some("arch = llama".to_owned()));
        assert_eq!(out, lines);
        assert_eq!(
            report,
            ReparseReport {
                rewritten: 0,
                unchanged: 0,
                skipped: 2
            }
        );
    }

    #[test]
    fn a_block_the_parser_now_reads_differently_is_rewritten_and_marked() {
        let line = to_dataset_json(&Record {
            cell: "abc".to_owned(),
            log: "a.log.gz".to_owned(),
            parsed: Parsed {
                arch: "stale".to_owned(),
                ..Parsed::default()
            },
            ..record()
        });
        let (out, report) = reparse(&[line], &|_| Some("arch = llama\n".to_owned()));
        assert_eq!(report.rewritten, 1);
        assert!(out[0].contains(r#""arch": "llama""#), "{}", out[0]);
        assert!(out[0].ends_with(r#""reparsed": true}"#), "{}", out[0]);
        // Everything outside the block is untouched, including the keys that
        // surround it.
        assert!(out[0].contains(r#""status": "ok""#));
        assert!(out[0].contains(r#""log": "a.log.gz""#));
        assert!(out[0].contains(r#""cell": "abc""#));

        // And a second pass over the result changes nothing.
        let (again, report) = reparse(&out, &|_| Some("arch = llama\n".to_owned()));
        assert_eq!(again, out);
        assert_eq!(report.unchanged, 1);
    }
}
