//! The two maintenance passes over an existing dataset: re-derive, and retire.
//!
//! Both are pure functions over the file's lines, with the log reader injected, so
//! either can be held against the checked-in dataset in a test without writing to
//! it. Both also rewrite only the lines they change — see [`crate::harness::json`]
//! for why that matters.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    harness::json::{member_span, splice_member, to_dataset_json},
    parse_log,
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
    let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
        return Reparsed::Skipped;
    };
    // A cell that never loaded carries an empty `parsed` by construction, and its
    // log holds only the failure. Parsing it anyway replaces that emptiness with a
    // full block of zeros, which reads as a measurement that found nothing rather
    // than as no measurement at all.
    if record["status"] != serde_json::json!("ok") {
        return Reparsed::Skipped;
    }
    let Some(log) = record["log"].as_str().filter(|log| !log.is_empty()) else {
        return Reparsed::Skipped;
    };
    let Some(text) = read_log(log) else {
        return Reparsed::Skipped;
    };
    let parsed = parse_log(&text);
    let ours = serde_json::to_value(&parsed).expect("Parsed serializes as an object");
    if equivalent(&ours, &record["parsed"]) {
        return Reparsed::Unchanged;
    }
    let Some(line) = splice_member(line, "parsed", &to_dataset_json(&parsed)) else {
        return Reparsed::Skipped;
    };
    Reparsed::Rewritten(set_member(&line, "reparsed", "true"))
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
    let records: Vec<serde_json::Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).unwrap_or(serde_json::Value::Null))
        .collect();
    let stale = stale_builds(&records, tolerance);

    let mut report = RetireReport {
        builds: stale.iter().cloned().collect(),
        retired: 0,
    };
    let mut out = Vec::with_capacity(lines.len());
    for (line, record) in lines.iter().zip(&records) {
        let key = (architecture(record), build(record));
        if record["status"] == serde_json::json!("ok") && stale.contains(&key) {
            report.retired += 1;
            out.push(
                splice_member(line, "status", "\"stale-runtime\"").unwrap_or_else(|| line.clone()),
            );
        } else {
            out.push(line.clone());
        }
    }
    (out, report)
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RetireReport {
    /// The (architecture, build) pairs a later build disagreed with.
    pub(crate) builds: Vec<(String, String)>,
    pub(crate) retired: usize,
}

/// Which (architecture, build) pairs a later build has been shown to disagree
/// with, on a cell measured under both.
fn stale_builds(records: &[serde_json::Value], tolerance: f64) -> BTreeSet<(String, String)> {
    // Per architecture and cell: what each build read, and when it read it.
    let mut readings: BTreeMap<(String, String), BTreeMap<String, (f64, String)>> = BTreeMap::new();
    for record in records {
        if record["status"] != serde_json::json!("ok") {
            continue;
        }
        // The driver's per-process total is the comparison: it is the one figure
        // every cell has on both forks, ik_llama printing no breakdown table at
        // all. A row without it cannot take part.
        let Some(used) = record["rss"]["gpu_used_mib"]
            .as_f64()
            .filter(|mib| *mib > 0.0)
        else {
            continue;
        };
        let cell = record["cell"].as_str().unwrap_or_default().to_owned();
        let when = record["provenance"]["measured_at_utc"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        readings
            .entry((architecture(record), cell))
            .or_default()
            .insert(build(record), (used, when));
    }

    let mut stale = BTreeSet::new();
    for ((arch, _cell), seen) in readings {
        for (build, (value, when)) in &seen {
            for (other_build, (other, other_when)) in &seen {
                if build == other_build || *value == 0.0 {
                    continue;
                }
                // Older *and* different: a build that disagrees with one measured
                // before it is the newer program, and it is the older row that is
                // stale.
                if (other - value).abs() / value > tolerance && when < other_when {
                    stale.insert((arch.clone(), build.clone()));
                }
            }
        }
    }
    stale
}

fn architecture(record: &serde_json::Value) -> String {
    record["parsed"]["arch"].as_str().unwrap_or("?").to_owned()
}

fn build(record: &serde_json::Value) -> String {
    record["provenance"]["runtime_sha256"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// Set a top-level member, adding it at the end when it is not there, which is
/// where the records already carrying it have it.
fn set_member(line: &str, key: &str, value: &str) -> String {
    if member_span(line, key).is_some() {
        return splice_member(line, key, value).unwrap_or_else(|| line.to_owned());
    }
    match line.rfind('}') {
        Some(close) => format!("{}, \"{key}\": {value}{}", &line[..close], &line[close..]),
        None => line.to_owned(),
    }
}

/// JSON equality that reads numbers as numbers.
///
/// Exact, not tolerant: both sides are transcribed from the same decimal text in
/// the same log, so any difference at all means the parser now reads something
/// differently — which is precisely the case a rewrite is for.
fn equivalent(ours: &serde_json::Value, theirs: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (ours, theirs) {
        (Value::Object(ours), Value::Object(theirs)) => {
            ours.len() == theirs.len()
                && ours.iter().all(|(key, ours)| {
                    theirs
                        .get(key)
                        .is_some_and(|theirs| equivalent(ours, theirs))
                })
        }
        (Value::Array(ours), Value::Array(theirs)) => {
            ours.len() == theirs.len()
                && ours
                    .iter()
                    .zip(theirs)
                    .all(|(ours, theirs)| equivalent(ours, theirs))
        }
        (Value::Number(ours), Value::Number(theirs)) => ours.as_f64() == theirs.as_f64(),
        (ours, theirs) => ours == theirs,
    }
}

#[cfg(test)]
mod oracle;

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cell: &str, arch: &str, build: &str, when: &str, gpu: u64) -> String {
        to_dataset_json(&serde_json::json!({
            "cell": cell,
            "status": "ok",
            "provenance": {"runtime_sha256": build, "measured_at_utc": when},
            "parsed": {"arch": arch},
            "rss": {"gpu_used_mib": gpu},
        }))
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
            vec![("glm-dsa".to_owned(), "oldbuild".to_owned())]
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
        let lines = vec![
            to_dataset_json(
                &serde_json::json!({"status": "ok", "log": "", "parsed": {"arch": "x"}}),
            ),
            to_dataset_json(
                &serde_json::json!({"status": "failed-to-load", "log": "a.log.gz", "parsed": {}}),
            ),
        ];
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
        let line = to_dataset_json(&serde_json::json!({
            "status": "ok", "log": "a.log.gz", "parsed": {"arch": "stale"}, "cell": "abc",
        }));
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
