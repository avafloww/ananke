//! Every factor the harness varies is either read by the calibration or listed
//! here as deliberately unread.
//!
//! Two `Factors` types describe one JSON object: `ananke_measure`'s, which the
//! harness writes and which is the authority, and `ananke_calibrate`'s, a tolerant
//! reader declaring only what the derivers use. That is a reasonable split — the
//! reader should not have to grow a field to keep parsing — but it has one failure
//! mode, and it is this campaign's signature bug: a factor the harness starts
//! varying that the derivation silently ignores, so a term is fitted across cells
//! that differ in a way the fit cannot see. Four wrong constants came from exactly
//! that shape, including `bool(n_cpu_moe)` in place of the count and a pairing key
//! missing `ngl`.
//!
//! So the omissions are enumerated rather than implicit. Adding a factor to the
//! harness fails this test until somebody says which list it belongs on.

use std::collections::BTreeSet;

use ananke_calibrate::record::read_ndjson;
use ananke_measure::record::Factors;

const MEASUREMENTS: &str = "../scripts/calibration/data/measurements.ndjson";

/// Factors the derivers read. Each is a knob that changes what a process holds.
const READ: &[&str] = &[
    "label",
    "model",
    "runtime",
    "gpus",
    "ctx",
    "ubatch",
    "parallel",
    "ngl",
    "split",
    "kv_type",
    "kv_unified",
    "flash_attn",
    "n_cpu_moe",
    "mmproj",
    "draft",
    "spec_type",
    "no_mmap",
    "rtr",
    "cram",
    "soak",
    "concurrency",
    "probe_prompt_tokens",
    "embeddings",
    "bench",
    "served",
    "numa",
    "extra",
];

/// Factors the calibration deliberately does not read, and why.
///
/// Nothing here may be a term in a memory model. Two are load-bearing enough to
/// have been checked against the data rather than argued about:
///
/// - `threads` was swept on purpose (`threads-4` through `threads-48`) and the
///   graph arena does not move by a byte across it. See
///   `the_thread_count_does_not_move_the_arena` below, which is that sweep as an
///   assertion.
/// - `batch` (`-b`, the logical batch) appears in four cells and never against an
///   otherwise-identical partner, so there is no evidence either way. It is omitted
///   because llama.cpp sizes its buffers from the *micro*-batch, which is read; if a
///   cell pair for it ever lands, this is the comment that was relying on that.
///
/// The rest are harness bookkeeping: how the cell was exercised, how loudly it
/// logged, and which questions asked for it.
const NOT_READ: &[&str] = &[
    // Sweeps to no measurable effect; see above.
    "threads",
    "batch",
    // How the cell was exercised, not what it held.
    "bench_turns",
    "probe_tokens",
    // The cell's own metadata: which questions wanted it, and whether it is a
    // repeat of one for noise measurement.
    "purpose",
    "repeat",
    // Log verbosity. Changes what the parser can find, never what the process
    // holds — and a cell measured without it simply has fewer parsed fields.
    "verbose_log",
];

/// The two lists together are exactly the harness's factor set.
///
/// The assertion runs both ways on purpose. A factor missing from both lists is the
/// bug this file exists to catch; a name in a list that the harness no longer has is
/// a stale comment claiming a decision about nothing.
#[test]
fn every_factor_is_classified() {
    let json = serde_json::to_value(Factors::default()).expect("the factors serialise");
    let actual: BTreeSet<String> = json
        .as_object()
        .expect("factors are an object")
        .keys()
        .cloned()
        .collect();

    let classified: BTreeSet<String> = READ
        .iter()
        .chain(NOT_READ.iter())
        .map(|s| s.to_string())
        .collect();

    let unclassified: Vec<&String> = actual.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "the harness varies {unclassified:?}, which the calibration neither reads nor \
         declares unread — decide which, and if unread say why"
    );

    let stale: Vec<&String> = classified.difference(&actual).collect();
    assert!(
        stale.is_empty(),
        "{stale:?} are classified but the harness no longer has them"
    );
}

/// No factor is on both lists.
#[test]
fn the_lists_do_not_overlap() {
    let read: BTreeSet<&&str> = READ.iter().collect();
    let unread: BTreeSet<&&str> = NOT_READ.iter().collect();
    let both: Vec<&&&str> = read.intersection(&unread).collect();
    assert!(both.is_empty(), "{both:?} are both read and not read");
}

/// The thread count does not change the graph arena.
///
/// This is the justification for `threads` being on the unread list, kept as a test
/// so that the justification is checked rather than remembered. The campaign swept 4
/// through 48 threads against three models; within each, the arena is identical.
#[test]
fn the_thread_count_does_not_move_the_arena() {
    let text = std::fs::read_to_string(MEASUREMENTS).expect("the dataset is readable");
    let records = read_ndjson(&text).expect("the dataset parses");

    // Cells whose label marks them as part of the thread sweep, grouped by the
    // model they were taken against.
    let mut sweeps: std::collections::BTreeMap<&str, Vec<(u32, f64)>> = Default::default();
    for record in &records {
        if record.status != "ok" || !record.factors.label.contains("threads") {
            continue;
        }
        let Some(threads) = thread_count(&record.factors.label) else {
            continue;
        };
        let Some(arena) = record.parsed.arena_mib.filter(|v| *v > 0.0) else {
            continue;
        };
        sweeps
            .entry(record.factors.model.as_str())
            .or_default()
            .push((threads, arena));
    }

    assert!(
        sweeps.len() >= 2,
        "the thread sweep covers at least two models, found {}",
        sweeps.len()
    );
    for (model, points) in &sweeps {
        assert!(
            points.len() >= 2,
            "{model}: a sweep needs more than one point, got {points:?}"
        );
        let first = points[0].1;
        for (threads, arena) in points {
            assert!(
                (arena - first).abs() < 0.01,
                "{model}: the arena moved to {arena} MiB at {threads} threads, from \
                 {first} — the thread count is not the inert factor this claims"
            );
        }
    }
}

/// The thread count a sweep cell's label ends in.
fn thread_count(label: &str) -> Option<u32> {
    let tail = label.rsplit_once("threads")?.1;
    let digits: String = tail
        .trim_start_matches(['-', '_'])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}
