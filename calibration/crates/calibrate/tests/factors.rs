//! Every factor the dataset carries is either read by the calibration or listed
//! here as deliberately unread.
//!
//! One `Factors` describes the JSON object now — [`ananke_dataset`]'s — but the
//! hazard the split used to carry has not gone away with it: a factor the
//! harness starts varying that the derivation never consults is a term fitted
//! across cells that differ in a way the fit cannot see. Four wrong constants in
//! this campaign came from exactly that shape, including `bool(n_cpu_moe)` in
//! place of the count and a pairing key missing `ngl`.
//!
//! So the omissions are enumerated rather than implicit. Adding a factor to the
//! schema fails this test until somebody says which list it belongs on.

use std::collections::{BTreeMap, BTreeSet};

use ananke_dataset::{Factors, Status, read_ndjson};

const MEASUREMENTS: &str = "../../data/measurements.ndjson";

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

/// The two lists together are exactly the schema's factor set.
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
        "the schema carries {unclassified:?}, which the calibration neither reads nor \
         declares unread — decide which, and if unread say why"
    );

    let stale: Vec<&String> = classified.difference(&actual).collect();
    assert!(
        stale.is_empty(),
        "{stale:?} are classified but the schema no longer has them"
    );
}

/// The schema's `Factors` has no serde attribute that hides a field.
///
/// `every_factor_is_classified` enumerates the keys of a serialised
/// `Factors::default()`, which is complete only while every field actually appears
/// there. Three attributes would break that, and one of them silently:
///
/// - `skip_serializing_if` on a field that is `None` by default,
/// - `flatten` on a map that is empty by default,
/// - `skip`, which also removes the field from `cell_id` — so two cells differing
///   only in that factor would share an identity and the second would never run,
///   which is the `cram` bug this campaign already had once.
///
/// None is present today. The check is on the source because serde attributes leave
/// nothing at runtime to interrogate.
#[test]
fn no_serde_attribute_hides_a_factor() {
    let source = std::fs::read_to_string("../dataset/src/record/factors.rs")
        .expect("the schema's source is readable");
    let block = struct_block(&source, "pub struct Factors {");
    for hazard in ["skip_serializing_if", "flatten", "skip)", "skip,", "skip]"] {
        assert!(
            !block.contains(hazard),
            "`{hazard}` appears in `Factors`: a field it hides is absent from a \
             serialised default, so `every_factor_is_classified` would not see it \
             and the calibration could ignore it in silence"
        );
    }
}

/// The body of a named struct, up to its closing brace.
fn struct_block<'a>(source: &'a str, header: &str) -> &'a str {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("`{header}` is in the source"));
    let rest = &source[start + header.len()..];
    let end = rest.find("\n}").expect("the struct block is closed");
    &rest[..end]
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

    // Cells whose label marks them as part of the thread sweep, grouped by
    // everything else that sizes the arena. Grouping by model alone would compare a
    // future thread cell taken at another context against these and report the
    // context's effect as the thread count's.
    let mut sweeps: BTreeMap<Configuration<'_>, Vec<ArenaAt>> = BTreeMap::new();
    for record in &records {
        if record.status != Status::Ok || !record.factors.label.contains("threads") {
            continue;
        }
        let Some(threads) = thread_count(&record.factors.label) else {
            continue;
        };
        let arena = record.parsed.arena_mib;
        if arena <= 0.0 {
            continue;
        }
        sweeps
            .entry((
                record.factors.model.as_str(),
                record.factors.ctx,
                record.factors.ubatch,
            ))
            .or_default()
            .push((threads, arena));
    }

    let comparable: Vec<_> = sweeps
        .iter()
        .filter(|(_, points)| points.len() >= 2)
        .collect();
    assert!(
        comparable.len() >= 2,
        "the thread sweep needs at least two configurations with more than one thread \
         count each, found {} among {:?}",
        comparable.len(),
        sweeps.keys().collect::<Vec<_>>()
    );
    for ((model, ctx, ubatch), points) in comparable {
        let counts: BTreeSet<u32> = points.iter().map(|(threads, _)| *threads).collect();
        assert!(
            counts.len() >= 2,
            "{model} ctx {ctx} ub {ubatch}: the same thread count repeated is not a \
             sweep, got {points:?}"
        );
        let first = points[0].1;
        for (threads, arena) in points {
            assert!(
                (arena - first).abs() < 0.01,
                "{model} ctx {ctx} ub {ubatch}: the arena moved to {arena} MiB at \
                 {threads} threads, from {first} — the thread count is not the inert \
                 factor this claims"
            );
        }
    }
}

/// Everything but the thread count that sizes the graph arena: the model, the
/// context, and the micro-batch.
type Configuration<'a> = (&'a str, u32, u32);

/// A thread count and the arena measured at it, in MiB.
type ArenaAt = (u32, f64);

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
