//! The proof: every committed row parses into the schema and comes back out.
//!
//! Three properties, checked separately because they fail for different
//! reasons and carry different verdicts.
//!
//! *Readability* says the schema models every field the dataset carries.
//! `deny_unknown_fields` turns a column the schema forgot into a parse error,
//! so a passing parse is the no-loss guarantee in one direction.
//!
//! *Completeness* says the writer spells every key it read. A key that goes in
//! and does not come out is a schema bug, and fails the run unconditionally.
//!
//! *Canonicity* says the schema also agrees with the committed lines on key
//! order and number formatting. The dataset does not yet satisfy that — it was
//! written by more than one writer over the campaign's life — so this one
//! reports what a one-time canonical rewrite would change rather than failing.
//! Once that rewrite lands, flip [`ROWS_MUST_BE_CANONICAL`] and it becomes the
//! gate.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use ananke_dataset::{Record, to_dataset_json};

/// Whether the committed dataset is expected to be in canonical form already.
///
/// False until the one-time rewrite lands; the non-canonical rows are counted
/// and described instead of failing the run.
const ROWS_MUST_BE_CANONICAL: bool = false;

const DATASET: &str = include_str!("../../../data/measurements.ndjson");

/// Path segments are joined with a slash, not a dot, because a GGUF metadata
/// key is *itself* dotted (`general.architecture`) and splitting on a dot would
/// report a flat map as if it were six levels of nesting.
const SEPARATOR: char = '/';

#[test]
fn every_row_parses_with_no_field_unaccounted_for() {
    for (index, line) in rows() {
        if let Err(error) = serde_json::from_str::<Record>(line) {
            panic!("line {index}: {error}");
        }
    }
}

#[test]
fn every_row_round_trips_to_the_bytes_it_came_from() {
    let mut rewritten = 0usize;
    let mut first = None;
    let mut lost = Tally::default();
    let mut defaulted = Tally::default();
    let mut reordered = Tally::default();
    let mut reformatted = 0usize;

    for (index, line) in rows() {
        let record: Record = serde_json::from_str(line).expect("the row parses");
        let written = to_dataset_json(&record);
        if written == line {
            continue;
        }
        rewritten += 1;
        first.get_or_insert(index);
        let difference = Difference::between(line, &written);
        lost.extend(difference.lost);
        defaulted.extend(difference.defaulted);
        reordered.extend(difference.reordered);
        reformatted += usize::from(difference.reformatted);
    }

    assert!(
        lost.is_empty(),
        "the writer dropped fields the reader accepted, which is a schema bug: {}",
        lost.describe()
    );

    if rewritten == 0 {
        return;
    }
    let report = format!(
        "{rewritten} of {} rows are not byte-canonical; the first is line {}.\n\
         A one-time canonical rewrite would:\n\
         \x20 spell {} defaulted key(s) the rows omit: {}\n\
         \x20 reorder the keys inside: {}\n\
         \x20 respell a number or a string in {reformatted} row(s)",
        rows().count(),
        first.unwrap_or_default(),
        defaulted.len(),
        defaulted.describe(),
        reordered.describe(),
    );
    if ROWS_MUST_BE_CANONICAL {
        panic!("{report}");
    }
    eprintln!("{report}");
}

/// How a re-serialised row disagrees with the committed one.
///
/// The distinction that matters is between a *lost field* — a schema bug — and
/// a reordering, a defaulted key, or a reformatting, all of which a one-time
/// canonical rewrite settles. A row can disagree in more than one way at once,
/// so every facet is collected rather than the first.
#[derive(Default)]
struct Difference {
    /// Paths the row carries and the writer did not emit.
    lost: Vec<String>,
    /// Paths the writer emits from a default because the row omits them.
    defaulted: Vec<String>,
    /// Objects whose children the two spell in a different order.
    reordered: Vec<String>,
    /// Whether a value the two agree on the existence of is spelled
    /// differently.
    reformatted: bool,
}

impl Difference {
    fn between(line: &str, written: &str) -> Self {
        let before = keys(line);
        let after = keys(written);
        let mut difference = Self {
            lost: before.difference(&after).cloned().collect(),
            defaulted: after.difference(&before).cloned().collect(),
            ..Self::default()
        };

        // Order is judged only over the keys the two have in common, so a
        // defaulted key does not read as a reordering as well.
        let kept: Vec<String> = keys_in_order(written)
            .into_iter()
            .filter(|key| before.contains(key))
            .collect();
        difference.reordered = differing_parents(&keys_in_order(line), &kept);

        // Values are judged with both sides rendered key-sorted, so only a
        // genuine difference in a number's or a string's spelling shows up —
        // and only where the key sets already agree, since a defaulted key is
        // a value difference too.
        difference.reformatted = difference.lost.is_empty()
            && difference.defaulted.is_empty()
            && sorted(line) != sorted(written);
        difference
    }
}

/// How many rows each path was seen in.
#[derive(Default)]
struct Tally(BTreeMap<String, usize>);

impl Tally {
    fn extend(&mut self, paths: Vec<String>) {
        for path in paths {
            *self.0.entry(path).or_default() += 1;
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    /// Most-affected first, so the headline cause reads first.
    fn describe(&self) -> String {
        if self.0.is_empty() {
            return "nothing".to_owned();
        }
        let mut counted: Vec<(&String, &usize)> = self.0.iter().collect();
        counted.sort_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
        counted
            .iter()
            .map(|(path, count)| format!("{path} ({count})"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Which objects the two spell their children in a different order inside.
///
/// Reported per parent rather than as one flat "the order differs", because the
/// answer a one-time rewrite needs is *which* blocks it touches.
fn differing_parents(before: &[String], after: &[String]) -> Vec<String> {
    let before = children(before);
    let after = children(after);
    before
        .iter()
        .filter(|(parent, order)| after.get(*parent) != Some(order))
        .map(|(parent, _)| parent.clone())
        .collect()
}

fn children(paths: &[String]) -> BTreeMap<String, Vec<String>> {
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in paths {
        let (parent, child) = path
            .rsplit_once(SEPARATOR)
            .unwrap_or(("the row", path.as_str()));
        grouped
            .entry(if parent.is_empty() { "the row" } else { parent }.to_owned())
            .or_default()
            .push(child.to_owned());
    }
    grouped
}

/// The row rendered with every object key sorted, which `serde_json::Value`
/// does by default because it is backed by a `BTreeMap`.
fn sorted(text: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(text).expect("the row is JSON");
    to_dataset_json(&value)
}

fn rows() -> impl Iterator<Item = (usize, &'static str)> {
    DATASET
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .filter(|(_, line)| !line.trim().is_empty())
}

/// Every key anywhere in the document, path-qualified so a rename at one depth
/// cannot cancel out against another.
fn keys(text: &str) -> BTreeSet<String> {
    keys_in_order(text).into_iter().collect()
}

/// The same paths, in the order the *text* spells them.
///
/// Scanned off the bytes rather than off a parsed `serde_json::Value`, which
/// would be useless here: `Value` is backed by a `BTreeMap`, so parsing sorts
/// the keys and erases the very thing this comparison is about.
fn keys_in_order(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    scan(text.as_bytes(), 0, "", &mut out).expect("the row is JSON");
    out
}

/// One value, returning the index one past its end.
fn scan(bytes: &[u8], at: usize, path: &str, out: &mut Vec<String>) -> Option<usize> {
    let mut at = skip_space(bytes, at);
    match bytes.get(at)? {
        b'{' => {
            at += 1;
            loop {
                at = skip_space(bytes, at);
                if bytes.get(at) == Some(&b'}') {
                    return Some(at + 1);
                }
                let name = string_span(bytes, at)?;
                let child = format!(
                    "{path}{SEPARATOR}{}",
                    std::str::from_utf8(&bytes[name.clone()]).ok()?
                );
                out.push(child.clone());
                at = skip_space(bytes, name.end + 1);
                if bytes.get(at) != Some(&b':') {
                    return None;
                }
                at = scan(bytes, at + 1, &child, out)?;
                at = skip_space(bytes, at);
                match bytes.get(at) {
                    Some(b',') => at += 1,
                    Some(b'}') => return Some(at + 1),
                    _ => return None,
                }
            }
        }
        b'[' => {
            // The index is deliberately not in the path: two tables differing
            // only in length is not a key rename.
            let child = format!("{path}[]");
            at += 1;
            loop {
                at = skip_space(bytes, at);
                if bytes.get(at) == Some(&b']') {
                    return Some(at + 1);
                }
                at = scan(bytes, at, &child, out)?;
                at = skip_space(bytes, at);
                match bytes.get(at) {
                    Some(b',') => at += 1,
                    Some(b']') => return Some(at + 1),
                    _ => return None,
                }
            }
        }
        b'"' => string_end(bytes, at),
        // A number or one of the three literals: everything up to the next
        // structural byte.
        _ => {
            let start = at;
            while let Some(&byte) = bytes.get(at) {
                if matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }
                at += 1;
            }
            (at > start).then_some(at)
        }
    }
}

fn skip_space(bytes: &[u8], mut at: usize) -> usize {
    while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        at += 1;
    }
    at
}

/// The span *inside* the quotes of the string starting at `at`.
fn string_span(bytes: &[u8], at: usize) -> Option<Range<usize>> {
    let end = string_end(bytes, at)?;
    Some(at + 1..end - 1)
}

/// One past the closing quote of the string starting at `at`.
fn string_end(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.get(at) != Some(&b'"') {
        return None;
    }
    let mut index = at + 1;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'\\' => index += 2,
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}
