//! The four buffer-size figures the loaders print, each with its history.

use ananke_dataset::Parsed;
use regex::Regex;

use crate::parse::patterns;

/// Read the four figures out of a log into the flat fields the record spells
/// them under.
///
/// Flat because the schema is: [`Parsed`] names each figure and its repeat list
/// as its own field, so there is nothing for an indexed accumulator to be
/// indexed by.
pub(crate) fn fill(text: &str, parsed: &mut Parsed) {
    (parsed.arena_mib, parsed.arena_mib_all) = series(text, &patterns::ARENA);
    (parsed.out_buf_mib, parsed.out_buf_mib_all) = series(text, &patterns::OUT_BUF);
    (parsed.cpu_kv_mib, parsed.cpu_kv_mib_all) = series(text, &patterns::CPU_KV);
    (parsed.cpu_model_mib, parsed.cpu_model_mib_all) = series(text, &patterns::CPU_MODEL);
}

/// One figure's value, plus every occurrence of it when there was more than
/// one.
///
/// The value is the *last* occurrence: the loader logs a reserve pass first and
/// then the real graph, with the same figure. A cell with `-md` or `--mmproj`
/// loads two models, though, so a figure can appear more than once with
/// genuinely different values — keeping every occurrence lets a later reader
/// tell the target's from the draft's rather than inheriting whichever this
/// pass happened to pick.
fn series(text: &str, pattern: &Regex) -> (f64, Option<Vec<f64>>) {
    let found: Vec<f64> = pattern
        .captures_iter(text)
        .filter_map(|caps| caps.get(1)?.as_str().parse().ok())
        .collect();
    (
        found.last().copied().unwrap_or(0.0),
        (found.len() > 1).then_some(found),
    )
}
