//! Laying a probe run out for a reader.
//!
//! Four sections, one per question, each printing the comparison that answers it
//! rather than the readings it was computed from. A probe is read once, by someone
//! who already suspects a number is wrong, so the useful output is the difference —
//! `before -> after, step N` — not a table to subtract by hand.

use std::{collections::BTreeMap, fmt::Write};

use crate::harness::probe::{
    Observations, Reading,
    plan::{Question, STEP_PREDICT, STEP_WORDS, StageKind, Tag},
};

/// Bytes as MiB, to one decimal. Every figure here is a difference of two readings
/// taken seconds apart, so a tenth is the smallest digit that means anything.
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn anon_mib(reading: &Reading) -> f64 {
    mib(reading.rss.rss_anon_kb * 1024)
}

/// Render the questions asked, in the order they build on each other.
pub fn render(observations: &Observations, questions: &[Question], model: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "probe: {model}");

    if questions.contains(&Question::Step) {
        step(&mut out, observations);
    }
    if questions.contains(&Question::Maps) {
        maps(&mut out, observations);
    }
    if questions.contains(&Question::Growth) {
        growth(&mut out, observations);
    }
    if questions.contains(&Question::Prefill) {
        prefill(&mut out, observations);
    }
    if !observations.failures.is_empty() {
        let _ = writeln!(out, "\nnot measured:");
        for failure in &observations.failures {
            let _ = writeln!(out, "  {failure}");
        }
    }
    out
}

/// One process against itself, before and after the request that steps it.
///
/// The sharper of the two mapping views: two models differ in everything at once,
/// while a process differs from itself only in the allocation being hunted.
fn step(out: &mut String, observations: &Observations) {
    let (Some(before), Some(after)) = (
        observations
            .readings
            .iter()
            .find(|r| r.tag == Tag::Idle && r.stage == StageKind::Shared),
        observations
            .readings
            .iter()
            .find(|r| r.tag == Tag::Stepped && r.stage == StageKind::Shared),
    ) else {
        return;
    };
    let _ = writeln!(
        out,
        "\nstep — one process, before and after its first request\n  \
         RssAnon {:.1} -> {:.1} MiB   step {:+.1}",
        anon_mib(before),
        anon_mib(after),
        anon_mib(after) - anon_mib(before)
    );
    let (Some(a), Some(b)) = (before.maps.as_ref(), after.maps.as_ref()) else {
        return;
    };
    let mut rows: Vec<(f64, &String)> = keys(a, b)
        .into_iter()
        .map(|k| {
            (
                mib(*b.get(k).unwrap_or(&0)) - mib(*a.get(k).unwrap_or(&0)),
                k,
            )
        })
        .filter(|(delta, _)| delta.abs() > 1.0)
        .collect();
    rows.sort_by(|x, y| y.0.total_cmp(&x.0));
    if rows.is_empty() {
        let _ = writeln!(out, "  no mapping moved by more than a MiB");
    }
    for (delta, name) in rows {
        let _ = writeln!(out, "  {delta:>+9.1} MiB  {name}");
    }
}

/// Where the process's anonymous memory lives once it has served a request.
fn maps(out: &mut String, observations: &Observations) {
    let Some(after) = observations
        .readings
        .iter()
        .find(|r| r.tag == Tag::Stepped && r.maps.is_some())
    else {
        return;
    };
    let Some(maps) = after.maps.as_ref() else {
        return;
    };
    let _ = writeln!(out, "\nmaps — where the anonymous memory lives");
    let mut rows: Vec<(&String, &u64)> = maps.iter().filter(|(_, v)| mib(**v) > 2.0).collect();
    rows.sort_by(|x, y| y.1.cmp(x.1));
    for (name, bytes) in rows {
        let _ = writeln!(out, "  {:>9.1} MiB  {name}", mib(*bytes));
    }
}

/// Whether the footprint accumulates with use, and whether the prompt cache is why.
///
/// Printed as a series per cache setting rather than a total: a leak and a cache
/// filling to its cap both grow, and only the shape tells them apart.
fn growth(out: &mut String, observations: &Observations) {
    // Keyed on the cache setting, which is what the two series actually differ in.
    let mut by_cram: BTreeMap<u32, Vec<(usize, f64)>> = BTreeMap::new();
    for reading in &observations.readings {
        // Index 0 is the stage's post-step reading, which every stage takes. Starting
        // one series there and another at idle would put the one-time step in one
        // total and not the other, and the difference would read as a cache effect.
        let index = match reading.tag {
            Tag::Stepped => 0,
            Tag::Growth(n) => n,
            _ => continue,
        };
        by_cram
            .entry(reading.cram_mib)
            .or_default()
            .push((index, anon_mib(reading)));
    }
    by_cram.retain(|_, series| series.len() > 1);
    if by_cram.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "\ngrowth — RssAnon MiB over repeated identical requests, from the post-step reading"
    );
    for (cram, mut series) in by_cram {
        series.sort_by_key(|(n, _)| *n);
        let figures: Vec<String> = series.iter().map(|(_, v)| format!("{v:.1}")).collect();
        let drift = series
            .last()
            .zip(series.first())
            .map(|((_, last), (_, first))| last - first)
            .unwrap_or(0.0);
        let _ = writeln!(
            out,
            "  cram {cram:<17} {}   total {drift:+.1}",
            figures.join(" -> ")
        );
    }
}

/// Whether the one-time step is sized by the prompt or by the generation.
///
/// The axes move independently, so the table reads down a column: holding
/// `n_predict` and growing the prompt separates them, and a diagonal sweep could not.
fn prefill(out: &mut String, observations: &Observations) {
    let mut points: BTreeMap<(usize, u32), (Option<f64>, Option<f64>)> = BTreeMap::new();
    for reading in &observations.readings {
        match reading.tag {
            Tag::PrefillBefore { words, n_predict } => {
                points.entry((words, n_predict)).or_default().0 = Some(anon_mib(reading));
            }
            Tag::PrefillAfter { words, n_predict } => {
                points.entry((words, n_predict)).or_default().1 = Some(anon_mib(reading));
            }
            // The shared stage's step is also the (64, 8) point. Every growth stage
            // takes that same pair, so this reads the shared one specifically
            // rather than whichever happened to be recorded last.
            Tag::Idle if reading.stage == StageKind::Shared => {
                points.entry((STEP_WORDS, STEP_PREDICT)).or_default().0 = Some(anon_mib(reading));
            }
            Tag::Stepped if reading.stage == StageKind::Shared => {
                points.entry((STEP_WORDS, STEP_PREDICT)).or_default().1 = Some(anon_mib(reading));
            }
            _ => {}
        }
    }
    if points.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "\nprefill — is the step sized by the prompt or the generation?\n  \
         {:>6}  {:>9}  {:>10}  {:>10}  {:>8}",
        "words", "n_predict", "before", "after", "step"
    );
    for ((words, n_predict), (before, after)) in points {
        let (Some(before), Some(after)) = (before, after) else {
            continue;
        };
        let _ = writeln!(
            out,
            "  {words:>6}  {n_predict:>9}  {before:>10.1}  {after:>10.1}  {:>+8.1}",
            after - before
        );
    }
}

fn keys<'a>(a: &'a BTreeMap<String, u64>, b: &'a BTreeMap<String, u64>) -> Vec<&'a String> {
    let mut all: Vec<&String> = a.keys().chain(b.keys()).collect();
    all.sort();
    all.dedup();
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{harness::probe::Reading, record::RssSnapshot};

    fn reading(tag: Tag, anon_mib: u64, maps: Option<BTreeMap<String, u64>>) -> Reading {
        Reading {
            tag,
            stage: StageKind::Shared,
            cram_mib: 0,
            rss: RssSnapshot {
                rss_total_kb: anon_mib * 1024,
                rss_anon_kb: anon_mib * 1024,
                rss_file_kb: 0,
                rss_shmem_kb: 0,
            },
            maps,
        }
    }

    /// The step section reports the difference, and names the mapping it landed in.
    /// That attribution is the finding the probe exists to produce — "it grew" is
    /// what the harness already said.
    #[test]
    fn the_step_section_attributes_the_growth() {
        let observations = Observations {
            readings: vec![
                reading(
                    Tag::Idle,
                    1000,
                    Some(BTreeMap::from([("[anon]".to_string(), 100 << 20)])),
                ),
                reading(
                    Tag::Stepped,
                    1300,
                    Some(BTreeMap::from([("[anon]".to_string(), 400 << 20)])),
                ),
            ],
            failures: Vec::new(),
        };
        let out = render(&observations, &[Question::Step], "m.gguf");
        assert!(out.contains("step +300.0"), "{out}");
        assert!(out.contains("+300.0 MiB  [anon]"), "{out}");
    }

    /// A failed stage is named rather than silently missing, so a short table is
    /// never mistaken for a converged one.
    #[test]
    fn failures_are_reported() {
        let observations = Observations {
            readings: Vec::new(),
            failures: vec!["growth (cram 8192): the box started paging".to_string()],
        };
        let out = render(&observations, &Question::ALL, "m.gguf");
        assert!(out.contains("not measured:"), "{out}");
        assert!(out.contains("started paging"), "{out}");
    }
}
