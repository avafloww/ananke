//! Generate campaign plans for the measurement harness.
//!
//! Plans are generated rather than hand-written so that the factor coverage is
//! stated once, as code, and can be re-derived or extended later. What each sweep
//! exists to close is written on the sweep itself: `coverage.rs` audits the
//! dataset for regimes measured at a single point in an axis their rule depends
//! on, and the sweeps here are the answers to that audit. The two are a pair, and
//! neither is much use without the other.
//!
//! Model paths come from `$LLM_DIR` so a plan is portable to another machine with
//! the same library — see [`library::Library`].
//!
//! ```text
//! cargo run -p ananke-calibrate --bin plan -- curves > curves.json
//! python measure.py --out data/calibration/curves.csv --plan curves.json
//! ```
//!
//! The output type is [`ananke_measure::record::Factors`], the harness's own
//! factor set: a plan is a list of cells the harness will run, so there is one
//! definition of what a cell is and adding a factor cannot leave the planner
//! behind.

pub mod library;
pub mod phases;
pub mod refinement;
pub mod thin_axes;

use std::{collections::HashMap, path::Path};

use ananke_measure::record::Factors;
use serde::Serialize;
use serde_json::ser::{PrettyFormatter, Serializer};

pub use crate::plan::library::Library;
use crate::plan::library::runtime_name;

/// One question, answered by the cells it builds.
pub type Sweep = fn(&Library) -> Vec<Factors>;

/// What each cell is for.
///
/// These are questions, not a schedule — [`all_cells`] merges them into one
/// ordered run, and a cell wanted by two questions is measured once and tagged
/// with both.
pub const QUESTIONS: &[(&str, Sweep)] = &[
    ("noise", phases::noise),
    ("factor-screen", phases::factor_screen),
    ("model-baseline", phases::model_baseline),
    ("curves", phases::curves),
    ("fork", phases::fork),
    ("switches", phases::switches),
    ("device-scaling", refinement::device_scaling),
    ("interior", refinement::interior),
    ("interactions", refinement::interactions),
    ("replication", refinement::replication),
    ("concurrency", refinement::concurrency),
    ("review-followup", refinement::review_followup),
    ("mtp-slots", refinement::mtp_slots),
    ("slot-batch", thin_axes::slot_batch),
    ("concurrency-models", thin_axes::concurrency_models),
    ("loose-ends", thin_axes::loose_ends),
    ("single-card", thin_axes::single_card),
    ("sparse-switches", thin_axes::sparse_switches),
    ("checkpoint-steady", thin_axes::checkpoint_steady),
    ("second-context", thin_axes::second_context),
    ("flash-attention", thin_axes::flash_attention),
    ("holdout", phases::holdout),
];

/// The cells one question asks for, or `None` if nothing answers to that name.
pub fn cells_for(question: &str, lib: &Library) -> Option<Vec<Factors>> {
    QUESTIONS
        .iter()
        .find(|(name, _)| *name == question)
        .map(|(_, build)| build(lib))
}

/// Every question's name, in the order a usage message should list them.
pub fn question_names() -> Vec<&'static str> {
    let mut names: Vec<_> = QUESTIONS.iter().map(|(name, _)| *name).collect();
    names.sort_unstable();
    names
}

/// Every configuration worth measuring, once each, in the cheapest order.
///
/// The questions are separate — a noise floor, a per-model baseline, a context
/// curve, a fork comparison, growth — but they are not separate *schedules*.
/// Running them as separate passes reloads each model once per question, and a
/// reload is the single most expensive thing here: the 205 GiB production quant
/// cannot even stay in the page cache alongside anything else, so every revisit
/// pays full disk cost again.
///
/// So the questions become tags and the schedule becomes one list, ordered so
/// consecutive cells disturb as little as possible.
pub fn all_cells(lib: &Library) -> Vec<Factors> {
    let mut cells = merged_cells(lib);
    let sizes = ModelSizes::measure(&cells);
    order_by_disturbance(&mut cells, &sizes);
    cells
}

/// Every question's cells, de-duplicated and tagged, in the order the questions
/// are listed.
///
/// Kept separate from the ordering because the ordering needs the models' sizes
/// off disk and this half is pure.
pub fn merged_cells(lib: &Library) -> Vec<Factors> {
    let mut cells: Vec<Factors> = Vec::new();
    let mut seen: HashMap<Factors, usize> = HashMap::new();
    for (name, build) in QUESTIONS {
        for cell in build(lib) {
            match seen.get(&identity(&cell)) {
                None => {
                    seen.insert(identity(&cell), cells.len());
                    cells.push(Factors {
                        purpose: vec![(*name).to_owned()],
                        ..cell
                    });
                }
                Some(&at) => {
                    let purpose = &mut cells[at].purpose;
                    if !purpose.iter().any(|p| p == name) {
                        purpose.push((*name).to_owned());
                    }
                }
            }
        }
    }
    cells
}

/// Order the run so consecutive cells disturb as little as possible: all of a
/// model's work happens while its weights are hot, and models run smallest first,
/// because the largest evicts everything behind it on the way past.
///
/// A stable sort, so cells the key cannot separate keep the order the questions
/// asked for them in.
pub fn order_by_disturbance(cells: &mut [Factors], sizes: &ModelSizes) {
    cells.sort_by(|a, b| disturbance(a, sizes).cmp(&disturbance(b, sizes)));
}

/// A plan, as the harness reads it.
///
/// One space of indentation rather than the usual two, matching what the harness
/// has always been fed.
pub fn to_json(cells: &[Factors]) -> String {
    let mut out = Vec::new();
    let mut ser = Serializer::with_formatter(&mut out, PrettyFormatter::with_indent(b" "));
    cells
        .serialize(&mut ser)
        .expect("a plan serialises to a string");
    String::from_utf8(out).expect("serde_json emits UTF-8")
}

/// Total bytes across each model's shards, so the run can go smallest first.
///
/// Cached per path because the same model appears in dozens of cells and the sort
/// asks for its size once per comparison.
pub struct ModelSizes {
    sizes: HashMap<String, u64>,
}

impl ModelSizes {
    /// Stat every model the plan mentions.
    pub fn measure(cells: &[Factors]) -> Self {
        let mut sizes = HashMap::new();
        for cell in cells {
            if cell.model.is_empty() || sizes.contains_key(&cell.model) {
                continue;
            }
            sizes.insert(cell.model.clone(), total_size(Path::new(&cell.model)));
        }
        Self { sizes }
    }

    fn get(&self, model: &str) -> u64 {
        if model.is_empty() {
            return 0;
        }
        self.sizes.get(model).copied().unwrap_or(UNREADABLE)
    }
}

/// What an unreadable model weighs, so it sorts last instead of first.
///
/// A path that does not resolve on this box is either a typo or a model that is
/// not downloaded yet; either way the run should reach it after everything it can
/// actually measure.
const UNREADABLE: u64 = 1 << 62;

/// The cell's identity: every factor except the label and the purpose tags.
///
/// The label names the cell and the purpose says why it was wanted, but two cells
/// with the same flags are the same measurement whatever they are called, and
/// measuring one configuration twice under two names is pure waste.
///
/// Structural equality over the whole factor set is deliberately the key rather
/// than a hand-listed subset. This is the one place a quietly-omitted field would
/// silently merge two genuinely different measurements, and a field missing from a
/// mapping has produced four wrong constants in this campaign.
fn identity(cell: &Factors) -> Factors {
    Factors {
        label: String::new(),
        purpose: Vec::new(),
        ..cell.clone()
    }
}

/// What it costs to move from one cell to the next.
///
/// Changing the model is the expensive move, so it is the outermost key and models
/// are ordered by size. Everything below it — the runtime, then the load path,
/// then placement, then the cache and batch knobs — is progressively cheaper to
/// vary while the same weights stay resident.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Disturbance<'a> {
    size: u64,
    model: &'a str,
    runtime: &'static str,
    no_mmap: bool,
    rtr: bool,
    gpus: &'a str,
    split: &'a str,
    ngl: u32,
    spec_type: &'a str,
    draft: &'a str,
    mmproj: &'a str,
    ctx: u32,
    ubatch: u32,
    parallel: u32,
    kv_type: &'a str,
    flash_attn: &'a str,
    /// Served cells first: an idle process is the odd one out, and reaching it
    /// last keeps the served ones adjacent.
    idle: bool,
    bench: bool,
}

fn disturbance<'a>(cell: &'a Factors, sizes: &ModelSizes) -> Disturbance<'a> {
    Disturbance {
        size: sizes.get(&cell.model),
        model: &cell.model,
        runtime: runtime_name(cell.runtime),
        no_mmap: cell.no_mmap,
        rtr: cell.rtr,
        gpus: &cell.gpus,
        split: cell.split.as_deref().unwrap_or(""),
        ngl: cell.ngl,
        spec_type: cell.spec_type.as_deref().unwrap_or(""),
        draft: cell.draft.as_deref().unwrap_or(""),
        mmproj: cell.mmproj.as_deref().unwrap_or(""),
        ctx: cell.ctx,
        ubatch: cell.ubatch,
        parallel: cell.parallel,
        kv_type: &cell.kv_type,
        flash_attn: &cell.flash_attn,
        idle: !cell.served,
        bench: cell.bench,
    }
}

/// Total bytes across a model's shards; unreadable sorts last.
fn total_size(first: &Path) -> u64 {
    let Some(name) = first.file_name().and_then(|n| n.to_str()) else {
        return UNREADABLE;
    };
    if !first.exists() {
        return UNREADABLE;
    }
    let Some(stem) = shard_stem(name) else {
        return first.metadata().map(|m| m.len()).unwrap_or(UNREADABLE);
    };
    let Some(parent) = first.parent() else {
        return UNREADABLE;
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return UNREADABLE;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let shard = entry.file_name();
        let Some(shard) = shard.to_str() else {
            continue;
        };
        if is_shard_of(shard, stem) {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    total
}

/// The stem of a sharded GGUF's file name, or `None` when the name is not a shard.
///
/// A sharded model is named `<stem>-00001-of-00004.gguf`, and only the first shard
/// is ever put in a plan, so the rest have to be found by name.
fn shard_stem(name: &str) -> Option<&str> {
    let base = name.strip_suffix(".gguf")?;
    let (rest, count) = base.rsplit_once("-of-")?;
    let (stem, index) = rest.rsplit_once('-')?;
    (is_five_digits(index) && is_five_digits(count)).then_some(stem)
}

/// Whether a file name is one of `stem`'s shards — the `{stem}-*-of-*.gguf` the
/// Python planner globbed for.
fn is_shard_of(name: &str, stem: &str) -> bool {
    name.strip_suffix(".gguf")
        .and_then(|base| base.strip_prefix(stem))
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|rest| rest.contains("-of-"))
}

fn is_five_digits(text: &str) -> bool {
    text.len() == 5 && text.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn shard_names_round_trip() {
        let stem = shard_stem("Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf");
        assert_eq!(stem, Some("Laguna-S-2.1-UD-IQ4_NL"));
        assert!(is_shard_of(
            "Laguna-S-2.1-UD-IQ4_NL-00003-of-00003.gguf",
            "Laguna-S-2.1-UD-IQ4_NL"
        ));
        assert!(!is_shard_of(
            "Laguna-S-2.1-UD-IQ4_NL-mmproj.gguf",
            "Laguna-S-2.1-UD-IQ4_NL"
        ));
    }

    #[test]
    fn an_unsharded_name_has_no_stem() {
        assert_eq!(shard_stem("gemma-3-27b-it-abliterated.q4_k_m.gguf"), None);
        assert_eq!(shard_stem("model-1-of-3.gguf"), None);
    }

    /// A cell wanted by two questions is one measurement carrying both tags.
    #[test]
    fn merging_tags_rather_than_duplicating() {
        let cells = merged_cells(&Library::rooted("/fake/llm"));
        let identities: HashSet<_> = cells.iter().map(identity).collect();
        assert_eq!(identities.len(), cells.len(), "every cell is distinct");
        assert!(
            cells.iter().any(|c| c.purpose.len() > 1),
            "some cell answers more than one question"
        );
    }
}
