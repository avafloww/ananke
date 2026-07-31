//! Regenerating `tuning.json`, or verifying the committed one against the data.
//!
//! Constants with a deriver are recomputed and their evidence rewritten from what
//! the dataset actually shows. Everything else keeps its declared value and reason —
//! a policy default, a value chosen for reachability, or one held pending another run
//! — because inventing a derivation for those would be exactly the dishonesty this is
//! meant to prevent.
//!
//! Not every constant can be derived from measurement, so each carries a `kind`:
//!
//! - `derived` — computed here from the dataset; the deriver runs and its result is
//!   what ships. If it cannot run, emitting fails.
//! - `policy` — a choice, not a measurement (a runtime's documented default).
//! - `structural` — read from llama.cpp's source, or arithmetic over the graph.
//! - `reachable` — measured, but the spread is wide enough that the value is chosen
//!   so every model lands inside the rolling correction's `[0.8, 1.5]` clamp rather
//!   than to minimise error.
//!
//! The kinds are *declared* rather than inferred: an earlier version read the
//! evidence text and guessed, which filed a structural fact under "reachable" and
//! missed two constants whose review note happened to be lowercase. A constant absent
//! from [`KINDS`] is an error, so adding one forces the question of what justifies it.

use std::collections::BTreeMap;

use ananke_tuning_schema::{Document, Kind};
use serde_json::Number;

use crate::{
    derive::{
        NestedTable, Scalar, Table, baseline, dataset,
        emit::tables::{Tables, write_tables},
        error::Result,
        graph,
        keys::{ArchCardsKey, ArchKey, VariantEnvironmentKey, VariantKey},
        mtp, pinned, recurrent,
        tuning::Tuning,
        vram,
    },
    record::Record,
};

pub mod tables;

pub use tables::check_table_signs;

/// What kind of thing each constant without a deriver is.
pub const KINDS: &[(&str, Kind)] = &[
    // ik_llama's `-amb` default, read from the runtime rather than fitted.
    ("IK_ATTENTION_CHUNK", Kind::Structural),
    // Read from llama.cpp's source or arithmetic over the graph; not fitted.
    ("KV_CACHE_PAD", Kind::Structural),
    ("TENSOR_MASK_BYTES_PER_TOKEN_PAIR", Kind::Structural),
    // A runtime's documented default, mirrored so the reservation and the runtime's
    // own cap are the same number.
    ("DEFAULT_CACHE_RAM_MB", Kind::Policy),
    ("DEFAULT_UBATCH", Kind::Policy),
    // Measured, but with a spread wide enough that the value is chosen so every
    // model lands inside the rolling correction's [0.8, 1.5] clamp rather than to
    // minimise error against any one of them.
    ("PROCESS_BASE_BYTES", Kind::Reachable),
    ("PROCESS_BASE_BYTES_PER_LAYER", Kind::Reachable),
    ("PROCESS_BASE_BYTES_MOE", Kind::Reachable),
    ("PINNED_EXTRA_BYTES", Kind::Reachable),
    (
        "DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16",
        Kind::Reachable,
    ),
    ("QUANTISED_KV_COMPUTE_BYTES_PER_CTX_TOKEN", Kind::Derived),
    ("DRAFT_MODEL_COMPUTE_MIB_PER_1K", Kind::Derived),
    // Has data, but the fit is contested and the value is held.
    ("DRAFT_MODEL_COMPUTE_MIB", Kind::Reachable),
    ("MTP_HOST_BYTES_EMBEDDED", Kind::Derived),
    ("MTP_HOST_BYTES_SEPARATE_DRAFT", Kind::Derived),
    ("MTP_HOST_MIB_PER_1K", Kind::Derived),
    // Read straight out of llama.cpp's own `rs_seq` field.
    ("SPEC_RECURRENT_ROLLBACK_DEPTH", Kind::Derived),
    // The worst of two measured vision configurations; see the deriver for why it is
    // not yet a rate.
    ("MMPROJ_GRAPH_BYTES", Kind::Reachable),
];

/// What `emit` produced, and everything it could not.
#[derive(Debug, Clone)]
pub struct Emitted {
    /// The document as it should be committed.
    pub document: Document,
    /// Constants whose value moved, as `NAME: old -> new`.
    pub changed: Vec<String>,
    /// Derivations that could not run. A non-empty list is a failure.
    pub failed: Vec<String>,
    /// Context for reading the numbers — how much of the dataset predates the
    /// installed runtime — and never a verdict.
    pub notes: Vec<String>,
    /// How many measurements survived de-duplication.
    pub measurements: usize,
    /// Kept on the outcome because `check_arena_model` needs them: the arena the
    /// derived constants were fitted against is only complete with these two terms.
    pub no_flash_attn_rates: Option<Table<VariantKey>>,
    pub quantised_cache_rates: Option<Table<ArchKey>>,
}

/// Recompute every derived constant and table, and return the document to commit.
///
/// `compute_model` is deliberately untouched: it is fitted by
/// [`crate::compute_model`], and carrying the committed section through unchanged is
/// what lets this half be checked on its own.
pub fn emit(rows: &[Record], tuning_text: &str) -> Result<Emitted> {
    let mut document = parse(tuning_text)?;
    let mut changed = Vec::new();
    let mut failed = Vec::new();
    let mut notes = Vec::new();

    // Superseded rows go first, so nothing downstream fits two programs at once. The
    // check then runs on what survives, where a disagreement can no longer be
    // explained by a stale duplicate and so is a real problem.
    let before = rows.len();
    let rows = dataset::latest_per_cell(rows);
    let superseded = before - rows.len();
    if superseded > 0 {
        notes.push(format!(
            "{superseded} row(s) superseded by a later measurement of the same cell \
             under a newer runtime"
        ));
    }
    notes.extend(dataset::report_stale_builds(&rows));
    if let Err(error) = dataset::check_runtime_builds(&rows, dataset::BUILD_TOLERANCE) {
        failed.push(format!("runtime builds: {error}"));
    }

    // Derivers that read a constant read it from `live`, which carries what *this*
    // run has derived so far, not what the last one committed. Reading `committed`
    // instead makes the output a function of (dataset, previous document): a value
    // that moves takes a second run to settle, and nothing says which run you are
    // looking at.
    //
    // That makes the order below load-bearing, and it is a genuine order rather than
    // a cycle — every constant read here is derived by a pass that excludes it:
    //
    //   ik_moe_rates, MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD  (the arena's MoE terms)
    //     -> no_flash_attn_rates -> baseline_offset
    //
    // The arena charges the MoE terms when it charges anything, but both are derived
    // with the MoE term switched off, so neither reads itself. `baseline_offset`
    // subtracts `no_flash_attn_rates`, and without them it silently subtracts zero and
    // folds a per-token arena term into a flat baseline.
    let mut live = Tuning::of(&document);

    let mut ik_moe: Option<Table<ArchCardsKey>> = None;
    match graph::ik_moe_per_nembd(&rows, &live) {
        Ok((_scalar, table)) => {
            live.set_ik_moe_rates(tables::ik_moe_rates(&table));
            ik_moe = Some(table);
        }
        Err(error) => failed.push(format!("ik MoE rates: cannot derive — {error}")),
    }

    // The arena's other MoE term, hoisted out of the scalar loop for the same reason:
    // `no_flash_attn_rates` and `baseline_offset` both charge it through the arena.
    match graph::mainline_tensor_moe(&rows, &live) {
        Ok(scalar) => thread(
            &mut live,
            "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD",
            scalar.value,
            &mut failed,
        ),
        Err(error) => failed.push(format!(
            "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD: cannot derive — {error}"
        )),
    }

    // Likewise the E-variant's per-layer term: the arena subtracts it, and so does the
    // flash-attention residual.
    match pinned::gemma_e_per_layer_token(&rows, &live) {
        Ok(scalar) => thread(
            &mut live,
            "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN",
            scalar.value,
            &mut failed,
        ),
        Err(error) => failed.push(format!(
            "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN: cannot derive — {error}"
        )),
    }

    // And the per-device baseline, which `baseline_offset` subtracts along with the
    // rest of the process-baseline model. It takes no tuning of its own, so hoisting
    // it costs nothing and is the difference between the offset being a residual over
    // this run's model and over the last one's.
    match baseline::per_device_bytes(&rows) {
        Ok(scalar) => thread(
            &mut live,
            "PROCESS_BASE_BYTES_PER_DEVICE",
            scalar.value,
            &mut failed,
        ),
        Err(error) => failed.push(format!(
            "PROCESS_BASE_BYTES_PER_DEVICE: cannot derive — {error}"
        )),
    }

    let slot_scaling = mtp::mtp_slot_scaling(&rows);

    let mut checkpoint: Option<Table<VariantKey>> = None;
    match baseline::checkpoint_headroom(&rows) {
        Ok(table) => checkpoint = Some(table),
        Err(error) => failed.push(format!("checkpoint headroom: cannot derive — {error}")),
    }

    let mut per_slot: Option<Table<ArchKey>> = None;
    match baseline::per_slot_bytes(&rows) {
        Ok(table) => per_slot = Some(table),
        Err(error) => failed.push(format!("per-slot host bytes: cannot derive — {error}")),
    }

    let mut table_less: Option<NestedTable> = None;
    match vram::table_less_observations(&rows) {
        Ok(table) => table_less = Some(table),
        Err(error) => failed.push(format!(
            "table-less compute observations: cannot derive — {error}"
        )),
    }

    let mut score: Option<Table<ArchKey>> = None;
    match vram::no_flash_attn_score(&rows) {
        Ok(table) => score = Some(table),
        Err(error) => failed.push(format!(
            "unfused-attention score rates: cannot derive — {error}"
        )),
    }

    // `None` for the quantised rates: that table is filled by a deriver running
    // *after* this one, so the term is absent from the residual the committed rates
    // were fitted against.
    let mut no_fa: Option<Table<VariantKey>> = None;
    match pinned::no_flash_attn_rates(&rows, &live, None) {
        Ok(table) => no_fa = Some(table),
        Err(error) => failed.push(format!("no-flash-attention rates: cannot derive — {error}")),
    }

    let mut baseline_table: Option<Table<VariantEnvironmentKey>> = None;
    let empty = BTreeMap::new();
    let no_fa_rates = no_fa.as_ref().map(|t| &t.by_key).unwrap_or(&empty);
    match baseline::baseline_offset(&rows, &live, no_fa_rates) {
        Ok((_scalar, table)) => baseline_table = Some(table),
        Err(error) => failed.push(format!("baseline offset: cannot derive — {error}")),
    }

    let mut tensor_base: Option<Table<ArchKey>> = None;
    match baseline::tensor_split_baseline(&rows, &live) {
        Ok((_scalar, table)) => tensor_base = Some(table),
        Err(error) => failed.push(format!("tensor-split baseline: cannot derive — {error}")),
    }

    let mut quantised: Option<Table<ArchKey>> = None;
    match pinned::quantised_cache_bytes(&rows) {
        Ok((_scalar, table)) => quantised = Some(table),
        Err(error) => failed.push(format!("quantised-cache rates: cannot derive — {error}")),
    }

    for (name, deriver) in derivers() {
        let Some(entry) = document.constants.get_mut(name) else {
            failed.push(format!("{name}: in DERIVERS but not in tuning.json"));
            continue;
        };
        let derived = match deriver(&rows, &live) {
            Ok(scalar) => scalar,
            Err(error) => {
                failed.push(format!("{name}: cannot derive — {error}"));
                continue;
            }
        };
        if entry.value.as_i64() != Some(derived.value) {
            changed.push(format!("{name}: {} -> {}", entry.value, derived.value));
        }
        entry.value = Number::from(derived.value);
        entry.evidence = derived.evidence;
        entry.kind = Kind::Derived;
        thread(&mut live, name, derived.value, &mut failed);
    }

    let derived_names: Vec<&str> = derivers().iter().map(|(name, _)| *name).collect();
    for (name, entry) in &mut document.constants {
        if derived_names.contains(&name.as_str()) {
            continue;
        }
        match KINDS.iter().find(|(known, _)| known == name) {
            Some((_, kind)) => entry.kind = *kind,
            None => failed.push(format!("{name}: no declared kind — add it to KINDS")),
        }
    }

    let mut draft_compute = None;
    match mtp::mtp_draft_compute(&rows) {
        Ok(fit) => draft_compute = Some(fit),
        Err(error) => failed.push(format!("MTP draft compute: cannot derive — {error}")),
    }

    write_tables(
        &mut document,
        Tables {
            baseline: baseline_table.as_ref(),
            tensor_base: tensor_base.as_ref(),
            draft_compute: draft_compute.as_ref(),
            slot_scaling: &slot_scaling,
            checkpoint: checkpoint.as_ref(),
            per_slot: per_slot.as_ref(),
            table_less: table_less.as_ref(),
            score: score.as_ref(),
            no_fa: no_fa.as_ref(),
            quantised: quantised.as_ref(),
            ik_moe: ik_moe.as_ref(),
        },
    );

    document.measurements = rows.len() as u64;
    Ok(Emitted {
        document,
        changed,
        failed,
        notes,
        measurements: rows.len(),
        no_flash_attn_rates: no_fa,
        quantised_cache_rates: quantised,
    })
}

/// `emit`, plus the two assertions made just before writing the file.
///
/// They guard the *write* rather than the check: a `--check` run is comparing two
/// documents and does not need the arena model re-verified, while a run that replaces
/// the committed constants does — a drifted arena model means every rate derived from
/// a residual over it was fitted against the wrong number.
pub fn emit_write(rows: &[Record], tuning_text: &str) -> Result<Emitted> {
    let emitted = emit(rows, tuning_text)?;
    let committed = Tuning::parse(tuning_text)
        .map_err(|e| crate::derive::error::DeriveError::malformed(e.to_string()))?;
    let rows = dataset::latest_per_cell(rows);
    let no_fa_fallback = BTreeMap::new();
    let quant_fallback = BTreeMap::new();
    crate::derive::arena::check_arena_model(
        &rows,
        &committed,
        emitted
            .no_flash_attn_rates
            .as_ref()
            .map(|t| &t.by_key)
            .unwrap_or(&no_fa_fallback),
        emitted
            .quantised_cache_rates
            .as_ref()
            .map(|t| &t.by_key)
            .unwrap_or(&quant_fallback),
        crate::derive::arena::ARENA_TOLERANCE_MIB,
    )?;
    check_table_signs(&emitted.document)?;
    Ok(emitted)
}

/// Verify the committed document against the dataset.
///
/// Returns `Ok(())` when it matches. The error names what moved, which is the whole
/// output CI needs: a value that differs means one of the two implementations is
/// wrong, and a value that matches while the evidence differs means the inputs moved
/// under a constant that happened to land in the same place.
pub fn emit_check(rows: &[Record], tuning_text: &str) -> Result<Emitted> {
    let emitted = emit(rows, tuning_text)?;
    let committed = parse(tuning_text)?;
    // A deriver that fails leaves its constant exactly as committed, so the documents
    // match and the check would otherwise pass — reporting agreement with the data
    // for a constant nothing re-derived. The failure is the finding; comparing
    // documents cannot see it.
    if !emitted.failed.is_empty() {
        return Err(crate::derive::error::DeriveError::disagreement(format!(
            "{} constant(s) could not be derived, so the committed values were not \
             checked against the data: {}",
            emitted.failed.len(),
            emitted.failed.join("; "),
        )));
    }
    if committed == emitted.document {
        return Ok(emitted);
    }
    let mut detail = emitted.changed.clone();
    detail.extend(emitted.failed.clone());
    if detail.is_empty() {
        detail.push("evidence text differs; re-run without --check to refresh".to_string());
    }
    Err(crate::derive::error::DeriveError::disagreement(format!(
        "tuning.json does not match the dataset ({} measurements): {}",
        emitted.measurements,
        detail.join("; "),
    )))
}

/// The committed document, or the parse error as a derivation failure.
fn parse(tuning_text: &str) -> Result<Document> {
    serde_json::from_str(tuning_text)
        .map_err(|e| crate::derive::error::DeriveError::malformed(e.to_string()))
}

/// One deriver per scalar constant, in the order the document declares them.
pub type ScalarDeriver = fn(&[Record], &Tuning) -> Result<Scalar>;

/// Public so [`crate::crossval`] can refit each of them on a subset: a fold has to
/// run the same deriver the shipped constant came from, or it is validating
/// something else.
pub fn derivers() -> Vec<(&'static str, ScalarDeriver)> {
    vec![
        ("MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD", |rows, tuning| {
            graph::mainline_tensor_moe(rows, tuning)
        }),
        ("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN", |rows, tuning| {
            pinned::gemma_e_per_layer_token(rows, tuning)
        }),
        ("PROCESS_BASE_BYTES_PER_DEVICE", |rows, _| {
            baseline::per_device_bytes(rows)
        }),
        ("MAINLINE_LAYER_SPLIT_MASK_COPIES", |rows, tuning| {
            graph::layer_split_copies(rows, tuning)
        }),
        ("IK_OP_OFFLOAD_MIN_BATCH", |rows, tuning| {
            graph::offload_min_batch(rows, tuning)
        }),
        ("DRAFT_MODEL_COMPUTE_MIB_PER_1K", |rows, _| {
            mtp::draft_compute_slope(rows)
        }),
        ("MTP_HOST_BYTES_EMBEDDED", |rows, _| {
            mtp::mtp_host_embedded(rows)
        }),
        ("MTP_HOST_BYTES_SEPARATE_DRAFT", |rows, _| {
            mtp::mtp_host_separate(rows)
        }),
        ("MTP_HOST_MIB_PER_1K", |rows, _| mtp::mtp_host_slope(rows)),
        ("SPEC_RECURRENT_ROLLBACK_DEPTH", |rows, _| {
            recurrent::spec_rollback_depth(rows)
        }),
        ("MMPROJ_GRAPH_BYTES", |rows, _| vram::mmproj_graph(rows)),
        ("MTP_UNACCOUNTED_MIB_PER_DEVICE", |rows, _| {
            mtp::mtp_unaccounted(rows)
        }),
    ]
}

/// Hand this run's derived value to the derivers that read it, recording a name the
/// document does not declare as a failure.
///
/// The ordering above is only worth writing down if a broken link in it is visible:
/// a rename that lands nowhere leaves the downstream derivers reading the committed
/// value, which produces a document that looks settled and is fitted against the
/// previous run.
fn thread(live: &mut Tuning, name: &str, value: i64, failed: &mut Vec<String>) {
    if let Err(error) = live.set_constant(name, value) {
        failed.push(format!(
            "{error}, so this run's derived value did not reach the derivers that \
             read it"
        ));
    }
}
