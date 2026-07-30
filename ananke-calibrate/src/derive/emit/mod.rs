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

use serde_json::{Value, json};

use crate::{
    derive::{
        NestedTable, Scalar, Table, baseline, dataset,
        emit::tables::{Tables, write_tables},
        error::Result,
        graph, mtp, pinned, recurrent,
        tuning::Tuning,
        vram,
    },
    record::Record,
};

pub mod tables;

pub use tables::{SIGNED_TABLES, check_table_signs};

/// What kind of thing each constant without a deriver is.
pub const KINDS: &[(&str, &str)] = &[
    // ik_llama's `-amb` default, read from the runtime rather than fitted.
    ("IK_ATTENTION_CHUNK", "structural"),
    // Read from llama.cpp's source or arithmetic over the graph; not fitted.
    ("KV_CACHE_PAD", "structural"),
    ("TENSOR_MASK_BYTES_PER_TOKEN_PAIR", "structural"),
    // A runtime's documented default, mirrored so the reservation and the runtime's
    // own cap are the same number.
    ("DEFAULT_CACHE_RAM_MB", "policy"),
    ("DEFAULT_UBATCH", "policy"),
    // Measured, but with a spread wide enough that the value is chosen so every
    // model lands inside the rolling correction's [0.8, 1.5] clamp rather than to
    // minimise error against any one of them.
    ("PROCESS_BASE_BYTES", "reachable"),
    ("PROCESS_BASE_BYTES_PER_LAYER", "reachable"),
    ("PROCESS_BASE_BYTES_MOE", "reachable"),
    ("PINNED_EXTRA_BYTES", "reachable"),
    ("DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16", "reachable"),
    ("QUANTISED_KV_COMPUTE_BYTES_PER_CTX_TOKEN", "derived"),
    ("DRAFT_MODEL_COMPUTE_MIB_PER_1K", "derived"),
    // Has data, but the fit is contested and the value is held.
    ("DRAFT_MODEL_COMPUTE_MIB", "reachable"),
    ("MTP_HOST_BYTES_EMBEDDED", "derived"),
    ("MTP_HOST_BYTES_SEPARATE_DRAFT", "derived"),
    ("MTP_HOST_MIB_PER_1K", "derived"),
    // Read straight out of llama.cpp's own `rs_seq` field.
    ("SPEC_RECURRENT_ROLLBACK_DEPTH", "derived"),
    // The worst of two measured vision configurations; see the deriver for why it is
    // not yet a rate.
    ("MMPROJ_GRAPH_BYTES", "reachable"),
];

/// What `emit` produced, and everything it could not.
#[derive(Debug, Clone)]
pub struct Emitted {
    /// The document as it should be committed.
    pub document: Value,
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
    pub no_flash_attn_rates: Option<Table>,
    pub quantised_cache_rates: Option<Table>,
}

/// Recompute every derived constant and table, and return the document to commit.
///
/// `compute_model` is deliberately untouched: it is fitted by
/// `scripts/calibration/compute_model.py` and its Rust counterpart, and carrying the
/// committed section through unchanged is what lets this half be checked on its own.
pub fn emit(rows: &[Record], tuning_text: &str) -> Result<Emitted> {
    let committed = Tuning::parse(tuning_text)
        .map_err(|e| crate::derive::error::DeriveError::malformed(e.to_string()))?;
    let mut document = committed.document().clone();
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

    // The dependency order is spelled out because it is load-bearing:
    //
    //   no_flash_attn_rates -> baseline_offset
    //
    // The latter subtracts the former's per-architecture rates, and without them it
    // silently subtracts zero and folds a per-token arena term into a flat baseline.
    let mut ik_moe: Option<Table> = None;
    match graph::ik_moe_per_nembd(&rows, &committed) {
        Ok((_scalar, table)) => ik_moe = Some(table),
        Err(error) => failed.push(format!("ik MoE rates: cannot derive — {error}")),
    }

    let slot_scaling = mtp::mtp_slot_scaling(&rows);

    let mut checkpoint: Option<Table> = None;
    match baseline::checkpoint_headroom(&rows) {
        Ok(table) => checkpoint = Some(table),
        Err(error) => failed.push(format!("checkpoint headroom: cannot derive — {error}")),
    }

    let mut per_slot: Option<Table> = None;
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

    let mut score: Option<Table> = None;
    match vram::no_flash_attn_score(&rows) {
        Ok(table) => score = Some(table),
        Err(error) => failed.push(format!(
            "unfused-attention score rates: cannot derive — {error}"
        )),
    }

    // `None` for the quantised rates, matching the Python: its table is filled by a
    // deriver that runs *after* this one, so the term is absent from the residual the
    // committed rates were fitted against.
    let mut no_fa: Option<Table> = None;
    match pinned::no_flash_attn_rates(&rows, &committed, None) {
        Ok(table) => no_fa = Some(table),
        Err(error) => failed.push(format!("no-flash-attention rates: cannot derive — {error}")),
    }

    let mut baseline_table: Option<Table> = None;
    let empty = BTreeMap::new();
    let no_fa_rates = no_fa.as_ref().map(|t| &t.by_arch).unwrap_or(&empty);
    match baseline::baseline_offset(&rows, &committed, no_fa_rates) {
        Ok((_scalar, table)) => baseline_table = Some(table),
        Err(error) => failed.push(format!("baseline offset: cannot derive — {error}")),
    }

    let mut tensor_base: Option<Table> = None;
    match baseline::tensor_split_baseline(&rows, &committed) {
        Ok((_scalar, table)) => tensor_base = Some(table),
        Err(error) => failed.push(format!("tensor-split baseline: cannot derive — {error}")),
    }

    let mut quantised: Option<Table> = None;
    match pinned::quantised_cache_bytes(&rows) {
        Ok((_scalar, table)) => quantised = Some(table),
        Err(error) => failed.push(format!("quantised-cache rates: cannot derive — {error}")),
    }

    for (name, deriver) in derivers() {
        let entry = document.get("constants").and_then(|c| c.get(name)).cloned();
        let Some(mut entry) = entry else {
            failed.push(format!("{name}: in DERIVERS but not in tuning.json"));
            continue;
        };
        let derived = match deriver(&rows, &committed) {
            Ok(scalar) => scalar,
            Err(error) => {
                failed.push(format!("{name}: cannot derive — {error}"));
                continue;
            }
        };
        if entry.get("value").and_then(Value::as_i64) != Some(derived.value) {
            let old = entry.get("value").cloned().unwrap_or(Value::Null);
            changed.push(format!("{name}: {old} -> {}", derived.value));
        }
        entry["value"] = json!(derived.value);
        entry["evidence"] = json!(derived.evidence);
        entry["kind"] = json!("derived");
        document["constants"][name] = entry;
    }

    let derived_names: Vec<&str> = derivers().iter().map(|(name, _)| *name).collect();
    let names: Vec<String> = document["constants"]
        .as_object()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default();
    for name in names {
        if derived_names.contains(&name.as_str()) {
            continue;
        }
        match KINDS.iter().find(|(known, _)| *known == name) {
            Some((_, kind)) => document["constants"][&name]["kind"] = json!(kind),
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

    document["measurements"] = json!(rows.len());
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

/// `emit`, plus the two assertions the Python makes just before writing the file.
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
    let empty = BTreeMap::new();
    crate::derive::arena::check_arena_model(
        &rows,
        &committed,
        emitted
            .no_flash_attn_rates
            .as_ref()
            .map(|t| &t.by_arch)
            .unwrap_or(&empty),
        emitted
            .quantised_cache_rates
            .as_ref()
            .map(|t| &t.by_arch)
            .unwrap_or(&empty),
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
    let committed: Value = serde_json::from_str(tuning_text)
        .map_err(|e| crate::derive::error::DeriveError::malformed(e.to_string()))?;
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

/// One deriver per scalar constant, in the order the Python declares them.
type ScalarDeriver = fn(&[Record], &Tuning) -> Result<Scalar>;

fn derivers() -> Vec<(&'static str, ScalarDeriver)> {
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
