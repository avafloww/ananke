//! Every deriver, held against the committed `tuning.json`.
//!
//! The committed document is the oracle: `emit --check` passes against the same
//! dataset, so a deriver that disagrees here means one of the two is wrong.
//! Integers are compared exactly — a loosened assertion would hide precisely the
//! drift these tests exist to catch.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use ananke_calibrate::{
    derive::{
        arena, baseline, dataset, emit, graph, mtp, pinned, recurrent,
        shape::{query_head_count, variant_key},
        stats::pad,
        tuning::Tuning,
        vram,
    },
    record::Record,
};
use serde_json::Value;

/// The dataset, de-duplicated the way `emit` does before anything reads it.
fn rows() -> Arc<Vec<Record>> {
    static ROWS: OnceLock<Arc<Vec<Record>>> = OnceLock::new();
    ROWS.get_or_init(|| {
        let path = format!(
            "{}/../../data/measurements.ndjson",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(path).expect("the dataset is committed");
        let loaded = dataset::load(&text).expect("every row parses");
        Arc::new(dataset::latest_per_cell(&loaded))
    })
    .clone()
}

fn tuning_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        let path = format!(
            "{}/../../../crates/tuning/tuning.json",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(path).expect("tuning.json is committed")
    })
}

fn tuning() -> &'static Tuning {
    static TUNING: OnceLock<Tuning> = OnceLock::new();
    TUNING.get_or_init(|| Tuning::parse(tuning_text()).expect("tuning.json is valid JSON"))
}

fn committed() -> &'static Value {
    static DOCUMENT: OnceLock<Value> = OnceLock::new();
    DOCUMENT.get_or_init(|| serde_json::from_str(tuning_text()).expect("valid JSON"))
}

/// A committed scalar constant.
fn constant(name: &str) -> i64 {
    committed()["constants"][name]["value"]
        .as_i64()
        .unwrap_or_else(|| {
            panic!("{name} is missing from the committed constants");
        })
}

/// A committed per-architecture table, `default` included under that name so a
/// deriver's fallback is checked too.
fn table(name: &str) -> BTreeMap<String, i64> {
    let entry = &committed()[name];
    let mut out: BTreeMap<String, i64> = entry["by_arch"]
        .as_object()
        .unwrap_or_else(|| panic!("{name} has no by_arch"))
        .iter()
        .map(|(arch, value)| (arch.clone(), value.as_i64().expect("an integer rate")))
        .collect();
    if let Some(default) = entry.get("default").and_then(Value::as_i64) {
        out.insert("default".to_string(), default);
    }
    out
}

/// The same shape a deriver's table takes, so the comparison is one assertion.
fn derived(table: &ananke_calibrate::derive::Table) -> BTreeMap<String, i64> {
    let mut out = table.by_arch.clone();
    out.insert("default".to_string(), table.worst());
    out
}

// --- The whole document -------------------------------------------------------

#[test]
fn the_committed_document_matches_the_dataset() {
    let emitted = emit::emit_check(&rows(), tuning_text()).expect("emit --check passes");
    assert_eq!(
        emitted.measurements, 582,
        "the dataset lost or gained cells"
    );
    assert!(emitted.failed.is_empty(), "{:?}", emitted.failed);
}

#[test]
fn the_arena_model_still_reproduces_the_measurements() {
    // Runs the pre-write assertions too: a drifted arena model means every rate
    // derived from a residual over it was fitted against the wrong number.
    emit::emit_write(&rows(), tuning_text()).expect("the arena model holds");
}

#[test]
fn the_recurrent_formula_still_reproduces_every_pool() {
    recurrent::check_recurrent_model(&rows(), recurrent::RECURRENT_TOLERANCE_MIB)
        .expect("the formula reproduces every pool");
}

#[test]
fn the_dataset_holds_no_cell_measured_differently_by_two_builds() {
    dataset::check_runtime_builds(&rows(), dataset::BUILD_TOLERANCE).expect("builds agree");
}

// --- Scalar constants ---------------------------------------------------------

#[test]
fn mainline_tensor_moe_matches() {
    let derived = graph::mainline_tensor_moe(&rows(), tuning()).expect("derives");
    assert_eq!(
        derived.value,
        constant("MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD")
    );
    assert_eq!(
        derived.evidence,
        evidence("MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD")
    );
}

#[test]
fn gemma_e_per_layer_token_matches() {
    let derived = pinned::gemma_e_per_layer_token(&rows(), tuning()).expect("derives");
    assert_eq!(
        derived.value,
        constant("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN")
    );
    assert_eq!(
        derived.evidence,
        evidence("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN")
    );
}

#[test]
fn per_device_bytes_matches() {
    let derived = baseline::per_device_bytes(&rows()).expect("derives");
    assert_eq!(derived.value, constant("PROCESS_BASE_BYTES_PER_DEVICE"));
    assert_eq!(derived.evidence, evidence("PROCESS_BASE_BYTES_PER_DEVICE"));
}

#[test]
fn layer_split_copies_matches() {
    let derived = graph::layer_split_copies(&rows(), tuning()).expect("derives");
    assert_eq!(derived.value, constant("MAINLINE_LAYER_SPLIT_MASK_COPIES"));
    assert_eq!(
        derived.evidence,
        evidence("MAINLINE_LAYER_SPLIT_MASK_COPIES")
    );
}

#[test]
fn offload_min_batch_matches() {
    let derived = graph::offload_min_batch(&rows(), tuning()).expect("derives");
    assert_eq!(derived.value, constant("IK_OP_OFFLOAD_MIN_BATCH"));
    assert_eq!(derived.evidence, evidence("IK_OP_OFFLOAD_MIN_BATCH"));
}

#[test]
fn draft_compute_slope_matches() {
    let derived = mtp::draft_compute_slope(&rows()).expect("derives");
    assert_eq!(derived.value, constant("DRAFT_MODEL_COMPUTE_MIB_PER_1K"));
    assert_eq!(derived.evidence, evidence("DRAFT_MODEL_COMPUTE_MIB_PER_1K"));
}

#[test]
fn mtp_host_costs_match() {
    let embedded = mtp::mtp_host_embedded(&rows()).expect("derives");
    assert_eq!(embedded.value, constant("MTP_HOST_BYTES_EMBEDDED"));
    assert_eq!(embedded.evidence, evidence("MTP_HOST_BYTES_EMBEDDED"));
    let separate = mtp::mtp_host_separate(&rows()).expect("derives");
    assert_eq!(separate.value, constant("MTP_HOST_BYTES_SEPARATE_DRAFT"));
    assert_eq!(separate.evidence, evidence("MTP_HOST_BYTES_SEPARATE_DRAFT"));
    let slope = mtp::mtp_host_slope(&rows()).expect("derives");
    assert_eq!(slope.value, constant("MTP_HOST_MIB_PER_1K"));
    assert_eq!(slope.evidence, evidence("MTP_HOST_MIB_PER_1K"));
}

#[test]
fn spec_rollback_depth_matches() {
    let derived = recurrent::spec_rollback_depth(&rows()).expect("derives");
    assert_eq!(derived.value, constant("SPEC_RECURRENT_ROLLBACK_DEPTH"));
    assert_eq!(derived.evidence, evidence("SPEC_RECURRENT_ROLLBACK_DEPTH"));
}

#[test]
fn mmproj_graph_matches() {
    let derived = vram::mmproj_graph(&rows()).expect("derives");
    assert_eq!(derived.value, constant("MMPROJ_GRAPH_BYTES"));
    assert_eq!(derived.evidence, evidence("MMPROJ_GRAPH_BYTES"));
}

#[test]
fn mtp_unaccounted_matches() {
    let derived = mtp::mtp_unaccounted(&rows()).expect("derives");
    assert_eq!(derived.value, constant("MTP_UNACCOUNTED_MIB_PER_DEVICE"));
    assert_eq!(derived.evidence, evidence("MTP_UNACCOUNTED_MIB_PER_DEVICE"));
}

// --- Rate tables --------------------------------------------------------------

#[test]
fn ik_moe_rates_match() {
    let (_scalar, rates) = graph::ik_moe_per_nembd(&rows(), tuning()).expect("derives");
    assert_eq!(derived(&rates), table("ik_moe_rates"));
}

#[test]
fn quantised_cache_rates_match() {
    let (_scalar, rates) = pinned::quantised_cache_bytes(&rows()).expect("derives");
    assert_eq!(derived(&rates), table("quantised_cache_rates"));
}

#[test]
fn no_flash_attn_rates_match() {
    // `None` for the quantised table, matching the order `emit` runs them in — see
    // `pinned::no_flash_attn_rates`.
    let rates = pinned::no_flash_attn_rates(&rows(), tuning(), None).expect("derives");
    assert_eq!(derived(&rates), table("no_flash_attn_rates"));
}

#[test]
fn baseline_offsets_match() {
    let rates = pinned::no_flash_attn_rates(&rows(), tuning(), None).expect("derives");
    let (_scalar, offsets) =
        baseline::baseline_offset(&rows(), tuning(), &rates.by_arch).expect("derives");
    // `default` is zero here rather than the worst offset: an unmeasured architecture
    // has no evidence either way, so it is not charged one.
    assert_eq!(offsets.by_arch, table_only("baseline_offset"));
    assert_eq!(committed()["baseline_offset"]["default"].as_i64(), Some(0));
}

/// The ordering dependency `emit` spells out, asserted rather than trusted.
#[test]
fn the_baseline_offset_refuses_to_run_without_the_flash_attention_rates() {
    let error = baseline::baseline_offset(&rows(), tuning(), &BTreeMap::new())
        .expect_err("an empty table is a refusal, not a zero");
    assert!(error.to_string().contains("has not run yet"), "{error}");
}

#[test]
fn tensor_split_baselines_match() {
    let (_scalar, rates) = baseline::tensor_split_baseline(&rows(), tuning()).expect("derives");
    assert_eq!(derived(&rates), table("tensor_split_baseline"));
}

#[test]
fn per_slot_host_bytes_match() {
    let rates = baseline::per_slot_bytes(&rows()).expect("derives");
    assert_eq!(derived(&rates), table("per_slot_host_bytes"));
}

#[test]
fn checkpoint_headroom_matches() {
    let rates = baseline::checkpoint_headroom(&rows()).expect("derives");
    assert_eq!(derived(&rates), table("checkpoint_headroom_bytes"));
}

#[test]
fn no_flash_attn_score_matches() {
    let rates = vram::no_flash_attn_score(&rows()).expect("derives");
    assert_eq!(derived(&rates), table("no_flash_attn_score_centibytes"));
}

#[test]
fn mtp_draft_compute_matches() {
    let fit = mtp::mtp_draft_compute(&rows()).expect("derives");
    assert_eq!(fit.bases, table_only("mtp_draft_compute_base_mib"));
    assert_eq!(fit.slopes, table_only("mtp_draft_compute_mib_per_1k"));
}

#[test]
fn table_less_observations_match() {
    let observed = vram::table_less_observations(&rows()).expect("derives");
    let want: BTreeMap<String, BTreeMap<String, i64>> =
        committed()["table_less_compute_observations"]["by_arch"]
            .as_object()
            .expect("an object")
            .iter()
            .map(|(arch, points)| {
                let inner = points
                    .as_object()
                    .expect("an object")
                    .iter()
                    .map(|(cell, value)| (cell.clone(), value.as_i64().expect("an integer")))
                    .collect();
                (arch.clone(), inner)
            })
            .collect();
    assert_eq!(observed.by_arch, want);
}

#[test]
fn the_mtp_slot_series_matches() {
    assert_eq!(
        mtp::mtp_slot_scaling(&rows()),
        committed()["mtp_slot_scaling"]["observed"]
            .as_str()
            .expect("a string"),
    );
}

// --- Plumbing -----------------------------------------------------------------

#[test]
fn the_arena_terms_are_whole_masks() {
    // A cell with no sliding window and a unified cache: the mask is exactly
    // `pad(ctx) x tokens x width`, so the model is arithmetic and can be checked
    // against the record rather than fitted.
    let rows = rows();
    let record = rows
        .iter()
        .find(|r| {
            r.parsed.n_swa == Some(0)
                && r.factors.flash_attn_on()
                && r.factors.parallel == Some(1)
                && r.parsed.n_embd.is_some_and(|v| v > 0)
                && !r.factors.runtime_is_ik()
        })
        .expect("the dataset has a dense single-slot cell");
    let terms = arena::arena_terms(record, true, tuning());
    let tokens = record.factors.tokens();
    let n_kv = pad(u64::from(record.factors.ctx), arena::KV_CACHE_PAD);
    assert_eq!(terms.mask, (n_kv * tokens * 2) as f64 / 1048576.0);
    assert_eq!(terms.swa_mask, 0.0);
    let n_embd = u64::from(record.parsed.n_embd.unwrap_or(0));
    assert_eq!(terms.hidden, (2 * n_embd * tokens * 4) as f64 / 1048576.0);
}

#[test]
fn the_variant_key_splits_the_gemma_family() {
    let rows = rows();
    let keys: Vec<String> = rows
        .iter()
        .map(|record| variant_key(record, false))
        .collect();
    for expected in ["gemma4", "gemma4+e", "gemma4+moe"] {
        assert!(
            keys.iter().any(|k| k == expected),
            "{expected} is not keyed apart"
        );
    }
}

#[test]
fn the_head_count_falls_back_where_the_gguf_omits_it() {
    let rows = rows();
    // laguna carries only `head_count_kv`, so its query heads have to be inferred
    // from `embedding_length / key_length`; a zero there left 9218 MiB unreserved.
    let laguna = rows
        .iter()
        .find(|r| r.parsed.arch.as_deref() == Some("laguna"))
        .expect("the dataset measures laguna");
    assert!(query_head_count(&laguna.parsed) > 0);
}

#[test]
fn superseded_rows_are_dropped() {
    let path = format!(
        "{}/../../data/measurements.ndjson",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("the dataset is committed");
    let loaded = dataset::load(&text).expect("every row parses");
    assert!(
        dataset::latest_per_cell(&loaded).len() < loaded.len(),
        "the dataset holds duplicate cells, so this is the case that matters",
    );
}

/// The output depends on the dataset, not on the document it is handed.
///
/// Several derivers read a constant they do not derive — the arena charges the two
/// MoE rates, the baseline offset subtracts the whole process-baseline model — and
/// reading those from the *committed* document made `emit` a function of (dataset,
/// previous run). A value that moved would then take two runs to settle, with
/// nothing to say which run you were looking at.
///
/// Handing it a document whose *derived* inputs are corrupted must produce the
/// committed one regardless: each is now derived earlier in the same pass.
///
/// The `reachable` constants are deliberately not poisoned. Those are chosen rather
/// than derived — picked so every model lands inside the rolling correction's clamp —
/// so `emit` reading them from the document is reading an input, and a residual taken
/// over a different baseline is *supposed* to differ.
#[test]
fn emit_ignores_the_derived_values_it_is_given() {
    let mut poisoned = committed().clone();
    for name in [
        "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD",
        "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN",
    ] {
        poisoned["constants"][name]["value"] = Value::from(999_999_i64);
    }
    poisoned["ik_moe_rates"]["default"] = Value::from(999_999_i64);
    poisoned["ik_moe_rates"]["by_arch"] = serde_json::json!({});

    let text = serde_json::to_string_pretty(&poisoned).expect("the document serialises");
    let emitted = emit::emit(&rows(), &text).expect("emit runs against a poisoned document");

    for name in ["baseline_offset", "no_flash_attn_rates", "ik_moe_rates"] {
        assert_eq!(
            &emitted.document[name],
            &committed()[name],
            "`{name}` moved when only the input document changed"
        );
    }
    for name in [
        "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD",
        "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN",
    ] {
        assert_eq!(
            emitted.document["constants"][name]["value"],
            committed()["constants"][name]["value"],
            "`{name}` moved when only the input document changed"
        );
    }
}

/// A deriver that cannot run fails the gate, even though the document still matches.
///
/// This is the failure `--check` is least able to see on its own: a failed deriver
/// leaves its constant exactly as committed, so the documents compare equal and the
/// check reports agreement with the data for a constant nothing re-derived. Emptying
/// the dataset makes every deriver fail at once, which is the cheapest way to ask
/// whether the failures are consulted at all.
#[test]
fn a_deriver_that_cannot_run_fails_the_check() {
    let error = emit::emit_check(&[], tuning_text())
        .expect_err("an empty dataset derives nothing, so the check must fail");
    let message = error.to_string();
    assert!(
        message.contains("could not be derived"),
        "the failure should name the derivations, not a value difference: {message}"
    );
}

fn evidence(name: &str) -> &'static str {
    committed()["constants"][name]["evidence"]
        .as_str()
        .unwrap_or_else(|| panic!("{name} has no evidence"))
}

/// A committed table's `by_arch` alone, for the tables whose `default` is not the
/// worst measured value.
fn table_only(name: &str) -> BTreeMap<String, i64> {
    committed()[name]["by_arch"]
        .as_object()
        .unwrap_or_else(|| panic!("{name} has no by_arch"))
        .iter()
        .map(|(arch, value)| (arch.clone(), value.as_i64().expect("an integer")))
        .collect()
}
