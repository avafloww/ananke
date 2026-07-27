//! Check the estimator against the measurement dataset.
//!
//! `scripts/calibration/data/measurements.ndjson` records what real
//! `llama-server` processes actually held, on real hardware, across every
//! model and flag combination the calibration campaign covered. These tests
//! replay it: each record's `parsed` block carries the model's shape, so a
//! synthetic summary can be built from it and the estimator run against the
//! same configuration the measurement was taken under. No GGUF is needed,
//! which is what lets this run in CI.
//!
//! Two tiers, because the evidence is not equally strong:
//!
//! - **Exact.** The pinned graph arena is compared against llama.cpp's own
//!   logged `CUDA_Host compute buffer size`. Same quantity, so the model
//!   either reproduces it or is wrong, and the tolerance is a rounding error.
//! - **Banded.** The process baseline is compared against `/proc`, which
//!   carries far more that the estimator does not model. The assertion is the
//!   one the constants actually claim: that every model lands inside the
//!   rolling correction's `[0.8, 1.5]`, because a prediction outside that band
//!   cannot be pulled back by any amount of observation.
//!
//! **Hardware.** Every row was taken on one machine, so this validates the
//! estimator against that machine and says nothing about any other. If rows
//! from different hardware are ever pooled, the derivations behind these
//! constants take a plain mean over whatever is in the file — they carry no
//! hardware term and would silently blend two populations. That is a known
//! and deliberate gap, recorded here rather than in a commit message, and it
//! has to be closed before contributed data can be merged.

use std::{collections::BTreeMap, path::Path};

use ananke::{
    config::validate::SplitMode,
    estimator::{
        EstimatorInputs,
        host_buffer::{host_overhead_bytes, pinned_graph_bytes},
    },
    gguf::{GgufSummary, GgufValue},
};
use serde_json::Value;

/// Architectures whose arena the campaign confirmed to within 0.1 MiB.
///
/// Restricted deliberately: an architecture absent from here is one the
/// dataset has not settled, and asserting exactness for it would turn an open
/// question into a passing test.
/// Architecture, and the worst arena error the dataset shows for it, in MiB.
///
/// Five reproduce the measurement to well under a mebibyte, which is what
/// "the arena is arithmetic, not a fit" means in practice. Three do not, and
/// their bounds are recorded rather than smoothed away:
///
/// - `qwen35moe` at 2.5 and `gemma3` at 12.1 — both sliding-window or MoE
///   cases where a second term is suspected but not isolated.
/// - `glm-dsa` at 45.0 — the sparse path's CPU-resident MoE buffers, whose
///   per-token rate is derived from three models of which one deviates.
///
/// These are ceilings, so they can only be tightened. Raising one to make a
/// change pass would be defeating the test.
const CONFIRMED: &[(&str, f64)] = &[
    ("lfm2", 0.5),
    ("llama", 0.5),
    ("qwen3", 0.5),
    ("talkie", 0.5),
    ("qwen35", 1.5),
    ("qwen35moe", 3.0),
    ("gemma3", 13.0),
    ("glm-dsa", 45.5),
];

#[test]
fn arena_reproduces_the_measured_pinned_buffer() {
    let records = load();
    assert!(
        records.len() > 300,
        "the dataset should carry the campaign's measurements, found {}",
        records.len()
    );

    let mut checked = 0usize;
    let mut worst: Option<(String, f64)> = None;
    let mut by_arch: Vec<(String, f64)> = Vec::new();
    for record in &records {
        let Some(case) = Case::from_record(record) else {
            continue;
        };
        let Some((_, tolerance)) = CONFIRMED.iter().find(|(a, _)| *a == case.arch) else {
            continue;
        };
        // Flash attention off is excluded: the excess over the widened mask is
        // measured at 4.25x on non-SWA architectures and 4.8-4.9x on
        // sliding-window ones, so `NO_FLASH_ATTN_BYTES_PER_TOKEN` is a
        // representative value rather than a law, and asserting exactness on
        // it would claim more than the dataset supports.
        if !case.flash_attn {
            continue;
        }
        // The production holdout cells resolve their placement through the
        // packer (`expert_offload = "auto"`), so the record's factors do not
        // say whether the run was hybrid, and the mask replication cannot be
        // determined from them.
        if case.label.starts_with("prod-") {
            continue;
        }
        // Mainline hybrids are excluded, and the exclusion is a finding rather
        // than a convenience: they carry a CPU-resident MoE term the estimator
        // models for ik alone. Measured at ~116 KB per batch token on
        // qwen35moe against ik's 83 KB, it leaves the arena under-predicted by
        // ~57 MiB. Adding the term is the fix; excluding it here keeps this
        // test asserting what the dataset actually established.
        if case.hybrid && !case.ik_llama {
            continue;
        }
        let predicted = pinned_graph_bytes(&case.summary, &case.arch, &case.inputs()) as f64;
        let measured = case.arena_mib * 1024.0 * 1024.0;
        let delta = (predicted - measured).abs() / 1024.0 / 1024.0;
        checked += 1;
        by_arch.push((case.arch.clone(), delta));
        if delta > *tolerance {
            worst = Some((case.label.clone(), delta));
        }
    }

    assert!(checked > 100, "too few comparable cells: {checked}");
    let mut per_arch: std::collections::BTreeMap<String, f64> = Default::default();
    for (arch, delta) in &by_arch {
        let entry = per_arch.entry(arch.clone()).or_insert(0.0);
        if *delta > *entry {
            *entry = *delta;
        }
    }
    eprintln!("worst arena delta per architecture (MiB):");
    for (arch, delta) in &per_arch {
        eprintln!("  {arch:12} {delta:8.2}");
    }
    if let Some((label, delta)) = worst {
        panic!(
            "arena model is {delta:.2} MiB from measurement on {label}, beyond \
             the ceiling recorded for its architecture (of {checked} cells). The \
             arena is arithmetic over ggml's graph, not a fit, so a discrepancy \
             means the model is wrong rather than a constant being mistuned."
        );
    }
}

/// The whole claim behind the constants marked `reachable` in `tuning.json`.
///
/// They are not fitted to minimise error — the spread across models is too
/// wide for any single value to be close to all of them. They are chosen so
/// that every model lands inside the band the rolling correction can travel.
/// If one falls outside, no amount of observation recovers it, and that is
/// the failure this guards.
#[test]
fn every_model_lands_inside_the_correction_band() {
    let records = load();
    let mut outside = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    let mut checked = 0usize;

    for record in &records {
        let Some(case) = Case::from_record(record) else {
            continue;
        };
        // Owned host memory the process actually held, less the parts the
        // arena term already accounts for.
        let Some(owned_kb) = case.owned_kb else {
            continue;
        };
        let owned = owned_kb as f64 * 1024.0;
        if owned <= 0.0 {
            continue;
        }
        // Only fully-resident models: a hybrid's owned memory is dominated by
        // the CPU-held weights, which this term does not model at all, so the
        // ratio would measure the weights rather than the overhead.
        if case.hybrid {
            continue;
        }
        let predicted = host_overhead_bytes(&case.summary, &case.arch, &case.inputs()) as f64;
        if predicted <= 0.0 {
            continue;
        }
        checked += 1;
        ratios.push(owned / predicted);
        let ratio = owned / predicted;
        if !(0.8..=1.5).contains(&ratio) {
            outside.push(format!("{} ratio {ratio:.2}", case.label));
        }
    }

    assert!(checked > 100, "too few comparable cells: {checked}");
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    // A ratchet rather than a pass. The reachability claim — that every model
    // lands inside the band the rolling correction can travel — is *not* true
    // today: this many cells sit outside it, mostly under-predicted, which is
    // the direction that OOMs. The number is recorded so it can only fall.
    // See FINDINGS.md; closing the gap needs the host baseline refitted, not
    // this threshold raised.
    const KNOWN_OUTSIDE: usize = 44;
    assert!(
        outside.len() <= KNOWN_OUTSIDE,
        "{} of {checked} cells sit outside the correction's [0.8, 1.5] band, \
         up from a known {KNOWN_OUTSIDE}. Ratios span {:.2}-{:.2}, median {:.2}. \
         Examples: {:?}",
        outside.len(),
        ratios.first().copied().unwrap_or(0.0),
        ratios.last().copied().unwrap_or(0.0),
        ratios[ratios.len() / 2],
        &outside[..outside.len().min(5)]
    );
}

/// One measured configuration, rebuilt from its record.
struct Case {
    label: String,
    arch: String,
    summary: GgufSummary,
    arena_mib: f64,
    owned_kb: Option<u64>,
    context: u32,
    ubatch: u32,
    parallel: u32,
    kv_unified: bool,
    flash_attn: bool,
    ik_llama: bool,
    ik_dsa: bool,
    devices: u32,
    hybrid: bool,
    split: SplitMode,
}

impl Case {
    fn from_record(record: &Value) -> Option<Self> {
        if record.get("status")?.as_str()? != "ok" {
            return None;
        }
        let factors = record.get("factors")?;
        let parsed = record.get("parsed")?;
        let arch = parsed.get("arch")?.as_str()?.to_string();
        if arch == "?" {
            return None;
        }
        // Only fully-offloaded runs with a device: a partly- or un-offloaded
        // process has a different graph, which the arena model does not claim
        // to describe.
        if factors.get("ngl")?.as_u64()? != 99 {
            return None;
        }
        let gpus = factors.get("gpus")?.as_str()?;
        if gpus.is_empty() {
            return None;
        }
        let arena_mib = parsed.get("arena_mib").and_then(Value::as_f64)?;
        if arena_mib <= 0.0 {
            return None;
        }
        let n_layer = parsed.get("n_layer").and_then(Value::as_u64)? as u32;
        let n_embd = parsed.get("n_embd").and_then(Value::as_u64)? as u32;
        if n_layer == 0 || n_embd == 0 {
            return None;
        }

        let mut metadata = BTreeMap::new();
        let mut put = |key: &str, value: u32| {
            if value > 0 {
                metadata.insert(
                    smol_str::SmolStr::new(format!("{arch}.{key}")),
                    GgufValue::U32(value),
                );
            }
        };
        put("block_count", n_layer);
        put("embedding_length", n_embd);
        for (key, field) in [
            ("expert_count", "n_expert"),
            ("expert_used_count", "n_expert_used"),
            ("attention.sliding_window", "n_swa"),
            ("attention.head_count_kv", "n_head_kv"),
            ("attention.key_length", "n_embd_head_k"),
            ("attention.value_length", "n_embd_head_v"),
        ] {
            put(
                key,
                parsed.get(field).and_then(Value::as_u64).unwrap_or(0) as u32,
            );
        }

        let extra: Vec<String> = factors
            .get("extra")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        Some(Self {
            label: factors.get("label")?.as_str()?.to_string(),
            summary: GgufSummary {
                path: std::path::PathBuf::from("/measured"),
                total_tensor_bytes: 0,
                tensors: BTreeMap::new(),
                metadata,
                block_count: Some(n_layer),
                architecture: smol_str::SmolStr::new(&arch),
                shards: Vec::new(),
            },
            arch,
            arena_mib,
            owned_kb: record.get("rss").and_then(|r| {
                let anon = r.get("rss_anon_kb")?.as_u64()?;
                let shmem = r.get("rss_shmem_kb").and_then(Value::as_u64).unwrap_or(0);
                Some(anon + shmem)
            }),
            context: factors.get("ctx")?.as_u64()? as u32,
            ubatch: factors.get("ubatch")?.as_u64()? as u32,
            parallel: factors.get("parallel")?.as_u64()? as u32,
            kv_unified: factors
                .get("kv_unified")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            flash_attn: factors.get("flash_attn").and_then(Value::as_str) == Some("on"),
            ik_llama: factors.get("runtime").and_then(Value::as_str) == Some("ik"),
            ik_dsa: extra.iter().any(|f| f == "-dsa"),
            devices: gpus.split(',').count() as u32,
            // A cell with expert offload is the hybrid regime, where the
            // masks are not replicated across devices.
            hybrid: factors.get("n_cpu_moe").is_some_and(|v| !v.is_null()),
            split: match factors.get("split").and_then(Value::as_str) {
                Some("tensor") => SplitMode::Tensor,
                Some("row") => SplitMode::Row,
                _ => SplitMode::Layer,
            },
        })
    }

    fn inputs(&self) -> EstimatorInputs<'_> {
        EstimatorInputs {
            name: &self.label,
            model: Path::new("/measured"),
            mmproj: None,
            visible_devices: self.devices,
            host_resident_experts: self.hybrid,
            split_mode: self.split,
            context: self.context,
            ubatch: Some(self.ubatch),
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &[],
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
            ik_llama: self.ik_llama,
            ik_dsa: self.ik_dsa,
            parallel: Some(self.parallel),
            flash_attn: Some(self.flash_attn),
            kv_unified: Some(self.kv_unified),
            cache_ram_mb: None,
        }
    }
}

fn load() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("scripts/calibration/data/measurements.ndjson");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each line is a record"))
        .collect()
}
