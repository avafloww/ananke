//! Compare the estimator's prediction against every comparable measured cell.
//!
//! This is the campaign's top-level accuracy signal. It runs the estimator and
//! the packer **in-process**, where the Python it replaces spawned
//! `cargo run --example estimate` once per cell — some two hundred subprocess
//! launches, each re-reading the GGUF.
//!
//! What it compares is the *prediction*, not the reservation. The reservation
//! carries slop the process is not expected to use — one layer's headroom above
//! all — and comparing that to a measurement reads the slop as error. On the
//! production Qwen3.6-27B cell the two differ by 472 MiB, which is the
//! difference between +1.1% and −0.1%.

use std::collections::BTreeMap;

use ananke_config::placement::{
    DeviceSlot, OffloadMode, PlacementInputs, PlacementPolicy, SplitMode,
};
use ananke_estimate::EstimatorInputs;
use ananke_placement::{
    Corrections,
    devices::{DeviceSnapshot, GpuSnapshot},
};

use crate::record::Record;

/// Host memory the packer is told it has. The campaign machine has 256 GiB and
/// nothing here depends on the exact figure — it only has to be large enough not
/// to bound an expert offload.
const CPU_CAPACITY_MIB: u64 = 256_000;

/// One cell's outcome.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub label: String,
    pub arch: String,
    pub predicted_mib: u64,
    pub measured_mib: u64,
    pub reserved_mib: u64,
    pub drift_pct: f64,
}

/// Why a cell was not comparable. Counted and reported, never silently dropped.
pub fn skip_reason(record: &Record, known_models: &[String]) -> Option<String> {
    let f = &record.factors;
    if record.status != "ok" {
        return Some(format!("status {}", record.status));
    }
    if record.gpu_used_mib().is_none_or(|v| v == 0) {
        return Some("no driver reading".into());
    }
    if !f.served {
        return Some("idle: no first-use allocations".into());
    }
    if !known_models.iter().any(|m| m == &f.model) {
        return Some("model not in models.toml".into());
    }
    // An operator-chosen `ngl` is a placement the packer did not choose, so the
    // two are not comparable. Expert offload is different: `--n-cpu-moe` leaves
    // `ngl` at 99 and the packer models it.
    if f.ngl != Some(99) && f.n_cpu_moe.is_none() {
        return Some(format!(
            "operator-chosen placement (ngl {})",
            f.ngl.unwrap_or(-1)
        ));
    }
    if f.embeddings {
        return Some("embedding modality".into());
    }
    None
}

/// The estimator inputs this cell was measured under.
pub fn estimator_inputs<'a>(record: &'a Record, model: &'a std::path::Path) -> EstimatorInputs<'a> {
    let f = &record.factors;
    let cards = record.gpu_ids().len().max(1) as u32;
    EstimatorInputs {
        name: &f.label,
        model,
        mmproj: f.mmproj.as_deref().map(std::path::Path::new),
        context: f.ctx,
        ubatch: f.ubatch,
        visible_devices: cards,
        host_resident_experts: f.n_cpu_moe.is_some(),
        split_mode: split_mode(f.split.as_deref()),
        cache_type_k: f.kv_type.as_deref(),
        cache_type_v: f.kv_type.as_deref(),
        override_tensor: &[],
        compute_buffer_mb: None,
        allow_fallback: false,
        mtp: f.spec_type.is_some(),
        draft_model: f.draft.as_deref().map(std::path::Path::new),
        ik_llama: f.runtime == "ik",
        // The fork's sparse-attention path is a separate flag, and the campaign
        // only ran it for the architecture that has one.
        ik_dsa: f.runtime == "ik" && record.parsed.arch.as_deref() == Some("glm-dsa"),
        parallel: f.parallel,
        flash_attn: f.flash_attn.as_deref().map(|v| v == "on"),
        kv_unified: Some(f.kv_unified),
        cache_ram_mb: f.cram,
    }
}

/// The placement this cell was measured under.
pub fn placement_inputs(record: &Record) -> PlacementInputs {
    let f = &record.factors;
    PlacementInputs {
        policy: if f.n_cpu_moe.is_some() {
            PlacementPolicy::Hybrid
        } else {
            PlacementPolicy::GpuOnly
        },
        split_mode: split_mode(f.split.as_deref()),
        gpu_allow: record.gpu_ids(),
        expert_offload: match f.n_cpu_moe {
            Some(n) => OffloadMode::Layers(n),
            None => OffloadMode::Off,
        },
        ik_llama: f.runtime == "ik",
        ..PlacementInputs::named(&f.label)
    }
}

/// The cards this cell ran on, at the capacities the campaign saw.
///
/// The driver reserves a little of each card, which is why a nominally 24576 MiB
/// 3090 reads ~24124 free. Pack against what was actually available.
pub fn snapshot(record: &Record) -> DeviceSnapshot {
    let ids = record.gpu_ids();
    let gpus = ids
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            let total = record
                .hardware
                .gpus
                .get(index)
                .map(|g| g.memory_total_mib)
                .unwrap_or(24_576)
                * 1024
                * 1024;
            GpuSnapshot {
                id,
                name: format!("GPU {id}"),
                total_bytes: total,
                free_bytes: total,
            }
        })
        .collect();
    DeviceSnapshot {
        gpus,
        cpu: Some(ananke_placement::devices::CpuSnapshot {
            total_bytes: CPU_CAPACITY_MIB * 1024 * 1024,
            available_bytes: CPU_CAPACITY_MIB * 1024 * 1024,
        }),
        // A validation run compares the estimator against a recorded
        // measurement; nothing here is time-dependent.
        taken_at_ms: 0,
    }
}

fn split_mode(split: Option<&str>) -> SplitMode {
    match split {
        Some("tensor") => SplitMode::Tensor,
        Some("row") => SplitMode::Row,
        _ => SplitMode::Layer,
    }
}

/// A key identifying the configuration, so repeats of one cell are counted once.
///
/// Two labels can share a configuration — a re-measurement, or the same point
/// reached from two sweeps — and they measure the same thing.
pub fn configuration_key(record: &Record) -> String {
    let f = &record.factors;
    format!(
        "{}|{}|{:?}|{:?}|{}|{:?}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|{:?}",
        f.model,
        f.ctx,
        f.ubatch,
        f.parallel,
        f.kv_unified,
        f.split,
        f.gpus,
        f.kv_type,
        f.ngl,
        f.n_cpu_moe,
        f.mmproj,
        f.draft,
        f.spec_type,
        f.runtime,
        f.flash_attn,
        f.cram,
    )
}

/// Per-GPU reservation and prediction totals out of a packed allocation.
pub fn gpu_totals(per_device: &BTreeMap<DeviceSlot, u64>) -> (u64, usize) {
    let mut total = 0;
    let mut cards = 0;
    for (slot, &bytes) in per_device {
        if matches!(slot, DeviceSlot::Gpu(_)) && bytes > 0 {
            total += bytes;
            cards += 1;
        }
    }
    (total / (1024 * 1024), cards)
}

/// The neutral corrections a validation run packs with: this compares the
/// estimator, not the rolling correction that trains on top of it.
pub const NEUTRAL: Corrections = Corrections::NEUTRAL;
