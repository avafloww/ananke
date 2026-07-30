//! Print every model's estimate and packing, for eyeballing.
//!
//! Where `scoreboard` compares the seven production models against measurements,
//! this covers every entry in `models.toml` and breaks each one down — which term
//! is big, where the weights landed, how many expert layers went to the host.
//!
//! Nothing here is a gate. It is the tool you reach for when a number looks wrong
//! and you want to see its parts.
//!
//! Two fields are easy to get wrong here, and both were:
//!
//! - `cache_ram_mb`. Dropping it from the mapping ignores gemma-4-31B-QAT's
//!   `cache_ram_mb = 0` and applies the 8192 MiB default instead — the whole of
//!   that model's CPU-slot difference. `ModelConfig` is `deny_unknown_fields` for
//!   this reason.
//! - Card capacity. The real 24576 MiB is used throughout, matching `scoreboard`.
//!   Rounding it to 24000, or applying it only when expert offload is set, changes
//!   how a tensor split apportions between two cards.

use std::process::ExitCode;

use ananke_calibrate::{
    models::{self, ModelConfig},
    validate::{NEUTRAL, snapshot_for},
};
use ananke_estimate::Estimate;
use ananke_fs::LocalFs;
use ananke_placement::{Packed, devices::DeviceId, pack_demand};

const MODELS_TOML: &str = "scripts/calibration/models.toml";
const MIB: u64 = 1024 * 1024;
const GIB: f64 = (1024 * 1024 * 1024) as f64;

/// One model's estimate, or why it has none.
enum Row {
    Estimated {
        name: String,
        arch: String,
        weights_gib: f64,
        compute_buffer_mb: u32,
        kv_total_mib: u64,
        mtp_mib: u64,
        gpu_vram_gib: f64,
        host_overhead_mib: u64,
        packed: Option<Packed>,
    },
    Failed {
        name: String,
        why: String,
    },
}

fn main() -> ExitCode {
    let as_json = std::env::args().any(|a| a == "--json");
    let configs = match models::load(std::path::Path::new(MODELS_TOML)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let fs = LocalFs;
    let rows: Vec<Row> = configs.iter().map(|c| estimate_one(&fs, c)).collect();

    if as_json {
        let json: Vec<serde_json::Value> = rows.iter().map(as_value).collect();
        match serde_json::to_string_pretty(&json) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("serialising: {e}");
                return ExitCode::from(2);
            }
        }
        return ExitCode::SUCCESS;
    }

    println!(
        "\n{:<30} {:<12} {:>8} {:>7} {:>8} {:>8} {:>10} {:>8}",
        "model", "arch", "weights", "cb/dev", "kv", "mtp", "gpu_vram", "host_oh"
    );
    println!("{}", "-".repeat(100));
    for row in &rows {
        match row {
            Row::Failed { name, why } => println!("{name:<30} ERROR: {}", truncate(why, 80)),
            Row::Estimated {
                name,
                arch,
                weights_gib,
                compute_buffer_mb,
                kv_total_mib,
                mtp_mib,
                gpu_vram_gib,
                host_overhead_mib,
                ..
            } => println!(
                "{name:<30} {arch:<12} {weights_gib:>7.1}G {compute_buffer_mb:>6}M \
                 {kv_total_mib:>7}M {mtp_mib:>7}M {gpu_vram_gib:>8.1}G {host_overhead_mib:>7}M"
            ),
        }
    }

    println!(
        "\n{:<28} {:>10} {:>10} {:>10} {:>10} {:>7}",
        "model", "gpu0", "gpu1", "cpu", "offload", "layers"
    );
    println!("{}", "-".repeat(80));
    for row in &rows {
        let Row::Estimated { name, packed, .. } = row else {
            continue;
        };
        let Some(packed) = packed else {
            println!("{name:<28} {:>10}", "not packed");
            continue;
        };
        let slot = |want: DeviceId| -> String {
            packed
                .allocation
                .bytes
                .get(&want)
                .map(|b| format!("{}M", b / MIB))
                .unwrap_or_else(|| "0M".to_string())
        };
        println!(
            "{name:<28} {:>10} {:>10} {:>10} {:>10} {:>7}",
            slot(DeviceId::Gpu(0)),
            slot(DeviceId::Gpu(1)),
            slot(DeviceId::Cpu),
            format!("{}M", packed.expert_offload_bytes / MIB),
            packed.expert_offload_layers
        );
    }
    ExitCode::SUCCESS
}

/// Estimate and pack one model.
fn estimate_one(fs: &LocalFs, config: &ModelConfig) -> Row {
    eprintln!("  {}...", config.name);
    let model = config.model_path();
    let mmproj = config.mmproj_path();
    let draft = config.draft_path();
    let inputs = config.estimator_inputs(&model, mmproj.as_deref(), draft.as_deref());
    let estimate = match ananke_estimate::estimate_from_path(fs, &inputs) {
        Ok(e) => e,
        Err(e) => {
            return Row::Failed {
                name: config.name.clone(),
                why: e.to_string(),
            };
        }
    };
    let placement = config.placement_inputs();
    let snap = snapshot_for(&placement.gpu_allow, &[]);
    // A model that does not fit is still worth breaking down; the packing detail is
    // simply absent for it.
    let packed = pack_demand(&estimate, &placement, &snap, NEUTRAL).ok();
    // The estimator's own notional GPU total, not the packer's reservation: this
    // column answers "what does the model need on a card", which is comparable
    // across models regardless of how a given machine ends up splitting it. The
    // reservation is what `scoreboard` reports, and the packing detail below shows
    // where it actually landed.
    let gpu_vram_mib = notional_gpu_mib(&estimate, config);
    Row::Estimated {
        name: config.name.clone(),
        arch: estimate.architecture.to_string(),
        weights_gib: estimate.weights_bytes as f64 / GIB,
        compute_buffer_mb: estimate.compute_buffer_mb,
        kv_total_mib: kv_total_mib(&estimate, config),
        mtp_mib: estimate.mtp_bytes / MIB,
        gpu_vram_gib: (gpu_vram_mib * MIB) as f64 / GIB,
        host_overhead_mib: estimate.host_overhead_bytes / MIB,
        packed,
    }
}

/// What the estimator says this model needs on a card, before placement.
///
/// The same formula the `estimate` example prints as `gpu_vram_mib`: GPU-resident
/// weights, plus the cache, plus the compute buffer on up to two cards, plus MTP.
/// Token embeddings are excluded because llama.cpp keeps them on the host.
///
/// Rounded once, at the point of display. Rounding to two decimals and then
/// formatting to one turns 36.0518 into 36.05 and then into 36.0 under
/// half-to-even; the correct value is 36.1.
fn notional_gpu_mib(estimate: &Estimate, config: &ModelConfig) -> u64 {
    let gpu_weights = estimate
        .weights_bytes
        .saturating_sub(estimate.non_layer.token_embd_bytes);
    gpu_weights
        .saturating_add(
            estimate
                .kv_per_token
                .saturating_mul(u64::from(config.context)),
        )
        .saturating_add(
            u64::from(estimate.compute_buffer_mb)
                .saturating_mul(u64::from(config.cards()).min(2))
                .saturating_mul(MIB),
        )
        .saturating_add(estimate.mtp_bytes)
        / MIB
}

/// The cache's whole cost at this model's context.
fn kv_total_mib(estimate: &Estimate, config: &ModelConfig) -> u64 {
    estimate
        .kv_per_token
        .saturating_mul(u64::from(config.context))
        / MIB
}

fn as_value(row: &Row) -> serde_json::Value {
    match row {
        Row::Failed { name, why } => serde_json::json!({"name": name, "error": why}),
        Row::Estimated {
            name,
            arch,
            weights_gib,
            compute_buffer_mb,
            kv_total_mib,
            mtp_mib,
            gpu_vram_gib,
            host_overhead_mib,
            packed,
        } => serde_json::json!({
            "name": name,
            "architecture": arch,
            "weights_gib": weights_gib,
            "compute_buffer_mb": compute_buffer_mb,
            "kv_total_mib": kv_total_mib,
            "mtp_mib": mtp_mib,
            "gpu_vram_gib": gpu_vram_gib,
            "host_overhead_mib": host_overhead_mib,
            "expert_offload_mib": packed
                .as_ref()
                .map(|p| p.expert_offload_bytes / MIB),
            "expert_offload_layers": packed.as_ref().map(|p| p.expert_offload_layers),
        }),
    }
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}
