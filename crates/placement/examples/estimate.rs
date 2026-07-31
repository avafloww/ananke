//! Run the estimator against a GGUF and dump JSON.
//!
//! Usage:
//!   cargo run --example estimate -- \
//!     --model /path/to/model.gguf \
//!     --context 8192 \
//!     [--mmproj /path/to/mmproj.gguf] \
//!     [--cache-type-k q8_0 --cache-type-v q8_0] \
//!     [--override-tensor '<regex>=<device>' ...] \
//!     [--n-cpu-moe N] \
//!     [--compute-buffer-mb N] \
//!     [--active-devices N] \
//!     [--mtp] \
//!     [--draft-model /path/to/draft.gguf] \
//!     [--allow-fallback] \
//!     [--parallel N] \
//!     [--flash-attn on|off] \
//!     [--kv-unified on|off] \
//!     [--cache-ram-mb N] \
//!     [--split-mode layer|row|tensor] \
//!     [--visible-devices N] \
//!     [--ik-llama] \
//!     [--ik-dsa] \
//!     [--host-resident-experts] \
//!     [--pack --gpu 24000 --gpu 24000 --cpu 256000]
//!
//! `--pack` runs the packer against the supplied device capacities and
//! reports the per-device allocation alongside the raw estimate. Without
//! `--pack`, only the estimator output is printed.
//!
//! `--n-cpu-moe N` forces N expert layers to CPU (`expert_offload = N`),
//! matching llama-server's `--n-cpu-moe` flag. `--host-resident-experts`
//! is shorthand for `expert_offload = auto`, which spills only the surplus
//! that does not fit on the GPUs.
//!
//! Unknown architectures hard-reject by default; pass `--allow-fallback`
//! to accept the coarse fallback (see `ananke_estimate::fallback`).

use std::{path::PathBuf, process};

use ananke_config::placement::{OffloadMode, PlacementInputs, PlacementPolicy, SplitMode};
use ananke_estimate::{self as estimator, EstimatorInputs};
use ananke_fs::LocalFs;
use ananke_placement::{
    Corrections,
    devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot},
    pack_demand,
};
use serde_json::json;

struct Args {
    model: PathBuf,
    mmproj: Option<PathBuf>,
    context: u32,
    ubatch: Option<u32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    override_tensor: Vec<String>,
    compute_buffer_mb: Option<u32>,
    active_devices: Option<u64>,
    allow_fallback: bool,
    mtp: bool,
    draft_model: Option<PathBuf>,
    parallel: Option<u32>,
    flash_attn: Option<bool>,
    kv_unified: Option<bool>,
    cache_ram_mb: Option<u32>,
    split_mode: SplitMode,
    visible_devices: u32,
    ik_llama: bool,
    ik_dsa: bool,
    expert_offload: OffloadMode,
    pack: bool,
    gpu_capacities_mib: Vec<u64>,
    cpu_capacity_mib: Option<u64>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut mmproj: Option<PathBuf> = None;
    let mut context: u32 = 4096;
    let mut ubatch: Option<u32> = None;
    let mut cache_type_k: Option<String> = None;
    let mut cache_type_v: Option<String> = None;
    let mut override_tensor: Vec<String> = Vec::new();
    let mut compute_buffer_mb: Option<u32> = None;
    let mut active_devices: Option<u64> = None;
    let mut allow_fallback = false;
    let mut mtp = false;
    let mut draft_model: Option<PathBuf> = None;
    let mut parallel: Option<u32> = None;
    let mut flash_attn: Option<bool> = None;
    let mut kv_unified: Option<bool> = None;
    let mut cache_ram_mb: Option<u32> = None;
    let mut split_mode = SplitMode::Layer;
    let mut visible_devices: u32 = 1;
    let mut visible_devices_given = false;
    let mut ik_llama = false;
    let mut ik_dsa = false;
    let mut expert_offload = OffloadMode::Off;
    let mut pack = false;
    let mut gpu_capacities_mib: Vec<u64> = Vec::new();
    let mut cpu_capacity_mib: Option<u64> = None;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().map(PathBuf::from),
            "--mmproj" => mmproj = it.next().map(PathBuf::from),
            "--context" => {
                context = it.next().and_then(|s| s.parse().ok()).unwrap_or(context);
            }
            "--ubatch" | "--ubatch-size" => ubatch = it.next().and_then(|s| s.parse().ok()),
            "--cache-type-k" => cache_type_k = it.next(),
            "--cache-type-v" => cache_type_v = it.next(),
            "--override-tensor" => {
                if let Some(rule) = it.next() {
                    override_tensor.push(rule);
                }
            }
            "--compute-buffer-mb" => compute_buffer_mb = it.next().and_then(|s| s.parse().ok()),
            "--active-devices" => active_devices = it.next().and_then(|s| s.parse().ok()),
            "--allow-fallback" => allow_fallback = true,
            "--mtp" => mtp = true,
            "--draft-model" => draft_model = it.next().map(PathBuf::from),
            "--parallel" => parallel = it.next().and_then(|s| s.parse().ok()),
            "--flash-attn" => {
                flash_attn = match it.next().as_deref() {
                    Some("on") | Some("true") | Some("1") => Some(true),
                    Some("off") | Some("false") | Some("0") => Some(false),
                    _ => None,
                };
            }
            "--kv-unified" => {
                kv_unified = match it.next().as_deref() {
                    Some("on") | Some("true") | Some("1") => Some(true),
                    Some("off") | Some("false") | Some("0") => Some(false),
                    _ => Some(true),
                };
            }
            "--cache-ram-mb" => cache_ram_mb = it.next().and_then(|s| s.parse().ok()),
            "--split-mode" => {
                split_mode = it
                    .next()
                    .and_then(|s| SplitMode::from_flag(&s))
                    .unwrap_or(SplitMode::Layer);
            }
            "--visible-devices" => {
                visible_devices = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                visible_devices_given = true;
            }
            "--ik-llama" => ik_llama = true,
            "--ik-dsa" => ik_dsa = true,
            "--host-resident-experts" => expert_offload = OffloadMode::Auto,
            "--n-cpu-moe" => {
                if let Some(n) = it.next().and_then(|s| s.parse().ok()) {
                    expert_offload = OffloadMode::Layers(n);
                }
            }
            "--pack" => pack = true,
            "--gpu" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    gpu_capacities_mib.push(v);
                }
            }
            "--cpu" => cpu_capacity_mib = it.next().and_then(|s| s.parse().ok()),
            _ => {
                eprintln!("unknown argument: {arg}");
                process::exit(2);
            }
        }
    }
    let Some(model) = model else {
        eprintln!("--model is required");
        process::exit(2);
    };
    // A tensor or row split usually implies multiple visible devices, so default
    // to two — but never override an explicit count. llama.cpp accepts
    // `--split-mode tensor` on a single card, and the dataset holds such cells;
    // silently estimating them as two-card charges them a second device's share
    // of every per-device term.
    if split_mode != SplitMode::Layer && !visible_devices_given {
        visible_devices = 2;
    }
    Args {
        model,
        mmproj,
        context,
        ubatch,
        cache_type_k,
        cache_type_v,
        override_tensor,
        compute_buffer_mb,
        active_devices,
        allow_fallback,
        mtp: mtp || draft_model.is_some(),
        draft_model,
        parallel,
        flash_attn,
        kv_unified,
        cache_ram_mb,
        split_mode,
        visible_devices,
        ik_llama,
        ik_dsa,
        expert_offload,
        pack,
        gpu_capacities_mib,
        cpu_capacity_mib,
    }
}

/// The placement the packer needs: policy, split mode, the GPU allow list, and
/// the expert-offload decision.
fn build_placement(args: &Args) -> PlacementInputs {
    PlacementInputs {
        policy: if args.expert_offload.is_enabled() {
            PlacementPolicy::Hybrid
        } else {
            PlacementPolicy::GpuOnly
        },
        split_mode: args.split_mode,
        gpu_allow: (0..args.visible_devices).collect(),
        expert_offload: args.expert_offload,
        ik_llama: args.ik_llama,
        override_tensor: args.override_tensor.clone(),
        ..PlacementInputs::named("estimate-example")
    }
}

fn build_snapshot(args: &Args) -> DeviceSnapshot {
    let gpus: Vec<GpuSnapshot> = args
        .gpu_capacities_mib
        .iter()
        .enumerate()
        .map(|(i, cap_mib)| GpuSnapshot {
            id: i as u32,
            name: format!("fake-{i}"),
            total_bytes: cap_mib * 1024 * 1024,
            free_bytes: cap_mib * 1024 * 1024,
        })
        .collect();
    let cpu = args.cpu_capacity_mib.map(|cap_mib| CpuSnapshot {
        total_bytes: cap_mib * 1024 * 1024,
        available_bytes: cap_mib * 1024 * 1024,
    });
    DeviceSnapshot {
        gpus,
        cpu,
        taken_at_ms: 0,
    }
}

fn main() {
    let args = parse_args();
    let inputs = EstimatorInputs {
        host_resident_experts: args.expert_offload.is_enabled(),
        visible_devices: args.visible_devices,
        split_mode: args.split_mode,
        name: "estimate-example",
        model: args.model.as_path(),
        mmproj: args.mmproj.as_deref(),
        context: args.context,
        ubatch: args.ubatch,
        cache_type_k: args.cache_type_k.as_deref(),
        cache_type_v: args.cache_type_v.as_deref(),
        override_tensor: &args.override_tensor,
        compute_buffer_mb: args.compute_buffer_mb,
        allow_fallback: args.allow_fallback,
        mtp: args.mtp,
        draft_model: args.draft_model.as_deref(),
        ik_llama: args.ik_llama,
        ik_dsa: args.ik_dsa,
        parallel: args.parallel,
        flash_attn: args.flash_attn,
        kv_unified: args.kv_unified,
        cache_ram_mb: args.cache_ram_mb,
    };

    let estimate = match estimator::estimate_from_path(&LocalFs, &inputs) {
        Ok(e) => e,
        Err(e) => {
            println!("{}", json!({"estimator_error": e.to_string()}));
            process::exit(1);
        }
    };

    let kv_total_bytes = estimate
        .kv_per_token
        .saturating_mul(estimate.context as u64);

    let active_devices = args.active_devices.unwrap_or(3);
    let cb_total_bytes = (estimate.compute_buffer_mb as u64)
        .saturating_mul(active_devices)
        .saturating_mul(1024 * 1024);
    let total_bytes = estimate
        .weights_bytes
        .saturating_add(kv_total_bytes)
        .saturating_add(cb_total_bytes)
        .saturating_add(estimate.mtp_bytes);

    let cpu_resident_bytes = estimate.non_layer.token_embd_bytes;
    let gpu_weights_bytes = estimate.weights_bytes.saturating_sub(cpu_resident_bytes);
    let gpu_total_bytes = gpu_weights_bytes
        .saturating_add(kv_total_bytes)
        .saturating_add(
            (estimate.compute_buffer_mb as u64)
                .saturating_mul(active_devices.min(2))
                .saturating_mul(1024 * 1024),
        )
        .saturating_add(estimate.mtp_bytes);

    let mut out = json!({
        "architecture": estimate.architecture.as_str(),
        "context": estimate.context,
        "weights_bytes": estimate.weights_bytes,
        "weights_gib": estimate.weights_bytes as f64 / 1024.0_f64.powi(3),
        "kv_per_token_bytes": estimate.kv_per_token,
        "kv_total_bytes": kv_total_bytes,
        "kv_total_mib": kv_total_bytes / (1024 * 1024),
        "compute_buffer_mb": estimate.compute_buffer_mb,
        "tensor_split_replicated_mib": estimate.tensor_split_replicated_bytes / (1024 * 1024),
        "mtp_bytes": estimate.mtp_bytes,
        "mtp_mib": estimate.mtp_bytes / (1024 * 1024),
        "output_buffer_bytes": estimate.output_buffer_bytes,
        "output_buffer_mib": estimate.output_buffer_bytes / (1024 * 1024),
        "host_overhead_mib": estimate.host_overhead_bytes / (1024 * 1024),
        "host_cache_mib": estimate.host_cache_bytes / (1024 * 1024),
        "per_layer_count": estimate.per_layer_bytes.as_ref().map(|v| v.len()),
        "non_layer_output_head_bytes": estimate.non_layer.output_head_bytes,
        "non_layer_token_embd_bytes": estimate.non_layer.token_embd_bytes,
        "non_layer_other_bytes": estimate.non_layer.other_bytes,
        "expert_layer_count": estimate.expert_layers.len(),
        "expert_tensor_count": estimate.expert_tensors.as_ref().map(|v| v.len()),
        "expert_total_bytes": estimate
            .expert_tensors
            .as_ref()
            .map(|v| v.iter().map(|e| e.bytes).sum::<u64>()),
        "override_tensor_bytes": estimate.override_tensor_bytes
            .iter()
            .map(|(k, v)| (format!("{k:?}"), serde_json::Value::from(*v)))
            .collect::<serde_json::Map<_, _>>(),
        "total_accounted_bytes": total_bytes,
        "total_accounted_mib": total_bytes / (1024 * 1024),
        "gpu_vram_bytes": gpu_total_bytes,
        "gpu_vram_mib": gpu_total_bytes / (1024 * 1024),
    });

    if args.pack {
        let placement = build_placement(&args);
        let snapshot = build_snapshot(&args);
        match pack_demand(&estimate, &placement, &snapshot, Corrections::NEUTRAL) {
            Ok(packed) => {
                let allocation: serde_json::Map<_, _> = packed
                    .allocation
                    .bytes
                    .iter()
                    .map(|(id, bytes)| {
                        let key = id.as_display().to_string();
                        (
                            key,
                            json!({
                                "bytes": bytes,
                                "mib": bytes / (1024 * 1024),
                                "gib": *bytes as f64 / 1024.0_f64.powi(3),
                            }),
                        )
                    })
                    .collect();
                out["placement"] = json!({
                    "allocation": allocation,
                    // The reservation above carries deliberate slop — one
                    // layer's worth of headroom, the prompt cache's cap — that
                    // the process is not expected to use. The prediction is
                    // what the rolling correction divides an observation by, so
                    // it is the figure to compare against a measurement.
                    "predicted_vram_bytes": packed.rolling.uncorrected_vram_bytes,
                    "predicted_vram_mib": packed.rolling.uncorrected_vram_bytes / (1024 * 1024),
                    "predicted_host_bytes": packed.rolling.uncorrected_host_bytes,
                    "gpu_weight_bytes": packed.rolling.gpu_weight_bytes,
                    "expert_offload_bytes": packed.expert_offload_bytes,
                    "expert_offload_mib": packed.expert_offload_bytes / (1024 * 1024),
                    "expert_offload_layers": packed.expert_offload_layers,
                });
            }
            Err(e) => {
                out["placement"] = json!({"error": e.to_string()});
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
