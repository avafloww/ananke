//! Compare the estimator against every comparable measured cell.
//!
//! The estimator and the packer run in-process, once per cell. Spawning
//! `cargo run --example estimate` instead would be two hundred-odd process
//! launches, each re-reading its GGUF.

use std::{collections::BTreeMap, path::Path, process::ExitCode};

use ananke_calibrate::{
    record::{Record, read_ndjson},
    validate::{
        Comparison, NEUTRAL, configuration_key, estimator_inputs, placement_inputs, skip_reason,
        snapshot,
    },
};
use ananke_fs::LocalFs;
use ananke_placement::{devices::DeviceId, pack_demand};

const MEASUREMENTS: &str = "calibration/data/measurements.ndjson";

fn main() -> ExitCode {
    let mut tolerance = 5.0_f64;
    let mut check = false;
    let mut arch_filter: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tolerance" => tolerance = args.next().and_then(|v| v.parse().ok()).unwrap_or(5.0),
            "--check" => check = true,
            "--arch" => arch_filter = args.next(),
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let text = match std::fs::read_to_string(MEASUREMENTS) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };
    let records = match read_ndjson(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parsing {MEASUREMENTS}: {e}");
            return ExitCode::from(2);
        }
    };

    // A cell whose model file has gone is not a validation failure.
    let known: Vec<String> = records
        .iter()
        .map(|r| r.factors.model.clone())
        .filter(|m| Path::new(m).exists())
        .collect();

    let fs = LocalFs;
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen: Vec<String> = Vec::new();
    let mut results: Vec<Comparison> = Vec::new();

    for record in &records {
        if let Some(reason) = skip_reason(record, &known) {
            *skipped.entry(reason).or_default() += 1;
            continue;
        }
        if let Some(want) = &arch_filter
            && record.parsed.arch.as_deref() != Some(want.as_str())
        {
            continue;
        }
        let key = configuration_key(record);
        if seen.contains(&key) {
            *skipped.entry("duplicate configuration".into()).or_default() += 1;
            continue;
        }
        seen.push(key);

        match compare(&fs, record) {
            Ok(comparison) => results.push(comparison),
            Err(reason) => *skipped.entry(reason).or_default() += 1,
        }
    }

    results.sort_by(|a, b| a.drift_pct.total_cmp(&b.drift_pct));
    println!(
        "{:38}{:18}{:>10}{:>10}{:>9}{:>10}",
        "label", "arch", "predicted", "measured", "drift", "reserved"
    );
    for row in &results {
        let flag = if row.drift_pct.abs() <= tolerance {
            ""
        } else {
            "  <-- outside"
        };
        println!(
            "{:38}{:18}{:>10}{:>10}{:>+8.1}%{:>10}{}",
            truncate(&row.label, 37),
            truncate(&row.arch, 17),
            row.predicted_mib,
            row.measured_mib,
            row.drift_pct,
            row.reserved_mib,
            flag
        );
    }

    let total_skipped: usize = skipped.values().sum();
    println!(
        "\n{} cells validated, {total_skipped} skipped",
        results.len()
    );
    let mut by_count: Vec<_> = skipped.iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, count) in by_count {
        println!("  {count:4}  {reason}");
    }

    if results.is_empty() {
        return ExitCode::SUCCESS;
    }
    let mut drifts: Vec<f64> = results.iter().map(|r| r.drift_pct).collect();
    drifts.sort_by(f64::total_cmp);
    let median = drifts[drifts.len() / 2];
    let mean = drifts.iter().sum::<f64>() / drifts.len() as f64;
    let outside = results
        .iter()
        .filter(|r| r.drift_pct.abs() > tolerance)
        .count();
    println!(
        "\nmedian {median:+.1}%  mean {mean:+.1}%  range {:+.1}% to {:+.1}%",
        drifts[0],
        drifts[drifts.len() - 1]
    );
    println!("{outside} of {} outside +/-{tolerance}%", results.len());
    if check && outside > 0 {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Estimate and pack one cell, and compare the prediction to the measurement.
fn compare(fs: &LocalFs, record: &Record) -> Result<Comparison, String> {
    let model = Path::new(&record.factors.model);
    let inputs = estimator_inputs(record, model);
    let estimate = ananke_estimate::estimate_from_path(fs, &inputs)
        .map_err(|_| "estimator refused the configuration".to_string())?;
    let placement = placement_inputs(record);
    let snap = snapshot(record);
    let packed = pack_demand(&estimate, &placement, &snap, NEUTRAL)
        .map_err(|_| "estimator refused the configuration".to_string())?;

    let predicted_mib = packed.rolling.uncorrected_vram_bytes / (1024 * 1024);
    if predicted_mib == 0 {
        return Err("nothing placed on a GPU".into());
    }
    let reserved_mib: u64 = packed
        .allocation
        .bytes
        .iter()
        .filter(|(id, _)| matches!(id, DeviceId::Gpu(_)))
        .map(|(_, &b)| b)
        .sum::<u64>()
        / (1024 * 1024);
    // llama.cpp's layer split spreads across every visible card; ananke packs a
    // model that fits onto one, deliberately. When the two land on different
    // numbers of cards the totals are not comparable — a second card is a second
    // CUDA context and a second compute buffer, ~450 MiB of real cost that
    // belongs to the placement rather than to the estimate.
    let cards_placed = packed
        .allocation
        .bytes
        .iter()
        .filter(|(id, b)| matches!(id, DeviceId::Gpu(_)) && **b > 0)
        .count();
    let cards_measured = record.cards_measured();
    if cards_measured > 0 && cards_placed != cards_measured {
        return Err(format!(
            "placed on {cards_placed} card(s), measured on {cards_measured}"
        ));
    }
    let measured_mib = record.gpu_used_mib().unwrap_or(0);
    Ok(Comparison {
        label: record.factors.label.clone(),
        arch: record.parsed.arch.clone().unwrap_or_else(|| "?".into()),
        predicted_mib,
        measured_mib,
        reserved_mib,
        drift_pct: 100.0 * (predicted_mib as f64 - measured_mib as f64) / measured_mib as f64,
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}
