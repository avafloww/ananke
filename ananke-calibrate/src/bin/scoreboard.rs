//! Compare packed estimates against production `nvidia-smi` totals.
//!
//! The top-level pass/fail signal for the estimation campaign: seven models the
//! operator actually serves, each estimated the way the daemon would and compared
//! against what the driver reported for the running process.
//!
//! The estimator and the packer run in-process, once per model.

use std::{collections::BTreeMap, path::Path, process::ExitCode};

use ananke_calibrate::{
    models,
    record::read_ndjson,
    validate::{NEUTRAL, snapshot_for},
};
use ananke_fs::LocalFs;
use ananke_placement::pack_demand;

const MEASUREMENTS: &str = "scripts/calibration/data/measurements.ndjson";
const MODELS_TOML: &str = "scripts/calibration/models.toml";
const TOLERANCE_PCT: f64 = 5.0;

/// A `models.toml` entry name against the `factors.label` of its production cell.
const PROD_LABELS: &[(&str, &str)] = &[
    ("qwen3.6-35b-a3b", "prod-qwen36-35b-a3b"),
    ("qwen3.6-27b", "prod-qwen36-27b"),
    ("gemma-4-31b-it-qat", "prod-gemma4-31b-qat"),
    ("deepseek-v4-flash", "prod-dsv4f"),
    ("glm-5.2", "prod-glm52"),
    ("laguna-s-2.1-iq4-nl", "prod-laguna"),
    ("talkie-1930-13b-it", "prod-talkie"),
];

fn main() -> ExitCode {
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
    let configs = match models::load(Path::new(MODELS_TOML)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    // The most recent reading per label, not the first. A runtime upgrade changes
    // what the same configuration costs, and the old row then describes a
    // different program: ik's DSA compute buffer shrank by a third between two
    // builds — GLM-5.2 measured 11524 MiB of it on one and 7794 on the next, at
    // byte-identical settings — so a stale row would have the estimator chasing a
    // number the installed runtime no longer produces. A re-measurement under
    // `prod-<name>-<suffix>` supersedes the original by date.
    let mut production: BTreeMap<&str, (u64, String)> = BTreeMap::new();
    for record in &records {
        let label = record.factors.label.as_str();
        if !label.starts_with("prod-") || record.status != "ok" {
            continue;
        }
        let Some(used) = record.gpu_used_mib() else {
            continue;
        };
        let when = record.provenance.measured_at_utc.clone();
        // A suffixed re-measurement counts for the label it re-measures.
        let base = label
            .rsplit_once('-')
            .map(|(head, _)| head)
            .unwrap_or(label);
        for candidate in [label, base] {
            if let Some((_, target)) = PROD_LABELS.iter().find(|(_, l)| *l == candidate) {
                let entry = production.entry(*target).or_insert((used, when.clone()));
                if when > entry.1 {
                    *entry = (used, when.clone());
                }
            }
        }
    }

    let fs = LocalFs;
    println!(
        "{:<24} {:>9} {:>9} {:>8}  measured",
        "Model", "Est GPU", "Prod GPU", "Drift"
    );
    let mut worst = 0.0_f64;
    for (name, label) in PROD_LABELS {
        let Some(config) = configs.iter().find(|c| c.name == *name) else {
            println!("{name:<24} {:>9} {:>9} {:>8}", "-", "-", "missing");
            continue;
        };
        let Some((measured, when)) = production.get(*label) else {
            println!("{name:<24} {:>9} {:>9} {:>8}", "-", "-", "missing");
            continue;
        };
        let model = config.model_path();
        let mmproj = config.mmproj_path();
        let draft = config.draft_path();
        let inputs = config.estimator_inputs(&model, mmproj.as_deref(), draft.as_deref());
        let estimate = match ananke_estimate::estimate_from_path(&fs, &inputs) {
            Ok(e) => e,
            Err(e) => {
                println!("{name:<24} estimator refused: {e}");
                continue;
            }
        };
        let placement = config.placement_inputs();
        let snap = snapshot_for(&placement.gpu_allow, &[]);
        let packed = match pack_demand(&estimate, &placement, &snap, NEUTRAL) {
            Ok(p) => p,
            Err(e) => {
                println!("{name:<24} packer refused: {e:?}");
                continue;
            }
        };
        // The reservation summed over the placement's GPU slots — the figure the
        // operator's cards actually hold.
        let est = ananke_calibrate::validate::gpu_reserved_mib(&packed);
        let drift = 100.0 * (est as f64 - *measured as f64) / *measured as f64;
        let flag = if drift.abs() <= TOLERANCE_PCT {
            ""
        } else {
            "  <-- FAIL"
        };
        worst = worst.max(drift.abs());
        println!(
            "{name:<24} {est:>9} {measured:>9} {drift:>+7.1}%  {}{flag}",
            &when[..when.len().min(10)]
        );
    }
    println!("\nworst drift: {worst:.1}% (tolerance {TOLERANCE_PCT:.0}%)");
    if worst <= TOLERANCE_PCT {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
