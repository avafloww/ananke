//! Compare packed estimates against production `nvidia-smi` totals.
//!
//! The top-level pass/fail signal for the estimation campaign: seven models the
//! operator actually serves, each estimated the way the daemon would and compared
//! against what the driver reported for the running process.
//!
//! The estimator and the packer run in-process, once per model.

use std::{collections::BTreeMap, path::Path, process::ExitCode};

use ananke_calibrate::{
    derive::dataset::instant,
    models,
    record::read_ndjson,
    validate::{NEUTRAL, snapshot_for},
};
use ananke_fs::LocalFs;
use ananke_measure::record::Status;
use ananke_placement::pack_demand;

const MEASUREMENTS: &str = "calibration/data/measurements.ndjson";
const MODELS_TOML: &str = "calibration/models.toml";
const TOLERANCE_PCT: f64 = 5.0;

/// An ISO 8601 timestamp truncated to its `YYYY-MM-DD` date, which is as precise
/// as a measurement's age needs to be read.
const ISO_DATE_LEN: usize = 10;

/// A `models.toml` entry name against the `factors.label` of its production cell.
struct ProdModel {
    config: &'static str,
    label: &'static str,
}

const PROD_LABELS: &[ProdModel] = &[
    ProdModel {
        config: "qwen3.6-35b-a3b",
        label: "prod-qwen36-35b-a3b",
    },
    ProdModel {
        config: "qwen3.6-27b",
        label: "prod-qwen36-27b",
    },
    ProdModel {
        config: "gemma-4-31b-it-qat",
        label: "prod-gemma4-31b-qat",
    },
    ProdModel {
        config: "deepseek-v4-flash",
        label: "prod-dsv4f",
    },
    ProdModel {
        config: "glm-5.2",
        label: "prod-glm52",
    },
    ProdModel {
        config: "laguna-s-2.1-iq4-nl",
        label: "prod-laguna",
    },
    ProdModel {
        config: "talkie-1930-13b-it",
        label: "prod-talkie",
    },
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
        if !label.starts_with("prod-") || record.status != Status::Ok {
            continue;
        }
        let Some(used) = record.rss.gpu_used_mib else {
            continue;
        };
        let when = record.provenance.measured_at_utc.clone();
        // A suffixed re-measurement counts for the label it re-measures.
        let base = label
            .rsplit_once('-')
            .map(|(head, _)| head)
            .unwrap_or(label);
        for candidate in [label, base] {
            if let Some(target) = PROD_LABELS.iter().find(|m| m.label == candidate) {
                let entry = production
                    .entry(target.label)
                    .or_insert((used, when.clone()));
                if instant(&when) > instant(&entry.1) {
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
    for ProdModel {
        config: name,
        label,
    } in PROD_LABELS
    {
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
            &when[..when.len().min(ISO_DATE_LEN)]
        );
    }
    println!("\nworst drift: {worst:.1}% (tolerance {TOLERANCE_PCT:.0}%)");
    if worst <= TOLERANCE_PCT {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
