//! The `compute_model` section of `tuning.json`, emitted from the fitted groups.

use std::collections::BTreeMap;

use ananke_config::placement::SplitMode;
use ananke_estimate::compute_model::{Columns, Scalars};
use ananke_tuning_schema::compute_model::{Entry, Fit, Section};

use crate::compute_model::{Coefficients, Group, Groups, Row, evaluate, fit};

/// The `compute_model` section for `tuning.json`, and per-group coverage notes.
///
/// Entries are ordered variant-guarded first, so a lookup that scans for the
/// first matching architecture finds the specific graph before the general one.
/// The `default` entry pools every mainline layer-split observation, so an
/// unmeasured architecture falls back to data rather than to whichever entry
/// happened to be listed first.
pub fn document_section(groups: &Groups) -> Result<(Section, Vec<String>), String> {
    let mut ordered: Vec<(&Group, &[Row])> = groups.iter().collect();
    // Largest group first, then by key. `variant` sorts `None` before `Some`,
    // which only ever decides ties the row counts already separate.
    ordered.sort_by(|(a, left), (b, right)| {
        right.len().cmp(&left.len()).then_with(|| {
            (&a.runtime, &a.split, &a.arch, a.variant)
                .cmp(&(&b.runtime, &b.split, &b.arch, b.variant))
        })
    });

    let mut entries: Vec<Entry> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for (key, points) in ordered {
        let label = match key.variant {
            Some(variant) => format!("{}/{}/{}@{variant}", key.runtime, key.split, key.arch),
            None => format!("{}/{}/{}", key.runtime, key.split, key.arch),
        };
        let Some((coefficients, worst)) = fit(points) else {
            notes.push(format!(
                "{label}: {} row(s), not enough to fit",
                points.len()
            ));
            continue;
        };
        let outside = points
            .iter()
            .filter(|p| (evaluate(&coefficients, &p.columns) - p.target).abs() / p.target > 0.05)
            .count();
        entries.push(Entry {
            archs: vec![key.arch.clone()],
            coefficients: rounded(&coefficients),
            evidence: format!(
                "non-negative weighted least squares over {} per-device \
                 observation(s); worst residual {:.1}%, {outside} outside +/-5%",
                points.len(),
                worst * 100.0
            ),
            runtime: (key.runtime != "mainline").then(|| key.runtime.clone()),
            split: key.split.as_flag().to_owned(),
            variant: key.variant.map(str::to_owned),
        });
        notes.push(format!(
            "{label}: {} rows, {outside} outside +/-5%, worst {:.1}%",
            points.len(),
            worst * 100.0
        ));
    }
    // Every entry lists exactly one architecture, so ordering the lists orders
    // the architectures.
    entries.sort_by(|a, b| {
        a.variant
            .is_none()
            .cmp(&b.variant.is_none())
            .then_with(|| a.archs.cmp(&b.archs))
            .then_with(|| a.split.cmp(&b.split))
    });

    let pooled: Vec<Row> = groups
        .iter()
        .filter(|(key, _)| key.runtime == "mainline" && key.split == SplitMode::Layer)
        .flat_map(|(_, rows)| rows.iter().cloned())
        .collect();
    let (coefficients, worst) =
        fit(&pooled).ok_or("no pooled mainline layer-split rows to fit a default from")?;

    let section = Section {
        comment: "One per-device compute model per (runtime, split, architecture), \
                  replacing `compute_buffer_curves`, `tensor_compute_curves` and its \
                  companion tables, and the `ik_compute_*` rates. Columns are \
                  dimensionally normalised so architectures of different width share \
                  coefficients; see calibration/crates/calibrate/src/compute_model/mod.rs for what \
                  each one counts and why. Coefficients are constrained non-negative \
                  because every column counts bytes of a buffer that either exists or \
                  does not. Ordered variant-guarded first."
            .to_string(),
        columns: column_names(),
        default: Fit {
            coefficients: rounded(&coefficients),
            evidence: format!(
                "pooled over {} mainline layer-split observations across every \
                 measured architecture; worst residual {:.1}%",
                pooled.len(),
                worst * 100.0
            ),
        },
        entries,
    };
    Ok((section, notes))
}

/// The column names, in the order `tuning.json` declares them.
fn column_names() -> Vec<String> {
    Columns::from_scalars(Scalars {
        ubatch: 0.0,
        n_kv: 0.0,
        ctx: 0.0,
        quantised: false,
        head_share: 0.0,
        n_vocab: 0.0,
        n_embd: 0.0,
        offloading: false,
        mask_copies: 0.0,
    })
    .by_name()
    .into_iter()
    .map(|(name, _)| name.to_string())
    .collect()
}

/// Coefficients at the precision `tuning.json` carries them.
fn rounded(coefficients: &Coefficients) -> BTreeMap<String, f64> {
    coefficients
        .iter()
        .map(|(name, value)| ((*name).to_string(), (value * 1e6).round() / 1e6))
        .collect()
}
