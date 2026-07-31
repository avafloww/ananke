//! The `compute_model` section of `tuning.json`, emitted from the fitted groups.

use std::collections::BTreeMap;

use ananke_estimate::compute_model::{Columns, Scalars};
use serde_json::{Value, json};

use crate::compute_model::{Coefficients, Group, Groups, Row, evaluate, fit};

/// The `compute_model` section for `tuning.json`, and per-group coverage notes.
///
/// Entries are ordered variant-guarded first, so a lookup that scans for the
/// first matching architecture finds the specific graph before the general one,
/// matching the convention the curve tables already use. The `default` entry
/// pools every mainline layer-split observation: with the columns dimensionally
/// normalised, pooling across architectures is what the design is for, and it
/// gives an architecture nobody has measured a fallback derived from data rather
/// than borrowed from whichever entry happened to be listed first.
pub fn document_section(groups: &Groups) -> Result<(Value, Vec<String>), String> {
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
            arch: key.arch.clone(),
            variant: key.variant,
            split: key.split.clone(),
            value: json!({
                "archs": [&key.arch],
                "variant": key.variant,
                "runtime": (key.runtime != "mainline").then(|| key.runtime.clone()),
                "split": &key.split,
                "coefficients": rounded(&coefficients),
                "evidence": format!(
                    "non-negative weighted least squares over {} per-device \
                     observation(s); worst residual {:.1}%, {outside} outside +/-5%",
                    points.len(),
                    worst * 100.0
                ),
            }),
        });
        notes.push(format!(
            "{label}: {} rows, {outside} outside +/-5%, worst {:.1}%",
            points.len(),
            worst * 100.0
        ));
    }
    entries.sort_by(|a, b| {
        a.variant
            .is_none()
            .cmp(&b.variant.is_none())
            .then_with(|| a.arch.cmp(&b.arch))
            .then_with(|| a.split.cmp(&b.split))
    });

    let pooled: Vec<Row> = groups
        .iter()
        .filter(|(key, _)| key.runtime == "mainline" && key.split == "layer")
        .flat_map(|(_, rows)| rows.iter().cloned())
        .collect();
    let (coefficients, worst) =
        fit(&pooled).ok_or("no pooled mainline layer-split rows to fit a default from")?;

    let section = json!({
        "$comment":
            "One per-device compute model per (runtime, split, architecture), \
             replacing `compute_buffer_curves`, `tensor_compute_curves` and its \
             companion tables, and the `ik_compute_*` rates. Columns are \
             dimensionally normalised so architectures of different width share \
             coefficients; see calibration/crates/calibrate/src/compute_model/mod.rs for what \
             each one counts and why. Coefficients are constrained non-negative \
             because every column counts bytes of a buffer that either exists or \
             does not. Ordered variant-guarded first.",
        "columns": column_names(),
        "entries": entries.into_iter().map(|e| e.value).collect::<Vec<_>>(),
        "default": {
            "coefficients": rounded(&coefficients),
            "evidence": format!(
                "pooled over {} mainline layer-split observations across every \
                 measured architecture; worst residual {:.1}%",
                pooled.len(),
                worst * 100.0
            ),
        },
    });
    Ok((section, notes))
}

/// A fitted entry, kept with its sort key alongside the JSON it will become.
struct Entry {
    arch: String,
    variant: Option<&'static str>,
    split: String,
    value: Value,
}

/// The column names, in the order `tuning.json` declares them.
fn column_names() -> Vec<&'static str> {
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
    .map(|(name, _)| name)
    .collect()
}

/// Coefficients at the precision `tuning.json` carries them.
fn rounded(coefficients: &Coefficients) -> Value {
    json!(
        coefficients
            .iter()
            .map(|(name, value)| (*name, (value * 1e6).round() / 1e6))
            .collect::<BTreeMap<_, _>>()
    )
}
