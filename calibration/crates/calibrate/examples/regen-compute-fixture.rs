//! Regenerate `tests/fixtures/compute_fit.json` from the current dataset.
//!
//! Run after any change that alters `compute_model::collect`/`fit`/`document_section`'s
//! output, or after the dataset itself changes (a new model's rows, corrected
//! measurements) — `compute_model.rs` pins the fitter's output to this fixture, and
//! this is how the fixture is kept in step rather than hand-edited.

use ananke_calibrate::{
    compute_model::{Groups, collect, document_section, fit},
    derive::dataset::latest_per_cell,
    record::read_ndjson,
};
use ananke_config::placement::SplitMode;
use ananke_measure::record::Status;
use serde_json::json;

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let data_path = format!("{manifest}/../../data/measurements.ndjson");
    let fixture_path = format!("{manifest}/tests/fixtures/compute_fit.json");

    let text = std::fs::read_to_string(&data_path).expect("the measurement file is checked in");
    let records: Vec<_> = read_ndjson(&text)
        .expect("every record parses")
        .into_iter()
        .filter(|r| r.status == Status::Ok)
        .collect();
    let rows = latest_per_cell(&records);
    let copies = ananke_estimate::tuning::MAINLINE_LAYER_SPLIT_MASK_COPIES as u32;
    let groups: Groups = collect(&rows, copies, false);

    let group_entries: Vec<_> = groups
        .iter()
        .map(|(key, rows)| {
            let (coefficients, worst) = fit(rows).unzip();
            json!({
                "runtime": key.runtime,
                "split": key.split.as_flag(),
                "arch": key.arch,
                "variant": key.variant,
                "rows": rows.len(),
                "coefficients": coefficients.map(|c| c.into_iter().collect::<std::collections::BTreeMap<_, _>>()),
                "worst": worst,
                "targets_sum": rows.iter().map(|r| r.target).sum::<f64>(),
            })
        })
        .collect();

    let pooled: Vec<_> = groups
        .iter()
        .filter(|(key, _)| key.runtime == "mainline" && key.split == SplitMode::Layer)
        .flat_map(|(_, rows)| rows.iter().cloned())
        .collect();
    let (pooled_coefficients, pooled_worst) = fit(&pooled).expect("the pooled default fits");

    let (section, notes) = document_section(&groups).expect("the section builds");

    let fixture = json!({
        "groups": group_entries,
        "pooled": {
            "rows": pooled.len(),
            "coefficients": pooled_coefficients.into_iter().collect::<std::collections::BTreeMap<_, _>>(),
            "worst": pooled_worst,
        },
        "notes": notes,
        "section": section,
    });

    std::fs::write(
        &fixture_path,
        serde_json::to_string_pretty(&fixture).unwrap() + "\n",
    )
    .expect("fixture writes");
    println!(
        "wrote {fixture_path} from {} groups, {} pooled rows",
        groups.len(),
        pooled.len()
    );
}
