//! Prints muse-glimmer's cells from the real campaign (every question,
//! merged and deduplicated via `all_cells`), for review before running
//! anything. Real `$LLM_DIR`, so paths resolve to what's actually on disk.
//!
//! Usage: LLM_DIR=... cargo run -p ananke-calibrate --example muse-glimmer-cells

use ananke_calibrate::plan::{all_cells, library::Library, to_json};

fn main() {
    let lib = Library::from_env();
    let mine: Vec<_> = all_cells(&lib)
        .into_iter()
        .filter(|c| c.label.contains("muse-glimmer"))
        .collect();
    println!("{}", to_json(&mine));
}
