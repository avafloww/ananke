//! Regenerates tests/fixtures/plans.json.gz from the current plan generator.
//!
//! Run after any change to the plan (a new question, a new model, a changed
//! cell shape) that makes tests/plan.rs fail on a stale fixture. Overwrites
//! the fixture unconditionally — diff it before committing.
//!
//! Usage: cargo run -p ananke-calibrate --example regen-plan-fixture

use std::{collections::BTreeMap, fs::File, io::Write};

use ananke_calibrate::plan::{QUESTIONS, all_cells, cells_for, library::Library, to_json};
use flate2::{Compression, write::GzEncoder};

const FAKE_ROOT: &str = "/fake/llm";

fn main() {
    let lib = Library::rooted(FAKE_ROOT);
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (name, _) in QUESTIONS {
        let cells = cells_for(name, &lib).expect("registered question");
        out.insert((*name).to_owned(), to_json(&cells));
    }
    out.insert("all".to_owned(), to_json(&all_cells(&lib)));

    let path = format!(
        "{}/tests/fixtures/plans.json.gz",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = serde_json::to_string(&out).expect("serialises");
    let mut encoder = GzEncoder::new(File::create(&path).expect("creates"), Compression::default());
    encoder.write_all(text.as_bytes()).expect("writes");
    encoder.finish().expect("finishes");
    println!("wrote {path}");
}
