//! Every plan the generator can emit, held against a recorded fixture.
//!
//! Generation is deterministic — the same arguments produce the same plan, byte
//! for byte — so the fixture is one capture of it under `LLM_DIR=/fake/llm`, for
//! all twenty-two questions plus `all`. Ordered lists are compared, not sets:
//! nothing here is random or set-iterated, and the run *order* is itself part of
//! what the planner decides, since the whole point of `all` is to visit each
//! model's cells while its weights are hot.
//!
//! The sort key deliberately carries no `thp` term. That factor was removed when
//! the llama.cpp build turned out to reject `--use-thp`, and the fixture was
//! captured without it.
//!
//! `/fake/llm` is a root that does not exist, so every model reads as unreadable
//! and the size term in the ordering is constant. That makes the fixture
//! reproducible on any box while still exercising the merge and the whole rest of
//! the sort key. The size term itself was checked separately against the real
//! library, and the shard-summing is unit-tested in `plan::tests`.

use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use ananke_calibrate::plan::{QUESTIONS, all_cells, cells_for, library::Library, to_json};
use flate2::read::GzDecoder;
use pretty_assertions::assert_eq;

/// The root the fixture was captured against. It must not resolve, so that the
/// ordering's model-size term is the same everywhere.
const FAKE_ROOT: &str = "/fake/llm";

#[test]
fn every_question_matches_the_fixture() {
    let expected = fixture();
    let lib = Library::rooted(FAKE_ROOT);
    assert!(
        !Path::new(FAKE_ROOT).exists(),
        "the fixture's library root must not exist on this box"
    );

    for (name, _) in QUESTIONS {
        let cells = cells_for(name, &lib).expect("the question is registered");
        let want = expected
            .get(*name)
            .expect("the fixture covers the question");
        assert_eq!(&to_json(&cells), want, "plan for {name}");
    }
}

/// The merged campaign: every question's cells de-duplicated, tagged with each
/// question that asked for them, and ordered smallest-model-first.
#[test]
fn the_whole_campaign_matches_the_fixture() {
    let expected = fixture();
    let cells = all_cells(&Library::rooted(FAKE_ROOT));
    assert_eq!(&to_json(&cells), expected.get("all").expect("captured"));
}

/// The harness reads a plan written with one space of indentation, so the
/// formatter is part of the contract rather than a presentation choice.
#[test]
fn one_space_of_indentation() {
    let cells = cells_for("noise", &Library::rooted(FAKE_ROOT)).expect("registered");
    let text = to_json(&cells);
    assert!(
        text.starts_with("[\n {\n  \"label\": \"noise\","),
        "{text:.40}"
    );
}

/// Every question in the fixture is still a question, so a rename cannot pass by
/// leaving a stale entry unchecked.
#[test]
fn the_fixture_names_no_retired_question() {
    let names: Vec<&str> = QUESTIONS.iter().map(|(name, _)| *name).collect();
    for name in fixture().keys() {
        assert!(
            name == "all" || names.contains(&name.as_str()),
            "{name} is in the fixture but not in QUESTIONS"
        );
    }
}

fn fixture() -> BTreeMap<String, String> {
    let path = format!(
        "{}/tests/fixtures/plans.json.gz",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut text = String::new();
    GzDecoder::new(File::open(&path).expect("the fixture opens"))
        .read_to_string(&mut text)
        .expect("the fixture decompresses");
    serde_json::from_str(&text).expect("the fixture parses")
}
