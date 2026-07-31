//! Re-parse every archived log and hold the result against the row recorded for
//! it at the time.
//!
//! The campaign's logs are kept precisely so a record can be rebuilt from them,
//! which makes them an oracle: the parser is correct exactly insofar as it
//! reproduces the `parsed` block that the same log already produced. A
//! disagreement is a finding either way round, so the failure message names the
//! log, the field, and both values rather than only asserting inequality.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use ananke_measure::{parse_log, record::Status};
use flate2::read::GzDecoder;
use serde_json::Value;

/// Floats are transcribed from the same decimal text on both sides, so they
/// should be bit-identical; the tolerance is here to describe a rounding
/// difference rather than to hide one.
const TOLERANCE: f64 = 1e-9;

#[test]
fn parsed_blocks_match_the_recorded_rows() {
    let data = data_dir();
    let measurements = std::fs::read_to_string(data.join("measurements.ndjson"))
        .expect("the campaign's measurements are checked in");

    let mut compared = 0usize;
    let mut skipped = 0usize;
    let mut device_rows: BTreeMap<usize, usize> = BTreeMap::new();
    let mut disagreements = Vec::new();

    for line in measurements.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line).expect("each line is one JSON record");
        let log = record["log"].as_str().unwrap_or_default();
        let status: Status = serde_json::from_value(record["status"].clone())
            .expect("every status in the dataset is one this crate knows");
        // A cell that never loaded carries an empty `parsed` by construction,
        // and its log holds only the failure, so there is nothing to compare.
        if status != Status::Ok || log.is_empty() || !data.join("logs").join(log).exists() {
            skipped += 1;
            continue;
        }

        let parsed = parse_log(&read_log(&data.join("logs").join(log)));
        let ours = serde_json::to_value(&parsed).expect("Parsed serializes as a JSON object");
        let theirs = &record["parsed"];

        *device_rows.entry(parsed.devices.len()).or_default() += 1;
        assert!(
            parsed.devices.len() <= 2,
            "{log}: {} device rows for a two-card box, which means more than one \
             breakdown table was read",
            parsed.devices.len()
        );

        if let Err(difference) = compare(&ours, theirs, "parsed") {
            disagreements.push(format!("{log}: {difference}"));
        }
        compared += 1;
    }

    assert!(
        disagreements.is_empty(),
        "{} of {compared} records disagree:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
    // A parser that silently matched nothing would otherwise pass: the dataset
    // is checked in, so its size is a fact the test can assert on.
    assert!(compared > 500, "only {compared} records were comparable");
    assert!(
        device_rows.get(&1).copied().unwrap_or_default() > 50
            && device_rows.get(&2).copied().unwrap_or_default() > 50,
        "both single-card and two-card breakdowns have to be exercised, got {device_rows:?}"
    );
    println!("compared {compared} records ({skipped} skipped), device rows {device_rows:?}");
}

/// The factor set is the other half of the record's schema, and a field the
/// writer spells differently from the harness would go unnoticed by the
/// `parsed` comparison above.
#[test]
fn every_recorded_factor_set_round_trips() {
    let measurements = std::fs::read_to_string(data_dir().join("measurements.ndjson"))
        .expect("the campaign's measurements are checked in");
    for line in measurements.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line).expect("each line is one JSON record");
        let factors: ananke_measure::record::Factors =
            serde_json::from_value(record["factors"].clone())
                .unwrap_or_else(|error| panic!("{}: {error}", record["cell"]));
        let ours = serde_json::to_value(&factors).expect("Factors serializes as a JSON object");
        // Older rows predate a factor and simply do not spell it, so the
        // comparison is one-way: every key the harness wrote has to survive.
        for (key, value) in record["factors"].as_object().expect("factors is an object") {
            assert_eq!(
                ours.get(key),
                Some(value),
                "{}: factor {key} did not round-trip",
                record["cell"]
            );
        }
    }
}

/// A comparator that accepted everything would make the oracle above vacuous,
/// so it is held to the three differences that matter: a value, a missing key,
/// and an array's order.
#[test]
fn the_comparator_catches_a_difference() {
    let ours = serde_json::json!({"compute_mib": 1032, "devices": [{"device": "CUDA0"}]});
    assert!(compare(&ours, &ours, "parsed").is_ok());
    for theirs in [
        serde_json::json!({"compute_mib": 2042, "devices": [{"device": "CUDA0"}]}),
        serde_json::json!({"devices": [{"device": "CUDA0"}]}),
        serde_json::json!({"compute_mib": 1032, "devices": [{"device": "CUDA1"}]}),
        serde_json::json!({"compute_mib": 1032, "devices": []}),
    ] {
        assert!(
            compare(&ours, &theirs, "parsed").is_err(),
            "{theirs} should not compare equal"
        );
    }
}

/// Compare two JSON trees, naming the path of the first difference.
///
/// Objects are compared on their whole key set, so a key one side omits is a
/// difference rather than a default; arrays element-wise, so device rows have to
/// agree on their order as well as their contents.
fn compare(ours: &Value, theirs: &Value, path: &str) -> Result<(), String> {
    match (ours, theirs) {
        (Value::Object(ours), Value::Object(theirs)) => {
            let keys: BTreeSet<&String> = ours.keys().chain(theirs.keys()).collect();
            for key in keys {
                match (ours.get(key), theirs.get(key)) {
                    (Some(ours), Some(theirs)) => compare(ours, theirs, &format!("{path}.{key}"))?,
                    (Some(extra), None) => return Err(format!("{path}.{key} only ours: {extra}")),
                    (None, Some(missing)) => {
                        return Err(format!("{path}.{key} only theirs: {missing}"));
                    }
                    (None, None) => unreachable!("the key came from one of the two maps"),
                }
            }
            Ok(())
        }
        (Value::Array(ours), Value::Array(theirs)) => {
            if ours.len() != theirs.len() {
                return Err(format!(
                    "{path} has {} entries against {}",
                    ours.len(),
                    theirs.len()
                ));
            }
            for (index, (ours, theirs)) in ours.iter().zip(theirs).enumerate() {
                compare(ours, theirs, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        (Value::Number(ours), Value::Number(theirs)) => {
            let (Some(ours), Some(theirs)) = (ours.as_f64(), theirs.as_f64()) else {
                return Err(format!(
                    "{path}: {ours} against {theirs} is not representable"
                ));
            };
            if (ours - theirs).abs() <= TOLERANCE * ours.abs().max(1.0) {
                Ok(())
            } else {
                Err(format!("{path}: {ours} against {theirs}"))
            }
        }
        (ours, theirs) if ours == theirs => Ok(()),
        (ours, theirs) => Err(format!("{path}: {ours} against {theirs}")),
    }
}

/// The logs are archived compressed, and were originally decoded with
/// replacement, so a stray non-UTF-8 byte stays a replacement character here
/// too rather than failing the read.
fn read_log(path: &Path) -> String {
    let mut bytes = Vec::new();
    GzDecoder::new(File::open(path).expect("the archived log opens"))
        .read_to_end(&mut bytes)
        .expect("the archived log decompresses");
    String::from_utf8_lossy(&bytes).into_owned()
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
