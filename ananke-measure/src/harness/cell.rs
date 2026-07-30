//! A cell's identity and its command line.
//!
//! A cell *is* its factor set, so [`crate::record::Factors`] is the only
//! definition of one: the plan writes it, the identity hashes it, the command
//! line is built from it, and the record carries it back. Adding a factor
//! therefore reaches all four at once, which is the property the Python was
//! built around and the one that keeps a flag from being passed to the server
//! while going unrecorded.

use sha2::{Digest, Sha256};

use crate::{
    harness::{
        error::{Error, ErrorKind},
        json::to_python_json,
    },
    record::{Factors, Runtime},
};

/// Stable identity, so a rerun skips what has already been measured.
///
/// Two exclusions carry the weight here.
///
/// The label and the purpose tags are left out: they name the cell and say why it
/// was wanted, but two cells with the same flags are the same measurement
/// whatever they are called, and measuring one configuration twice under two
/// names is pure waste.
///
/// Fields still at their default are left out too, and that is what makes the
/// schema extensible. Hashing every field means adding or removing one changes
/// the identity of *every* cell ever measured, so the harness stops recognising
/// its own dataset and re-measures all of it — days of GPU time, against a
/// different llama.cpp build. Excluding defaults costs nothing, because two cells
/// differing in a field still hash differently (one of them is not at the
/// default), and it makes a new defaulted field free.
///
/// Everything else is in, deliberately and without exception. A factor missing
/// from this key is two different measurements sharing one identity, and the
/// second silently never runs; a `cram` omitted from a cell identity is one of
/// the bugs this campaign actually shipped.
pub(crate) fn cell_id(factors: &Factors) -> String {
    let mine = serde_json::to_value(factors).expect("Factors serializes as a JSON object");
    let defaults =
        serde_json::to_value(Factors::default()).expect("Factors serializes as a JSON object");
    let (Some(mine), Some(defaults)) = (mine.as_object(), defaults.as_object()) else {
        unreachable!("Factors is a struct, so both serialize as objects");
    };
    // A `BTreeMap` because the payload is hashed with sorted keys: the map's
    // order *is* part of the identity.
    let payload: std::collections::BTreeMap<&String, &serde_json::Value> = mine
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "label" | "purpose"))
        .filter(|(key, value)| defaults.get(*key) != Some(*value))
        .collect();
    let digest = Sha256::digest(to_python_json(&payload).as_bytes());
    // Twelve hex characters, as every existing row spells it.
    format!("{digest:x}")[..12].to_owned()
}

/// The server's command line, in the order the campaign ran it.
///
/// The order matters less than the completeness: every factor that reaches the
/// server has to appear here, or a row describes a process that was not the one
/// measured.
pub(crate) fn argv(factors: &Factors, binary: &str, port: u16) -> Vec<String> {
    let mut argv: Vec<String> = [
        binary,
        "-m",
        &factors.model,
        "-c",
        &factors.ctx.to_string(),
        "-ub",
        &factors.ubatch.to_string(),
        "-ngl",
        &factors.ngl.to_string(),
        "-np",
        &factors.parallel.to_string(),
        "-cram",
        &factors.cram.to_string(),
        "-fa",
        &factors.flash_attn,
        "-ctk",
        &factors.kv_type,
        "-ctv",
        &factors.kv_type,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]
    .iter()
    .map(|argument| (*argument).to_owned())
    .collect();

    // The two runtimes gate their buffer-size logging differently, and without
    // those lines there is nothing to measure.
    if factors.verbose_log {
        argv.extend(match factors.runtime {
            Runtime::Mainline => ["-lv".to_owned(), "5".to_owned()],
            Runtime::Ik => ["--verbosity".to_owned(), "1".to_owned()],
        });
    }
    let optional: [(&str, Option<String>); 8] = [
        ("-b", factors.batch.map(|value| value.to_string())),
        ("--split-mode", factors.split.clone()),
        (
            "--n-cpu-moe",
            factors.n_cpu_moe.map(|value| value.to_string()),
        ),
        ("--mmproj", factors.mmproj.clone()),
        ("-md", factors.draft.clone()),
        ("--spec-type", factors.spec_type.clone()),
        ("-t", factors.threads.map(|value| value.to_string())),
        ("--numa", factors.numa.clone()),
    ];
    for (flag, value) in optional {
        if let Some(value) = value {
            argv.push(flag.to_owned());
            argv.push(value);
        }
    }
    for (flag, on) in [
        ("-kvu", factors.kv_unified),
        ("--no-mmap", factors.no_mmap),
        ("-rtr", factors.rtr),
        ("--embeddings", factors.embeddings),
    ] {
        if on {
            argv.push(flag.to_owned());
        }
    }
    argv.extend(factors.extra.iter().cloned());
    argv
}

/// Read a campaign plan: a JSON list of objects, each a factor set.
///
/// Strict about keys the plan spells and this harness does not know, rather than
/// letting serde drop them. A dropped key is a cell measured under different
/// flags than the plan asked for, recorded as though it had been measured under
/// the plan's — the exact shape of several of this campaign's wrong constants.
pub(crate) fn load_plan(text: &str) -> Result<Vec<Factors>, Error> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(text)
        .map_err(|error| Error::new(ErrorKind::Plan, format!("read the plan: {error}")))?;
    let known = serde_json::to_value(Factors::default()).expect("Factors serializes as an object");
    let known = known.as_object().expect("Factors is a struct");
    let mut cells = Vec::with_capacity(entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        let object = entry
            .as_object()
            .ok_or_else(|| Error::new(ErrorKind::Plan, format!("cell {index} is not an object")))?;
        if let Some(unknown) = object.keys().find(|key| !known.contains_key(*key)) {
            return Err(Error::new(
                ErrorKind::Plan,
                format!(
                    "cell {index} spells `{unknown}`, which is not a factor this harness knows; \
                     regenerate the plan against the current factor set"
                ),
            ));
        }
        cells.push(
            serde_json::from_value(entry)
                .map_err(|error| Error::new(ErrorKind::Plan, format!("cell {index}: {error}")))?,
        );
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> Factors {
        Factors {
            label: "glm-2card-moe40".to_owned(),
            model: "/models/glm.gguf".to_owned(),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            ctx: 65536,
            n_cpu_moe: Some(40),
            split: Some("layer".to_owned()),
            kv_unified: true,
            extra: vec!["-dsa".to_owned()],
            ..Factors::default()
        }
    }

    #[test]
    fn the_command_line_carries_every_factor_it_should() {
        let argv = argv(&cell(), "ik-llama-server", 18099);
        let joined = argv.join(" ");
        assert!(joined.starts_with("ik-llama-server -m /models/glm.gguf -c 65536 -ub 512"));
        assert!(
            joined.contains("--verbosity 1"),
            "ik spells verbosity its own way: {joined}"
        );
        assert!(joined.contains("--split-mode layer --n-cpu-moe 40"));
        assert!(joined.ends_with("-kvu -dsa"), "flags then extras: {joined}");
        assert!(!joined.contains("-lv"), "that is mainline's spelling");
    }

    #[test]
    fn the_label_and_the_purpose_do_not_change_the_identity() {
        let mut renamed = cell();
        renamed.label = "something-else".to_owned();
        renamed.purpose = vec!["switches".to_owned()];
        assert_eq!(cell_id(&cell()), cell_id(&renamed));
    }

    /// The bug class the identity exists to prevent: a factor that moves memory
    /// and leaves the key alone, so the second configuration is skipped as
    /// already measured.
    #[test]
    fn every_factor_moves_the_identity() {
        let base = cell();
        let baseline = cell_id(&base);
        let mutations: Vec<(&str, Factors)> = vec![
            (
                "cram",
                Factors {
                    cram: 8192,
                    ..base.clone()
                },
            ),
            (
                "ctx",
                Factors {
                    ctx: 32768,
                    ..base.clone()
                },
            ),
            (
                "ubatch",
                Factors {
                    ubatch: 1024,
                    ..base.clone()
                },
            ),
            (
                "ngl",
                Factors {
                    ngl: 0,
                    ..base.clone()
                },
            ),
            (
                "n_cpu_moe",
                Factors {
                    n_cpu_moe: Some(30),
                    ..base.clone()
                },
            ),
            (
                "parallel",
                Factors {
                    parallel: 4,
                    ..base.clone()
                },
            ),
            (
                "kv_type",
                Factors {
                    kv_type: "q8_0".to_owned(),
                    ..base.clone()
                },
            ),
            (
                "kv_unified",
                Factors {
                    kv_unified: false,
                    ..base.clone()
                },
            ),
            (
                "runtime",
                Factors {
                    runtime: Runtime::Mainline,
                    ..base.clone()
                },
            ),
            (
                "gpus",
                Factors {
                    gpus: "0".to_owned(),
                    ..base.clone()
                },
            ),
            (
                "repeat",
                Factors {
                    repeat: 1,
                    ..base.clone()
                },
            ),
            (
                "extra",
                Factors {
                    extra: Vec::new(),
                    ..base.clone()
                },
            ),
            (
                "bench",
                Factors {
                    bench: true,
                    ..base.clone()
                },
            ),
            (
                "probe_prompt_tokens",
                Factors {
                    probe_prompt_tokens: 64,
                    ..base.clone()
                },
            ),
            (
                "verbose_log",
                Factors {
                    verbose_log: false,
                    ..base.clone()
                },
            ),
        ];
        for (factor, mutated) in mutations {
            assert_ne!(
                cell_id(&mutated),
                baseline,
                "{factor} left the identity alone"
            );
        }
    }

    /// The dataset is the oracle: every one of the campaign's rows carries both its
    /// factor set and the id the Python harness hashed from it, so the two hashers
    /// have to agree on all of them or this harness does not recognise its own
    /// measurements and re-measures the lot.
    #[test]
    fn every_recorded_cell_id_is_reproduced() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scripts/calibration/data/measurements.ndjson");
        let measurements =
            std::fs::read_to_string(path).expect("the campaign's measurements are checked in");
        let mut compared = 0usize;
        for line in measurements.lines().filter(|line| !line.trim().is_empty()) {
            let record: serde_json::Value =
                serde_json::from_str(line).expect("each line is one JSON record");
            let factors: Factors = serde_json::from_value(record["factors"].clone())
                .unwrap_or_else(|error| panic!("{}: {error}", record["cell"]));
            assert_eq!(
                serde_json::json!(cell_id(&factors)),
                record["cell"],
                "{} hashes differently",
                record["factors"]["label"]
            );
            compared += 1;
        }
        assert!(compared > 600, "only {compared} rows were compared");
    }

    #[test]
    fn a_plan_key_the_harness_does_not_know_is_an_error_rather_than_a_silent_drop() {
        let plan = r#"[{"label": "a", "model": "/m.gguf", "ctx": 4096, "thp": false}]"#;
        let error = load_plan(plan).expect_err("`thp` is no longer a factor");
        assert!(error.to_string().contains("thp"), "{error}");

        let plan = r#"[{"label": "a", "model": "/m.gguf", "ctx": 4096}]"#;
        let cells = load_plan(plan).expect("a plan of known keys loads");
        assert_eq!(cells[0].ctx, 4096);
        assert_eq!(
            cells[0].ubatch, 512,
            "an unspelled factor takes its default"
        );
    }
}
