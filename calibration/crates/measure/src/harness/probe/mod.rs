//! Attributing host-memory growth that a single measurement cannot separate.
//!
//! The measurement harness samples one process once per configuration. That is the
//! right shape for fitting constants, but it cannot tell a term allocated once from
//! one that accumulates with use — both read as "this model holds more than the model
//! predicts". A probe varies one thing at a time against a fresh server and reports
//! the series, which is what separates them.
//!
//! Nothing is written. This is the tool you reach for when a number looks wrong and
//! you want to see its parts, so the output is for a reader rather than for the
//! dataset — a probe run answers a different question from a measured cell and would
//! only muddy `measurements.ndjson` if it landed there.
//!
//! What the questions established, and why they are shaped as they are, is in
//! `calibration/docs/findings.md` under "The per-model residual is a first-request
//! step". The ordering rule that keeps them valid is in [`plan`].

use std::{collections::BTreeMap, path::Path, time::Duration};

use crate::{
    harness::{
        probe::plan::{Sample, Stage, StageKind, Step, Tag},
        run::{
            PORT_WAIT,
            child::{spawn_server, stop_child},
            readiness::{Readiness, ReadinessWait, wait_for_port, wait_for_ready},
            watchdog::SwapWatchdog,
        },
        sys::Deps,
    },
    record::{FULLY_OFFLOADED, Factors, FlashAttn, KvType, RssSnapshot, Runtime},
};

pub mod plan;
mod report;

pub use plan::Question;
pub use report::render;

/// How long a server gets to load before the probe gives up on it.
const LOAD_TIMEOUT: Duration = Duration::from_secs(1800);

/// Allocation lags the response: the completion returns before the runtime has
/// finished growing, and a reading taken immediately misses part of the step.
const SETTLE: Duration = Duration::from_secs(2);
/// How far swap may grow past the baseline before a stage is abandoned.
const SWAP_LIMIT_GIB: f64 = 4.0;
/// A completion's own timeout. Generous because a 400-word prefill on a hybrid is
/// slow, and a probe that gives up early reports a step that never finished.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

/// What to probe, and with what.
pub struct Options<'a> {
    pub model: &'a str,
    pub binary: &'a str,
    pub gpus: &'a str,
    pub context: u32,
    pub port: u16,
    pub log_dir: &'a Path,
    pub questions: Vec<Question>,
}

/// One reading, and what it was for.
#[derive(Debug, Clone)]
pub struct Reading {
    pub tag: Tag,
    pub stage: StageKind,
    /// Repeated on the reading so the report can group by cache setting without
    /// re-deriving it from the stage.
    pub cram_mib: u32,
    pub rss: RssSnapshot,
    /// Anonymous bytes per mapping, when the plan asked for the breakdown.
    pub maps: Option<BTreeMap<String, u64>>,
}

/// Everything a run observed, for [`render`] to lay out.
#[derive(Debug, Default)]
pub struct Observations {
    pub readings: Vec<Reading>,
    /// Stages that could not be measured, with why. A probe reports what it got
    /// rather than failing whole: a model that will not load at one cache setting
    /// still says something at the others.
    pub failures: Vec<String>,
}

/// Run the plan, returning what it saw.
///
/// Each stage is a server of its own, torn down before the next starts. That is the
/// expensive part and it is not avoidable: the step being hunted happens on a
/// process's first request, so a stage that reuses one measures nothing.
pub fn probe(deps: &Deps, options: &Options<'_>) -> Observations {
    let mut observations = Observations::default();
    let stages = plan::plan(&options.questions);

    for stage in &stages {
        if let Err(error) = run_stage(deps, options, stage, &mut observations) {
            observations
                .failures
                .push(format!("{}: {error}", stage.label()));
        }
    }
    observations
}

fn run_stage(
    deps: &Deps,
    options: &Options<'_>,
    stage: &Stage,
    observations: &mut Observations,
) -> Result<(), String> {
    if !wait_for_port(deps, options.port, PORT_WAIT) {
        return Err(format!("port {} never came free", options.port));
    }

    let factors = factors_for(options, stage);
    let log_path = options
        .log_dir
        .join(format!("probe-{}.log", stage.cram_mib));
    let mut child = spawn_server(deps, &factors, options.binary, &log_path, options.port)
        .map_err(|e| format!("spawning: {e}"))?;

    let spawned_at = deps.clock.elapsed();
    // The same limit the campaign runs with: a hybrid that overcommits pages rather
    // than failing, and the measurement is worthless once that starts.
    let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), SWAP_LIMIT_GIB);
    let readiness = wait_for_ready(
        deps,
        ReadinessWait {
            child: child.as_mut(),
            port: options.port,
            spawned_at,
            timeout: LOAD_TIMEOUT,
            watchdog: &mut watchdog,
        },
    );
    let result = match readiness {
        Readiness::Loaded { .. } => {
            let pid = child.pid();
            walk(deps, options, stage, pid, observations);
            Ok(())
        }
        Readiness::Exited(status) => Err(format!("the server exited with {status}")),
        Readiness::TimedOut => Err("the server never became healthy".to_string()),
        Readiness::Swapping(gib) => Err(format!("the box started paging ({gib:.1} GiB)")),
    };

    stop_child(deps, child.as_mut());
    result
}

/// Walk one stage's steps in order. The plan says what may be read when, and
/// this does exactly that and nothing else — that is the whole of the
/// contamination rule at runtime.
fn walk(
    deps: &Deps,
    options: &Options<'_>,
    stage: &Stage,
    pid: u32,
    observations: &mut Observations,
) {
    for step in &stage.steps {
        match step {
            Step::Sample(sample) => {
                if let Some(reading) = read(deps, pid, stage, sample) {
                    observations.readings.push(reading);
                }
            }
            Step::Request { words, n_predict } => {
                let prompt = vec!["word"; *words].join(" ");
                let body = serde_json::json!({
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "cache_prompt": false,
                });
                if deps
                    .http
                    .post(options.port, "/completion", &body, REQUEST_TIMEOUT)
                    .is_none()
                {
                    observations
                        .failures
                        .push(format!("{}: a completion did not return", stage.label()));
                }
                deps.clock.sleep(SETTLE);
            }
        }
    }
}

fn read(deps: &Deps, pid: u32, stage: &Stage, sample: &Sample) -> Option<Reading> {
    let rss = deps.procfs.status(pid)?;
    let maps = if sample.with_maps {
        deps.procfs.smaps_anon(pid)
    } else {
        None
    };
    Some(Reading {
        tag: sample.tag.clone(),
        stage: stage.kind,
        cram_mib: stage.cram_mib,
        rss,
        maps,
    })
}

/// A stage's server, as the harness's own spawn path wants it.
///
/// Deliberately the same `Factors` the measurement harness builds a command line
/// from, so a probe and a cell differ in what they *ask*, not in how the server is
/// started. A flag that changes for a measured cell changes here too.
fn factors_for(options: &Options<'_>, stage: &Stage) -> Factors {
    Factors {
        label: format!("probe-{}", stage.label()),
        model: options.model.to_string(),
        runtime: Runtime::Mainline,
        gpus: options.gpus.to_string(),
        ctx: options.context,
        ubatch: 512,
        parallel: 1,
        ngl: FULLY_OFFLOADED,
        kv_type: KvType::F16,
        flash_attn: FlashAttn::On,
        cram: stage.cram_mib,
        ..Factors::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes};

    fn snapshot(anon_mib: u64) -> RssSnapshot {
        RssSnapshot {
            rss_total_kb: anon_mib * 1024,
            rss_anon_kb: anon_mib * 1024,
            rss_file_kb: 0,
            rss_shmem_kb: 0,
        }
    }

    fn fakes() -> Fakes {
        Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new()
                .with_status(snapshot(1024))
                .with_smaps(BTreeMap::from([
                    ("[anon]".to_string(), 900 * 1024 * 1024),
                    ("/usr/lib/libcuda.so".to_string(), 124 * 1024 * 1024),
                ])),
            FakeGpu::new(),
            FakeHttp::new().with_reply(serde_json::json!({"content": "ok"})),
        )
    }

    /// The battery runs end to end against the in-memory world, which is the whole
    /// reason for porting it: the ordering rule is checkable without a GPU, a model,
    /// or half an hour.
    #[test]
    fn the_battery_runs_against_the_fakes() {
        let fakes = fakes();
        let deps = fakes.deps();
        let observations = probe(
            &deps,
            &Options {
                model: "/models/fake.gguf",
                binary: "llama-server",
                gpus: "0",
                context: 32768,
                port: 18080,
                log_dir: Path::new("/tmp"),
                questions: Question::ALL.to_vec(),
            },
        );
        assert!(
            observations.failures.is_empty(),
            "{:?}",
            observations.failures
        );
        assert!(
            observations
                .readings
                .iter()
                .any(|r| matches!(r.tag, Tag::Idle)),
            "the idle reading is missing"
        );
        assert!(
            observations
                .readings
                .iter()
                .any(|r| matches!(r.tag, Tag::Stepped)),
            "the stepped reading is missing"
        );
    }

    /// One server per stage, torn down before the next: a stage that reused a server
    /// would have nothing fresh to read, which is the failure the plan is built to
    /// prevent.
    #[test]
    fn every_stage_gets_its_own_server() {
        let fakes = fakes();
        let deps = fakes.deps();
        let questions = Question::ALL.to_vec();
        let expected = plan::server_loads(&plan::plan(&questions));
        probe(
            &deps,
            &Options {
                model: "/models/fake.gguf",
                binary: "llama-server",
                gpus: "0",
                context: 32768,
                port: 18080,
                log_dir: Path::new("/tmp"),
                questions,
            },
        );
        assert_eq!(fakes.spawner.processes().len(), expected);
    }

    /// A stage whose server never loads is reported and the rest still run. A model
    /// that will not fit at one cache setting still says something at the others.
    #[test]
    fn a_dead_stage_does_not_take_the_run_with_it() {
        let fakes = Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new().with_status(snapshot(1024)),
            FakeGpu::new(),
            FakeHttp::new().never_healthy(),
        );
        let deps = fakes.deps();
        let observations = probe(
            &deps,
            &Options {
                model: "/models/fake.gguf",
                binary: "llama-server",
                gpus: "0",
                context: 32768,
                port: 18080,
                log_dir: Path::new("/tmp"),
                questions: vec![Question::Growth],
            },
        );
        assert_eq!(observations.failures.len(), 2, "both stages report");
        assert!(observations.readings.is_empty());
    }
}
