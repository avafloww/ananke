//! What to measure, as data, before anything is spawned.
//!
//! The plan exists so the one rule that makes a probe valid is written down once and
//! can be checked without a GPU: **a reading taken on a fresh process must have no
//! request before it in its stage.** The step this tool hunts happens on the first
//! batched prefill and never again, so a "before" sampled after any request measures
//! nothing — and measures it plausibly, as a number in the right range rather than an
//! error.
//!
//! Stages share a server where the ordering allows it and start a new one where it
//! does not. `prefill` needs seven servers because each of its points needs a first
//! request of its own; `growth` needs one per `-cram` setting to watch the series
//! from zero. Everything else rides the first stage.

/// One question the battery answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Question {
    /// Where the residual lives, by mapping.
    Maps,
    /// Which mapping the first prefill's allocation lands in.
    Step,
    /// Whether it accumulates with use, and whether the prompt cache is why.
    Growth,
    /// Whether the step is sized by the prompt or by the generation.
    Prefill,
}

impl Question {
    pub const ALL: [Question; 4] = [Self::Maps, Self::Step, Self::Growth, Self::Prefill];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Maps => "maps",
            Self::Step => "step",
            Self::Growth => "growth",
            Self::Prefill => "prefill",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|q| q.as_str() == name)
    }
}

/// One server, and the ordered things done to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// Shown in the output, and in the failure when a server dies.
    pub label: String,
    /// `-cram`, the prompt-cache cap in MiB.
    pub cram_mib: u32,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Read the process's memory and file it under `tag`.
    Sample(Sample),
    /// Drive one completion. Everything sampled after this is post-request.
    Request { words: usize, n_predict: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub tag: Tag,
    /// Whether to read the per-mapping breakdown as well as the totals. It costs a
    /// `/proc/<pid>/smaps` parse, which is far from free on a large process.
    pub with_maps: bool,
    /// Whether this reading is only meaningful on a process that has served nothing.
    pub needs_fresh: bool,
}

/// What a reading is for, which is what the report groups on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tag {
    /// Before the first request, on a fresh process.
    Idle,
    /// After the request that steps it.
    Stepped,
    /// The nth reading of the growth series.
    Growth(usize),
    /// One point of the prompt/generation sweep, before and after.
    PrefillBefore {
        words: usize,
        n_predict: u32,
    },
    PrefillAfter {
        words: usize,
        n_predict: u32,
    },
}

/// The prompt and generation lengths the sweep varies, independently.
///
/// Holding one axis while moving the other is what separates "the step is sized by
/// the prompt" from "by the generation"; a diagonal sweep cannot.
pub const PREFILL_POINTS: [(usize, u32); 7] = [
    (1, 8),
    (8, 8),
    (64, 8),
    (400, 8),
    (1, 64),
    (1, 256),
    (8, 256),
];

/// The `-cram` settings the growth series is measured at.
///
/// Zero disables the prompt cache, so a footprint that still grows is growing for
/// another reason; the default lets it fill, which is what a cache reaching its cap
/// looks like against a leak that does not stop.
pub const GROWTH_CRAM_MIB: [u32; 2] = [0, 8192];

/// How many completions the growth series drives after the first.
const GROWTH_REQUESTS: usize = 5;

/// The request that steps the process, and one point of the prefill sweep.
///
/// 64 words because the step saturates by then and is absent at one, so it is the
/// cheapest request that provokes the whole of it.
const STEP_WORDS: usize = 64;
const STEP_PREDICT: u32 = 8;

/// Build the ordered plan for the questions asked.
///
/// The first stage carries everything that can share one server: the idle reading,
/// the step, the growth series at `-cram 0`, and the prefill sweep's `(64, 8)` point,
/// which is the same request the step uses. Sharing is worth the care because the
/// numbers then describe one process rather than four, measured minutes apart.
pub fn plan(questions: &[Question]) -> Vec<Stage> {
    let wants = |q: Question| questions.contains(&q);
    let mut stages = Vec::new();

    // The shared stage. Present if anything at all needs a fresh server, which is
    // everything except a prefill-only run — that one wants its own seven.
    let shared_stage = wants(Question::Maps) || wants(Question::Step) || wants(Question::Growth);
    if shared_stage {
        let mut steps = vec![Step::Sample(Sample {
            tag: Tag::Idle,
            with_maps: wants(Question::Maps) || wants(Question::Step),
            needs_fresh: true,
        })];
        steps.push(Step::Request {
            words: STEP_WORDS,
            n_predict: STEP_PREDICT,
        });
        steps.push(Step::Sample(Sample {
            tag: Tag::Stepped,
            with_maps: wants(Question::Maps) || wants(Question::Step),
            needs_fresh: false,
        }));
        if wants(Question::Growth) {
            steps.extend(growth_series());
        }
        stages.push(Stage {
            label: format!("shared (cram {})", GROWTH_CRAM_MIB[0]),
            cram_mib: GROWTH_CRAM_MIB[0],
            steps,
        });
    }

    // The growth series' other cache settings need their own server each: a series
    // that starts after the cache has already filled shows none of the filling.
    //
    // Shaped exactly like the shared stage — fresh read, the step request, then the
    // series — so every series begins at the same point in a process's life. A series
    // anchored before the step and one anchored after are not comparable, and their
    // totals differ by the step rather than by anything about the cache.
    if wants(Question::Growth) {
        for &cram in &GROWTH_CRAM_MIB[1..] {
            let mut steps = vec![
                Step::Sample(Sample {
                    tag: Tag::Idle,
                    with_maps: false,
                    needs_fresh: true,
                }),
                Step::Request {
                    words: STEP_WORDS,
                    n_predict: STEP_PREDICT,
                },
                Step::Sample(Sample {
                    tag: Tag::Stepped,
                    with_maps: false,
                    needs_fresh: false,
                }),
            ];
            steps.extend(growth_series());
            stages.push(Stage {
                label: format!("growth (cram {cram})"),
                cram_mib: cram,
                steps,
            });
        }
    }

    if wants(Question::Prefill) {
        for &(words, n_predict) in &PREFILL_POINTS {
            // The shared stage already ran this point — but only if there is one.
            // Testing whether *any* stage exists instead drops the point as soon as
            // the sweep has pushed its own first, which is silent: the table comes
            // back one row short and every remaining row is right.
            if shared_stage && (words, n_predict) == (STEP_WORDS, STEP_PREDICT) {
                continue;
            }
            stages.push(Stage {
                label: format!("prefill (words {words}, predict {n_predict})"),
                cram_mib: 0,
                steps: vec![
                    Step::Sample(Sample {
                        tag: Tag::PrefillBefore { words, n_predict },
                        with_maps: false,
                        needs_fresh: true,
                    }),
                    Step::Request { words, n_predict },
                    Step::Sample(Sample {
                        tag: Tag::PrefillAfter { words, n_predict },
                        with_maps: false,
                        needs_fresh: false,
                    }),
                ],
            });
        }
    }

    stages
}

/// The repeated identical requests, sampled after each.
///
/// The series proper starts from the stage's `Stepped` reading, which every stage
/// takes: growth asks whether the footprint accumulates *with use*, and the one-time
/// step is a separate question that `step` and `prefill` already answer.
fn growth_series() -> Vec<Step> {
    let mut steps = Vec::new();
    for n in 1..=GROWTH_REQUESTS {
        steps.push(Step::Request {
            words: 6,
            n_predict: 16,
        });
        steps.push(Step::Sample(Sample {
            tag: Tag::Growth(n),
            with_maps: false,
            needs_fresh: false,
        }));
    }
    steps
}

/// How many servers a plan loads, which is what its wall-clock is.
pub fn server_loads(stages: &[Stage]) -> usize {
    stages.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule the whole module exists for: nothing that has to be fresh may be
    /// sampled after a request.
    ///
    /// A violation does not announce itself. The step happens once, so a "before"
    /// taken afterwards reports the *stepped* figure — a plausible number, in range,
    /// simply measuring the wrong thing, and the conclusion drawn from it would be
    /// that there is no step at all.
    #[test]
    fn nothing_fresh_is_sampled_after_a_request() {
        for stage in plan(&Question::ALL) {
            let mut requested = false;
            for step in &stage.steps {
                match step {
                    Step::Request { .. } => requested = true,
                    Step::Sample(sample) => assert!(
                        !(sample.needs_fresh && requested),
                        "`{}` samples {:?} as fresh after a request",
                        stage.label,
                        sample.tag
                    ),
                }
            }
        }
    }

    /// Every stage starts by reading a process that has served nothing, because that
    /// is the only reading a later one can be compared against.
    #[test]
    fn every_stage_opens_with_a_fresh_reading() {
        for stage in plan(&Question::ALL) {
            let Some(Step::Sample(first)) = stage.steps.first() else {
                panic!("`{}` does not open with a reading", stage.label);
            };
            assert!(
                first.needs_fresh,
                "`{}` opens with a reading it does not require to be fresh",
                stage.label
            );
        }
    }

    /// The whole battery costs eight loads, not the eleven of running each question
    /// separately: the shared stage covers `maps`, `step`, `growth` at `cram 0`, and
    /// the prefill sweep's `(64, 8)` point.
    #[test]
    fn sharing_saves_three_loads() {
        let together = server_loads(&plan(&Question::ALL));
        let apart: usize = Question::ALL
            .iter()
            .map(|q| server_loads(&plan(&[*q])))
            .sum();
        assert_eq!(together, 8, "the shared plan");
        assert_eq!(apart, 11, "one question at a time");
    }

    /// Every growth series starts at the same point in a process's life.
    ///
    /// One anchored before the step and one after differ by the step, which is a
    /// one-time cost and nothing to do with the cache the series is comparing. The
    /// totals would then look like a cache difference and be read as one.
    #[test]
    fn the_growth_series_are_anchored_alike() {
        let stages = plan(&[Question::Growth]);
        let shapes: Vec<Vec<&Step>> = stages
            .iter()
            .map(|stage| {
                stage
                    .steps
                    .iter()
                    .skip_while(|step| !matches!(step, Step::Sample(s) if s.tag == Tag::Stepped))
                    .collect()
            })
            .collect();
        assert!(shapes.len() > 1, "there is more than one cache setting");
        assert!(
            shapes.windows(2).all(|pair| pair[0] == pair[1]),
            "the series differ in shape after the step"
        );
        for stage in &stages {
            assert!(
                stage
                    .steps
                    .iter()
                    .any(|s| matches!(s, Step::Sample(s) if s.tag == Tag::Stepped)),
                "`{}` has no post-step anchor to start its series from",
                stage.label
            );
        }
    }

    /// A single question still produces a usable plan rather than half of the shared
    /// one, since `--only` exists for when a full battery is too slow.
    #[test]
    fn one_question_plans_only_what_it_needs() {
        assert_eq!(server_loads(&plan(&[Question::Step])), 1);
        assert_eq!(server_loads(&plan(&[Question::Maps])), 1);
        assert_eq!(server_loads(&plan(&[Question::Growth])), 2);
        assert_eq!(
            server_loads(&plan(&[Question::Prefill])),
            PREFILL_POINTS.len()
        );
    }

    /// The growth series is measured at every cache setting, and each starts from a
    /// process that has served nothing.
    #[test]
    fn each_cache_setting_gets_its_own_server() {
        let stages = plan(&[Question::Growth]);
        let crams: Vec<u32> = stages.iter().map(|s| s.cram_mib).collect();
        assert_eq!(crams, GROWTH_CRAM_MIB.to_vec());
    }
}
