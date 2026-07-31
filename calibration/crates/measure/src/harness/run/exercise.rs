//! Put the server through enough work to allocate what it allocates lazily.
//!
//! Several host buffers are sized on first use rather than at load, so an idle
//! process under-reports: serving one first request was measured moving host
//! memory from -2 MiB to +238 across models, predicted by neither vocabulary,
//! model size, nor architecture.
//!
//! Three shapes of work, because they reach different terms:
//!
//! - the warm-up probe, whose *prompt length* is what matters. llama.cpp takes a
//!   context checkpoint while decoding a prompt, spaced by
//!   `--checkpoint-min-step` (8192 tokens), so a few-token prompt captures part
//!   of one checkpoint and a longer one captures more: 11 MiB at one token, 274
//!   at sixty-four, 431 once past the spacing. How many tokens it *generates* is
//!   a timing knob — the step is identical at 8, 4096, and 12288.
//! - `soak` with `concurrency`, which is the only way to touch slots past the
//!   first: a strictly serial probe leaves per-slot state unallocated however
//!   many requests it makes.
//! - `bench`, an agent-shaped conversation that checkpoints memory *against
//!   tokens*. The RSS trace answers "did memory move"; it cannot answer "per
//!   what", because nothing in a time series says how many tokens were generated
//!   between two samples — a footprint that grows with generation and one that
//!   grows with wall-clock look identical on a clock.

use std::time::Duration;

use serde::{Deserialize, Serialize, de::IgnoredAny};

use crate::{
    harness::{
        run::{bench, watchdog::SwapWatchdog},
        sys::{Deps, post_json},
    },
    record::{Checkpoint, DEFAULT_PROBE_PROMPT_TOKENS, Factors, GpuUsage},
};

/// Long enough for a 200 GiB hybrid's first prompt, which is minutes of prefill.
const PROBE_TIMEOUT: Duration = Duration::from_secs(600);
const SOAK_TIMEOUT: Duration = Duration::from_secs(300);
const GROWTH_TIMEOUT: Duration = Duration::from_secs(900);
const GROWTH_MAX_TOKENS: u32 = 512;

/// How much of the context a growth run is allowed to fill before it stops. Past
/// that point the server evicts, and the footprint stops being a function of what
/// was generated — which is the whole quantity the run exists to measure.
const CONTEXT_WRAP_FRACTION: f64 = 0.85;

pub(crate) fn exercise(
    deps: &Deps,
    factors: &Factors,
    port: u16,
    pid: u32,
    watchdog: &mut SwapWatchdog,
) -> Vec<Checkpoint> {
    if !factors.served {
        return Vec::new();
    }
    if factors.embeddings {
        return embeddings(deps, factors, port, pid);
    }
    // One word per token, near enough: what matters is which side of the
    // checkpoint spacing the prompt falls on, not the exact count.
    let prompt = if factors.probe_prompt_tokens <= DEFAULT_PROBE_PROMPT_TOKENS {
        // The literal the whole campaign was measured with, so existing cells
        // keep their meaning as well as their identity.
        "Count to twenty.".to_owned()
    } else {
        vec!["word"; factors.probe_prompt_tokens as usize].join(" ")
    };
    let warmup = [message(Role::User, &prompt)];
    post_json::<_, IgnoredAny>(
        deps.http.as_ref(),
        port,
        "/v1/chat/completions",
        &chat_request(&warmup, factors.probe_tokens),
        PROBE_TIMEOUT,
    );

    if factors.bench {
        return growth(deps, factors, port, pid, watchdog);
    }

    let mut prompt = "Explain memory allocation.".to_owned();
    for round in 0..factors.soak {
        prompt.push_str(&format!(
            " Also cover point {round} in detail, at length, with examples."
        ));
        let messages = [message(Role::User, &prompt)];
        let request = chat_request(&messages, 256);
        // Overlapping requests, because a strictly serial probe never reaches a
        // slot past the first.
        std::thread::scope(|scope| {
            for _ in 0..factors.concurrency.max(1) {
                let http = deps.http.clone();
                let request = &request;
                scope.spawn(move || {
                    post_json::<_, IgnoredAny>(
                        http.as_ref(),
                        port,
                        "/v1/chat/completions",
                        request,
                        SOAK_TIMEOUT,
                    )
                });
            }
        });
        if watchdog.check(deps.procfs.as_ref()).is_some() {
            break;
        }
    }
    Vec::new()
}

/// An embedding model has no generation, so requests drive it instead. A growth
/// cell issues many and checkpoints the same way rather than being skipped: the
/// question is whether repeated embedding calls accumulate anything.
fn embeddings(deps: &Deps, factors: &Factors, port: u16, pid: u32) -> Vec<Checkpoint> {
    let rounds = if factors.bench {
        factors.bench_turns
    } else {
        1
    };
    let mut checkpoints = Vec::new();
    for round in 0..rounds {
        let input = format!("calibration probe {round} {}", "token ".repeat(64));
        post_json::<_, IgnoredAny>(
            deps.http.as_ref(),
            port,
            "/v1/embeddings",
            &EmbeddingsRequest { input: &input },
            Duration::from_secs(120),
        );
        if !factors.bench {
            continue;
        }
        // No token accounting: an embedding response reports none, so the
        // checkpoint is memory against *round* rather than against tokens.
        checkpoints.push(checkpoint(deps, pid, round + 1, Tokens::default(), None));
    }
    checkpoints
}

/// Drive an agent-shaped conversation, checkpointing memory against tokens so
/// growth is fittable against cumulative tokens and against KV depth separately.
/// Replies are fed back in, so the context grows the way an agent's does and the
/// prompt cache sees a real prefix rather than filler.
fn growth(
    deps: &Deps,
    factors: &Factors,
    port: u16,
    pid: u32,
    watchdog: &mut SwapWatchdog,
) -> Vec<Checkpoint> {
    // `-cram` serialises prompts that have been *evicted* from a slot. One
    // strictly-growing conversation shares a prefix and never evicts anything, so
    // it would measure cram 0 and cram 8192 identically for a reason that has
    // nothing to do with the cache. Alternating distinct conversations is what
    // makes a slot's prompt get displaced and the cache get used.
    let markers: &[&str] = if factors.cram > 0 {
        &["", " Prefer Rust.", " Prefer Python.", " Answer tersely."]
    } else {
        &[""]
    };
    let mut conversations: Vec<Vec<Message>> = markers
        .iter()
        .map(|marker| vec![message(Role::System, &format!("{}{marker}", bench::SYSTEM))])
        .collect();

    let mut checkpoints = Vec::new();
    let mut generated = 0u64;
    for turn in 0..factors.bench_turns {
        let which = turn as usize % conversations.len();
        let prompt = bench::PROMPTS[turn as usize % bench::PROMPTS.len()];
        conversations[which].push(message(Role::User, prompt));
        // Not `deny_unknown_fields`: a completion response carries far more than
        // this reads, and an unrecognised field must be ignored, not rejected.
        let Some(response) = post_json::<_, ChatCompletionResponse>(
            deps.http.as_ref(),
            port,
            "/v1/chat/completions",
            &chat_request(&conversations[which], GROWTH_MAX_TOKENS),
            GROWTH_TIMEOUT,
        ) else {
            break;
        };
        let Some(choice) = response.choices.into_iter().next() else {
            break;
        };
        conversations[which].push(message(Role::Assistant, &choice.message.content));
        let tokens = Tokens::from_usage(response.usage);
        generated += tokens.completion;
        checkpoints.push(checkpoint(
            deps,
            pid,
            turn + 1,
            Tokens {
                generated,
                ..tokens
            },
            Some(which as u32),
        ));
        // Stop before the context wraps.
        if tokens.kv_depth() as f64 > f64::from(factors.ctx) * CONTEXT_WRAP_FRACTION {
            break;
        }
        if watchdog.check(deps.procfs.as_ref()).is_some() {
            break;
        }
    }
    checkpoints
}

/// What the server said the turn cost, which is what ties a memory reading to
/// the work that produced it.
#[derive(Debug, Default, Clone, Copy)]
struct Tokens {
    prompt: u64,
    completion: u64,
    generated: u64,
}

impl Tokens {
    fn from_usage(usage: Usage) -> Self {
        Self {
            prompt: usage.prompt_tokens,
            completion: usage.completion_tokens,
            generated: 0,
        }
    }

    /// The term that scales with context rather than with use.
    fn kv_depth(&self) -> u64 {
        self.prompt + self.completion
    }
}

fn checkpoint(
    deps: &Deps,
    pid: u32,
    turn: u32,
    tokens: Tokens,
    conversation: Option<u32>,
) -> Checkpoint {
    let gpu = deps.gpu.per_process_mib(pid);
    Checkpoint {
        turn: u64::from(turn),
        at_utc: deps.clock.now_utc(),
        prompt_tokens: tokens.prompt,
        completion_tokens: tokens.completion,
        generated_tokens_total: tokens.generated,
        kv_depth_tokens: tokens.kv_depth(),
        rss: deps.procfs.status(pid).unwrap_or_default(),
        gpu: GpuUsage {
            total_mib: (!gpu.is_empty()).then(|| gpu.values().sum()),
            used_mib: gpu,
        },
        conversation,
    }
}

/// The harness only ever plays these three parts, so the role is a closed set
/// rather than an arbitrary string.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize)]
struct Message {
    role: Role,
    content: String,
}

fn message(role: Role, content: &str) -> Message {
    Message {
        role,
        content: content.to_owned(),
    }
}

/// `model` is a placeholder: the request goes straight to the server under
/// measurement, which serves whatever it was started with.
const MODEL_PLACEHOLDER: &str = "m";

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    max_tokens: u32,
}

fn chat_request<'a>(messages: &'a [Message], max_tokens: u32) -> ChatCompletionRequest<'a> {
    ChatCompletionRequest {
        model: MODEL_PLACEHOLDER,
        messages,
        max_tokens,
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    input: &'a str,
}

/// A `/v1/chat/completions` response. llama-server's replies carry far more
/// fields than these (timings, model, id, …), so this deliberately does not
/// `deny_unknown_fields` — an unrecognised field is ignored, not rejected.
#[derive(Debug, Default, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Default, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes},
        record::RssSnapshot,
    };

    fn reply(prompt_tokens: u64, completion_tokens: u64) -> serde_json::Value {
        serde_json::json!({
            "choices": [{"message": {"content": "here is some code"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        })
    }

    fn fakes(http: FakeHttp) -> Fakes {
        Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new().with_status(RssSnapshot {
                rss_total_kb: 4_000,
                rss_anon_kb: 3_000,
                rss_file_kb: 900,
                rss_shmem_kb: 100,
            }),
            FakeGpu::new().with_used_mib(&[(0, 12_000)]),
            http,
        )
    }

    #[test]
    fn an_unserved_cell_is_left_alone() {
        let fakes = fakes(FakeHttp::new());
        let factors = Factors {
            served: false,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        assert!(exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog).is_empty());
        assert!(fakes.http.requests().is_empty());
    }

    #[test]
    fn the_short_probe_sends_the_prompt_the_campaign_was_measured_with() {
        let fakes = fakes(FakeHttp::new().with_reply(reply(4, 64)));
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        exercise(&fakes.deps(), &Factors::default(), 18099, 1, &mut watchdog);
        let requests = fakes.http.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, "/v1/chat/completions");
        assert_eq!(
            requests[0].1["messages"][0]["content"],
            serde_json::json!("Count to twenty.")
        );
        assert_eq!(requests[0].1["max_tokens"], serde_json::json!(64));
    }

    #[test]
    fn a_longer_probe_prompt_is_padded_to_the_asked_for_token_count() {
        let fakes = fakes(FakeHttp::new().with_reply(reply(64, 8)));
        let factors = Factors {
            probe_prompt_tokens: 64,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);
        let prompt = fakes.http.requests()[0].1["messages"][0]["content"]
            .as_str()
            .expect("the prompt is a string")
            .to_owned();
        assert_eq!(prompt.split_whitespace().count(), 64);
    }

    #[test]
    fn a_growth_run_checkpoints_every_turn_and_feeds_the_reply_back() {
        let fakes = fakes(FakeHttp::new().with_reply(reply(300, 100)));
        let factors = Factors {
            bench: true,
            bench_turns: 3,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        let checkpoints = exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);

        assert_eq!(checkpoints.len(), 3);
        assert_eq!(checkpoints[2].generated_tokens_total, 300, "cumulative");
        assert_eq!(checkpoints[2].kv_depth_tokens, 400);
        assert_eq!(checkpoints[2].gpu.total_mib, Some(12_000));
        assert_eq!(checkpoints[2].conversation, Some(0));
        // The warm-up probe, then one request per turn, each carrying the whole
        // conversation so far: system, then user/assistant per completed turn.
        let requests = fakes.http.requests();
        assert_eq!(requests.len(), 4);
        let last = requests[3].1["messages"]
            .as_array()
            .expect("messages is a list");
        assert_eq!(last.len(), 1 + 2 * 2 + 1);
        assert_eq!(last[2]["role"], serde_json::json!("assistant"));
    }

    /// `cram` is what makes eviction happen, and eviction is what fills the
    /// prompt cache — one strictly growing conversation would measure the cache
    /// as free.
    #[test]
    fn a_cram_cell_alternates_conversations_so_slots_evict() {
        let fakes = fakes(FakeHttp::new().with_reply(reply(300, 100)));
        let factors = Factors {
            bench: true,
            bench_turns: 5,
            cram: 8192,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        let checkpoints = exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);
        let conversations: Vec<Option<u32>> = checkpoints
            .iter()
            .map(|checkpoint| checkpoint.conversation)
            .collect();
        assert_eq!(
            conversations,
            vec![Some(0), Some(1), Some(2), Some(3), Some(0)]
        );
    }

    #[test]
    fn a_growth_run_stops_before_the_context_wraps() {
        // 3600 + 512 tokens a turn against a 4096 window: the first turn is
        // already past 85% of it.
        let fakes = fakes(FakeHttp::new().with_reply(reply(3_600, 512)));
        let factors = Factors {
            bench: true,
            bench_turns: 40,
            ctx: 4096,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        assert_eq!(
            exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog).len(),
            1
        );
    }

    #[test]
    fn an_embedding_cell_drives_the_embeddings_route_instead() {
        let fakes = fakes(FakeHttp::new().with_reply(serde_json::json!({"data": []})));
        let factors = Factors {
            embeddings: true,
            bench: true,
            bench_turns: 3,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        let checkpoints = exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);
        assert_eq!(checkpoints.len(), 3);
        assert_eq!(checkpoints[0].kv_depth_tokens, 0);
        assert!(
            fakes
                .http
                .requests()
                .iter()
                .all(|(path, _)| path == "/v1/embeddings")
        );
    }

    /// A soak's whole point is overlapping requests, so the count has to be
    /// `soak × concurrency` and not one per round.
    #[test]
    fn a_soak_issues_one_request_per_slot_per_round() {
        let fakes = fakes(FakeHttp::new().with_reply(reply(10, 10)));
        let factors = Factors {
            soak: 3,
            concurrency: 4,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 4.0);
        exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);
        assert_eq!(fakes.http.requests().len(), 1 + 3 * 4);
    }

    #[test]
    fn a_soak_stops_when_the_box_starts_paging() {
        let fakes = Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new()
                .with_status(RssSnapshot::default())
                .with_swap_growth_gib(1.0),
            FakeGpu::new(),
            FakeHttp::new().with_reply(reply(10, 10)),
        );
        let factors = Factors {
            soak: 20,
            ..Factors::default()
        };
        let mut watchdog = SwapWatchdog::start(fakes.procfs.as_ref(), 2.0);
        exercise(&fakes.deps(), &factors, 18099, 1, &mut watchdog);
        assert!(watchdog.tripped().is_some());
        // Three rounds, not twenty: the swap check happens after each.
        assert_eq!(fakes.http.requests().len(), 1 + 3);
    }
}
