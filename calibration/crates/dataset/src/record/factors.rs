//! One measurable configuration.

use serde::{Deserialize, Serialize};

/// Every field is a factor that could plausibly move host memory, and every one
/// is recorded, so a row is a complete description of the process that produced
/// it.
///
/// `serde(default)` on the container: a record written before a factor existed
/// simply does not spell it, and the cell identity deliberately excludes
/// defaulted fields, so adding one has to stay free. It pairs with
/// `deny_unknown_fields` — a key that is *absent* is a version, a key that is
/// *unrecognised* is a drift.
///
/// The dataset settles two disagreements between the old reader and the old
/// writer. `ngl` is `u32`, not `i32`: no committed row is negative, and the
/// harness only ever writes an unsigned count. `ubatch`, `parallel`, `cram`,
/// `soak`, `concurrency`, and `probe_prompt_tokens` are plain counts rather
/// than `Option`s, because every committed row spells all six.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Factors {
    pub label: String,
    pub model: String,
    /// What questions this configuration answers. A tag rather than a
    /// schedule: it does not take part in the cell's identity, so one
    /// measurement serves every question that asked for it.
    pub purpose: Vec<String>,
    pub runtime: Runtime,
    /// `CUDA_VISIBLE_DEVICES`, verbatim.
    pub gpus: String,
    pub ctx: u32,
    pub ubatch: u32,
    pub batch: Option<u32>,
    pub parallel: u32,
    pub ngl: u32,
    pub split: Option<String>,
    pub kv_type: String,
    pub kv_unified: bool,
    pub flash_attn: String,
    pub n_cpu_moe: Option<u32>,
    pub mmproj: Option<String>,
    pub draft: Option<String>,
    pub spec_type: Option<String>,
    pub threads: Option<u32>,
    pub numa: Option<String>,
    pub cram: u32,
    pub no_mmap: bool,
    pub rtr: bool,
    pub embeddings: bool,
    pub served: bool,
    /// How many tokens the warm-up probe generates. Memory does not depend on
    /// it — the first-request step is identical at `n_predict` 8, 4096, and
    /// 12288 — so it is a timing knob, not a factor.
    pub probe_tokens: u32,
    /// How long the warm-up probe's *prompt* is, in tokens. This is what moves
    /// host memory: llama.cpp's server takes a context checkpoint while
    /// decoding a prompt, spaced by `--checkpoint-min-step` (8192 tokens), so
    /// the step measures 11 MiB at one token against 274 at sixty-four, and 431
    /// once past the spacing.
    pub probe_prompt_tokens: u32,
    pub soak: u32,
    pub concurrency: u32,
    /// Drive the vendored coding-agent benchmark instead of one short request,
    /// so the context grows the way an agent's does and the prompt cache fills
    /// on representative tokens.
    pub bench: bool,
    pub bench_turns: u32,
    /// Whether to raise the loader's log verbosity. Needed to read the buffer
    /// sizes, but verbose logging serialises graph ops, so growth runs turn it
    /// off: their subject is memory over time, and the arena is already known
    /// from the matching non-growth cell.
    pub verbose_log: bool,
    pub extra: Vec<String>,
    /// Distinguishes otherwise identical cells, so repeats can measure the
    /// noise floor instead of collapsing into one resume key.
    pub repeat: u32,
}

impl Default for Factors {
    fn default() -> Self {
        Self {
            label: String::new(),
            model: String::new(),
            purpose: Vec::new(),
            runtime: Runtime::Mainline,
            gpus: "0".to_owned(),
            ctx: 32768,
            ubatch: 512,
            batch: None,
            parallel: 1,
            ngl: 99,
            split: None,
            kv_type: "f16".to_owned(),
            kv_unified: false,
            flash_attn: "on".to_owned(),
            n_cpu_moe: None,
            mmproj: None,
            draft: None,
            spec_type: None,
            threads: None,
            numa: None,
            cram: 0,
            no_mmap: false,
            rtr: false,
            embeddings: false,
            served: true,
            probe_tokens: 64,
            probe_prompt_tokens: 4,
            soak: 0,
            concurrency: 1,
            bench: false,
            bench_turns: 40,
            verbose_log: true,
            extra: Vec::new(),
            repeat: 0,
        }
    }
}

/// Which llama.cpp the cell was measured against. The two forks size the graph
/// arena by different rules, so the fork is a factor rather than a detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    #[default]
    Mainline,
    Ik,
}

impl Factors {
    /// The split mode, with llama.cpp's default named rather than left absent.
    pub fn split_or_layer(&self) -> &str {
        self.split
            .as_deref()
            .filter(|split| !split.is_empty())
            .unwrap_or("layer")
    }

    /// How many cards the cell was pinned to, with `default` for an empty pin.
    ///
    /// An empty `gpus` means one card to most callers and zero to the
    /// device-scaling deriver, so the fallback is the caller's to choose.
    pub fn cards_or(&self, default: usize) -> usize {
        if self.gpus.is_empty() {
            default
        } else {
            self.gpus.split(',').count()
        }
    }

    /// The physical GPU ids this cell was pinned to.
    pub fn gpu_ids(&self) -> Vec<u32> {
        self.gpus
            .split(',')
            .filter(|id| !id.is_empty())
            .filter_map(|id| id.parse().ok())
            .collect()
    }
}
