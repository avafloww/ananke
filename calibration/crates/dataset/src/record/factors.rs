//! One measurable configuration.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Every factor that could plausibly move host memory, so a row is a complete
/// description of the process that produced it.
///
/// `serde(default)` with `deny_unknown_fields`: a key that is *absent* is an
/// older version and must stay free to add, while a key that is *unrecognised*
/// is a drift and must fail.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Factors {
    pub label: String,
    pub model: String,
    /// What questions this configuration answers. Deliberately outside the
    /// cell's identity, so one measurement serves every question that asked for
    /// it.
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
    /// A timing knob, not a factor: the first-request step is identical at
    /// `n_predict` 8, 4096, and 12288.
    pub probe_tokens: u32,
    /// The warm-up probe's *prompt* length, which does move host memory:
    /// llama.cpp's server takes a context checkpoint while decoding a prompt,
    /// spaced by `--checkpoint-min-step` (8192 tokens), so the step measures 11
    /// MiB at one token against 274 at sixty-four and 431 past the spacing.
    pub probe_prompt_tokens: u32,
    pub soak: u32,
    pub concurrency: u32,
    /// Drive the vendored coding-agent benchmark instead of one short request,
    /// so the context grows the way an agent's does and the prompt cache fills
    /// on representative tokens.
    pub bench: bool,
    pub bench_turns: u32,
    /// Needed to read the buffer sizes, but verbose logging serialises graph
    /// ops, so growth runs turn it off: their subject is memory over time, and
    /// the arena is already known from the matching non-growth cell.
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
            ubatch: DEFAULT_UBATCH,
            batch: None,
            parallel: 1,
            ngl: FULLY_OFFLOADED,
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

impl Runtime {
    /// How the fork is spelled in the record, in a cell's label, and in every
    /// report keyed on it — deliberately the same word in all three.
    pub fn name(self) -> &'static str {
        match self {
            Runtime::Mainline => "mainline",
            Runtime::Ik => "ik",
        }
    }
}

impl fmt::Display for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
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

    /// How many cards were named, ignoring empty entries, never below one.
    pub fn cards_nonempty(&self) -> usize {
        self.gpus
            .split(',')
            .filter(|id| !id.is_empty())
            .count()
            .max(1)
    }

    pub fn gpu_ids(&self) -> Vec<u32> {
        self.gpus
            .split(',')
            .filter(|id| !id.is_empty())
            .filter_map(|id| id.parse().ok())
            .collect()
    }

    /// Batch tokens the graph actually processes: a context shorter than the
    /// batch caps it.
    pub fn tokens(&self) -> u64 {
        u64::from(self.ctx).min(u64::from(self.ubatch_or_default()))
    }

    /// A zero reads as absent — llama.cpp's default — rather than as a batch of
    /// nothing.
    pub fn ubatch_or_default(&self) -> u32 {
        match self.ubatch {
            0 => DEFAULT_UBATCH,
            value => value,
        }
    }

    pub fn flash_attn_on(&self) -> bool {
        self.flash_attn == "on"
    }

    /// A partly- or un-offloaded process builds a different graph, so most
    /// derivations hold this fixed rather than modelling across it.
    pub fn fully_offloaded(&self) -> bool {
        self.ngl == FULLY_OFFLOADED
    }

    /// A hybrid does not replicate the graph's masks across cards, which
    /// several derivers turn on.
    pub fn is_hybrid(&self) -> bool {
        self.n_cpu_moe.unwrap_or(0) != 0
    }

    pub fn has_spec(&self) -> bool {
        self.spec_type
            .as_deref()
            .is_some_and(|spec| !spec.is_empty())
    }

    pub fn runtime_is_ik(&self) -> bool {
        self.runtime == Runtime::Ik
    }
}

/// llama.cpp's own micro-batch default, which a cell that names no `-ub` runs
/// at.
const DEFAULT_UBATCH: u32 = 512;

/// The `-ngl` the campaign spells "every layer on a device" as. Any count at or
/// above the model's layer total does it; 99 is the one every cell uses.
const FULLY_OFFLOADED: u32 = 99;
