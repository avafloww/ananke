//! One measurable configuration.

use std::fmt;

use ananke_config::{flags::cache_type, placement::SplitMode};
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
    pub split: Option<SplitMode>,
    pub kv_type: KvType,
    pub kv_unified: bool,
    pub flash_attn: FlashAttn,
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
            kv_type: KvType::F16,
            kv_unified: false,
            flash_attn: FlashAttn::On,
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
            probe_prompt_tokens: DEFAULT_PROBE_PROMPT_TOKENS,
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
/// arena by different rules, so the fork is a factor rather than a detail — and
/// the same fork marker the daemon's runtime config carries.
pub use ananke_config::runtime::Runtime;

/// What `-fa` was set to, which decides whether the attention pass is fused.
///
/// Guardrail: `Auto` is not a synonym for either of the others. llama.cpp
/// resolves it at load time against the backend, so a record spelling it says
/// what was *asked for* and not what happened — and the KQ mask's element width
/// is a factor of two on the answer. Every deriver that fits against that term
/// excludes `Auto` rather than guessing, via [`FlashAttn::fused`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum FlashAttn {
    /// `-fa on`, fused.
    #[default]
    On,
    /// `-fa off`, unfused: the graph materialises the score matrix.
    Off,
    /// `-fa auto`, the recent builds' default: fused where the backend can.
    Auto,
}

impl FlashAttn {
    /// The KQ mask's element width in bytes when the pass is fused (f16).
    pub const FUSED_MASK_BYTES: u64 = 2;
    /// The KQ mask's element width in bytes when it is not (f32).
    pub const UNFUSED_MASK_BYTES: u64 = 4;

    /// How the flag is spelled on the command line and in the record.
    pub fn name(self) -> &'static str {
        match self {
            FlashAttn::On => "on",
            FlashAttn::Off => "off",
            FlashAttn::Auto => "auto",
        }
    }

    /// Whether the pass is fused, or `None` where only the runtime knows.
    pub fn fused(self) -> Option<bool> {
        match self {
            FlashAttn::On => Some(true),
            FlashAttn::Off => Some(false),
            FlashAttn::Auto => None,
        }
    }

    /// Whether the *charge* is the unfused one. `Auto` is charged unfused so
    /// an unresolved cell over-reserves rather than under-reserving; the
    /// derivers that *fit* the unfused terms use [`Self::fused`] instead, so no
    /// unresolved cell reaches a fit.
    pub fn charged_unfused(self) -> bool {
        self.fused() != Some(true)
    }

    /// The KQ mask's element width, f32 wherever the charge is the unfused one.
    pub fn mask_element_bytes(self) -> u64 {
        if self.charged_unfused() {
            Self::UNFUSED_MASK_BYTES
        } else {
            Self::FUSED_MASK_BYTES
        }
    }
}

impl fmt::Display for FlashAttn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What `-ctk`/`-ctv` were set to. Both halves of the cache take one type in
/// this campaign, so the record carries one field.
///
/// Closed on llama.cpp's own list rather than free text, so a spelling the
/// runtime would reject fails the plan reader instead of the server.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum KvType {
    /// Unquantised 32-bit float.
    #[serde(rename = "f32")]
    F32,
    /// Unquantised 16-bit float, llama.cpp's default and the campaign's.
    #[default]
    #[serde(rename = "f16")]
    F16,
    /// Unquantised brain float.
    #[serde(rename = "bf16")]
    Bf16,
    /// 8-bit quantised, the only quantised type the campaign measured.
    #[serde(rename = "q8_0")]
    Q80,
    /// 5-bit quantised, with a per-block offset.
    #[serde(rename = "q5_1")]
    Q51,
    /// 5-bit quantised.
    #[serde(rename = "q5_0")]
    Q50,
    /// 4-bit quantised, with a per-block offset.
    #[serde(rename = "q4_1")]
    Q41,
    /// 4-bit quantised.
    #[serde(rename = "q4_0")]
    Q40,
    /// 4-bit non-linear quantised.
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
}

impl KvType {
    /// How the type is spelled on the command line and in the record.
    pub fn name(self) -> &'static str {
        match self {
            KvType::F32 => cache_type::F32,
            KvType::F16 => cache_type::F16,
            KvType::Bf16 => cache_type::BF16,
            KvType::Q80 => cache_type::Q8_0,
            KvType::Q51 => cache_type::Q5_1,
            KvType::Q50 => cache_type::Q5_0,
            KvType::Q41 => cache_type::Q4_1,
            KvType::Q40 => cache_type::Q4_0,
            KvType::Iq4Nl => cache_type::IQ4_NL,
        }
    }

    /// Whether the cache is stored quantised, which costs extra pinned memory.
    ///
    /// The predicate is [`cache_type::is_quantised`], shared with
    /// `ananke_estimate::host_buffer`, so the rate is fitted over exactly the
    /// rows it is later charged to.
    pub fn is_quantised(self) -> bool {
        cache_type::is_quantised(self.name())
    }
}

impl fmt::Display for KvType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Factors {
    /// The split mode, with llama.cpp's default named rather than left absent.
    pub fn split_or_layer(&self) -> SplitMode {
        self.split.unwrap_or_default()
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

    /// Definitely fused. `-fa auto` is neither this nor [`Self::flash_attn_off`],
    /// so a deriver that filters on both never fits an unresolved cell.
    pub fn flash_attn_on(&self) -> bool {
        self.flash_attn.fused() == Some(true)
    }

    /// Definitely unfused.
    pub fn flash_attn_off(&self) -> bool {
        self.flash_attn.fused() == Some(false)
    }

    /// Whether the cell ran a quantised KV cache.
    pub fn kv_quantised(&self) -> bool {
        self.kv_type.is_quantised()
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

/// The probe prompt a cell gets when it asks for no particular length. Too short
/// to reach a checkpoint, which is what makes it the control half of every
/// checkpoint-headroom pair.
pub const DEFAULT_PROBE_PROMPT_TOKENS: u32 = 4;

/// The `-ngl` the campaign spells "every layer on a device" as. Any count at or
/// above the model's layer total does it; this is the one every cell uses.
pub const FULLY_OFFLOADED: u32 = 99;

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire and the flag are one spelling.
    ///
    /// `name()` reaches the server's command line and the serde rename reaches
    /// the dataset and the payload a cell's identity is hashed over. They are
    /// written apart, so a drift between them would re-hash every cell in the
    /// campaign while the harness went on passing the old flag.
    #[test]
    fn a_factors_flag_and_its_recorded_spelling_are_the_same_word() {
        let flash = [FlashAttn::On, FlashAttn::Off, FlashAttn::Auto];
        for variant in flash {
            match variant {
                FlashAttn::On | FlashAttn::Off | FlashAttn::Auto => {}
            }
            assert_eq!(
                serde_json::to_string(&variant).expect("a unit variant serializes"),
                format!("\"{}\"", variant.name())
            );
        }

        let kv = [
            KvType::F32,
            KvType::F16,
            KvType::Bf16,
            KvType::Q80,
            KvType::Q51,
            KvType::Q50,
            KvType::Q41,
            KvType::Q40,
            KvType::Iq4Nl,
        ];
        for variant in kv {
            match variant {
                KvType::F32
                | KvType::F16
                | KvType::Bf16
                | KvType::Q80
                | KvType::Q51
                | KvType::Q50
                | KvType::Q41
                | KvType::Q40
                | KvType::Iq4Nl => {}
            }
            assert_eq!(
                serde_json::to_string(&variant).expect("a unit variant serializes"),
                format!("\"{}\"", variant.name())
            );
        }
        assert_eq!(kv.len(), cache_type::ALL.len());
    }
}
