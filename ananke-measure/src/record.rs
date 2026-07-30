//! The NDJSON record a measured cell is written as.
//!
//! Nested rather than flat: the hardware block, the per-device VRAM split, and
//! the factor set all have their own shapes, and a fixed column set would force
//! every one of them into a lowest common denominator — which is what
//! previously capped the GPU breakdown at two devices and made adding a factor
//! a schema migration.
//!
//! This is the writer's half only. The sampling that fills `trace`,
//! `checkpoints`, and `rss` still lives in the Python harness, so the types
//! here describe what it writes rather than producing it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};

use crate::parse::Parsed;

/// Bumped whenever a record's shape changes in a way an analysis must notice.
///
/// 1: the original flat CSV-era rows. 2: nested NDJSON with hardware and
/// traces. 3: generic per-device breakdown (tensor-split included), per-process
/// GPU memory, model identity, first-occurrence metadata, retained log tails.
pub const SCHEMA: u32 = 3;

/// One measured cell.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub schema: u32,
    /// The cell's stable identity, so a rerun skips what has already been
    /// measured.
    pub cell: String,
    pub status: Status,
    /// Facts that make a stale row identifiable later: when, on what box, and
    /// against which binary.
    pub provenance: BTreeMap<String, String>,
    pub hardware: Hardware,
    pub factors: Factors,
    pub parsed: Parsed,
    /// Peak resident memory, with the final reading and the growth since
    /// startup alongside it.
    pub rss: BTreeMap<String, Metric>,
    /// The end of a failed run's log, so a bad record says why it is bad.
    pub log_tail: String,
    /// The archived log's file name, which is what makes a record
    /// re-parseable rather than merely re-readable.
    pub log: String,
    /// The full time series, not just its summary: growth is a shape, and a
    /// peak alone cannot distinguish "allocated on first use" from "still
    /// climbing when we stopped looking".
    pub trace: Vec<Sample>,
    /// Memory against tokens, which a time series alone cannot give.
    pub checkpoints: Vec<Checkpoint>,
    /// Set when the `parsed` block was rebuilt from the archived log, so an
    /// analysis can tell which rows carry the newer fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reparsed: Option<bool>,
}

/// Why a record holds what it holds. Every reader takes `Ok` and skips the
/// rest, so a status it does not understand is a status it correctly ignores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Ok,
    /// Another server still held the port, so nothing was measured.
    PortBusy,
    FailedToLoad,
    Timeout,
    SkippedInsufficientMemory,
    HarnessError,
    /// A runtime upgrade invalidated the row; it keeps its data and its
    /// archived log, and only the status changes.
    StaleRuntime,
}

/// One measurable configuration.
///
/// Every field is a factor that could plausibly move host memory, and every one
/// is recorded, so a row is a complete description of the process that produced
/// it.
///
/// The struct carries `#[serde(default)]` because a record written before a
/// factor existed simply does not spell it: the cell identity deliberately
/// excludes defaulted fields, so adding one has to stay free.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Mainline,
    Ik,
}

/// The machine, in enough detail to key a calibration curve on.
///
/// Several terms are hardware-specific rather than universal, so a constant
/// fitted on one box is only transferable to another if you can tell the two
/// apart.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Hardware {
    pub gpus: Vec<Gpu>,
    pub cpu: Cpu,
    pub mem_total_gib: f64,
    pub kernel: String,
    /// The tuning skill's first sanity check: a `powersave` governor pins cores
    /// to the base clock and silently halves CPU-bound throughput.
    pub cpu_governor: String,
    pub transparent_hugepage: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Gpu {
    pub name: String,
    pub memory_total_mib: u64,
    pub compute_capability: String,
    pub driver: String,
}

/// Whatever `lscpu` and `/proc/cpuinfo` named; a field neither reports is left
/// absent rather than guessed.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Cpu {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cores_per_socket: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sockets: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa_nodes: Option<String>,
}

/// One entry of the `rss` summary. Nearly all of them are counts; the load
/// duration rides along beside them and is not.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Metric {
    Whole(i64),
    Fractional(f64),
}

/// One resident-memory sample, on the same two-second cadence ananke's
/// snapshotter uses — a single snapshot measures a different quantity than the
/// daemon does.
#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub t_seconds: f64,
    pub at_utc: String,
    #[serde(flatten)]
    pub rss: RssSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_used_mib: Option<u64>,
    #[serde(flatten)]
    pub gpu_per_device: GpuUsage,
}

/// One turn's memory reading, against the tokens that produced it.
#[derive(Debug, Clone, Serialize)]
pub struct Checkpoint {
    pub turn: u32,
    pub at_utc: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub generated_tokens_total: u64,
    /// What the server is holding, which is the term that scales with context
    /// rather than with use.
    pub kv_depth_tokens: u64,
    #[serde(flatten)]
    pub rss: RssSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_used_mib: Option<u64>,
    #[serde(flatten)]
    pub gpu_per_device: GpuUsage,
    /// Which of the alternating conversations this turn belongs to; absent
    /// outside a growth run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<u32>,
}

/// The `/proc/<pid>/status` resident-memory breakdown, in kB — the same three
/// figures ananke's `ProcFs` reads.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RssSnapshot {
    pub rss_total_kb: u64,
    pub rss_anon_kb: u64,
    pub rss_file_kb: u64,
    pub rss_shmem_kb: u64,
}

/// Per-process VRAM, split by card index.
///
/// llama.cpp's own breakdown attributes what it allocated; the driver counts
/// the CUDA context and everything else besides, and ik_llama does not print
/// the breakdown table at all, so for every ik cell this is the only per-device
/// source.
#[derive(Debug, Clone, Default)]
pub struct GpuUsage {
    pub used_mib: BTreeMap<u32, u64>,
}

impl Serialize for GpuUsage {
    /// Flattened into the sample as `gpu<index>_used_mib`, and absent
    /// altogether when the driver reported nothing.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.used_mib.len()))?;
        for (index, mib) in &self.used_mib {
            map.serialize_entry(&format!("gpu{index}_used_mib"), mib)?;
        }
        map.end()
    }
}
