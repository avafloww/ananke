//! The measurement record as it appears in `measurements.ndjson`.
//!
//! Only the fields the calibration tools read are declared; serde ignores the
//! rest, so the schema here can lag the harness without breaking. The harness is
//! still Python (`scripts/calibration/measure.py`) and remains the authority on
//! what a record contains — this is a reader.
//!
//! Field names match the JSON exactly rather than being renamed to Rust
//! conventions, because the NDJSON is the interchange format between the two
//! halves and a rename is a silent way for them to disagree.

use std::collections::BTreeMap;

use serde::Deserialize;

/// One measured cell.
#[derive(Debug, Clone, Deserialize)]
pub struct Record {
    /// A hash of the factors, so two records sharing one describe the same
    /// configuration. Absent from the oldest schema versions.
    #[serde(default)]
    pub cell: Option<String>,
    pub status: String,
    pub factors: Factors,
    #[serde(default)]
    pub parsed: Parsed,
    #[serde(default)]
    pub rss: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub hardware: Hardware,
    #[serde(default)]
    pub provenance: Provenance,
    /// Path to the captured runtime log, quoted in a disagreement message so a
    /// failing cell can be opened rather than merely counted.
    #[serde(default)]
    pub log: Option<String>,
}

/// The configuration the cell was measured under. One field per knob the
/// harness varies.
#[derive(Debug, Clone, Deserialize)]
pub struct Factors {
    #[serde(default)]
    pub label: String,
    pub model: String,
    pub ctx: u32,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub parallel: Option<u32>,
    #[serde(default)]
    pub kv_unified: bool,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub gpus: String,
    #[serde(default)]
    pub kv_type: Option<String>,
    #[serde(default)]
    pub ngl: Option<i32>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub draft: Option<String>,
    #[serde(default)]
    pub spec_type: Option<String>,
    #[serde(default)]
    pub flash_attn: Option<String>,
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub cram: Option<u32>,
    #[serde(default)]
    pub embeddings: bool,
    #[serde(default)]
    pub served: bool,
    /// A benchmark cell runs a whole agentic workload, so its host figures
    /// include the prompt cache and several rounds of growth. Every host
    /// deriver excludes them.
    #[serde(default)]
    pub bench: bool,
    /// `--no-mmap` reads the weights into anonymous memory instead of mapping
    /// them, which moves them into the counter the host derivers read. Cells
    /// that set it are excluded from any term that models overhead rather than
    /// weights.
    #[serde(default)]
    pub no_mmap: bool,
    /// ik_llama's run-time repack, which forces `--no-mmap`.
    #[serde(default)]
    pub rtr: bool,
    #[serde(default)]
    pub numa: Option<String>,
    /// How many rounds the probe grew its prompt over. It costs memory of its
    /// own, so it belongs in every pairing key.
    #[serde(default)]
    pub soak: Option<u32>,
    /// How many requests the probe issued at once. The per-slot host term is
    /// measured against this rather than against `parallel`, since an idle slot
    /// costs nothing.
    #[serde(default)]
    pub concurrency: Option<u32>,
    /// How long the probe's prompt was. Past `CHECKPOINT_MIN_STEP` the process
    /// holds more than one context checkpoint, which is a different steady
    /// state and not what the baseline terms model.
    #[serde(default)]
    pub probe_prompt_tokens: Option<u32>,
    /// Raw extra runtime arguments, searched for flags the record schema has no
    /// field of its own for (ik's `-dsa`).
    #[serde(default)]
    pub extra: Vec<String>,
}

/// What the harness read back out of the runtime's own logs.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Parsed {
    #[serde(default)]
    pub arch: Option<String>,
    /// Hidden width. Absent where the runtime's log did not name it.
    #[serde(default)]
    pub n_embd: Option<u32>,
    #[serde(default)]
    pub n_vocab: Option<u64>,
    /// Set on the Gemma E-variants, which keep their embeddings per layer on the
    /// host. That is a different graph under the same architecture string, so it
    /// discriminates a fit variant.
    #[serde(default)]
    pub per_layer_token_embd: Option<bool>,
    /// llama.cpp's memory breakdown table, one row per device — or a single fused
    /// `Meta()` row under a tensor split. Empty for a runtime that prints no
    /// table at all, which is ik.
    #[serde(default)]
    pub devices: Vec<Device>,
    /// The runtime's own buffer lines, one entry per context it created. The only
    /// route to a per-device figure where there is no breakdown table.
    #[serde(default)]
    pub contexts: Vec<Context>,
    /// The pinned graph arena llama.cpp reports as its `CUDA_Host compute
    /// buffer size`, in MiB. Absent for a cell whose log did not print it.
    #[serde(default)]
    pub arena_mib: Option<f64>,
    #[serde(default)]
    pub n_layer: Option<u32>,
    /// The sliding-attention window, zero on a model without one.
    #[serde(default)]
    pub n_swa: Option<u32>,
    #[serde(default)]
    pub n_expert: Option<u32>,
    #[serde(default)]
    pub n_expert_used: Option<u32>,
    /// Query heads. Absent where the GGUF omits `attention.head_count`, which
    /// is why `query_head_count` has a fallback.
    #[serde(default)]
    pub n_head: Option<u32>,
    #[serde(default)]
    pub n_head_kv: Option<u32>,
    #[serde(default)]
    pub n_embd_head_k: Option<u32>,
    #[serde(default)]
    pub n_embd_head_v: Option<u32>,
    /// Weights the runtime placed on the host, in MiB.
    #[serde(default)]
    pub cpu_model_mib: Option<f64>,
    /// The GGUF's own metadata, verbatim. Every value in the dataset is an
    /// integer, but it is kept as `Value` so a string-valued key cannot make a
    /// record unreadable.
    #[serde(default)]
    pub gguf_kv: BTreeMap<String, serde_json::Value>,
    /// llama.cpp's `[mtmd] adding N MiB to fit_params_target` figure, per
    /// device name. It covers the projector's weights *and* its graph.
    #[serde(default)]
    pub mmproj_reserved_mib: BTreeMap<String, f64>,
    /// The summed `clip_model_loader` tensor sizes, which is what isolates the
    /// graph term out of the reservation above.
    #[serde(default)]
    pub mmproj_tensor_bytes: Option<u64>,
    #[serde(default)]
    pub clip_image_size: Option<u32>,
    #[serde(default)]
    pub clip_n_merge: Option<u32>,
    /// llama.cpp's `[spec] estimated memory usage of MTP context is N MiB`.
    #[serde(default)]
    pub mtp_context_mib: Option<f64>,
    /// `{arch}.nextn_predict_layers`: how many trailing blocks the embedded MTP
    /// head spans.
    #[serde(default)]
    pub nextn_predict_layers: Option<u32>,
    /// Everything the schema above does not name, which is how the per-card
    /// `gpu{index}_model_mib` columns are reached without eight fields.
    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}

/// One row of llama.cpp's memory breakdown table.
///
/// A tensor split reports a single fused row whose `total`, `free`, and `self`
/// are summed across cards but whose `model`, `kv`, and `compute` columns are one
/// card's share. The row is recognised by its `device` name starting `Meta`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Device {
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub model_mib: f64,
    #[serde(default)]
    pub kv_mib: f64,
    #[serde(default)]
    pub compute_mib: f64,
    #[serde(default)]
    pub unaccounted_mib: f64,
}

/// One llama context's buffer lines, keyed by backend name then by role
/// (`model`, `kv`, `rs`, `compute`, `output`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Context {
    #[serde(default)]
    pub buffers: BTreeMap<String, BTreeMap<String, f64>>,
    /// The attention caches this context allocated, one entry per span of
    /// layers sharing a pool.
    #[serde(default)]
    pub kv_pools: Vec<KvPool>,
    /// The recurrent-state module, on a hybrid or recurrent model. A context
    /// creates at most one.
    #[serde(default)]
    pub rs_pool: Option<RsPool>,
}

/// One `llama_kv_cache` pool, as the load log summarises it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct KvPool {
    #[serde(default)]
    pub total_mib: f64,
    /// How many layers the pool spans — not necessarily how many allocate.
    #[serde(default)]
    pub layers: u32,
    #[serde(default)]
    pub seqs: u32,
}

/// One `llama_memory_recurrent` module, as the load log summarises it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RsPool {
    #[serde(default)]
    pub total_mib: f64,
    /// The span the module covers. The attention layers inside it do not
    /// allocate, so this is larger than the number of recurrent layers.
    #[serde(default)]
    pub layers: u32,
    #[serde(default)]
    pub seqs: u32,
    /// llama.cpp's `n_rs_seq`: extra copies held so a speculative draft can be
    /// rolled back. Zero without speculative decoding.
    #[serde(default)]
    pub rs_seq: u32,
    #[serde(default)]
    pub r_mib: f64,
    #[serde(default)]
    pub s_mib: f64,
}

/// The machine the cell ran on.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Hardware {
    #[serde(default)]
    pub gpus: Vec<HardwareGpu>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HardwareGpu {
    pub memory_total_mib: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Provenance {
    #[serde(default)]
    pub model_key: String,
    /// When the cell was measured, ISO-8601. Compared as a string, which sorts
    /// correctly for that format and avoids a date dependency.
    #[serde(default)]
    pub measured_at_utc: String,
    /// The runtime binary's digest. A cell id hashes the *factors*, and the
    /// binary is not one of them, so this is the only thing that distinguishes
    /// two readings of one configuration taken under different builds.
    #[serde(default)]
    pub runtime_sha256: String,
}

impl Record {
    /// One `rss` sample, in KiB, absent reading as zero the way the Python does.
    pub fn rss_kb(&self, key: &str) -> i64 {
        self.rss.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
    }

    /// Host memory the process *owns*, in bytes.
    ///
    /// `RssAnon + RssShmem`, not `VmRSS`: `cudaMallocHost` is accounted as
    /// shmem, and `RssFile` is the mapped GGUF, which llama.cpp populates and
    /// then leaves resident as clean reclaimable pages.
    pub fn owned_bytes(&self) -> i64 {
        (self.rss_kb("rss_anon_kb") + self.rss_kb("rss_shmem_kb")) * 1024
    }

    /// The same figure in whole MiB, truncated.
    pub fn owned_mib(&self) -> i64 {
        (self.rss_kb("rss_anon_kb") + self.rss_kb("rss_shmem_kb")) / 1024
    }

    /// The driver's total for the whole process, in MiB.
    pub fn gpu_used_mib(&self) -> Option<u64> {
        self.rss.get("gpu_used_mib").and_then(|v| v.as_u64())
    }

    /// One card's driver reading, in MiB.
    ///
    /// Keyed by the *physical* id: the sampler records `gpu{id}_used_mib` while
    /// the loader's breakdown rows are in visible order, so a cell pinned to
    /// GPU 1 has its usage under `gpu1_used_mib` and its breakdown row under
    /// `CUDA0`. A zero reads as absent, as it does on the Python side.
    pub fn gpu_card_used_mib(&self, card: &str) -> Option<f64> {
        self.rss
            .get(&format!("gpu{card}_used_mib"))
            .and_then(|v| v.as_f64())
            .filter(|v| *v != 0.0)
    }

    /// How many cards the driver reported usage on.
    ///
    /// `gpu_used_mib` is the process total and is deliberately excluded; the
    /// per-card keys are `gpu{physical id}_used_mib`.
    pub fn cards_measured(&self) -> usize {
        self.rss
            .iter()
            .filter(|(key, value)| {
                key.starts_with("gpu")
                    && key.ends_with("_used_mib")
                    && *key != "gpu_used_mib"
                    && value.as_u64().is_some_and(|v| v > 0)
            })
            .count()
    }

    /// The physical GPU ids this cell was pinned to.
    pub fn gpu_ids(&self) -> Vec<u32> {
        self.factors
            .gpus
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect()
    }
}

impl Factors {
    /// The split mode, with llama.cpp's default named rather than left absent.
    pub fn split_or_layer(&self) -> &str {
        self.split
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("layer")
    }

    /// How many cards the cell was pinned to, with `default` for an empty pin.
    ///
    /// The Python spells this several ways at different call sites — an empty
    /// `gpus` reads as one card in most and as zero in the device-scaling
    /// deriver — so the fallback is the caller's to choose.
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
            .filter(|g| !g.is_empty())
            .count()
            .max(1)
    }

    /// Batch tokens the graph actually processes: a context shorter than the
    /// batch caps it.
    pub fn tokens(&self) -> u64 {
        u64::from(self.ctx).min(u64::from(self.ubatch.unwrap_or(0)))
    }

    /// The physical batch, with llama.cpp's default named where the cell did not
    /// set one. Zero reads as absent, as it does on the Python side.
    pub fn ubatch_or_default(&self) -> u32 {
        match self.ubatch.unwrap_or(0) {
            0 => 512,
            value => value,
        }
    }

    pub fn flash_attn_on(&self) -> bool {
        self.flash_attn.as_deref() == Some("on")
    }

    /// Whether any expert layers were pushed to the host. A hybrid does not
    /// replicate the graph's masks across cards, which several derivers turn on.
    pub fn is_hybrid(&self) -> bool {
        self.n_cpu_moe.unwrap_or(0) != 0
    }

    pub fn has_spec(&self) -> bool {
        self.spec_type.as_deref().is_some_and(|s| !s.is_empty())
    }

    pub fn runtime_is_ik(&self) -> bool {
        self.runtime == "ik"
    }
}

impl Parsed {
    /// One card's share of the weights, from the breakdown table's own columns.
    pub fn gpu_model_mib(&self, index: usize) -> f64 {
        self.other
            .get(&format!("gpu{index}_model_mib"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// A GGUF metadata integer, absent reading as zero.
    pub fn gguf_int(&self, key: &str) -> i64 {
        self.gguf_kv.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
    }
}

/// Read an NDJSON measurement file, skipping blank lines.
pub fn read_ndjson(text: &str) -> Result<Vec<Record>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}
