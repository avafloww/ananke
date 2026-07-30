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
}

impl Record {
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

/// Read an NDJSON measurement file, skipping blank lines.
pub fn read_ndjson(text: &str) -> Result<Vec<Record>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}
