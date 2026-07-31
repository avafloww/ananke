//! Where and on what the cell ran.

use serde::{Deserialize, Serialize};

/// Facts that make a stale row identifiable later.
///
/// Every value is a `String` even where it names a number: the harness wrote
/// this as a string map, `model_bytes` is spelled as quoted digits in all 643
/// committed rows, and retyping it would change the bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Provenance {
    /// When the cell was measured, ISO-8601. Compared as a string, which sorts
    /// correctly for that format and avoids a date dependency.
    pub measured_at_utc: String,
    pub measured_at_local: String,
    pub host: String,
    pub binary: String,
    pub ananke_rev: String,
    pub runtime_version: String,
    /// A cell id hashes the *factors*, and the binary is not one of them, so
    /// this digest is the only thing distinguishing two readings of one
    /// configuration taken under different builds.
    pub runtime_sha256: String,
    pub ananke_dirty: String,
    pub model_file_at: String,
    pub model_key: String,
    pub model_quant: String,
    pub model_bytes: String,
}

/// The machine, in enough detail to key a calibration curve on: several terms
/// are hardware-specific, so a constant fitted on one box is only transferable
/// to another if you can tell the two apart.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gpu {
    pub name: String,
    pub memory_total_mib: u64,
    pub compute_capability: String,
    pub driver: String,
}

/// Whatever `lscpu` and `/proc/cpuinfo` named; a field neither reports is left
/// absent rather than guessed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
