//! Runtime-flavour vocabulary for llama-cpp services: NUMA strategy, the
//! mainline-vs-ik_llama runtime split, and the expert-offload mode.

use ananke_config::flags;

use crate::config::validate::{flag_variant, variant_flag};

/// NUMA thread-and-memory placement strategy for a llama-cpp service,
/// emitted as llama.cpp's `--numa <strategy>`. Resolved from the `numa`
/// config value; unset emits no flag (llama.cpp's default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaStrategy {
    /// Spread worker threads across all nodes and interleave the model's
    /// memory allocation across them — balances memory-bandwidth load on
    /// multi-node / multi-CCD hosts (e.g. Threadripper).
    Distribute,
    /// Confine threads and allocation to a single NUMA node.
    Isolate,
    /// Defer placement to an external `numactl` mask.
    Numactl,
}

impl NumaStrategy {
    /// Variant ↔ flag binding, sourced from
    /// [`ananke_config::flags::numa`] (see [`SplitMode::VARIANTS`]).
    const VARIANTS: &'static [(Self, &'static str)] = &[
        (Self::Distribute, flags::numa::DISTRIBUTE),
        (Self::Isolate, flags::numa::ISOLATE),
        (Self::Numactl, flags::numa::NUMACTL),
    ];

    /// The `--numa` flag value.
    pub fn as_flag(self) -> &'static str {
        variant_flag(Self::VARIANTS, self)
    }

    /// Parse an accepted `numa` string into a variant.
    pub fn from_flag(s: &str) -> Option<Self> {
        flag_variant(Self::VARIANTS, s)
    }

    /// Accepted values as a quoted list for operator-facing errors.
    pub fn valid_values() -> String {
        flags::quoted_list(flags::numa::ALL)
    }
}

/// Serving runtime for a llama-cpp-template service, mirroring
/// [`crate::config::parse::RawRuntime`]. `IkLlama` carries the fork's
/// validated knobs and switches spawn/estimation to the fork's flag and
/// memory conventions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Runtime {
    #[default]
    LlamaCpp,
    IkLlama(IkSettings),
}

impl Runtime {
    /// The fork settings, when this is the fork runtime.
    pub fn ik(&self) -> Option<&IkSettings> {
        match self {
            Runtime::LlamaCpp => None,
            Runtime::IkLlama(ik) => Some(ik),
        }
    }
}

/// Validated ik_llama.cpp settings. See
/// [`crate::config::parse::RawIkSettings`] for per-field semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IkSettings {
    /// `-mla` kernel mode (0-3).
    pub mla: Option<u32>,
    /// DSA sparse attention (`-dsa -fidx`).
    pub dsa: bool,
    /// `-amb` attention scratch cap in MiB.
    pub attn_max_batch: Option<u32>,
    /// `-rtr` runtime repacking.
    pub runtime_repack: bool,
}

/// MoE expert-offload policy for a llama-cpp service. Resolved from the
/// `expert_offload` config value. The packer reads this to decide whether and
/// how much expert weight to move off the GPU when the model doesn't fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadMode {
    /// No expert offload. The model packs whole layers, spilling entire layers
    /// to CPU only under a CPU-allowing placement.
    #[default]
    Off,
    /// The packer keeps each expert on its layer's home GPU while that GPU has
    /// room, then greedily spills the experts that don't fit — to the most-free
    /// other GPU first, then to CPU — so only the surplus over live VRAM moves.
    Auto,
    /// The packer offloads the experts of exactly the `N` tail-most
    /// expert-bearing layers, regardless of fit.
    Layers(u32),
}

impl OffloadMode {
    /// Whether any expert offload is requested (i.e. not [`OffloadMode::Off`]).
    pub fn is_enabled(self) -> bool {
        !matches!(self, OffloadMode::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::{test_fixtures::parse_and_merge, validate};

    #[test]
    fn numa_vocab_is_single_sourced_and_complete() {
        for &(variant, flag) in NumaStrategy::VARIANTS {
            assert_eq!(variant.as_flag(), flag);
            assert_eq!(NumaStrategy::from_flag(flag), Some(variant));
        }
        for variant in [
            NumaStrategy::Distribute,
            NumaStrategy::Isolate,
            NumaStrategy::Numactl,
        ] {
            match variant {
                NumaStrategy::Distribute | NumaStrategy::Isolate | NumaStrategy::Numactl => {}
            }
            assert!(
                NumaStrategy::VARIANTS.iter().any(|&(v, _)| v == variant),
                "{variant:?} missing from NumaStrategy::VARIANTS"
            );
        }
        assert_eq!(NumaStrategy::from_flag("bogus"), None);
        assert_eq!(
            NumaStrategy::valid_values(),
            flags::quoted_list(flags::numa::ALL)
        );
    }
    #[test]
    fn expert_offload_parses_auto_and_count() {
        let auto = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&auto).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Auto
        );

        let count = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = 16
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&count).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Layers(16)
        );
    }
    #[test]
    fn ik_runtime_parses_and_validates() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 131072
spec_type = "mtp:n_max=4,p_min=0.5"
runtime = { kind = "ik-llama", mla = 1, dsa = true, attn_max_batch = 512 }
lifecycle = "persistent"
"#,
        );
        let e = validate(&cfg).unwrap();
        let svc = &e.services[0];
        let lc = svc.llama_cpp().unwrap();
        let ik = lc.runtime.ik().expect("ik runtime");
        assert_eq!(ik.mla, Some(1));
        assert!(ik.dsa);
        assert_eq!(ik.attn_max_batch, Some(512));
        assert!(!ik.runtime_repack);
    }

    #[test]
    fn ik_runtime_rejects_unknown_keys_in_table() {
        // deny_unknown_fields must hold through the internally-tagged
        // enum's newtype variant — a typo in the runtime table is a hard
        // error, not a silent no-op.
        let err = crate::config::parse::parse_toml(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
runtime = { kind = "ik-llama", mla = 1, n_cpu_moe = 4 }
"#,
            std::path::Path::new("/fake/ananke.toml"),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("n_cpu_moe"),
            "unknown runtime key must be rejected, got: {err}"
        );
    }

    #[test]
    fn ik_runtime_gates_spec_type_dialects() {
        // Mainline service with ik-dialect spec_type.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
spec_type = "mtp:n_max=4"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("ik_llama syntax"), "got: {err}");

        // ik service with mainline-dialect spec_type.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
spec_type = "draft-mtp"
runtime = { kind = "ik-llama" }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("mainline syntax"), "got: {err}");
    }

    #[test]
    fn ik_dsa_requires_f16_kv() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
cache_type_k = "q8_0"
flash_attn = true
runtime = { kind = "ik-llama", dsa = true }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("requires f16 KV"), "got: {err}");
    }

    #[test]
    fn rejects_attn_max_batch_zero() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
runtime = { kind = "ik-llama", attn_max_batch = 0 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("attn_max_batch must be > 0"),
            "got: {err}"
        );
    }
    #[test]
    fn expert_offload_requires_hybrid_placement() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("expert_offload requires placement=hybrid"),
            "got: {err}"
        );
    }

    #[test]
    fn expert_offload_rejects_sharded_split() {
        // A sharded (tensor/row) split has no CPU half, so it cannot honour an
        // expert offload to host RAM — reject the combination explicitly rather
        // than leaving the operator to infer it from the placement constraints.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("expert_offload cannot be combined with devices.split"),
            "got: {err}"
        );
    }
}
