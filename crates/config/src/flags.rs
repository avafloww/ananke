//! Canonical vocabularies for enum-valued config fields.
//!
//! Each accepted string literal for a validated enum field (`devices.split`,
//! `numa`, `expert_offload`, …) lives here exactly once. Both the schema
//! documentation (this crate's [`crate::docs`]) and the daemon's config
//! validation (the `ananke` crate's `NumaStrategy`, `SplitMode`,
//! `OffloadMode`) reference these constants, so adding or renaming an
//! accepted value is a one-line change that cannot leave the parser, the
//! error messages, and the docs disagreeing.

/// `devices.split` / `--split-mode` values.
pub mod split_mode {
    /// Whole-layer pipeline across GPUs (the default).
    pub const LAYER: &str = "layer";
    /// Tensor parallelism, llama.cpp's older implementation.
    pub const ROW: &str = "row";
    /// Tensor parallelism, llama.cpp's newer implementation.
    pub const TENSOR: &str = "tensor";
    /// Every accepted value, in declaration order.
    pub const ALL: &[&str] = &[LAYER, ROW, TENSOR];
}

/// `numa` / `--numa` values.
pub mod numa {
    /// Spread threads and interleave memory across all NUMA nodes.
    pub const DISTRIBUTE: &str = "distribute";
    /// Confine threads and allocation to a single node.
    pub const ISOLATE: &str = "isolate";
    /// Defer placement to an external `numactl` mask.
    pub const NUMACTL: &str = "numactl";
    /// Every accepted value, in declaration order.
    pub const ALL: &[&str] = &[DISTRIBUTE, ISOLATE, NUMACTL];
}

/// `cache_type_k` / `cache_type_v` / `-ctk` / `-ctv` values.
///
/// The config field is a free-form string rather than a validated enum, so this
/// vocabulary is llama.cpp's rather than the schema's; it exists so the
/// estimator and the calibration agree on which spellings mean a quantised
/// cache.
pub mod cache_type {
    /// Unquantised 32-bit float.
    pub const F32: &str = "f32";
    /// Unquantised 16-bit float, llama.cpp's default.
    pub const F16: &str = "f16";
    /// Unquantised brain float.
    pub const BF16: &str = "bf16";
    /// 8-bit quantised.
    pub const Q8_0: &str = "q8_0";
    /// 5-bit quantised, with a per-block offset.
    pub const Q5_1: &str = "q5_1";
    /// 5-bit quantised.
    pub const Q5_0: &str = "q5_0";
    /// 4-bit quantised, with a per-block offset.
    pub const Q4_1: &str = "q4_1";
    /// 4-bit quantised.
    pub const Q4_0: &str = "q4_0";
    /// 4-bit non-linear quantised.
    pub const IQ4_NL: &str = "iq4_nl";
    /// Every accepted value, in llama.cpp's own declaration order.
    pub const ALL: &[&str] = &[F32, F16, BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, IQ4_NL];

    /// Whether a cache type stores the cache quantised.
    ///
    /// Guardrail: `f16` and `f32` are the unquantised forms and *everything*
    /// else — `bf16` included — is charged the quantised rate. That rate was
    /// fitted against this exact partition, so narrowing it means refitting
    /// `quantised_cache_rates`, not just editing this line.
    pub fn is_quantised(value: &str) -> bool {
        !value.eq_ignore_ascii_case(F16) && !value.eq_ignore_ascii_case(F32)
    }
}

/// `expert_offload` string values. The field also accepts an integer layer
/// count, which has no fixed string form and so is not listed here.
pub mod expert_offload {
    /// No expert offload; whole-layer CPU spill only.
    pub const OFF: &str = "off";
    /// The packer offloads the minimum experts to fit live VRAM.
    pub const AUTO: &str = "auto";
    /// Every accepted string value, in declaration order.
    pub const ALL: &[&str] = &[OFF, AUTO];
}

/// Render a value vocabulary as a quoted, comma-separated list for
/// operator-facing docs and validation errors, e.g. `"layer", "row",
/// "tensor"`.
pub fn quoted_list(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ")
}
