//! The tables: one rate per architecture, plus the two sections that record
//! observations rather than rates.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The shape nearly every table takes: a rate per key, and what an unlisted key
/// inherits.
///
/// Nothing in the document says which vocabulary the keys are drawn from —
/// architecture, architecture-and-variant, architecture-and-cards — and a lookup
/// at the wrong one misses silently. The deriving side names that vocabulary in
/// a type; here the keys are the strings the file spells.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateTable {
    #[serde(rename = "$comment")]
    pub comment: String,
    pub by_arch: BTreeMap<String, i64>,
    pub default: i64,
}

impl RateTable {
    /// The rate for one key, or the table's default.
    pub fn rate(&self, key: &str) -> i64 {
        self.by_arch.get(key).copied().unwrap_or(self.default)
    }
}

/// Which rate table one is, so a reader that must say so does not spell the
/// section name a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateTableName {
    BaselineOffset,
    CheckpointHeadroomBytes,
    DraftModelComputeMib,
    DraftModelComputeMibPer1k,
    IkMoeRates,
    MtpDraftComputeBaseMib,
    MtpDraftComputeMibPer1k,
    NoFlashAttnRates,
    NoFlashAttnScoreCentibytes,
    PerSlotHostBytes,
    QuantisedCacheRates,
    TensorSplitBaseline,
}

impl RateTableName {
    /// The key the document spells it under.
    pub fn as_str(self) -> &'static str {
        match self {
            RateTableName::BaselineOffset => "baseline_offset",
            RateTableName::CheckpointHeadroomBytes => "checkpoint_headroom_bytes",
            RateTableName::DraftModelComputeMib => "draft_model_compute_mib",
            RateTableName::DraftModelComputeMibPer1k => "draft_model_compute_mib_per_1k",
            RateTableName::IkMoeRates => "ik_moe_rates",
            RateTableName::MtpDraftComputeBaseMib => "mtp_draft_compute_base_mib",
            RateTableName::MtpDraftComputeMibPer1k => "mtp_draft_compute_mib_per_1k",
            RateTableName::NoFlashAttnRates => "no_flash_attn_rates",
            RateTableName::NoFlashAttnScoreCentibytes => "no_flash_attn_score_centibytes",
            RateTableName::PerSlotHostBytes => "per_slot_host_bytes",
            RateTableName::QuantisedCacheRates => "quantised_cache_rates",
            RateTableName::TensorSplitBaseline => "tensor_split_baseline",
        }
    }

    /// Whether the table's values may be negative.
    ///
    /// The generator emits an unsigned table through `as u64`, which turns a
    /// negative into the table's default — the value vanishes with no error
    /// anywhere. So the two sides read this rather than each carrying a list:
    /// the generator picks its emitter by it, and the emitter refuses to write a
    /// negative into a table it says is unsigned.
    pub fn signed(self) -> bool {
        matches!(self, RateTableName::BaselineOffset)
    }
}

/// `mtp_slot_scaling`, which reports what was observed rather than a fitted rate
/// and so has no per-architecture breakdown to fall back from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotScaling {
    #[serde(rename = "$comment")]
    pub comment: String,
    pub observed: String,
}

/// Observations keyed by architecture and then by the configuration each was
/// taken at. No `default`: nothing reads these, they are written down so a
/// hand-held value's justification cannot go stale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableLessObservations {
    #[serde(rename = "$comment")]
    pub comment: String,
    pub by_arch: BTreeMap<String, BTreeMap<String, i64>>,
}
