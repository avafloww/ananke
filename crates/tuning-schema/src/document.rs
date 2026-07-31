//! The document itself, and the scalar constants that make up most of it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Number;

use crate::{
    compute_model::Section,
    tables::{ObservationTable, RateTable, RateTableName, SlotScaling},
};

/// `tuning.json`, whole.
///
/// Fields are declared in the committed file's key order, which is alphabetical,
/// and reordering one rewrites the file. `deny_unknown_fields` so a section this
/// crate stops declaring is a parse error rather than a term silently dropped on
/// the next write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    #[serde(rename = "$comment")]
    pub comment: String,
    pub baseline_offset: RateTable,
    pub checkpoint_headroom_bytes: RateTable,
    pub compute_model: Section,
    pub constants: BTreeMap<String, Constant>,
    /// The measurement dataset the document was derived from.
    pub dataset: String,
    /// The binary that wrote it.
    pub generator: String,
    /// The machine every measurement behind it was taken on.
    pub hardware: String,
    pub ik_moe_rates: RateTable,
    /// How many rows survived de-duplication.
    pub measurements: u64,
    pub mtp_draft_compute_base_mib: RateTable,
    pub mtp_draft_compute_mib_per_1k: RateTable,
    pub mtp_slot_scaling: SlotScaling,
    pub no_flash_attn_rates: RateTable,
    pub no_flash_attn_score_centibytes: RateTable,
    pub per_slot_host_bytes: RateTable,
    pub quantised_cache_rates: RateTable,
    pub table_less_compute_observations: ObservationTable,
    pub tensor_split_baseline: RateTable,
}

/// One scalar constant: the value the estimator compiles in, and its warrant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constant {
    pub doc: String,
    pub evidence: String,
    pub kind: Kind,
    /// The Rust type the generated constant takes.
    #[serde(rename = "type")]
    pub ty: Type,
    /// Kept as a JSON number so an integer does not acquire a decimal point on
    /// the way back out, which is the difference between `4` and `4.0` in the
    /// generated source.
    pub value: Number,
}

/// What justifies a constant, as declared rather than inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Computed from the dataset by a deriver.
    Derived,
    /// A choice — typically a runtime's documented default.
    Policy,
    /// Read from llama.cpp's source, or arithmetic over the graph.
    Structural,
    /// Measured, but chosen so every model stays inside the rolling
    /// correction's clamp rather than to minimise error.
    Reachable,
}

/// The type of a generated constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    U32,
    U64,
    F64,
}

impl Document {
    /// Every per-architecture rate table, in the document's key order.
    ///
    /// The name travels with the table so a reader that has to say which one it
    /// is — the sign check, the generator — does not spell it again.
    pub fn rate_tables(&self) -> [(RateTableName, &RateTable); 10] {
        use RateTableName::{
            BaselineOffset, CheckpointHeadroomBytes, IkMoeRates, MtpDraftComputeBaseMib,
            MtpDraftComputeMibPer1k, NoFlashAttnRates, NoFlashAttnScoreCentibytes,
            PerSlotHostBytes, QuantisedCacheRates, TensorSplitBaseline,
        };
        [
            (BaselineOffset, &self.baseline_offset),
            (CheckpointHeadroomBytes, &self.checkpoint_headroom_bytes),
            (IkMoeRates, &self.ik_moe_rates),
            (MtpDraftComputeBaseMib, &self.mtp_draft_compute_base_mib),
            (MtpDraftComputeMibPer1k, &self.mtp_draft_compute_mib_per_1k),
            (NoFlashAttnRates, &self.no_flash_attn_rates),
            (
                NoFlashAttnScoreCentibytes,
                &self.no_flash_attn_score_centibytes,
            ),
            (PerSlotHostBytes, &self.per_slot_host_bytes),
            (QuantisedCacheRates, &self.quantised_cache_rates),
            (TensorSplitBaseline, &self.tensor_split_baseline),
        ]
    }
}

impl Type {
    /// The type as the generated source spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Type::U32 => "u32",
            Type::U64 => "u64",
            Type::F64 => "f64",
        }
    }

    /// Whether the constant is a float, and so must keep a decimal point.
    pub fn is_float(self) -> bool {
        matches!(self, Type::F64)
    }
}
