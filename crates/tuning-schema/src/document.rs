//! The document itself, and the scalar constants that make up most of it.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    compute_model::Section,
    tables::{RateTable, RateTableName, SlotScaling, TableLessObservations},
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
    pub draft_model_compute_mib: RateTable,
    pub draft_model_compute_mib_per_1k: RateTable,
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
    pub table_less_compute_observations: TableLessObservations,
    pub tensor_split_baseline: RateTable,
}

/// One scalar constant: the value the estimator compiles in, and its warrant.
///
/// Not `deny_unknown_fields`, because serde cannot combine that with the
/// flattened value. An unrecognised key is instead caught by `emit --check`,
/// which compares whole documents: a key this drops does not come back out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constant {
    pub doc: String,
    pub evidence: String,
    pub kind: Kind,
    /// The declared type and the number, as one thing. JSON has a single
    /// number type, so the tag chooses between `u32` and `u64` and decides
    /// whether the generated literal carries a decimal point.
    #[serde(flatten)]
    pub value: ConstantValue,
}

/// A constant's value, tagged with the Rust type the generator emits for it.
///
/// Adjacently tagged, which is the `{"type": …, "value": …}` pair the document
/// already spells. A number that is not its declared type fails to parse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum ConstantValue {
    U32(u32),
    U64(u64),
    F64(f64),
}

impl fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
        }
    }
}

/// A derived number that does not fit the type its constant declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoesNotFit {
    pub declared: &'static str,
    pub value: i64,
}

impl fmt::Display for DoesNotFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "derived {} does not fit {}", self.value, self.declared)
    }
}

impl std::error::Error for DoesNotFit {}

impl ConstantValue {
    /// The number as a float, whatever type it declares. Readers that only
    /// need its magnitude — a residual, a rate — do not care which it is.
    pub fn as_f64(self) -> f64 {
        match self {
            Self::U32(v) => f64::from(v),
            Self::U64(v) => v as f64,
            Self::F64(v) => v,
        }
    }

    /// The number, where the declared type is one that holds an integer.
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::U32(v) => Some(i64::from(v)),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::F64(_) => None,
        }
    }

    /// The same declared type, carrying a freshly derived number.
    ///
    /// Keeping the variant is what stops a re-derivation from quietly changing
    /// the type the generator emits, and range-checks the number against it.
    pub fn with_derived(self, value: i64) -> Result<Self, DoesNotFit> {
        let fits = |declared| DoesNotFit { declared, value };
        match self {
            Self::U32(_) => u32::try_from(value).map(Self::U32).map_err(|_| fits("u32")),
            Self::U64(_) => u64::try_from(value).map(Self::U64).map_err(|_| fits("u64")),
            Self::F64(_) => Ok(Self::F64(value as f64)),
        }
    }

    /// The Rust type the generated constant takes.
    pub fn type_name(self) -> &'static str {
        match self {
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::F64(_) => "f64",
        }
    }
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

impl Document {
    /// Every per-architecture rate table, in the document's key order.
    ///
    /// The name travels with the table so a reader that has to say which one it
    /// is — the sign check, the generator — does not spell it again.
    pub fn rate_tables(&self) -> [(RateTableName, &RateTable); 12] {
        use RateTableName::{
            BaselineOffset, CheckpointHeadroomBytes, DraftModelComputeMib,
            DraftModelComputeMibPer1k, IkMoeRates, MtpDraftComputeBaseMib, MtpDraftComputeMibPer1k,
            NoFlashAttnRates, NoFlashAttnScoreCentibytes, PerSlotHostBytes, QuantisedCacheRates,
            TensorSplitBaseline,
        };
        [
            (BaselineOffset, &self.baseline_offset),
            (CheckpointHeadroomBytes, &self.checkpoint_headroom_bytes),
            (DraftModelComputeMib, &self.draft_model_compute_mib),
            (
                DraftModelComputeMibPer1k,
                &self.draft_model_compute_mib_per_1k,
            ),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed document round-trips byte for byte.
    ///
    /// Two things at once. Every constant's declared type agrees with the
    /// number beside it, or parsing fails. And a key this schema does not model
    /// would be dropped on the way through and show up as a difference here —
    /// which is the guarantee `deny_unknown_fields` would give, except that
    /// serde cannot combine it with the flattened value.
    #[test]
    fn the_committed_document_round_trips() {
        let text = include_str!("../../tuning/tuning.json");
        let document: Document = serde_json::from_str(text).expect("the document parses");
        assert!(!document.constants.is_empty(), "the document has constants");
        let out = serde_json::to_string_pretty(&document).expect("serialises") + "\n";
        assert_eq!(out, text, "the document did not survive a round trip");
    }

    /// A number that is not its declared type fails to parse, rather than
    /// reaching the generator and producing Rust that will not compile.
    #[test]
    fn a_number_that_is_not_its_declared_type_is_refused() {
        let entry = r#"{"doc":"d","evidence":"e","kind":"derived","type":"u64","value":4.5}"#;
        let error = serde_json::from_str::<Constant>(entry).expect_err("4.5 is not a u64");
        assert!(error.to_string().contains("u64"), "{error}");
    }
}
