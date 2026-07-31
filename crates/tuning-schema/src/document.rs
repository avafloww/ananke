//! The document itself, and the scalar constants that make up most of it.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};
use serde_json::Number;

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
    /// generated source. Read it through [`Constant::typed`], which is the only
    /// place the declared type and the written number are checked against each
    /// other.
    pub value: Number,
}

/// A constant's value, as the type its entry declares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstantValue {
    U32(u32),
    U64(u64),
    F64(f64),
}

/// A declared type and a written number that do not agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueMismatch {
    pub declared: Type,
    pub written: String,
}

impl fmt::Display for ValueMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "read {} as {}", self.written, self.declared.as_str())
    }
}

impl std::error::Error for ValueMismatch {}

impl Constant {
    /// The value as the type the entry declares, or the disagreement.
    ///
    /// JSON has one number type, so `type` is not a restatement of `value`: it
    /// chooses between `u32` and `u64`, and decides whether the generated
    /// literal carries a decimal point. Nothing but this checks that the two
    /// describe the same number.
    pub fn typed(&self) -> Result<ConstantValue, ValueMismatch> {
        let mismatch = || ValueMismatch {
            declared: self.ty,
            written: self.value.to_string(),
        };
        match self.ty {
            Type::U32 => self
                .value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .map(ConstantValue::U32)
                .ok_or_else(mismatch),
            Type::U64 => self
                .value
                .as_u64()
                .map(ConstantValue::U64)
                .ok_or_else(mismatch),
            Type::F64 => self
                .value
                .as_f64()
                .map(ConstantValue::F64)
                .ok_or_else(mismatch),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every committed constant's declared type agrees with the number written
    /// beside it. A disagreement generates Rust that does not compile, which is
    /// a worse place to find out than here.
    #[test]
    fn every_committed_constant_reads_as_the_type_it_declares() {
        let text = include_str!("../../tuning/tuning.json");
        let document: Document = serde_json::from_str(text).expect("the document parses");
        for (name, constant) in &document.constants {
            if let Err(error) = constant.typed() {
                panic!("{name}: {error}");
            }
        }
        assert!(!document.constants.is_empty(), "the document has constants");
    }

    /// A float declared as an integer is caught, rather than truncating.
    #[test]
    fn a_number_that_is_not_its_declared_type_is_refused() {
        let constant = Constant {
            doc: String::new(),
            evidence: String::new(),
            kind: Kind::Derived,
            ty: Type::U64,
            value: serde_json::Number::from_f64(4.5).expect("finite"),
        };
        assert!(constant.typed().is_err());
    }
}
