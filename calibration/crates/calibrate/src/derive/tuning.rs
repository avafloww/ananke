//! Reading the constants the estimator already compiles in.
//!
//! Several derivers need a constant they do not derive. Those were copied by
//! hand once, and the copy went stale the moment the constant was re-derived,
//! inflating every residual computed for an ik mixture of experts until someone
//! noticed. `tuning.json` is read instead.
//!
//! Absence is deliberately not papered over with one defaulting reader: a
//! renamed or misspelled constant read as `0.0` does not stop a derivation, it
//! just fits every residual against the wrong model.

use std::{fmt, marker::PhantomData};

use ananke_tuning_schema::{Document, RateTable};
use serde_json::Number;

use crate::derive::{
    error::DeriveError,
    keys::{ArchCardsKey, ArchKey},
};

/// The committed tuning document, as the derivers see it.
#[derive(Debug, Clone)]
pub struct Tuning {
    document: Document,
}

/// A constant the document does not declare. Its own type rather than a
/// [`DeriveError`], because the fault is in `tuning.json` rather than in the
/// dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownConstant {
    name: String,
}

impl UnknownConstant {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for UnknownConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: no such constant in tuning.json", self.name)
    }
}

impl std::error::Error for UnknownConstant {}

impl From<UnknownConstant> for DeriveError {
    fn from(error: UnknownConstant) -> Self {
        DeriveError::malformed(error.to_string())
    }
}

impl Tuning {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        Ok(Self {
            document: serde_json::from_str(text)?,
        })
    }

    /// A view over a document being rebuilt, so a deriver reads the value this
    /// run produced rather than the one the last run committed.
    pub fn of(document: &Document) -> Self {
        Self {
            document: document.clone(),
        }
    }

    /// Replace one constant's value, for [`Self::of`]'s caller to thread the
    /// freshly derived figure through to the derivers that consume it.
    ///
    /// Errors on an undeclared name rather than doing nothing: a silent no-op
    /// leaves every downstream residual fitted against the previous run's value
    /// with nothing to say so.
    pub fn set_constant(&mut self, name: &str, value: i64) -> Result<(), UnknownConstant> {
        let entry = self
            .document
            .constants
            .get_mut(name)
            .ok_or_else(|| UnknownConstant {
                name: name.to_string(),
            })?;
        entry.value = Number::from(value);
        Ok(())
    }

    /// Replace the ik MoE rate table, as [`Self::set_constant`] does for a
    /// scalar. The one table a deriver reads back within a run.
    pub fn set_ik_moe_rates(&mut self, table: RateTable) {
        self.document.ik_moe_rates = table;
    }

    pub fn constant(&self, name: &str) -> Option<i64> {
        self.constant_f64(name).map(|value| value as i64)
    }

    pub fn constant_f64(&self, name: &str) -> Option<f64> {
        self.document
            .constants
            .get(name)
            .and_then(|entry| entry.value.as_f64())
    }

    /// [`Self::constant_f64`] for the reader whose model is wrong without the
    /// value, so the miss stops the derivation instead of zeroing a term of it.
    pub fn required_f64(&self, name: &str) -> Result<f64, UnknownConstant> {
        self.constant_f64(name).ok_or_else(|| UnknownConstant {
            name: name.to_string(),
        })
    }

    /// The bootstrap reader, and the only honest use of a default: a term being
    /// added for the first time has no entry to read. Anything that expects the
    /// constant to be there wants [`Self::constant`] or [`Self::required_f64`].
    pub fn constant_or(&self, name: &str, default: i64) -> i64 {
        self.constant(name).unwrap_or(default)
    }

    /// ik's MoE rate table, keyed `{arch}@{cards}` as
    /// [`crate::derive::graph::ik_moe_per_nembd`] writes it.
    pub fn ik_moe_rates(&self) -> RateTableView<'_, ArchCardsKey> {
        RateTableView::new(&self.document.ik_moe_rates)
    }

    /// The ik MoE rate the arena model charges, looked up by architecture alone
    /// against a table no row of which is keyed that way — so every call misses
    /// and takes `default`.
    ///
    /// Frozen, not overlooked. Every constant in the document was fitted against
    /// an arena that charged the default here, so resolving the lookup properly
    /// would move the model out from under all of them. The miss is spelled out
    /// by [`ArchCardsKey::without_cards_frozen_miss`] rather than left to a
    /// vocabulary mismatch that reads like a working lookup.
    pub fn ik_moe_rate_frozen_arch_miss(&self, arch: &ArchKey) -> i64 {
        self.ik_moe_rates()
            .rate(&ArchCardsKey::without_cards_frozen_miss(arch))
    }

    /// mainline's host-resident MoE rate under a tensor split, per unit of
    /// hidden size. The arena model charges it and derives it too, so the value
    /// read here is the previous run's. Defaulting because its one caller,
    /// [`crate::derive::arena::arena_terms`], has no failure path to return.
    pub fn mainline_tensor_moe_per_nembd(&self) -> i64 {
        self.constant_or(
            "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD",
            MAINLINE_TENSOR_MOE_DEFAULT,
        )
    }
}

/// One committed rate table, read at the vocabulary its rows are keyed by.
///
/// The document does not distinguish a table keyed by architecture from one
/// keyed `{arch}@{cards}`, and a lookup at the wrong vocabulary quietly takes
/// `default`. `K` is the reader stating which it is, so the mismatch has to be
/// written on purpose.
pub struct RateTableView<'a, K> {
    table: &'a RateTable,
    key: PhantomData<fn(&K)>,
}

impl<'a, K: fmt::Display> RateTableView<'a, K> {
    /// The rate for one key, or the table's `default`.
    pub fn rate(&self, key: &K) -> i64 {
        self.table.rate(&key.to_string())
    }

    /// What an unlisted key inherits.
    pub fn default(&self) -> i64 {
        self.table.default
    }

    fn new(table: &'a RateTable) -> Self {
        Self {
            table,
            key: PhantomData,
        }
    }
}

/// mainline's per-`n_embd` tensor-split MoE rate to fall back on before the
/// constant exists.
const MAINLINE_TENSOR_MOE_DEFAULT: i64 = 57;

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed document. [`Tuning`] parses the whole of it, so a
    /// hand-written fragment would be a second and laxer schema.
    fn document() -> Tuning {
        Tuning::parse(include_str!("../../../../../crates/tuning/tuning.json"))
            .expect("the committed document parses")
    }

    /// The defect this file exists to close: a name the document does not declare
    /// used to be written nowhere and reported nowhere, so the ordering between
    /// derivers dissolved into every downstream fit reading a stale value.
    #[test]
    fn setting_an_unknown_constant_is_reported() {
        let mut tuning = document();
        let error = tuning
            .set_constant("MISSPELLED", 1)
            .expect_err("an undeclared name is an error, not a no-op");
        assert_eq!(error.name(), "MISSPELLED");
        assert!(
            error.to_string().contains("MISSPELLED"),
            "the report names the constant, since `emit` joins these into one line"
        );
        assert_eq!(
            tuning.constant("MISSPELLED"),
            None,
            "and nothing was written"
        );
    }

    #[test]
    fn setting_a_declared_constant_writes_it() {
        let mut tuning = document();
        tuning
            .set_constant("KV_CACHE_PAD", 9)
            .expect("the name is declared");
        assert_eq!(tuning.constant("KV_CACHE_PAD"), Some(9));
        assert_eq!(tuning.required_f64("KV_CACHE_PAD"), Ok(9.0));
    }

    #[test]
    fn a_missing_constant_reads_as_absent_rather_than_zero() {
        let tuning = document();
        assert_eq!(tuning.constant_f64("ABSENT"), None);
        assert!(tuning.required_f64("ABSENT").is_err());
        assert_eq!(
            tuning.constant_or("ABSENT", 3),
            3,
            "the bootstrap reader defaults"
        );
    }
}
