//! Leave-one-model-out cross-validation over the derived constants.
//!
//! The campaign's analysis protocol asks for accuracy to be reported this way
//! rather than from the `holdout` question, and for a reason worth restating: the
//! holdout's models are in the fitting set like every other row, so the scoreboard's
//! drift is in-sample and optimistic. Cross-validation costs no extra measurement —
//! it refits what is already on disk — and it is the only generalisation figure this
//! dataset can produce.
//!
//! The question asked of each constant is the one an operator has: *a model this
//! campaign never measured is about to be served — how wrong is the shared constant
//! going to be for it?* So for each model in turn the constant is refitted from
//! every other model's cells, and compared against what that model's own cells say
//! it should have been. The gap between those two is the error the estimator would
//! have made, and it is the number [`Fold::generalisation`] reports.
//!
//! Two limits, both structural, and neither hidden in the output.
//!
//! A deriver that cannot fit without some model has found something more
//! interesting than an error: that constant rests on one model, so it is a
//! model-specific fit wearing an architecture constant's name. `PLAN.md` predicts
//! five of these — the architectures with exactly one model each — and they cannot
//! be cross-validated at all, by construction rather than by omission.
//!
//! And the derivers read the *committed* `tuning.json` for constants they do not
//! themselves derive. A strict fold would refit those too; this does not, so a
//! constant downstream of another carries a little of the full dataset with it. That
//! makes these figures optimistic in a bounded way, which is worth knowing and is
//! still a great deal better than in-sample.

use std::collections::BTreeSet;

use crate::{
    derive::{dataset, emit::derivers, tuning::Tuning},
    record::Record,
};

/// What a fit produced, or why it did not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    Value(i64),
    /// The deriver refused. Carries its reason, because "no cell meets the filters"
    /// and "the cells disagree" are different findings.
    None(String),
}

impl Fit {
    pub fn value(&self) -> Option<i64> {
        match self {
            Fit::Value(v) => Some(*v),
            Fit::None(_) => None,
        }
    }
}

/// One constant, held out against one model.
#[derive(Debug, Clone)]
pub struct Fold {
    pub constant: &'static str,
    /// The model's file name, which is how the dataset labels it.
    pub model: String,
    /// Fitted from every model but this one — what the estimator would have shipped
    /// had this model never been measured.
    pub without: Fit,
    /// Fitted from this model's cells alone — what this model says the constant is.
    pub alone: Fit,
}

impl Fold {
    /// The relative error of predicting this model's constant from the others.
    ///
    /// `None` when either side did not fit, which is not a pass: it means the fold
    /// could not be evaluated, and the caller has to say so rather than count it as
    /// agreement.
    pub fn generalisation(&self) -> Option<f64> {
        let (without, alone) = (self.without.value()?, self.alone.value()?);
        if alone == 0 {
            return None;
        }
        Some(100.0 * (without - alone) as f64 / alone as f64)
    }
}

/// One constant's standing across every fold.
#[derive(Debug, Clone)]
pub struct ConstantReport {
    pub constant: &'static str,
    /// Fitted from the whole dataset — the value that ships.
    pub full: Fit,
    pub folds: Vec<Fold>,
}

impl ConstantReport {
    /// Folds that produced a comparison.
    pub fn evaluated(&self) -> Vec<&Fold> {
        self.folds
            .iter()
            .filter(|f| f.generalisation().is_some())
            .collect()
    }

    /// The largest relative error over the evaluated folds.
    pub fn worst(&self) -> Option<(&Fold, f64)> {
        self.evaluated()
            .into_iter()
            .filter_map(|fold| fold.generalisation().map(|e| (fold, e)))
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
    }

    /// Models whose removal leaves the constant unfittable.
    ///
    /// A constant with any of these does not generalise off the models it was fitted
    /// on — there is no evidence that it would.
    pub fn load_bearing(&self) -> Vec<&str> {
        self.folds
            .iter()
            .filter(|f| f.without.value().is_none())
            .map(|f| f.model.as_str())
            .collect()
    }
}

/// Cross-validate every scalar constant against every model in the dataset.
pub fn cross_validate(rows: &[Record], tuning: &Tuning) -> Vec<ConstantReport> {
    // The same de-duplication `emit` applies, so a fold fits one program rather than
    // a cell and its supersession together.
    let rows = dataset::latest_per_cell(rows);
    let models = models(&rows);

    derivers()
        .into_iter()
        .map(|(constant, derive)| {
            let fit = |subset: &[Record]| match derive(subset, tuning) {
                Ok(scalar) => Fit::Value(scalar.value),
                Err(error) => Fit::None(error.to_string()),
            };
            let folds = models
                .iter()
                .map(|model| Fold {
                    constant,
                    model: model.clone(),
                    without: fit(&excluding(&rows, model)),
                    alone: fit(&only(&rows, model)),
                })
                .collect();
            ConstantReport {
                constant,
                full: fit(&rows),
                folds,
            }
        })
        .collect()
}

/// Every model the dataset measured, by file name.
pub fn models(rows: &[Record]) -> Vec<String> {
    rows.iter()
        .map(|r| model_name(&r.factors.model))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// A model path's file name, which is short enough to tabulate and still unique
/// across the campaign's models.
pub fn model_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}

fn excluding(rows: &[Record], model: &str) -> Vec<Record> {
    rows.iter()
        .filter(|r| model_name(&r.factors.model) != model)
        .cloned()
        .collect()
}

fn only(rows: &[Record], model: &str) -> Vec<Record> {
    rows.iter()
        .filter(|r| model_name(&r.factors.model) == model)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::read_ndjson;

    const MEASUREMENTS: &str = "../../data/measurements.ndjson";
    const TUNING: &str = "../../../crates/tuning/tuning.json";

    fn dataset() -> Vec<Record> {
        let text = std::fs::read_to_string(MEASUREMENTS).expect("the dataset is readable");
        read_ndjson(&text)
            .expect("the dataset parses")
            .into_iter()
            .filter(|r| r.status == "ok")
            .collect()
    }

    fn tuning() -> Tuning {
        Tuning::parse(&std::fs::read_to_string(TUNING).expect("tuning.json is readable"))
            .expect("tuning.json parses")
    }

    /// The real campaign cross-validates, and the folds are real folds.
    #[test]
    fn the_campaign_cross_validates() {
        let reports = cross_validate(&dataset(), &tuning());
        assert!(!reports.is_empty(), "there are constants to validate");
        assert!(
            reports.iter().any(|r| !r.evaluated().is_empty()),
            "at least one constant has a fold with both sides fitted"
        );
    }

    /// A fold's `without` really excludes its model.
    ///
    /// The cheap way to get this wrong is to filter on the whole path against a
    /// short name and match nothing, which leaves every fold fitted on the full
    /// dataset and reports perfect generalisation everywhere.
    #[test]
    fn a_fold_excludes_exactly_its_model() {
        let rows = dataset();
        let model = model_name(&rows[0].factors.model);
        let without = excluding(&rows, &model);
        let alone = only(&rows, &model);

        assert!(!alone.is_empty(), "the model has cells");
        assert!(!without.is_empty(), "and it is not the only model");
        assert_eq!(without.len() + alone.len(), rows.len(), "no row is lost");
        assert!(
            without
                .iter()
                .all(|r| model_name(&r.factors.model) != model),
            "no cell of the held-out model survives the exclusion"
        );
    }

    /// A constant fitted from the whole dataset agrees with what ships.
    ///
    /// The fold values mean nothing if the full fit does not reproduce the committed
    /// constant, since then the cross-validation is describing some other model.
    #[test]
    fn the_full_fit_reproduces_the_committed_constants() {
        let tuning = tuning();
        for report in cross_validate(&dataset(), &tuning) {
            let Some(fitted) = report.full.value() else {
                continue;
            };
            let committed = tuning.constant(report.constant, i64::MIN);
            if committed == i64::MIN {
                continue;
            }
            assert_eq!(
                fitted, committed,
                "{}: cross-validation fits {fitted}, tuning.json ships {committed}",
                report.constant
            );
        }
    }

    /// Generalisation is undefined, not zero, when either side did not fit.
    #[test]
    fn an_unevaluable_fold_has_no_error() {
        let fold = Fold {
            constant: "X",
            model: "m".into(),
            without: Fit::None("no cell meets the filters".into()),
            alone: Fit::Value(10),
        };
        assert_eq!(fold.generalisation(), None);
    }
}
