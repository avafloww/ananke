//! The `compute_model` section: one coefficient set per fitted group.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The `compute_model` section as `tuning.json` carries it. Named fields and
/// strict on both sides, so a key this crate stops writing — or one a hand-edit
/// adds — is a parse error rather than a silently dropped term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    #[serde(rename = "$comment")]
    pub comment: String,
    /// The design columns, in the order the coefficients are declared against.
    pub columns: Vec<String>,
    /// What an architecture nobody has measured falls back to.
    pub default: Fit,
    pub entries: Vec<Entry>,
}

/// One fitted (runtime, split, architecture) entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub archs: Vec<String>,
    pub coefficients: BTreeMap<String, f64>,
    pub evidence: String,
    /// `None` for mainline, so the common case reads as an absent guard rather
    /// than as a runtime the estimator has to match by name.
    pub runtime: Option<String>,
    pub split: String,
    pub variant: Option<String>,
}

/// A fit on its own, without the group it applies to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fit {
    pub coefficients: BTreeMap<String, f64>,
    pub evidence: String,
}
