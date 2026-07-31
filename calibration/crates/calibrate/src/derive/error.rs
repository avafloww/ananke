//! Why a derivation stops.
//!
//! Two cases: "no cell the deriver can use" and "the cells do not agree".
//! `emit` reports the constant as not derived either way, so the distinction is
//! informational rather than structural — but it is the interesting half of the
//! message, so it is kept as a kind rather than folded into the text.

use std::fmt;

/// What kind of thing went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// No cell in the dataset meets the deriver's filters, so there is nothing
    /// to reduce. Usually means a campaign has not run the sweep yet.
    NoData,
    /// The cells behind a constant do not agree, so no single value fits them.
    ///
    /// This is not defensive. Ten conclusions in the campaign were drawn from a
    /// median that described none of its members, and each produced a plausible
    /// law; a spread wider than the tolerance is treated as a failure to have
    /// grouped properly rather than as noise to average over.
    Disagreement,
    /// The dataset itself is unreadable or malformed.
    Malformed,
}

/// A derivation that could not produce a value.
#[derive(Debug, Clone)]
pub struct DeriveError {
    kind: ErrorKind,
    message: String,
}

impl DeriveError {
    pub fn no_data(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::NoData,
            message: message.into(),
        }
    }

    pub fn disagreement(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Disagreement,
            message: message.into(),
        }
    }

    pub fn malformed(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Malformed,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for DeriveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DeriveError {}

pub type Result<T> = std::result::Result<T, DeriveError>;
