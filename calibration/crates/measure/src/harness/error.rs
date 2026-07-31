//! The harness's one error type.
//!
//! Everything a run can fail *at* is an operator-facing condition — a plan that
//! does not read, a dataset that is not there, a log directory that cannot be
//! created — so the messages are lowercase sentence fragments that read after
//! "failed to". A cell that fails to *measure* is not an error at all: it is a
//! record with a status, because the campaign must survive it.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    /// The plan does not describe cells this harness can run.
    Plan,
    /// The measurements file, or an archived log beside it.
    Dataset,
    /// Anything the filesystem refused.
    Io,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    detail: String,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub(crate) fn io(detail: impl fmt::Display) -> Self {
        Self::new(ErrorKind::Io, detail.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let subject = match self.kind {
            ErrorKind::Plan => "plan",
            ErrorKind::Dataset => "dataset",
            ErrorKind::Io => "filesystem",
        };
        write!(formatter, "{subject}: {}", self.detail)
    }
}

impl std::error::Error for Error {}
