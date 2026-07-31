//! Reading and extending the NDJSON the campaign accumulates.
//!
//! Self-describing per line, so a run that adds a factor or a parsed field
//! appends happily beside older records instead of needing a schema migration —
//! and an analysis reads what each record actually carries.
//!
//! Every write goes through [`to_dataset_json`], which reproduces the spacing and
//! escaping of the lines already there. The file is checked in as the campaign's
//! oracle, and a line whose spacing differs from its neighbours is a diff that
//! says nothing.
//!
//! The filesystem arrives as a [`Files`] rather than being reached for directly,
//! for the reason every other capability here does: the loop above this one appends
//! a row per cell and notifies its driver, and that contract cannot be checked
//! against a scratch directory without also being a test of the disk.

use std::{collections::BTreeSet, io::Read, path::Path};

use serde::Deserialize;

use crate::{
    harness::{
        error::{Error, ErrorKind},
        sys::Files,
        to_dataset_json,
    },
    record::{Record, Status},
};

pub(crate) fn read_lines(files: &dyn Files, path: &Path) -> Result<Vec<String>, Error> {
    let bytes = files.read(path).ok_or_else(|| {
        Error::new(
            ErrorKind::Dataset,
            format!("read {}: no such file", path.display()),
        )
    })?;
    Ok(String::from_utf8_lossy(&bytes)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect())
}

/// The cells that no longer need measuring.
///
/// Only a completed measurement ends a cell. A cell skipped because the box was
/// momentarily full is not measured, and a rerun with memory free must retry it
/// rather than inherit the skip forever; load failures are likewise worth
/// retrying.
pub(crate) fn already_measured(files: &dyn Files, path: &Path) -> Result<BTreeSet<String>, Error> {
    if files.read(path).is_none() {
        return Ok(BTreeSet::new());
    }
    let mut measured = BTreeSet::new();
    for line in read_lines(files, path)? {
        let probe: MembershipProbe = serde_json::from_str(&line)
            .map_err(|error| Error::new(ErrorKind::Dataset, format!("read a record: {error}")))?;
        if probe.status == Status::Ok {
            measured.insert(probe.cell);
        }
    }
    Ok(measured)
}

/// Just enough of [`Record`] to answer "does this cell still need measuring".
///
/// `Record` itself only derives `Serialize` — the writer side owns the exact
/// spacing via [`to_dataset_json`] — so this is its own small `Deserialize`
/// shape rather than a cast through `serde_json::Value`, which lets a status
/// spelling drift or a non-object row compare `Null != "ok"` and silently pass
/// as "not yet measured", costing a re-measurement of a cell that has already
/// used the GPU time. `status` uses the crate's own [`Status`] enum so a typo
/// or a schema change fails to deserialize instead of comparing unequal.
#[derive(Deserialize)]
struct MembershipProbe {
    cell: String,
    status: Status,
}

pub(crate) fn append(files: &dyn Files, path: &Path, record: &Record) -> Result<(), Error> {
    let line = format!("{}\n", to_dataset_json(record));
    files.append(path, line.as_bytes()).map_err(Error::io)
}

/// Rewrite the whole file. Used by the maintenance passes, which have already
/// decided that every line they are not changing stays byte for byte as it was.
pub(crate) fn write_lines(files: &dyn Files, path: &Path, lines: &[String]) -> Result<(), Error> {
    let mut text = lines.join("\n");
    text.push('\n');
    files.write(path, text.as_bytes()).map_err(Error::io)
}

/// Keep the load log alongside the record, compressed.
///
/// The parsers read a handful of kinds of line. Everything else the loader prints
/// — and everything a future question turns out to need — is only recoverable if
/// the log itself survives, and a log left in a temporary directory does not. They
/// compress to tens of KiB, which is a small price for making a record
/// re-parseable rather than merely re-readable.
///
/// Returns `Err` rather than an empty name on any failure: an empty
/// `Record::log` is indistinguishable from archiving being disabled, and this
/// is the field that makes a record re-parseable later. An archive directory
/// that silently never receives a file turns every subsequent `ok` row into
/// one that can never be re-derived, with only a skip count as the symptom.
pub(crate) fn archive_log(
    files: &dyn Files,
    log_path: &Path,
    archive_dir: &Path,
) -> Result<String, Error> {
    let stem = log_path.file_stem().ok_or_else(|| {
        Error::new(
            ErrorKind::Dataset,
            format!("archive {}: no file stem", log_path.display()),
        )
    })?;
    let name = format!("{}.log.gz", stem.to_string_lossy());
    let source = files.read(log_path).ok_or_else(|| {
        Error::new(
            ErrorKind::Dataset,
            format!("archive {}: source log is unreadable", log_path.display()),
        )
    })?;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &source).map_err(|error| {
        Error::new(
            ErrorKind::Dataset,
            format!("compress {}: {error}", log_path.display()),
        )
    })?;
    let compressed = encoder.finish().map_err(|error| {
        Error::new(
            ErrorKind::Dataset,
            format!("compress {}: {error}", log_path.display()),
        )
    })?;
    files
        .write(&archive_dir.join(&name), &compressed)
        .map_err(Error::io)?;
    Ok(name)
}

/// Read one archived log back. Decoded with replacement, so a stray non-UTF-8
/// byte stays a replacement character rather than failing the read.
pub(crate) fn read_archived_log(files: &dyn Files, path: &Path) -> Option<String> {
    let compressed = files.read(path)?;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut bytes)
        .ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}
