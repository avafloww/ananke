//! Reading and extending the NDJSON the campaign accumulates.
//!
//! Self-describing per line, so a run that adds a factor or a parsed field
//! appends happily beside older records instead of needing a schema migration —
//! and an analysis reads what each record actually carries.
//!
//! Every write goes through [`to_python_json`], which reproduces the Python
//! harness's `json.dumps` defaults. That is not nostalgia: the file is checked in
//! as the campaign's oracle, and a line whose spacing differs from its neighbours
//! is a diff that says nothing.

use std::{
    collections::BTreeSet,
    io::{Read, Write},
    path::Path,
};

use crate::{
    harness::{
        error::{Error, ErrorKind},
        json::to_python_json,
    },
    record::Record,
};

pub(crate) fn read_lines(path: &Path) -> Result<Vec<String>, Error> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        Error::new(
            ErrorKind::Dataset,
            format!("read {}: {error}", path.display()),
        )
    })?;
    Ok(text
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
pub(crate) fn already_measured(path: &Path) -> Result<BTreeSet<String>, Error> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let mut measured = BTreeSet::new();
    for line in read_lines(path)? {
        let record: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| Error::new(ErrorKind::Dataset, format!("read a record: {error}")))?;
        if record["status"] == serde_json::json!("ok")
            && let Some(cell) = record["cell"].as_str()
        {
            measured.insert(cell.to_owned());
        }
    }
    Ok(measured)
}

pub(crate) fn append(path: &Path, record: &Record) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::io)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(Error::io)?;
    writeln!(file, "{}", to_python_json(record)).map_err(Error::io)
}

/// Rewrite the whole file. Used by the maintenance passes, which have already
/// decided that every line they are not changing stays byte for byte as it was.
pub(crate) fn write_lines(path: &Path, lines: &[String]) -> Result<(), Error> {
    let mut text = lines.join("\n");
    text.push('\n');
    std::fs::write(path, text).map_err(Error::io)
}

/// Keep the load log alongside the record, compressed.
///
/// The parsers read a handful of kinds of line. Everything else the loader prints
/// — and everything a future question turns out to need — is only recoverable if
/// the log itself survives, and a log left in a temporary directory does not. They
/// compress to tens of KiB, which is a small price for making a record
/// re-parseable rather than merely re-readable.
pub(crate) fn archive_log(log_path: &Path, archive_dir: &Path) -> String {
    let Some(stem) = log_path.file_stem() else {
        return String::new();
    };
    let name = format!("{}.log.gz", stem.to_string_lossy());
    if std::fs::create_dir_all(archive_dir).is_err() {
        return String::new();
    }
    let Ok(mut source) = std::fs::File::open(log_path) else {
        return String::new();
    };
    let Ok(target) = std::fs::File::create(archive_dir.join(&name)) else {
        return String::new();
    };
    let mut encoder = flate2::write::GzEncoder::new(target, flate2::Compression::default());
    if std::io::copy(&mut source, &mut encoder).is_err() || encoder.finish().is_err() {
        return String::new();
    }
    name
}

/// Read one archived log back. The originals were decoded with replacement, so a
/// stray non-UTF-8 byte stays a replacement character here too rather than
/// failing the read.
pub(crate) fn read_archived_log(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(file)
        .read_to_end(&mut bytes)
        .ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}
