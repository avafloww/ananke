//! The filesystem, as the harness uses it.
//!
//! Three primitives rather than a general filesystem: the dataset is read whole,
//! appended a line at a time, and rewritten whole, and the archived logs are read
//! and written whole. Nothing here seeks, streams, or lists a directory, so a fake
//! is a map from path to bytes and the phases above it become testable without a
//! scratch directory.
//!
//! Reads answer `None` for a missing file rather than an error. Every caller treats
//! absence as a legitimate state — a campaign's first cell has no dataset yet, and a
//! record measured before the logs were archived has no log — and distinguishing it
//! from a permission error would give them a case none of them have anything to do
//! with.

use std::path::Path;

pub trait Files: Send + Sync {
    /// The file's bytes, or `None` if it is not there.
    fn read(&self, path: &Path) -> Option<Vec<u8>>;

    /// Replace the file, creating parent directories as needed.
    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;

    /// Add to the end of the file, creating it and its parents as needed.
    ///
    /// Separate from `write` because it is what makes a campaign survive being
    /// killed: the dataset is only ever extended, so a run that dies between cells
    /// leaves every completed row intact.
    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;
}

pub struct LocalFiles;

impl Files for LocalFiles {
    fn read(&self, path: &Path) -> Option<Vec<u8>> {
        std::fs::read(path).ok()
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        create_parents(path)?;
        std::fs::write(path, bytes)
    }

    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        use std::io::Write;

        create_parents(path)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(bytes)
    }
}

fn create_parents(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => std::fs::create_dir_all(parent),
        _ => Ok(()),
    }
}

#[cfg(any(test, feature = "test-fakes"))]
pub use fake::FakeFiles;

#[cfg(any(test, feature = "test-fakes"))]
mod fake {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    use parking_lot::Mutex;

    use crate::harness::sys::files::Files;

    /// An in-memory filesystem. Directories are implicit: a path is a key.
    #[derive(Default)]
    pub struct FakeFiles {
        contents: Mutex<BTreeMap<PathBuf, Vec<u8>>>,
    }

    impl FakeFiles {
        pub fn new() -> Self {
            Self::default()
        }

        /// Start with a file already in place.
        pub fn with_file(self, path: impl Into<PathBuf>, contents: impl AsRef<[u8]>) -> Self {
            self.contents
                .lock()
                .insert(path.into(), contents.as_ref().to_vec());
            self
        }

        /// What a path holds now, as text.
        pub fn text(&self, path: impl AsRef<Path>) -> Option<String> {
            self.contents
                .lock()
                .get(path.as_ref())
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        }

        /// The non-empty lines of a path, which is how the dataset is read.
        pub fn lines(&self, path: impl AsRef<Path>) -> Vec<String> {
            self.text(path)
                .unwrap_or_default()
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_owned)
                .collect()
        }

        /// Every path written to, in order.
        pub fn paths(&self) -> Vec<PathBuf> {
            self.contents.lock().keys().cloned().collect()
        }
    }

    impl Files for FakeFiles {
        fn read(&self, path: &Path) -> Option<Vec<u8>> {
            self.contents.lock().get(path).cloned()
        }

        fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
            self.contents.lock().insert(path.to_owned(), bytes.to_vec());
            Ok(())
        }

        fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
            self.contents
                .lock()
                .entry(path.to_owned())
                .or_default()
                .extend_from_slice(bytes);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Appending to a file that is not there creates it, which is the campaign's
    /// first cell.
    #[test]
    fn appending_to_nothing_starts_a_file() {
        let files = FakeFiles::new();
        files
            .append(Path::new("data/out.ndjson"), b"one\n")
            .unwrap();
        files
            .append(Path::new("data/out.ndjson"), b"two\n")
            .unwrap();
        assert_eq!(files.lines("data/out.ndjson"), ["one", "two"]);
    }

    /// A missing file reads as absent, not as an error.
    #[test]
    fn a_missing_file_is_none() {
        assert_eq!(FakeFiles::new().read(Path::new("nowhere")), None);
    }

    /// `write` replaces; `append` does not.
    #[test]
    fn writing_replaces_what_appending_extends() {
        let files = FakeFiles::new().with_file("a", "first\n");
        files.append(Path::new("a"), b"second\n").unwrap();
        assert_eq!(files.text("a").as_deref(), Some("first\nsecond\n"));
        files.write(Path::new("a"), b"third\n").unwrap();
        assert_eq!(files.text("a").as_deref(), Some("third\n"));
    }
}
