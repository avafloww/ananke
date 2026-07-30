//! Committing the dataset as it is measured.
//!
//! Behind a trait for the usual reason: the campaign's *decisions* — when to
//! commit, what to say, and what to do when the commit fails — are worth testing,
//! and none of them should need a repository.
//!
//! Every operation is scoped to an explicit path list. A bare `git commit` takes
//! everything staged, which would sweep an operator's half-staged work into a data
//! commit if they happened to be staging while an overnight campaign ran. The
//! Python was careful about this and the care is worth keeping.

use std::{path::PathBuf, process::Command};

/// The version control the campaign commits through.
pub trait Vcs {
    /// Stage exactly these paths.
    fn stage(&self, paths: &[PathBuf]) -> Result<(), String>;

    /// Whether any of these paths has staged content.
    fn has_staged(&self, paths: &[PathBuf]) -> bool;

    /// Commit exactly these paths with this message.
    fn commit(&self, message: &str, paths: &[PathBuf]) -> Result<(), String>;
}

/// Whatever `git` is on the path, run from the repository root.
pub struct LocalGit {
    root: PathBuf,
}

impl LocalGit {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn git(&self, args: &[&str], paths: &[PathBuf]) -> Result<std::process::Output, String> {
        // `git commit -m msg --` with nothing after the separator is not a no-op: it
        // is a bare commit, which takes the whole index. An empty list reaching here
        // is the one input that turns this module into the hazard it exists to
        // prevent, so it fails rather than running.
        if paths.is_empty() {
            return Err(format!(
                "refusing to run `git {}` against no paths: an empty pathspec commits \
                 the whole index",
                args.join(" ")
            ));
        }
        let mut command = Command::new("git");
        command.current_dir(&self.root).args(args).arg("--");
        for path in paths {
            command.arg(path);
        }
        command
            .output()
            .map_err(|e| format!("running git {}: {e}", args.join(" ")))
    }
}

impl Vcs for LocalGit {
    fn stage(&self, paths: &[PathBuf]) -> Result<(), String> {
        let output = self.git(&["add"], paths)?;
        if output.status.success() {
            return Ok(());
        }
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }

    fn has_staged(&self, paths: &[PathBuf]) -> bool {
        self.git(&["diff", "--cached", "--quiet"], paths)
            .is_ok_and(|output| !output.status.success())
    }

    fn commit(&self, message: &str, paths: &[PathBuf]) -> Result<(), String> {
        let output = self.git(&["commit", "-m", message], paths)?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(stderr
            .lines()
            .next()
            .unwrap_or("git commit failed")
            .to_string())
    }
}

/// Commit the measured data, tolerating both nothing-to-do and failure.
///
/// A failed commit is reported and swallowed. Signing can fail if the key has gone
/// away, and losing the remainder of an overnight run to that would be worse than
/// an uncommitted dataset: the rows are on disk either way, and the next commit
/// picks them up.
pub fn commit_data(vcs: &dyn Vcs, paths: &[PathBuf], message: &str) -> Outcome {
    if paths.is_empty() {
        return Outcome::Failed("no paths to commit".to_string());
    }
    if let Err(error) = vcs.stage(paths) {
        return Outcome::Failed(error);
    }
    if !vcs.has_staged(paths) {
        return Outcome::NothingToDo;
    }
    match vcs.commit(message, paths) {
        Ok(()) => Outcome::Committed,
        Err(error) => Outcome::Failed(error),
    }
}

/// What `commit_data` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Committed,
    /// Nothing new was measured since the last commit.
    NothingToDo,
    /// The data is still on disk; the campaign continues.
    Failed(String),
}

#[cfg(any(test, feature = "test-fakes"))]
pub use fake::FakeGit;

#[cfg(any(test, feature = "test-fakes"))]
mod fake {
    use std::path::PathBuf;

    use parking_lot::Mutex;

    use crate::campaign::git::Vcs;

    /// Records what it was asked to do, and can be told to fail.
    #[derive(Default)]
    pub struct FakeGit {
        state: Mutex<State>,
    }

    #[derive(Default)]
    struct State {
        staged: bool,
        commits: Vec<String>,
        paths: Vec<Vec<PathBuf>>,
        fail_commit: Option<String>,
    }

    impl FakeGit {
        /// A repository in which the given paths have something to commit.
        pub fn dirty() -> Self {
            let fake = Self::default();
            fake.state.lock().staged = true;
            fake
        }

        /// Something new landed on disk, as appending a cell's row does.
        ///
        /// A commit clears the staged flag, so without this a driver test would see
        /// `NothingToDo` for every commit after the first — the fake would be
        /// modelling a campaign that measures nothing.
        pub fn touch(&self) {
            self.state.lock().staged = true;
        }

        /// Make every commit fail, as a missing signing key would.
        pub fn failing(reason: &str) -> Self {
            let fake = Self::dirty();
            fake.state.lock().fail_commit = Some(reason.to_string());
            fake
        }

        pub fn commits(&self) -> Vec<String> {
            self.state.lock().commits.clone()
        }

        /// The path list each commit was scoped to.
        pub fn committed_paths(&self) -> Vec<Vec<PathBuf>> {
            self.state.lock().paths.clone()
        }
    }

    impl Vcs for FakeGit {
        fn stage(&self, _paths: &[PathBuf]) -> Result<(), String> {
            Ok(())
        }

        fn has_staged(&self, _paths: &[PathBuf]) -> bool {
            self.state.lock().staged
        }

        fn commit(&self, message: &str, paths: &[PathBuf]) -> Result<(), String> {
            let mut state = self.state.lock();
            if let Some(reason) = state.fail_commit.clone() {
                return Err(reason);
            }
            state.commits.push(message.to_string());
            state.paths.push(paths.to_vec());
            state.staged = false;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> Vec<PathBuf> {
        vec![PathBuf::from("data/measurements.ndjson")]
    }

    #[test]
    fn a_clean_tree_commits_nothing() {
        let git = FakeGit::default();
        assert_eq!(
            commit_data(&git, &paths(), "data: 1 measurement"),
            Outcome::NothingToDo
        );
        assert!(git.commits().is_empty());
    }

    #[test]
    fn a_measured_row_is_committed_scoped_to_its_paths() {
        let git = FakeGit::dirty();
        assert_eq!(
            commit_data(&git, &paths(), "data: 1 measurement"),
            Outcome::Committed
        );
        assert_eq!(git.commits(), vec!["data: 1 measurement".to_string()]);
        assert_eq!(git.committed_paths(), vec![paths()]);
    }

    /// An empty path list commits nothing rather than everything.
    ///
    /// This is the module's whole purpose stated as a test. `git commit -m msg --`
    /// with no pathspec after the separator is a *bare* commit: it takes the entire
    /// index, including whatever the operator was staging while the campaign ran.
    /// `data_paths` is a public field with no validation, so the check has to be
    /// here rather than at the one call site that happens to fill it correctly.
    #[test]
    fn no_paths_commits_nothing() {
        let git = FakeGit::dirty();
        assert!(matches!(
            commit_data(&git, &[], "data: 1 measurement"),
            Outcome::Failed(_)
        ));
        assert!(git.commits().is_empty());
    }

    /// The real git refuses an empty pathspec too, before reaching the process.
    ///
    /// `FakeGit` cannot show this: it does not run git, so it would accept an empty
    /// list and report the same `Committed` as any other. Checking `LocalGit`
    /// directly is what pins the behaviour that matters.
    #[test]
    fn local_git_refuses_an_empty_pathspec() {
        let git = LocalGit::at(".");
        let error = git
            .commit("data: 1 measurement", &[])
            .expect_err("an empty pathspec is refused");
        assert!(error.contains("whole index"), "{error}");
        assert!(git.stage(&[]).is_err());
        assert!(!git.has_staged(&[]), "an empty pathspec has nothing staged");
    }

    /// A campaign outlives a commit failure.
    ///
    /// The rows are on disk; giving up here would throw away the hours of
    /// measurement still to come for a problem that has nothing to do with them.
    #[test]
    fn a_failed_commit_is_reported_not_fatal() {
        let git = FakeGit::failing("gpg: signing failed: No secret key");
        let outcome = commit_data(&git, &paths(), "data: 1 measurement");
        assert_eq!(
            outcome,
            Outcome::Failed("gpg: signing failed: No secret key".to_string())
        );
        assert!(git.commits().is_empty());
    }
}
