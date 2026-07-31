//! Git operations for the release flow: branch and tag preconditions, and
//! the final commit/tag creation.

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use crate::release::Error;

pub(super) fn repo_root() -> Result<PathBuf, Error> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git rev-parse --show-toplevel".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

pub(super) fn ensure_clean_tree(repo: &Path) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["status", "--porcelain"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git status --porcelain".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    if !output.stdout.is_empty() {
        let lines = String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .lines()
            .count();
        return Err(Error::DirtyTree { lines });
    }
    Ok(())
}

pub(super) fn ensure_on_branch(
    repo: &Path,
    expected: &str,
    allow_mismatch: bool,
) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git rev-parse --abbrev-ref HEAD".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let actual = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if actual != expected && !allow_mismatch {
        return Err(Error::WrongBranch {
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

pub(super) fn ensure_tag_absent(repo: &Path, tag: &str) -> Result<(), Error> {
    let status = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", &format!("refs/tags/{tag}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(Error::GitSpawn)?;
    if status.success() {
        return Err(Error::TagExists(tag.to_string()));
    }
    Ok(())
}

pub(super) fn git(repo: &Path, args: &[&str]) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: format!("git {}", args.join(" ")),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}
