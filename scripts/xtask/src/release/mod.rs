//! Release flow: bump the workspace version, commit, and tag locally.
//!
//! The flow deliberately stops before any remote-affecting operation
//! (no `git push`, no `gh release`) so the operator can inspect the
//! commit and the tag before publishing them. Once `git push origin
//! v<version>` lands, the Release workflow (`.github/workflows/release.yml`)
//! takes over and produces the actual GitHub release.

mod git;
mod plan;

use std::{fmt, path::PathBuf};

use clap::Parser;
use git::{ensure_clean_tree, ensure_on_branch, ensure_tag_absent, git, repo_root};
use plan::{Plan, read_workspace_version, validate_version};

#[derive(Parser)]
pub struct Args {
    /// New workspace version (without leading `v`), e.g. 0.2.0.
    pub version: String,

    /// Proceed even if the working tree has uncommitted changes.
    #[arg(long)]
    pub allow_dirty: bool,

    /// Branch on which a release is allowed. Defaults to `main`.
    #[arg(long, default_value = "main")]
    pub branch: String,

    /// Proceed even if the current branch does not match `--branch`.
    #[arg(long)]
    pub allow_branch_mismatch: bool,

    /// Print the actions that would be taken without performing them.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(args: Args) -> Result<(), Error> {
    validate_version(&args.version)?;

    let repo = repo_root()?;
    let cargo_toml = repo.join("Cargo.toml");
    let cargo_lock = repo.join("Cargo.lock");
    let package_json = repo.join("frontend/package.json");
    let package_lock = repo.join("frontend/package-lock.json");

    let old = read_workspace_version(&cargo_toml)?;
    if old == args.version {
        return Err(Error::SameVersion { version: old });
    }

    if !args.allow_dirty {
        ensure_clean_tree(&repo)?;
    }
    ensure_on_branch(&repo, &args.branch, args.allow_branch_mismatch)?;
    let tag = format!("v{}", args.version);
    ensure_tag_absent(&repo, &tag)?;

    let plan = Plan::build(
        &cargo_toml,
        &cargo_lock,
        &package_json,
        &package_lock,
        &old,
        &args.version,
    )?;

    println!("Bumping workspace version: {old} -> {}", args.version);
    println!("Tag to create:             {tag}");
    println!("Files to update:");
    for file in &plan.files {
        println!("  {}", file.path.display());
    }

    if args.dry_run {
        println!();
        println!("Dry run; no files written.");
        return Ok(());
    }

    plan.write()?;

    git(
        &repo,
        &[
            "add",
            "Cargo.toml",
            "Cargo.lock",
            "frontend/package.json",
            "frontend/package-lock.json",
        ],
    )?;
    git(&repo, &["commit", "-m", &format!("chore(release): {tag}")])?;
    git(&repo, &["tag", "-a", &tag, "-m", &format!("Release {tag}")])?;

    println!();
    println!("Created commit and annotated tag {tag}.");
    println!("Push when ready:");
    println!("  git push --follow-tags");
    Ok(())
}

#[derive(Debug)]
pub enum Error {
    InvalidVersion {
        value: String,
        reason: String,
    },
    SameVersion {
        version: String,
    },
    DirtyTree {
        lines: usize,
    },
    WrongBranch {
        expected: String,
        actual: String,
    },
    TagExists(String),
    GitSpawn(std::io::Error),
    GitCommand {
        command: String,
        stderr: String,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    TomlParse {
        path: PathBuf,
        source: toml_edit::TomlError,
    },
    JsonParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    MissingKey {
        path: PathBuf,
        key: String,
    },
    VersionMismatch {
        path: PathBuf,
        package: String,
        expected: String,
        found: String,
    },
    WorkspaceCratesIncomplete {
        found: usize,
        expected: usize,
    },
    UnexpectedMatchCount {
        path: PathBuf,
        needle: String,
        found: usize,
        expected: usize,
    },
    CargoMetadata(cargo_metadata::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion { value, reason } => {
                write!(f, "invalid version `{value}`: {reason}")
            }
            Self::SameVersion { version } => {
                write!(f, "workspace is already at version {version}")
            }
            Self::DirtyTree { lines } => write!(
                f,
                "working tree has {lines} uncommitted change(s); rerun with --allow-dirty to override"
            ),
            Self::WrongBranch { expected, actual } => write!(
                f,
                "on branch `{actual}`, expected `{expected}`; rerun with --allow-branch-mismatch to override"
            ),
            Self::TagExists(tag) => write!(f, "tag `{tag}` already exists"),
            Self::GitSpawn(source) => write!(f, "failed to spawn `git`: {source}"),
            Self::GitCommand { command, stderr } => {
                let trimmed = stderr.trim();
                if trimmed.is_empty() {
                    write!(f, "`{command}` failed")
                } else {
                    write!(f, "`{command}` failed: {trimmed}")
                }
            }
            Self::Io { path, source } => write!(f, "i/o error on {}: {source}", path.display()),
            Self::TomlParse { path, source } => {
                write!(f, "failed to parse {} as TOML: {source}", path.display())
            }
            Self::JsonParse { path, source } => {
                write!(f, "failed to parse {} as JSON: {source}", path.display())
            }
            Self::MissingKey { path, key } => {
                write!(f, "missing key `{key}` in {}", path.display())
            }
            Self::VersionMismatch {
                path,
                package,
                expected,
                found,
            } => write!(
                f,
                "{}: package `{package}` is at version {found}, expected {expected}",
                path.display()
            ),
            Self::WorkspaceCratesIncomplete { found, expected } => write!(
                f,
                "found {found} of {expected} expected workspace crate entries in Cargo.lock"
            ),
            Self::UnexpectedMatchCount {
                path,
                needle,
                found,
                expected,
            } => write!(
                f,
                "expected {expected} occurrence(s) of `{needle}` in {}, found {found}",
                path.display()
            ),
            Self::CargoMetadata(source) => write!(f, "failed to read workspace metadata: {source}"),
        }
    }
}

impl std::error::Error for Error {}
