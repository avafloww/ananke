//! Version-bump plan: validate the target version and compute the new
//! contents of `Cargo.toml`, `Cargo.lock`, `package.json`, and
//! `package-lock.json` without writing anything until the caller commits.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use toml_edit::{DocumentMut, value};

use crate::release::Error;

pub(super) struct Plan {
    pub(super) files: Vec<PlannedFile>,
}

pub(super) struct PlannedFile {
    pub(super) path: PathBuf,
    new_content: String,
}

impl Plan {
    pub(super) fn build(
        cargo_toml: &Path,
        cargo_lock: &Path,
        package_json: &Path,
        package_lock: &Path,
        old: &str,
        new: &str,
    ) -> Result<Self, Error> {
        let members = workspace_member_names(cargo_toml)?;
        let files = vec![
            PlannedFile {
                path: cargo_toml.to_path_buf(),
                new_content: rewrite_cargo_toml(cargo_toml, new)?,
            },
            PlannedFile {
                path: cargo_lock.to_path_buf(),
                new_content: rewrite_cargo_lock(cargo_lock, &members, old, new)?,
            },
            PlannedFile {
                path: package_json.to_path_buf(),
                new_content: rewrite_package_json(package_json, old, new)?,
            },
            PlannedFile {
                path: package_lock.to_path_buf(),
                new_content: rewrite_package_lock(package_lock, old, new)?,
            },
        ];
        Ok(Self { files })
    }

    pub(super) fn write(&self) -> Result<(), Error> {
        for file in &self.files {
            write(&file.path, &file.new_content)?;
        }
        Ok(())
    }
}

fn workspace_member_names(cargo_toml: &Path) -> Result<Vec<String>, Error> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .manifest_path(cargo_toml)
        .no_deps()
        .exec()
        .map_err(Error::CargoMetadata)?;
    Ok(metadata
        .workspace_packages()
        .into_iter()
        .map(|p| p.name.clone())
        .collect())
}

pub(super) fn validate_version(v: &str) -> Result<(), Error> {
    if let Some(stripped) = v.strip_prefix('v') {
        return Err(Error::InvalidVersion {
            value: v.to_string(),
            reason: format!("drop the leading `v` (use `{stripped}` instead)"),
        });
    }
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let parts: Vec<&str> = core.split('.').collect();
    let core_ok = parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
    if !core_ok {
        return Err(Error::InvalidVersion {
            value: v.to_string(),
            reason: "expected MAJOR.MINOR.PATCH semver core".to_string(),
        });
    }
    Ok(())
}

pub(super) fn read_workspace_version(path: &Path) -> Result<String, Error> {
    let content = read(path)?;
    let doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let v = doc
        .get("workspace")
        .and_then(|w| w.as_table())
        .and_then(|w| w.get("package"))
        .and_then(|p| p.as_table())
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace.package.version".to_string(),
        })?;
    Ok(v.to_string())
}

fn rewrite_cargo_toml(path: &Path, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    let mut doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let workspace = doc
        .get_mut("workspace")
        .and_then(|w| w.as_table_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace".to_string(),
        })?;
    let package = workspace
        .get_mut("package")
        .and_then(|p| p.as_table_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace.package".to_string(),
        })?;
    package["version"] = value(new_version);
    Ok(doc.to_string())
}

fn rewrite_cargo_lock(
    path: &Path,
    members: &[String],
    old: &str,
    new_version: &str,
) -> Result<String, Error> {
    let content = read(path)?;
    let mut doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let packages = doc
        .get_mut("package")
        .and_then(|p| p.as_array_of_tables_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "package".to_string(),
        })?;
    let mut bumped = 0usize;
    for pkg in packages.iter_mut() {
        let name = pkg
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or_default()
            .to_string();
        if !members.iter().any(|m| m == &name) {
            continue;
        }
        let current = pkg
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if current != old {
            return Err(Error::VersionMismatch {
                path: path.to_path_buf(),
                package: name,
                expected: old.to_string(),
                found: current,
            });
        }
        pkg["version"] = value(new_version);
        bumped += 1;
    }
    if bumped != members.len() {
        return Err(Error::WorkspaceCratesIncomplete {
            found: bumped,
            expected: members.len(),
        });
    }
    Ok(doc.to_string())
}

/// The one field the bump reads. npm owns the rest of the schema, so
/// everything else is ignored rather than modelled.
#[derive(Deserialize)]
struct PackageJson {
    version: Option<String>,
}

fn rewrite_package_json(path: &Path, old: &str, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    let parsed: PackageJson =
        serde_json::from_str(&content).map_err(|source| Error::JsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let current = parsed.version.as_deref().unwrap_or_default();
    if current != old {
        return Err(Error::VersionMismatch {
            path: path.to_path_buf(),
            package: "frontend".to_string(),
            expected: old.to_string(),
            found: current.to_string(),
        });
    }
    let needle = format!("\"version\": \"{old}\"");
    let count = content.matches(&needle).count();
    if count != 1 {
        return Err(Error::UnexpectedMatchCount {
            path: path.to_path_buf(),
            needle,
            found: count,
            expected: 1,
        });
    }
    Ok(content.replacen(&needle, &format!("\"version\": \"{new_version}\""), 1))
}

fn rewrite_package_lock(path: &Path, old: &str, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    // The project's own version appears twice near the top of the file
    // (top-level + `packages[""]`). Every other version field belongs to
    // a dependency under a `node_modules/...` key. Bounding the search to
    // the head keeps the substitution surgical without parsing the whole
    // lockfile (which would reorder keys without `preserve_order`).
    let cutoff = content.find("\"node_modules/").unwrap_or(content.len());
    let (head, tail) = content.split_at(cutoff);
    let needle = format!("\"version\": \"{old}\"");
    let count = head.matches(&needle).count();
    if count != 2 {
        return Err(Error::UnexpectedMatchCount {
            path: path.to_path_buf(),
            needle,
            found: count,
            expected: 2,
        });
    }
    let new_head = head.replace(&needle, &format!("\"version\": \"{new_version}\""));
    Ok(format!("{new_head}{tail}"))
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write(path: &Path, content: &str) -> Result<(), Error> {
    fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_version_accepts_semver_core() {
        assert!(validate_version("0.1.0").is_ok());
        assert!(validate_version("1.2.3").is_ok());
        assert!(validate_version("10.20.30").is_ok());
        assert!(validate_version("1.0.0-alpha.1").is_ok());
        assert!(validate_version("1.0.0+build.5").is_ok());
    }

    #[test]
    fn validate_version_rejects_leading_v() {
        let err = validate_version("v1.2.3").unwrap_err();
        assert!(matches!(err, Error::InvalidVersion { .. }));
    }

    #[test]
    fn validate_version_rejects_non_semver() {
        assert!(validate_version("1.2").is_err());
        assert!(validate_version("1.2.3.4").is_err());
        assert!(validate_version("1.x.0").is_err());
        assert!(validate_version("").is_err());
    }
}
