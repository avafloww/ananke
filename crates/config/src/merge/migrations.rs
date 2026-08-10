//! Resolve `migrate_from` chains into an ordered list of service renames.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, BTreeSet};

use ananke_errors::ExpectedError;
use smol_str::SmolStr;

use crate::parse::{RawConfig, RawService};

/// One service rename produced by a `migrate_from` chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// The service's previous name, as recorded in the database.
    pub old_name: SmolStr,
    /// The name it is being migrated to.
    pub new_name: SmolStr,
}

/// Resolve `migrate_from` chains into an ordered list of (old, new) pairs.
///
/// Returns pairs in topological order (sources before dependents) so the
/// database layer can reparent sequentially. Cycles are errors.
pub fn resolve_migrations(cfg: &mut RawConfig) -> Result<Vec<Migration>, ExpectedError> {
    let mut out: Vec<Migration> = Vec::new();
    let mut by_name: BTreeMap<SmolStr, &RawService> = BTreeMap::new();
    for s in &cfg.services {
        let name = s.common().name.clone().ok_or_else(|| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                "service without a name during migrate_from resolution".to_string(),
            )
        })?;
        by_name.insert(name, s);
    }

    let mut visiting: BTreeSet<SmolStr> = BTreeSet::new();
    let mut visited: BTreeSet<SmolStr> = BTreeSet::new();

    fn visit(
        name: &SmolStr,
        by_name: &BTreeMap<SmolStr, &RawService>,
        visiting: &mut BTreeSet<SmolStr>,
        visited: &mut BTreeSet<SmolStr>,
        out: &mut Vec<Migration>,
    ) -> Result<(), ExpectedError> {
        if visited.contains(name) {
            return Ok(());
        }
        if !visiting.insert(name.clone()) {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("migrate_from cycle involving {name}"),
            ));
        }
        if let Some(svc) = by_name.get(name)
            && let Some(old) = &svc.common().migrate_from
        {
            if by_name.contains_key(old) {
                visit(old, by_name, visiting, visited, out)?;
            }
            out.push(Migration {
                old_name: old.clone(),
                new_name: name.clone(),
            });
        }
        visiting.remove(name);
        visited.insert(name.clone());
        Ok(())
    }

    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for n in &names {
        visit(n, &by_name, &mut visiting, &mut visited, &mut out)?;
    }

    // Clear the migrate_from field on services so downstream code doesn't re-process.
    for svc in cfg.services.iter_mut() {
        svc.common_mut().migrate_from = None;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::{resolve_inheritance, test_support::parse};

    #[test]
    fn migrate_from_chain_resolved_in_order() {
        let mut cfg = parse(
            r#"
[[service]]
name = "c"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12002
migrate_from = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "a"
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        let b_idx = migrations.iter().position(|m| m.new_name == "b").unwrap();
        let c_idx = migrations.iter().position(|m| m.new_name == "c").unwrap();
        assert!(b_idx < c_idx);
        assert_eq!(migrations[b_idx].old_name, "a");
        assert_eq!(migrations[c_idx].old_name, "b");
    }

    #[test]
    fn migrate_from_missing_source_is_warning_not_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "does-not-exist"
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].old_name, "does-not-exist");
    }
}
