//! Topological resolution of `extends` chains: cycle detection, dispatch to
//! the per-template field merge, and same-template enforcement.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, BTreeSet};

use ananke_errors::ExpectedError;
use smol_str::SmolStr;

use crate::{
    merge::field_merge::{merge_command, merge_llama_cpp},
    parse::{RawConfig, RawService},
};

/// Resolve every service's `extends` chain, merging inherited fields into
/// each service and enforcing that a service only extends the same template.
pub fn resolve_inheritance(cfg: &mut RawConfig) -> Result<(), ExpectedError> {
    // Index services by name; require names and disallow duplicates.
    let mut by_name: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    for s in std::mem::take(&mut cfg.services) {
        let name = s.common().name.clone().ok_or_else(|| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                "service block missing name".into(),
            )
        })?;
        if by_name.insert(name.clone(), s).is_some() {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("duplicate service name: {name}"),
            ));
        }
    }

    // Topologically resolve each service's extends chain.
    let mut resolved: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for name in &names {
        resolve_one(name, &by_name, &mut resolved, &mut BTreeSet::new())?;
    }

    cfg.services = resolved.into_values().collect();
    Ok(())
}

fn resolve_one(
    name: &SmolStr,
    source: &BTreeMap<SmolStr, RawService>,
    resolved: &mut BTreeMap<SmolStr, RawService>,
    stack: &mut BTreeSet<SmolStr>,
) -> Result<(), ExpectedError> {
    if resolved.contains_key(name) {
        return Ok(());
    }
    if !stack.insert(name.clone()) {
        return Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("extends cycle involving service {name}"),
        ));
    }

    let raw = source.get(name).cloned().ok_or_else(|| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("service {name} not found during extends resolution"),
        )
    })?;

    let merged = match raw.common().extends.clone() {
        None => raw,
        Some(parent_name) => {
            if !source.contains_key(&parent_name) {
                return Err(ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<config>"),
                    format!("service {name} extends {parent_name} which does not exist"),
                ));
            }
            resolve_one(&parent_name, source, resolved, stack)?;
            let parent = resolved
                .get(&parent_name)
                .ok_or_else(|| {
                    ExpectedError::config_unparseable(
                        std::path::PathBuf::from("<config>"),
                        format!("service {name} extends {parent_name} which resolved to nothing"),
                    )
                })?
                .clone();
            merge_service(&parent, &raw, name)?
        }
    };

    stack.remove(name);
    resolved.insert(name.clone(), merged);
    Ok(())
}

fn merge_service(
    parent: &RawService,
    child: &RawService,
    child_name: &SmolStr,
) -> Result<RawService, ExpectedError> {
    match (parent, child) {
        (RawService::LlamaCpp(p), RawService::LlamaCpp(c)) => Ok(RawService::LlamaCpp(Box::new(
            merge_llama_cpp(p, c, child_name)?,
        ))),
        (RawService::Command(p), RawService::Command(c)) => Ok(RawService::Command(Box::new(
            merge_command(p, c, child_name)?,
        ))),
        _ => Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!(
                "service {child_name}: template `{}` does not match parent's template `{}`; \
                 cross-template extends is not allowed",
                child.template_label(),
                parent.template_label(),
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::test_support::{find_llama, parse};

    #[test]
    fn transitive_extends() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
context = 4096

[[service]]
name = "b"
template = "llama-cpp"
extends = "a"
port = 11001

[[service]]
name = "c"
template = "llama-cpp"
extends = "b"
port = 11002
context = 32768
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let c = find_llama(&cfg, "c");
        assert_eq!(c.context, Some(32768));
        assert_eq!(c.model.as_ref().unwrap().to_str(), Some("/m/a.gguf"));
    }
    #[test]
    fn cycle_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11001
extends = "a"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("cycle"));
    }
    #[test]
    fn missing_extends_target_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "does-not-exist"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }
    #[test]
    fn cross_template_extends_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "command"
port = 11000
command = ["/bin/true"]

[[service]]
name = "child"
template = "llama-cpp"
extends = "base"
port = 11001
model = "/m/a.gguf"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("template") && msg.contains("cross-template"),
            "unexpected error: {msg}"
        );
    }
}
