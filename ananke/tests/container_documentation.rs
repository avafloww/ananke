//! Hold `docs/configuration.md`'s container section to the config it
//! documents.
//!
//! The generator keeps the field tables honest, but the prose fragment is
//! hand-written: its TOML examples and its operational claims are only as
//! true as the last person to edit them. These tests parse every example
//! and check that the contract an operator has to know before running a
//! container is actually stated somewhere.

use std::path::Path;

use ananke_config::{parse_toml, resolve_inheritance, validate};

/// The container section of the generated configuration reference.
fn container_section() -> String {
    let doc = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/configuration.md"),
    )
    .expect("docs/configuration.md must exist");
    let start = doc
        .find("\n## Container Workloads")
        .expect("configuration.md must document container workloads");
    let rest = &doc[start + 1..];
    // Up to the next `##` heading, so a later section's examples don't leak
    // into this one's assertions.
    match rest[2..].find("\n## ") {
        Some(end) => rest[..end + 3].to_string(),
        None => rest.to_string(),
    }
}

/// Every fenced ```toml block in `section`.
fn toml_blocks(section: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = section;
    while let Some(open) = rest.find("```toml\n") {
        rest = &rest[open + "```toml\n".len()..];
        let close = rest.find("```").expect("unterminated toml fence");
        blocks.push(rest[..close].to_string());
        rest = &rest[close + 3..];
    }
    blocks
}

#[test]
fn container_documentation_examples_parse() {
    let blocks = toml_blocks(&container_section());
    assert!(
        blocks.len() >= 3,
        "expected the container docs to carry the llama.cpp, command, and overview examples; found {}",
        blocks.len()
    );

    for (i, block) in blocks.iter().enumerate() {
        let mut raw = parse_toml(block, Path::new("/docs/configuration.md"))
            .unwrap_or_else(|e| panic!("container example {i} does not parse: {e}\n{block}"));
        resolve_inheritance(&mut raw)
            .unwrap_or_else(|e| panic!("container example {i} fails inheritance: {e}"));
        let effective = validate(&raw)
            .unwrap_or_else(|e| panic!("container example {i} fails validation: {e}\n{block}"));

        // Each example is a container service, or the docs are illustrating
        // something other than what the section is for.
        for svc in &effective.services {
            assert!(
                svc.container.is_some(),
                "container example {i} declares service `{}` with no [service.container] block",
                svc.name
            );
        }
    }
}

#[test]
fn container_documentation_covers_operational_contract() {
    let section = container_section();

    // Every field the descriptor table declares must also be reachable from
    // this section, so a field can't be added without landing in the docs.
    for id in [
        "container",
        "container_mounts",
        "container_extra_publications",
    ] {
        let table = ananke_config::docs::all_sections()
            .into_iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("descriptor table missing section `{id}`"));
        for f in &table.fields {
            assert!(
                section.contains(&format!("`{}`", f.name)),
                "container docs never mention the `{}` field from section `{id}`",
                f.name
            );
        }
    }

    // The operational topics an operator has to know before running one.
    for (topic, needle) in [
        ("image prerequisites", "neither pulls nor builds"),
        ("runtime selection", "podman"),
        ("environment allowlisting", "env_passthrough"),
        ("secret handling", "never leave the runtime invocation"),
        ("host networking", "network = \"host\""),
        ("loopback-only publication", "127.0.0.1:<private_port>"),
        ("mount path translation", "chat_template_file"),
        ("opaque paths", "extra_args"),
        ("CDI GPU injection", "nvidia.com/gpu=${id}"),
        ("cgroup attribution", "cgroup"),
        ("detached lifecycle", "logs --follow"),
        ("explicit cleanup", "`--rm` is deliberately not used"),
        ("restart reconciliation", "owner UUID"),
        ("shutdown sweep", "overruns `shutdown_timeout`"),
        ("uncatchable signals", "SIGKILL"),
        ("combined log stream", "`combined`"),
    ] {
        assert!(
            section.contains(needle),
            "container docs do not cover {topic} (looked for {needle:?})"
        );
    }
}
