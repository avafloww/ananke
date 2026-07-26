//! Dry-run every `{placeholder}` a service's argv can contain at validate
//! time, so a typo fails `config validate` rather than a runtime spawn.

use smol_str::SmolStr;

use crate::{config::validate::fail, errors::ExpectedError};

/// Resolve every `{placeholder}` in `argv` against a synthetic context
/// covering every substitution the supervisor can produce. Propagates
/// the first [`SubstituteError`] as a config error with `field` + `name`
/// context, so a typo like `{prot}` fails `config validate` rather than
/// slipping through to a runtime `StartFailure`.
pub(crate) fn check_placeholders(
    name: &SmolStr,
    field: &str,
    argv: &[String],
) -> Result<(), ExpectedError> {
    use crate::{
        devices::{Allocation, DeviceId},
        templates::{PlaceholderContext, substitute},
    };
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        // `None` so a `{reserve_mb}` placeholder on a dynamic allocation
        // trips the `ReserveMbOnDynamic` branch at config time, not
        // later. Static allocations re-validate at spawn time against
        // the real static_reserve_mb.
        static_reserve_mb: None,
    };
    for (i, arg) in argv.iter().enumerate() {
        substitute(arg, &ctx)
            .map_err(|e| fail(format!("service {name}: {field}[{i}] {arg:?}: {e}")))?;
    }
    Ok(())
}

/// Dry-run a llama-cpp `launcher` argv at validate time. Identical
/// purpose to [`check_placeholders`] but tolerates the `{args}` splat
/// (which would otherwise be rejected by [`substitute`]). Surfaces
/// typos like `{prot}` and misuses like `--foo={args}` as config errors
/// rather than runtime `StartFailure`s.
pub(crate) fn check_launcher_placeholders(
    name: &SmolStr,
    argv: &[String],
) -> Result<(), ExpectedError> {
    use crate::{
        devices::{Allocation, DeviceId},
        templates::{PlaceholderContext, substitute_launcher_argv},
    };
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        static_reserve_mb: None,
    };
    substitute_launcher_argv(argv, &[], &ctx)
        .map_err(|e| fail(format!("service {name}: launcher: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::config::validate::{test_fixtures::parse_and_merge, validate};

    #[test]
    fn launcher_accepts_well_formed_template() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["/opt/podman-wrap.sh", "{model}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(
            lc.launcher.as_deref(),
            Some(
                &[
                    "/opt/podman-wrap.sh".to_string(),
                    "{model}".into(),
                    "{args}".into()
                ][..]
            )
        );
    }

    #[test]
    fn launcher_rejects_unknown_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "{bogus}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("{bogus}") && msg.contains("launcher"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn launcher_rejects_splat_embedded_in_arg() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "--foo={args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("{args}"));
    }

    #[test]
    fn launcher_rejects_empty_argv() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = []
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("launcher"));
    }
}
