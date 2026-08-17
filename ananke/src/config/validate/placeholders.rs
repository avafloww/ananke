//! Dry-run every `{placeholder}` a service's argv can contain at validate
//! time, so a typo fails `config validate` rather than a runtime spawn.

use std::collections::BTreeMap;

use ananke_errors::ExpectedError;
use ananke_templates::stale_placeholder_names;
use smol_str::SmolStr;
use tracing::warn;

use crate::config::validate::{PlaceholderChecker, fail};

/// The daemon's placeholder dry-run checker: validates `command`,
/// `shutdown_command`, and llama-cpp `launcher` argv at config time.
pub struct DaemonPlaceholderChecker;

impl PlaceholderChecker for DaemonPlaceholderChecker {
    fn check(&self, name: &SmolStr, field: &str, argv: &[String]) -> Result<(), ExpectedError> {
        match field {
            "launcher" => check_launcher_placeholders(name, argv),
            _ => check_placeholders(name, field, argv),
        }
    }

    fn check_env(
        &self,
        name: &SmolStr,
        field: &str,
        env: &BTreeMap<String, String>,
    ) -> Result<(), ExpectedError> {
        for (key, value) in env {
            check_one(name, &format!("{field}.{key}"), value)?;
        }
        Ok(())
    }
}

/// Resolve every `{placeholder}` in `argv` against a synthetic context
/// covering every substitution the supervisor can produce. Propagates
/// the first [`SubstituteError`] as a config error with `field` + `name`
/// context, so a typo like `${prot}` fails `config validate` rather than
/// slipping through to a runtime `StartFailure`.
pub(crate) fn check_placeholders(
    name: &SmolStr,
    field: &str,
    argv: &[String],
) -> Result<(), ExpectedError> {
    for (i, arg) in argv.iter().enumerate() {
        check_one(name, &format!("{field}[{i}]"), arg)?;
    }
    Ok(())
}

/// Dry-run one value — an argv entry or an env value — against a synthetic
/// context covering every substitution the supervisor can produce. `entry`
/// names the value in full, so a failure says where it is.
pub(crate) fn check_one(name: &SmolStr, entry: &str, value: &str) -> Result<(), ExpectedError> {
    use ananke_placement::devices::{Allocation, DeviceId};
    use ananke_templates::{PlaceholderContext, substitute};
    let mut alloc_bytes = BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        // `None` so a `${reserve_mb}` placeholder on a dynamic allocation
        // trips the `ReserveMbOnDynamic` branch at config time, not
        // later. Static allocations re-validate at spawn time against
        // the real static_reserve_mb.
        static_reserve_mb: None,
        listen_host: None,
        host_port: 0,
    };
    warn_on_stale_form(name, entry, value);
    substitute(value, &ctx).map_err(|e| fail(format!("service {name}: {entry} {value:?}: {e}")))?;
    Ok(())
}

/// Say so when an entry contains a placeholder name written in the old
/// `{name}` form. It resolves to nothing now, so a config carried over
/// from before the grammar changed launches with the literal text in its
/// argv and no error anywhere.
///
/// A warning rather than an error: a bare `{` is ordinary payload, and a
/// JSON or format-string argument may hold `{model}` on purpose. Write
/// `$${model}` to say that deliberately and silence this.
fn warn_on_stale_form(name: &SmolStr, entry: &str, value: &str) {
    for stale in stale_placeholder_names(value) {
        warn!(
            service = %name,
            entry = %entry,
            "`{{{stale}}}` is not a placeholder; write `${{{stale}}}`, or `$${{{stale}}}` for the literal text"
        );
    }
}

/// Dry-run a llama-cpp `launcher` argv at validate time. Identical
/// purpose to [`check_placeholders`] but tolerates the `${args}` splat
/// (which would otherwise be rejected by [`substitute`]). Surfaces
/// typos like `${prot}` and misuses like `--foo={args}` as config errors
/// rather than runtime `StartFailure`s.
pub(crate) fn check_launcher_placeholders(
    name: &SmolStr,
    argv: &[String],
) -> Result<(), ExpectedError> {
    use ananke_placement::devices::{Allocation, DeviceId};
    use ananke_templates::{PlaceholderContext, substitute_launcher_argv};
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        static_reserve_mb: None,
        listen_host: None,
        host_port: 0,
    };
    for (i, arg) in argv.iter().enumerate() {
        warn_on_stale_form(name, &format!("launcher[{i}]"), arg);
    }
    substitute_launcher_argv(argv, &[], &ctx)
        .map_err(|e| fail(format!("service {name}: launcher: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::config::validate::{
        DaemonPlaceholderChecker, test_fixtures::parse_and_merge, validate_with_checks,
    };

    fn validate(
        cfg: &ananke_config::parse::RawConfig,
    ) -> Result<ananke_config::validate::EffectiveConfig, ananke_errors::ExpectedError> {
        validate_with_checks(cfg, &DaemonPlaceholderChecker)
    }

    #[test]
    fn launcher_accepts_well_formed_template() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["/opt/podman-wrap.sh", "${model}", "${args}"]
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
                    "${model}".into(),
                    "${args}".into()
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
launcher = ["wrap.sh", "${model}", "${bogus}", "${args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("${bogus}") && msg.contains("launcher"),
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
launcher = ["wrap.sh", "${model}", "--foo=${args}"]
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

    #[test]
    fn command_service_rejects_typo_in_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port=${prot}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("command[1]") && msg.contains("${prot}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_rejects_typo_in_shutdown_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port=${port}"]
shutdown_command = ["stop", "${bogus}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("shutdown_command[1]") && msg.contains("${bogus}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_typo_in_an_env_value_is_a_config_error() {
        // Env values are substituted at spawn like any argv entry, so a
        // typo in one has to fail here rather than at the launch it breaks.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port=${port}"]
env = { ENDPOINT = "http://localhost:${prot}" }
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("env.ENDPOINT") && msg.contains("${prot}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_typo_in_a_container_env_value_is_a_config_error() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port=${listen_port}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1

[service.container]
image = "example:latest"
network = "host"
env = { BIND = "${lsiten_host}" }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("container.env.BIND") && msg.contains("${lsiten_host}"),
            "unexpected error: {err}"
        );
    }
}
