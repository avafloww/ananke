//! Substitute `{port}`/`{listen_port}`, `{listen_host}`, `{host_port}`,
//! `{gpu_ids}`, `{reserve_mb}`, `{model}`, `{name}` in command-template
//! argv and env values.

use std::collections::BTreeMap;

use ananke_devices::{Allocation, DeviceId};

#[derive(Debug, Clone)]
pub struct PlaceholderContext<'a> {
    pub name: &'a str,
    /// The port the workload should bind, resolving `{listen_port}` and its
    /// `{port}` alias. For a native process this is the ananke private
    /// port; for a bridge container it is `container_port`, so the argv
    /// binds the in-container side of the publication.
    pub port: u16,
    pub model: Option<&'a str>,
    pub allocation: &'a Allocation,
    /// Only populated for single-device static allocations; `None` on
    /// dynamic or multi-device, where `{reserve_mb}` is a config error.
    pub static_reserve_mb: Option<u64>,
    /// Container listening interface for `{listen_host}`. `None` renders
    /// the native-process default of `127.0.0.1`; a bridge-networked
    /// container supplies `0.0.0.0` so the published loopback host port
    /// maps onto the in-container all-interfaces listener. Host-networked
    /// containers keep `127.0.0.1`.
    pub listen_host: Option<&'a str>,
    /// Host-side port for `{host_port}`. For a native process this equals
    /// [`Self::port`]; for a bridge container it is the ananke private
    /// port while [`Self::port`] is the container port.
    pub host_port: u16,
}

#[derive(Debug)]
pub enum SubstituteError {
    ReserveMbOnDynamic,
    ReserveMbMultiDevice,
    UnknownPlaceholder(String),
    /// The launcher splat `{args}` must occupy a launcher entry on its
    /// own; it cannot be embedded inside a larger argv string because
    /// the expansion produces multiple arguments, not a single one.
    SplatInsideArg,
}

impl std::fmt::Display for SubstituteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstituteError::ReserveMbOnDynamic => {
                write!(f, "{{reserve_mb}} is invalid with a dynamic allocation")
            }
            SubstituteError::ReserveMbMultiDevice => {
                write!(
                    f,
                    "{{reserve_mb}} is valid only with a single-device static allocation"
                )
            }
            SubstituteError::UnknownPlaceholder(s) => {
                write!(f, "unknown placeholder {{{s}}}")
            }
            SubstituteError::SplatInsideArg => {
                write!(
                    f,
                    "splat placeholder {{args}} must be the entire launcher entry, \
                     not embedded inside a larger string"
                )
            }
        }
    }
}

impl std::error::Error for SubstituteError {}

/// Substitute every `{placeholder}` in `input` using `ctx`. Returns a
/// fresh owned String. Unknown placeholders produce a hard error so
/// typos surface rather than leaking literal `{oops}` into the argv.
///
/// Only an identifier between braces is treated as a placeholder, so
/// brace-delimited content that could never be one — a JSON argument such
/// as vLLM's `--diffusion-config '{"canvas_length": 256}'` — passes through
/// untouched. `{{` and `}}` remain available as explicit escapes for a
/// literal `{` / `}`.
pub fn substitute(input: &str, ctx: &PlaceholderContext<'_>) -> Result<String, SubstituteError> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    // How many literal `{` are currently open. The `{{`/`}}` escapes apply
    // only at the top level: inside a literal brace group the closing `}}`
    // of nested JSON is two real braces, not one escaped one.
    let mut depth = 0usize;
    while !rest.is_empty() {
        // `{{` → literal `{`.
        if depth == 0
            && let Some(after) = rest.strip_prefix("{{")
        {
            out.push('{');
            rest = after;
            continue;
        }
        // `}}` → literal `}`.
        if depth == 0
            && let Some(after) = rest.strip_prefix("}}")
        {
            out.push('}');
            rest = after;
            continue;
        }
        if let Some(after_brace) = rest.strip_prefix('{') {
            // Placeholder: consume up to the matching `}`, but only if what
            // is between the braces could be a placeholder name at all.
            if let Some(close) = after_brace.find('}')
                && is_placeholder_key(&after_brace[..close])
            {
                let replacement = resolve(&after_brace[..close], ctx)?;
                out.push_str(&replacement);
                rest = &after_brace[close + 1..];
                continue;
            }
            // An unmatched `{`, or a brace group holding something no
            // placeholder could be named — a JSON object, say. Literal.
            out.push('{');
            depth += 1;
            rest = after_brace;
            continue;
        }
        if let Some(after_brace) = rest.strip_prefix('}') {
            // A `}` closes a literal group, or has nothing to close at all;
            // either way it is literal. Without this arm the scanner makes
            // no progress on one and spins forever.
            out.push('}');
            depth = depth.saturating_sub(1);
            rest = after_brace;
            continue;
        }
        // Regular char run up to the next `{` or `}`.
        let next = rest.find(['{', '}']).unwrap_or(rest.len());
        out.push_str(&rest[..next]);
        rest = &rest[next..];
    }
    Ok(out)
}

/// Whether `key` is shaped like a placeholder name — an identifier.
///
/// Every placeholder ananke defines is one, so anything else between braces
/// is content that merely happens to be brace-delimited. The distinction
/// matters because command templates routinely carry JSON: vLLM's
/// `--diffusion-config '{"canvas_length": 256}'` is an argument, not a
/// typo'd `{port}`, and treating it as the latter fails the launch.
///
/// A typo still errors, because a typo is identifier-shaped: `{prot}` is
/// rejected as an unknown placeholder exactly as before.
fn is_placeholder_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn resolve(key: &str, ctx: &PlaceholderContext<'_>) -> Result<String, SubstituteError> {
    match key {
        // `listen_port` is the name to write for the port the workload
        // should bind. `port` is its compatibility alias, kept because
        // every pre-container command template in the wild uses it.
        "port" | "listen_port" => Ok(ctx.port.to_string()),
        "name" => Ok(ctx.name.to_string()),
        "model" => Ok(ctx.model.unwrap_or("").to_string()),
        // `listen_host` resolves to the container/host listening interface.
        "listen_host" => Ok(ctx.listen_host.unwrap_or("127.0.0.1").to_string()),
        // `host_port` is the host-side private port when a command needs
        // both sides of a bridge publication.
        "host_port" => Ok(ctx.host_port.to_string()),
        "gpu_ids" => {
            let mut ids: Vec<u32> = ctx
                .allocation
                .bytes
                .keys()
                .filter_map(|id| match id {
                    DeviceId::Cpu => None,
                    DeviceId::Gpu(n) => Some(*n),
                })
                .collect();
            ids.sort_unstable();
            Ok(ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
        }
        // `vram_mb` is a device-specific alias, accepted so command templates
        // in the wild keep launching. `reserve_mb` is the name to write, since
        // the reservation lands on the CPU device just as readily as on a
        // GPU.
        "reserve_mb" | "vram_mb" => ctx
            .static_reserve_mb
            .map(|mb| mb.to_string())
            .ok_or(SubstituteError::ReserveMbOnDynamic),
        other => Err(SubstituteError::UnknownPlaceholder(other.to_string())),
    }
}

/// Substitute a llama-cpp `launcher` argv template, expanding the splat
/// `{args}` placeholder to the full list of llama-server flags ananke
/// would otherwise have emitted. `{args}` must occupy a launcher entry
/// on its own — `["--foo={args}"]` is rejected because the expansion
/// produces multiple argv entries, not a single one.
///
/// Every other launcher entry passes through [`substitute`], so the
/// usual placeholders (`{model}`, `{port}`, `{name}`, `{gpu_ids}`) are
/// resolved in-place.
pub fn substitute_launcher_argv(
    launcher: &[String],
    llama_args: &[String],
    ctx: &PlaceholderContext<'_>,
) -> Result<Vec<String>, SubstituteError> {
    let mut out: Vec<String> = Vec::with_capacity(launcher.len() + llama_args.len());
    for entry in launcher {
        if entry == "{args}" {
            out.extend(llama_args.iter().cloned());
            continue;
        }
        if entry.contains("{args}") {
            return Err(SubstituteError::SplatInsideArg);
        }
        out.push(substitute(entry, ctx)?);
    }
    Ok(out)
}

/// Apply substitution across a whole argv vector and env map. Stops at
/// the first substitution error.
pub fn substitute_argv(
    argv: &[String],
    env: &BTreeMap<String, String>,
    ctx: &PlaceholderContext<'_>,
) -> Result<(Vec<String>, BTreeMap<String, String>), SubstituteError> {
    let argv_out: Vec<String> = argv
        .iter()
        .map(|a| substitute(a, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let mut env_out = BTreeMap::new();
    for (k, v) in env {
        env_out.insert(k.clone(), substitute(v, ctx)?);
    }
    Ok((argv_out, env_out))
}

#[cfg(test)]
mod tests {
    use ananke_config::placement::DeviceSlot;

    use super::*;

    fn alloc_gpu0_only() -> Allocation {
        let mut map = std::collections::BTreeMap::new();
        map.insert(DeviceSlot::Gpu(0), 6000);
        Allocation::from_override(&map)
    }

    fn alloc_cpu_only() -> Allocation {
        let mut map = std::collections::BTreeMap::new();
        map.insert(DeviceSlot::Cpu, 1000);
        Allocation::from_override(&map)
    }

    #[test]
    fn substitutes_common_placeholders() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: Some("/m/x.gguf"),
            allocation: &alloc,
            static_reserve_mb: Some(6000),
            listen_host: None,
            host_port: 0,
        };
        let out = substitute(
            "python main.py --port {port} --model {model} --gpu {gpu_ids} --vram {reserve_mb}",
            &ctx,
        )
        .unwrap();
        assert_eq!(
            out,
            "python main.py --port 8188 --model /m/x.gguf --gpu 0 --vram 6000"
        );
    }

    /// `{vram_mb}` is the device-specific alias for the device-neutral
    /// `{reserve_mb}`. Command templates in the wild use it, so it has to
    /// keep resolving identically.
    #[test]
    fn legacy_vram_mb_placeholder_still_resolves() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: Some(6000),
            listen_host: None,
            host_port: 0,
        };
        assert_eq!(substitute("{vram_mb}", &ctx).unwrap(), "6000");
        assert_eq!(substitute("{reserve_mb}", &ctx).unwrap(), "6000");
    }

    #[test]
    fn reserve_mb_on_dynamic_fails() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        let err = substitute("--vram {reserve_mb}", &ctx).unwrap_err();
        assert!(matches!(err, SubstituteError::ReserveMbOnDynamic));
    }

    #[test]
    fn gpu_ids_empty_for_cpu_only() {
        let alloc = alloc_cpu_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        let out = substitute("{gpu_ids}", &ctx).unwrap();
        assert_eq!(out, "");
    }

    #[test]
    fn unknown_placeholder_errors() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        let err = substitute("{bogus}", &ctx).unwrap_err();
        assert!(matches!(err, SubstituteError::UnknownPlaceholder(_)));
    }

    #[test]
    fn literal_braces_pass_through() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        // No close brace → literal.
        let out = substitute("prefix {not closed", &ctx).unwrap();
        assert_eq!(out, "prefix {not closed");
    }

    #[test]
    fn double_braces_escape_to_literals() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        // `{{` / `}}` are escapes; the embedded script keeps its braces.
        let out = substitute("print(d[{{'k': 1}}]) on {port}", &ctx).unwrap();
        assert_eq!(out, "print(d[{'k': 1}]) on 8188");
    }

    #[test]
    fn json_arguments_are_not_placeholders() {
        // vLLM and friends take JSON on the command line. A brace group
        // holding something no placeholder could be named passes through
        // verbatim rather than failing the launch.
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8000,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: Some("0.0.0.0"),
            host_port: 40000,
        };
        for json in [
            r#"{"canvas_length": 256}"#,
            r#"{"max_new_tokens": null}"#,
            r#"{"enable_thinking": true}"#,
            r#"{"image": 7}"#,
            r#"{"max_soft_tokens": 1120}"#,
            // Nested, and with a placeholder alongside it.
            r#"{"a": {"b": 1}}"#,
        ] {
            assert_eq!(substitute(json, &ctx).unwrap(), json, "input: {json}");
        }
        assert_eq!(
            substitute(r#"--cfg {"k": 1} --port {port}"#, &ctx).unwrap(),
            r#"--cfg {"k": 1} --port 8000"#
        );

        // A typo is still identifier-shaped, so it still errors.
        assert!(matches!(
            substitute("{prot}", &ctx).unwrap_err(),
            SubstituteError::UnknownPlaceholder(_)
        ));
    }

    #[test]
    fn unmatched_close_brace_terminates() {
        // A `}` with nothing to close used to leave the scanner making no
        // progress, hanging the daemon on config load.
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: 0,
        };
        assert_eq!(substitute("a}b", &ctx).unwrap(), "a}b");
        assert_eq!(substitute("}", &ctx).unwrap(), "}");
        assert_eq!(substitute("}{port}", &ctx).unwrap(), "}8188");
    }
}
