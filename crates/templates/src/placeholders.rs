//! Substitute `${port}`/`${listen_port}`, `${listen_host}`, `${host_port}`,
//! `${gpu_ids}`, `${reserve_mb}`, `${model}`, `${name}` in command-template
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
    /// `${` with no closing brace. Left as a literal it would silently
    /// reach the argv; a placeholder the author meant to write is more
    /// likely than a payload containing a bare `${`.
    UnterminatedPlaceholder(String),
    /// The launcher splat `{args}` must occupy a launcher entry on its
    /// own; it cannot be embedded inside a larger argv string because
    /// the expansion produces multiple arguments, not a single one.
    SplatInsideArg,
}

impl std::fmt::Display for SubstituteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstituteError::ReserveMbOnDynamic => {
                write!(f, "`${{reserve_mb}}` is invalid with a dynamic allocation")
            }
            SubstituteError::ReserveMbMultiDevice => {
                write!(
                    f,
                    "`${{reserve_mb}}` is valid only with a single-device static allocation"
                )
            }
            SubstituteError::UnknownPlaceholder(s) => {
                write!(f, "unknown placeholder `${{{s}}}`")
            }
            SubstituteError::UnterminatedPlaceholder(s) => {
                write!(
                    f,
                    "unterminated placeholder near `{s}`: expected a closing `}}`, \
                     or write `$$` for a literal dollar"
                )
            }
            SubstituteError::SplatInsideArg => {
                write!(
                    f,
                    "splat placeholder `${{args}}` must be the entire launcher entry, \
                     not embedded inside a larger string"
                )
            }
        }
    }
}

impl std::error::Error for SubstituteError {}

/// Substitute every `${placeholder}` in `input` using `ctx`. Returns a
/// fresh owned String.
///
/// Only `${` and `$$` are special. A bare `{` never is, so an argument
/// carrying JSON, a Jinja template, or a Python format string passes
/// through untouched — the parser is never asked to guess which braces
/// belong to ananke and which to the program being launched. Guessing is
/// what made the previous `{name}` grammar wrong for those payloads.
///
/// - `${name}` resolves a placeholder. An unknown name is an error, so a
///   typo surfaces instead of reaching the argv.
/// - `$$` is a literal `$`, which is how a literal `${...}` is written.
/// - `$` before anything else is itself, since payloads carry bare dollars
///   far more often than they carry `${`.
/// - `${` with no closing brace is an error rather than a silent literal.
pub fn substitute(input: &str, ctx: &PlaceholderContext<'_>) -> Result<String, SubstituteError> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while !rest.is_empty() {
        // Copy everything up to the next `$`; none of it can be special.
        let Some(at) = rest.find('$') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        rest = &rest[at..];

        if let Some(after) = rest.strip_prefix("$$") {
            out.push('$');
            rest = after;
            continue;
        }
        if let Some(after) = rest.strip_prefix("${") {
            let Some(close) = after.find('}') else {
                return Err(SubstituteError::UnterminatedPlaceholder(
                    rest.chars().take(24).collect(),
                ));
            };
            out.push_str(&resolve(&after[..close], ctx)?);
            rest = &after[close + 1..];
            continue;
        }
        // A `$` that opens nothing.
        out.push('$');
        rest = &rest[1..];
    }
    Ok(out)
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
/// `${args}` placeholder to the full list of llama-server flags ananke
/// would otherwise have emitted. `${args}` must occupy a launcher entry
/// on its own — `["--foo=${args}"]` is rejected because the expansion
/// produces multiple argv entries, not a single one.
///
/// Every other launcher entry passes through [`substitute`], so the
/// usual placeholders (`${model}`, `${port}`, `${name}`, `${gpu_ids}`) are
/// resolved in-place.
pub fn substitute_launcher_argv(
    launcher: &[String],
    llama_args: &[String],
    ctx: &PlaceholderContext<'_>,
) -> Result<Vec<String>, SubstituteError> {
    let mut out: Vec<String> = Vec::with_capacity(launcher.len() + llama_args.len());
    for entry in launcher {
        if entry == "${args}" {
            out.extend(llama_args.iter().cloned());
            continue;
        }
        if entry.contains("${args}") {
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
        map.insert(DeviceSlot::Cpu, 6000);
        Allocation::from_override(&map)
    }

    fn ctx<'a>(alloc: &'a Allocation, reserve: Option<u64>) -> PlaceholderContext<'a> {
        PlaceholderContext {
            name: "demo",
            port: 8188,
            model: Some("/m/x.gguf"),
            allocation: alloc,
            static_reserve_mb: reserve,
            listen_host: None,
            host_port: 40000,
        }
    }

    #[test]
    fn substitutes_every_placeholder() {
        let alloc = alloc_gpu0_only();
        let out = substitute(
            "python main.py --port ${port} --model ${model} --gpu ${gpu_ids} --vram ${reserve_mb}",
            &ctx(&alloc, Some(6000)),
        )
        .unwrap();
        assert_eq!(
            out,
            "python main.py --port 8188 --model /m/x.gguf --gpu 0 --vram 6000"
        );
    }

    #[test]
    fn endpoint_placeholders_resolve() {
        let alloc = alloc_gpu0_only();
        let mut c = ctx(&alloc, None);
        assert_eq!(
            substitute("${listen_host}:${listen_port}", &c).unwrap(),
            "127.0.0.1:8188"
        );
        assert_eq!(substitute("${host_port}", &c).unwrap(), "40000");
        c.listen_host = Some("0.0.0.0");
        assert_eq!(substitute("${listen_host}", &c).unwrap(), "0.0.0.0");
    }

    #[test]
    fn legacy_vram_mb_placeholder_still_resolves() {
        let alloc = alloc_gpu0_only();
        let c = ctx(&alloc, Some(6000));
        assert_eq!(substitute("${vram_mb}", &c).unwrap(), "6000");
        assert_eq!(substitute("${reserve_mb}", &c).unwrap(), "6000");
    }

    #[test]
    fn reserve_mb_on_dynamic_fails() {
        let alloc = alloc_gpu0_only();
        assert!(matches!(
            substitute("${reserve_mb}", &ctx(&alloc, None)).unwrap_err(),
            SubstituteError::ReserveMbOnDynamic
        ));
    }

    #[test]
    fn gpu_ids_empty_for_cpu_only() {
        let alloc = alloc_cpu_only();
        assert_eq!(substitute("${gpu_ids}", &ctx(&alloc, None)).unwrap(), "");
    }

    #[test]
    fn unknown_placeholder_errors() {
        let alloc = alloc_gpu0_only();
        // A typo is identifier-shaped and must not reach the argv.
        assert!(matches!(
            substitute("${prot}", &ctx(&alloc, None)).unwrap_err(),
            SubstituteError::UnknownPlaceholder(_)
        ));
    }

    #[test]
    fn braces_are_never_special() {
        // The whole point of the grammar: a payload's own braces are its
        // own. No JSON, Jinja, or format string needs escaping.
        let alloc = alloc_gpu0_only();
        let c = ctx(&alloc, None);
        for input in [
            r#"{"canvas_length": 256}"#,
            r#"{"a": {"b": 1}}"#,
            r#"{"max_new_tokens": null}"#,
            "{% for m in messages %}{{ m.role }}{% endfor %}",
            // A bare brace group is now just text.
            "{port}",
            "}",
            "{",
            "{{",
            "}}",
            "{a{b}c}",
        ] {
            assert_eq!(substitute(input, &c).unwrap(), input, "input: {input:?}");
        }
        // And alongside a real placeholder.
        assert_eq!(
            substitute(r#"--cfg {"k": 1} --port ${port}"#, &c).unwrap(),
            r#"--cfg {"k": 1} --port 8188"#
        );
    }

    #[test]
    fn dollars_that_open_nothing_are_literal() {
        let alloc = alloc_gpu0_only();
        let c = ctx(&alloc, None);
        for input in ["$", "cost: $5", "a$b", "$PATH", "$"] {
            assert_eq!(substitute(input, &c).unwrap(), input, "input: {input:?}");
        }
    }

    #[test]
    fn double_dollar_escapes_a_placeholder() {
        // How a literal `${port}` is written.
        let alloc = alloc_gpu0_only();
        let c = ctx(&alloc, None);
        assert_eq!(substitute("$${port}", &c).unwrap(), "${port}");
        assert_eq!(substitute("$$", &c).unwrap(), "$");
        // Each `$$` yields one `$`; the `{port}` left over is literal.
        assert_eq!(substitute("$$$${port}", &c).unwrap(), "$${port}");
        // `$$` then `${port}` is a literal dollar followed by a *resolved*
        // placeholder — escaping covers the `$`, not what follows it.
        assert_eq!(substitute("${port}$$${port}", &c).unwrap(), "8188$8188");
    }

    #[test]
    fn an_unterminated_placeholder_is_an_error() {
        // Silently literal would put `${port` in the argv, which is never
        // what the author meant.
        let alloc = alloc_gpu0_only();
        assert!(matches!(
            substitute("--port ${port", &ctx(&alloc, None)).unwrap_err(),
            SubstituteError::UnterminatedPlaceholder(_)
        ));
    }

    #[test]
    fn scanning_terminates_on_anything() {
        // Every branch consumes at least one byte, and slicing is on
        // `$`/`}` byte positions, so multi-byte text cannot split a char.
        let alloc = alloc_gpu0_only();
        let c = ctx(&alloc, None);
        for input in ["$${", "${", "$", "", "é$é", "${é}", "日本$$語", "$$$"] {
            let _ = substitute(input, &c);
        }
        assert_eq!(substitute("é$$é", &c).unwrap(), "é$é");
    }
}
