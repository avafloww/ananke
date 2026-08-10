//! Render argv for `Command`-template services: the main `command` and its
//! optional `shutdown_command` sibling, both substituted through the same
//! `{port}` / `{gpu_ids}` / `{reserve_mb}` / `{model}` / `{name}` placeholder
//! rules.

use std::collections::BTreeMap;

use ananke_config::validate::{ServiceConfig, TemplateConfig};
use ananke_devices::{Allocation, cuda_env};

use crate::spawn::SpawnConfig;

/// Assemble the [`PlaceholderContext`] a command-template argv renders
/// against. Shared by spawn-time and shutdown-time so both paths resolve
/// `{port}` / `{gpu_ids}` / `{reserve_mb}` / `{name}` identically.
fn placeholder_context<'a>(
    svc: &'a ServiceConfig,
    alloc: &'a Allocation,
) -> ananke_templates::PlaceholderContext<'a> {
    use ananke_config::AllocationMode;
    let static_reserve_mb = match svc.allocation_mode {
        AllocationMode::Static { reserve_mb } => Some(reserve_mb),
        _ => None,
    };
    ananke_templates::PlaceholderContext {
        name: &svc.name,
        port: svc.private_port,
        // Command template has no model path; {model} resolves to empty.
        model: None,
        allocation: alloc,
        static_reserve_mb,
    }
}

/// Render any command-template argv (the main `command` or the sibling
/// `shutdown_command`) under the same substitution rules. Hard-fails on
/// substitution errors — callers surface them as `StartFailure` /
/// shutdown-run warnings rather than launching with literal
/// `{placeholder}` tokens in argv.
fn render_command_like(
    argv: &[String],
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Result<SpawnConfig, ananke_templates::SubstituteError> {
    let binary = argv.first().cloned().unwrap_or_default();
    let tail: Vec<String> = argv.iter().skip(1).cloned().collect();

    let ctx = placeholder_context(svc, alloc);
    let user_env: BTreeMap<String, String> = svc.env.clone();
    let (args, env_substituted) = ananke_templates::substitute_argv(&tail, &user_env, &ctx)?;

    let mut env = BTreeMap::new();
    for (k, v) in env_substituted {
        env.insert(k, v);
    }
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    Ok(SpawnConfig {
        binary,
        args,
        env,
        env_inherit: svc.env_inherit,
    })
}

/// Render argv for the optional `shutdown_command` sibling of a
/// command-template service, if one is configured. Returns `None` when
/// the service has no shutdown command or isn't a command-template
/// service. Propagates substitution errors so the caller logs them
/// (instead of launching the shutdown with unresolved `{placeholder}`s).
pub fn render_shutdown_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Option<Result<SpawnConfig, ananke_templates::SubstituteError>> {
    let TemplateConfig::Command(cmd_cfg) = &svc.template_config else {
        return None;
    };
    let argv = cmd_cfg.shutdown_command.as_ref()?;
    if argv.is_empty() {
        return None;
    }
    Some(render_command_like(argv, svc, alloc))
}

/// Render argv for a `Command`-template service. Substitutes `{port}`,
/// `{gpu_ids}`, `{reserve_mb}`, `{model}`, `{name}`.
pub(super) fn render_command_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Result<SpawnConfig, ananke_templates::SubstituteError> {
    let TemplateConfig::Command(cmd_cfg) = &svc.template_config else {
        unreachable!("render_command_argv called on non-command service")
    };
    render_command_like(&cmd_cfg.command, svc, alloc)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ananke_config::validate::{
        AllocationMode, DeviceSlot, PlacementPolicy, test_fixtures::minimal_command_service,
    };
    use ananke_devices::Allocation;

    use crate::spawn::render_argv;

    #[test]
    fn command_template_renders_placeholders() {
        let command_argv = vec![
            "python".into(),
            "main.py".into(),
            "--port".into(),
            "{port}".into(),
        ];
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 6144);
        let mut svc = minimal_command_service("comfy", command_argv);
        svc.port = 8188;
        svc.private_port = 48188;
        svc.placement_override = placement.clone();
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.allocation_mode = AllocationMode::Static { reserve_mb: 6144 };
        let alloc = Allocation::from_override(&placement);
        let cfg = render_argv(&svc, &alloc, None).unwrap();
        assert_eq!(cfg.binary, "python");
        assert!(
            cfg.args.iter().any(|a| a == "48188"),
            "expected port substituted; got {:?}",
            cfg.args
        );
        assert!(
            cfg.args.iter().all(|a| a != "{port}"),
            "raw placeholder leaked into args: {:?}",
            cfg.args
        );
    }
}
