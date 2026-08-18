//! Validate a config file offline and render what each service would
//! launch: parse, resolve inheritance, validate, then for every
//! containerized service render the exact runtime `create` argv.
//!
//! `anankectl server-config validate` does the parse-and-validate half over
//! the management API, which requires a daemon already running with a config
//! good enough to have started. This is the version for the config you have
//! not deployed yet — a generated one in particular, where the first sign of
//! a mistake would otherwise be the daemon refusing to boot, or worse, a
//! container coming up bound to the wrong endpoint or unable to find its
//! weights.
//!
//! ```sh
//! cargo run -p ananke-supervise --example validate-config -- /path/to/ananke.toml
//! ```
//!
//! The reservation sizes are *not* what this checks: rendering uses a
//! nominal allocation rather than running the estimator, because the
//! estimator needs the GGUFs. What it does check is the shape of the launch
//! — image, entrypoint, argv, mount translation, publication, devices.

use std::{collections::BTreeMap, path::PathBuf, process::ExitCode};

use ananke_config::{
    parse_toml, resolve_inheritance,
    validate::{ContainerNetwork, DeviceSlot, ServiceConfig, TemplateConfig, validate},
};
use ananke_devices::Allocation;
use ananke_supervise::spawn::container::render_container_spec;

/// Stand-in reservation for services whose real one comes from the
/// estimator. Only the device set reaches the rendered argv (through CDI
/// expansion and `CUDA_VISIBLE_DEVICES`), so the size is immaterial here.
const NOMINAL_RESERVE_MB: u64 = 1_024;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: validate-config <path-to-ananke.toml>");
        return ExitCode::from(2);
    };

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };

    let mut raw = match parse_toml(&source, &path) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    if let Err(e) = resolve_inheritance(&mut raw) {
        return fail(e);
    }
    let effective = match validate(&raw) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };

    println!(
        "{}: {} service(s)\n",
        path.display(),
        effective.services.len()
    );
    let mut failed = false;
    for svc in &effective.services {
        if let Err(e) = describe(svc) {
            eprintln!("  !! {}: {e}\n", svc.name);
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn describe(svc: &ServiceConfig) -> Result<(), Box<dyn std::error::Error>> {
    let template = match &svc.template_config {
        TemplateConfig::LlamaCpp(_) => "llama-cpp",
        TemplateConfig::Command(_) => "command",
    };
    println!(
        "{} [{template}]  :{} -> :{}",
        svc.name, svc.port, svc.private_port
    );

    let Some(container) = &svc.container else {
        println!("  host process\n");
        return Ok(());
    };

    let alloc = nominal_allocation(svc);
    let spec = render_container_spec(svc, &alloc, None, 0, "00000000-0000-4000-8000-000000000000")?;
    let executable = spec
        .runtime_executable
        .clone()
        .unwrap_or_else(|| spec.runtime.executable().to_string());
    let argv = ananke_system::container::render_create_argv(&executable, &spec);

    println!(
        "  {} {} ({} net{})",
        container.runtime.as_str(),
        container.image,
        match container.network {
            ContainerNetwork::Bridge => "bridge",
            ContainerNetwork::Host => "host",
        },
        match (spec.host_port, spec.container_port) {
            (Some(h), Some(c)) => format!(", publishes 127.0.0.1:{h} -> {c}"),
            _ => String::new(),
        }
    );
    println!("  create: {}", argv.join(" "));
    println!();
    Ok(())
}

/// Build the allocation a render needs. An explicit `placement_override`
/// is used verbatim; otherwise the service's first allowed GPU stands in
/// for whatever the estimator would have picked.
fn nominal_allocation(svc: &ServiceConfig) -> Allocation {
    if !svc.placement_override.is_empty() {
        return Allocation::from_override(&svc.placement_override);
    }
    let mut map = BTreeMap::new();
    let gpu = svc.gpu_allow.first().copied().unwrap_or(0);
    map.insert(DeviceSlot::Gpu(gpu), NOMINAL_RESERVE_MB);
    Allocation::from_override(&map)
}

fn fail(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("invalid: {e}");
    ExitCode::FAILURE
}
