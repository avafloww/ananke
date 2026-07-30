//! Placement and lifecycle vocabulary: which template a service uses, how it
//! claims memory, when it runs, and the global device reserves it sees.

use std::collections::BTreeMap;

use crate::config::validate::gib_to_mib;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    LlamaCpp,
    Command,
}

impl Template {
    pub fn as_str(self) -> &'static str {
        match self {
            Template::LlamaCpp => "llamacpp",
            Template::Command => "command",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    /// Llama-cpp services: placement decided by estimator/override; mode absent.
    None,
    /// A fixed reservation. Named device-neutrally because it lands on the
    /// CPU device for a cpu-only command service just as readily as on a GPU.
    Static { reserve_mb: u64 },
    Dynamic {
        min_mb: u64,
        max_mb: u64,
        min_borrower_runtime_ms: u64,
    },
}

impl AllocationMode {
    /// Resolve an allocation mode from a `(template, mode)` pair plus the
    /// associated reservation knobs. Shared by the TOML validator and the
    /// oneshot API so both paths agree on the semantics of `"static"`,
    /// `"dynamic"`, and the llama-cpp exclusions.
    ///
    /// The returned error is a bare sentence fragment; the caller is
    /// expected to prepend context (e.g. `service {name}: `).
    pub fn from_parts(
        template: Template,
        mode: Option<&str>,
        reserve_gb: Option<f32>,
        min_reserve_gb: Option<f32>,
        max_reserve_gb: Option<f32>,
        min_borrower_runtime_ms: u64,
    ) -> Result<AllocationMode, String> {
        match (template, mode) {
            (Template::LlamaCpp, Some(m)) => Err(format!(
                "allocation.mode `{m}` invalid for llama-cpp (use placement_override or estimator)"
            )),
            (Template::LlamaCpp, None) => Ok(AllocationMode::None),
            (Template::Command, Some("static")) => {
                let gb = reserve_gb
                    .ok_or_else(|| "allocation.mode=static requires reserve_gb".to_string())?;
                Ok(AllocationMode::Static {
                    reserve_mb: gib_to_mib(gb),
                })
            }
            (Template::Command, Some("dynamic")) => {
                let min = min_reserve_gb
                    .ok_or_else(|| "allocation.mode=dynamic requires min_reserve_gb".to_string())?;
                let max = max_reserve_gb
                    .ok_or_else(|| "allocation.mode=dynamic requires max_reserve_gb".to_string())?;
                if max <= min {
                    return Err("max_reserve_gb must be > min_reserve_gb".to_string());
                }
                Ok(AllocationMode::Dynamic {
                    min_mb: gib_to_mib(min),
                    max_mb: gib_to_mib(max),
                    min_borrower_runtime_ms,
                })
            }
            (Template::Command, Some(other)) => Err(format!("unknown allocation.mode `{other}`")),
            (Template::Command, None) => {
                Err("command template requires allocation.mode (static|dynamic)".to_string())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Persistent,
    OnDemand,
}

impl Lifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::Persistent => "persistent",
            Lifecycle::OnDemand => "ondemand",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Filters {
    pub strip_params: Vec<String>,
    pub set_params: BTreeMap<String, serde_json::Value>,
}

pub use ananke_config::placement::{DeviceReserves, PlacementPolicy};

#[derive(Debug, Clone)]
pub struct HealthSettings {
    /// HTTP path to probe for readiness. `None` means no health check —
    /// the service transitions to Running immediately after spawn.
    pub http_path: Option<String>,
    pub timeout_ms: u64,
    pub probe_interval_ms: u64,
}

pub use ananke_config::placement::DeviceSlot;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::{
        parse::parse_toml,
        validate::{test_fixtures::parse_and_merge, validate},
    };

    #[test]
    fn rejects_oneshot_lifecycle_in_service_block() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "oneshot"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("oneshot"));
    }

    #[test]
    fn phase2_accepts_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn default_lifecycle_is_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn parses_filters() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
filters.strip_params = ["temperature"]
filters.set_params = { max_tokens = 4096 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let s = &ec.services[0];
        assert_eq!(s.filters.strip_params, vec!["temperature"]);
        assert!(s.filters.set_params.contains_key("max_tokens"));
    }

    #[test]
    fn parses_idle_timeout() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
idle_timeout = "5m"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].idle_timeout_ms, 300_000);
    }
    #[test]
    fn command_template_with_static_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.reserve_gb = 6
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert_eq!(svc.template(), Template::Command);
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Static { reserve_mb: 6144 }
        ));
    }

    /// `vram_gb` / `min_vram_gb` / `max_vram_gb` were the names before the
    /// reservation was recognised as device-neutral. Configs on disk still use
    /// them, so they have to keep parsing to the same allocation.
    #[test]
    fn legacy_vram_gb_keys_still_parse() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "static-legacy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.vram_gb = 6

[[service]]
name = "dynamic-legacy"
template = "command"
command = ["python", "main.py"]
port = 8189
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_vram_gb = 4
allocation.max_vram_gb = 20
"#,
        );
        let ec = validate(&cfg).unwrap();
        let mode = |name: &str| {
            ec.services
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("service {name} must be present"))
                .allocation_mode
        };
        assert!(matches!(
            mode("static-legacy"),
            AllocationMode::Static { reserve_mb: 6144 }
        ));
        assert!(matches!(
            mode("dynamic-legacy"),
            AllocationMode::Dynamic {
                min_mb: 4096,
                max_mb: 20480,
                ..
            }
        ));
    }

    #[test]
    fn command_template_with_dynamic_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_reserve_gb = 4
allocation.max_reserve_gb = 20
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Dynamic {
                min_mb: 4096,
                max_mb: 20480,
                ..
            }
        ));
    }
    #[test]
    fn llama_cpp_allocation_mode_rejected_at_parse() {
        // With a tagged enum, `allocation` isn't a field on the llama-cpp
        // variant; serde rejects it before the validator runs.
        let res = parse_toml(
            r#"
[[service]]
name = "llama"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
allocation.mode = "static"
allocation.reserve_gb = 4
"#,
            Path::new("/t"),
        );
        assert!(
            res.is_err(),
            "expected parse error for allocation on llama-cpp; got {:?}",
            res.ok()
        );
    }
    #[test]
    fn dynamic_rejects_max_le_min() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
allocation.mode = "dynamic"
allocation.min_reserve_gb = 10
allocation.max_reserve_gb = 5
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("max_reserve_gb"));
    }
}
