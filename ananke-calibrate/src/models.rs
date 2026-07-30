//! The production model configurations, read from `models.toml`.
//!
//! Each entry mirrors what a `ServiceConfig` would carry for that model, so an
//! estimate taken here is the one the daemon would produce. The file is
//! gitignored because its paths are system-specific; `models.toml.example`
//! documents the shape.

use std::path::{Path, PathBuf};

use ananke_config::placement::{OffloadMode, PlacementInputs, PlacementPolicy, SplitMode};
use ananke_estimate::EstimatorInputs;
use serde::Deserialize;

/// Where the model files live, relative to which `ModelConfig::model` is
/// resolved. The campaign machine keeps them all under one root.
pub const MODEL_ROOT: &str = "/mnt/ssd0/ai/llm";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsFile {
    #[serde(default)]
    pub model: Vec<ModelConfig>,
}

/// One production model's configuration.
///
/// `deny_unknown_fields` deliberately: a mistyped key here would otherwise be
/// dropped in silence, and the estimate would come out confidently wrong. That is
/// exactly what happened during the port — the file says `mtp = true` and this
/// struct first declared `spec_type`, so MTP was ignored for three models and the
/// scoreboard read 16% low on one of them.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub draft_model: Option<String>,
    pub context: u32,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub parallel: Option<u32>,
    #[serde(default)]
    pub flash_attn: Option<bool>,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    #[serde(default)]
    pub cache_ram_mb: Option<u32>,
    #[serde(default)]
    pub split_mode: Option<String>,
    #[serde(default)]
    pub visible_devices: Option<u32>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub expert_offload: Option<String>,
    #[serde(default)]
    pub ik_llama: bool,
    #[serde(default)]
    pub ik_dsa: bool,
    /// Whether the service runs `--spec-type draft-mtp`.
    #[serde(default)]
    pub mtp: bool,
    /// Whether the operator has opted into the coarse fallback for an
    /// architecture no family estimator recognises.
    #[serde(default)]
    pub allow_fallback: bool,
    #[serde(default)]
    pub gpus: Option<Vec<u32>>,
}

impl ModelConfig {
    /// The GGUF's absolute path.
    pub fn model_path(&self) -> PathBuf {
        Path::new(MODEL_ROOT).join(&self.model)
    }

    /// The vision projector's absolute path, if this model has one.
    pub fn mmproj_path(&self) -> Option<PathBuf> {
        self.mmproj.as_ref().map(|p| Path::new(MODEL_ROOT).join(p))
    }

    /// The separate draft GGUF's absolute path, if this model has one.
    pub fn draft_path(&self) -> Option<PathBuf> {
        self.draft_model
            .as_ref()
            .map(|p| Path::new(MODEL_ROOT).join(p))
    }

    /// How many cards this model spans.
    pub fn cards(&self) -> u32 {
        self.visible_devices
            .or_else(|| self.gpus.as_ref().map(|g| g.len() as u32))
            .unwrap_or(2)
            .max(1)
    }

    /// The split this model runs under.
    pub fn split(&self) -> SplitMode {
        match self.split_mode.as_deref() {
            Some("tensor") => SplitMode::Tensor,
            Some("row") => SplitMode::Row,
            _ => SplitMode::Layer,
        }
    }

    /// How much of a mixture of experts is host-resident.
    pub fn offload(&self) -> OffloadMode {
        match (self.n_cpu_moe, self.expert_offload.as_deref()) {
            (Some(n), _) => OffloadMode::Layers(n),
            (None, Some("auto")) => OffloadMode::Auto,
            _ => OffloadMode::Off,
        }
    }

    /// The estimator inputs for this model.
    ///
    /// `model`, `mmproj`, and `draft` are passed in because `EstimatorInputs`
    /// borrows its paths and they must outlive the returned value.
    pub fn estimator_inputs<'a>(
        &'a self,
        model: &'a Path,
        mmproj: Option<&'a Path>,
        draft: Option<&'a Path>,
    ) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: &self.name,
            model,
            mmproj,
            context: self.context,
            ubatch: self.ubatch,
            visible_devices: self.cards(),
            host_resident_experts: self.offload().is_enabled(),
            split_mode: self.split(),
            cache_type_k: self.cache_type_k.as_deref(),
            cache_type_v: self.cache_type_v.as_deref(),
            override_tensor: &[],
            compute_buffer_mb: None,
            allow_fallback: self.allow_fallback,
            mtp: self.mtp,
            draft_model: draft,
            ik_llama: self.ik_llama,
            ik_dsa: self.ik_dsa,
            parallel: self.parallel,
            flash_attn: self.flash_attn,
            kv_unified: self.kv_unified,
            cache_ram_mb: self.cache_ram_mb,
        }
    }

    /// The placement for this model.
    pub fn placement_inputs(&self) -> PlacementInputs {
        PlacementInputs {
            policy: if self.offload().is_enabled() {
                PlacementPolicy::Hybrid
            } else {
                PlacementPolicy::GpuOnly
            },
            split_mode: self.split(),
            gpu_allow: self
                .gpus
                .clone()
                .unwrap_or_else(|| (0..self.cards()).collect()),
            expert_offload: self.offload(),
            ik_llama: self.ik_llama,
            ..PlacementInputs::named(&self.name)
        }
    }
}

/// Read `models.toml`.
pub fn load(path: &Path) -> Result<Vec<ModelConfig>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "{}: {e} — copy models.toml.example and edit the paths",
            path.display()
        )
    })?;
    let parsed: ModelsFile =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parsed.model)
}
