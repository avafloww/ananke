//! The template-tagged `RawService` enum that dispatches between the
//! llama-cpp and command service variants, plus the shared accessors both
//! variants expose through it.

use serde::Deserialize;

use crate::config::parse::{
    DEFAULT_START_QUEUE_DEPTH, RawCommandService, RawLlamaCppService, RawServiceCommon,
};

/// Template-tagged service: the `template = "llama-cpp" | "command"` field
/// selects a variant. Each variant flattens `RawServiceCommon` so all shared
/// fields appear at the top level of the service table in TOML.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "template", rename_all = "kebab-case")]
pub enum RawService {
    /// Both variants are boxed: each inner struct flattens
    /// `RawServiceCommon` and so runs to several hundred bytes, and the
    /// llama-cpp side carries ~1 KiB of optional knobs on top. Boxing
    /// keeps `RawService` pointer-sized and the two variants uniform,
    /// mirroring the boxing on the validated
    /// [`crate::config::validate::TemplateConfig`].
    LlamaCpp(Box<RawLlamaCppService>),
    Command(Box<RawCommandService>),
}

impl RawService {
    pub fn common(&self) -> &RawServiceCommon {
        match self {
            RawService::LlamaCpp(s) => &s.common,
            RawService::Command(s) => &s.common,
        }
    }

    pub fn common_mut(&mut self) -> &mut RawServiceCommon {
        match self {
            RawService::LlamaCpp(s) => &mut s.common,
            RawService::Command(s) => &mut s.common,
        }
    }

    pub fn template_label(&self) -> &'static str {
        match self {
            RawService::LlamaCpp(_) => "llama-cpp",
            RawService::Command(_) => "command",
        }
    }

    /// Return the start queue depth, falling back to `DEFAULT_START_QUEUE_DEPTH`
    /// when unset.
    pub fn start_queue_depth(&self) -> usize {
        self.common()
            .start_queue_depth
            .unwrap_or(DEFAULT_START_QUEUE_DEPTH)
    }
}
