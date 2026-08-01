//! The `general.architecture` values ananke recognises.
//!
//! llama.cpp identifies a model's graph by this one string, and everything
//! ananke estimates depends on which graph it is. Parsing it once at the
//! reader, into a type whose variants are the architectures the estimator has
//! actually been calibrated against, means a name it does not know cannot
//! reach a per-family estimator by accident — it arrives as
//! [`Architecture::Unknown`], and the estimator refuses rather than guessing.
//!
//! Adding a variant here is not enough to support an architecture. See the
//! family modules in `ananke-estimate`, which say which graph each variant's
//! tensors and metadata are read as.

use std::fmt;

use smol_str::SmolStr;

/// A model's `general.architecture`.
///
/// [`Architecture::Unknown`] carries the name it was given so a diagnostic can
/// print it. It is deliberately not `PartialEq`-equal to any known variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Architecture {
    Llama,
    Llama4,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen3VlMoe,
    Qwen35,
    Qwen35Moe,
    Mistral,
    Mixtral,
    Gemma,
    Gemma2,
    Gemma3,
    Gemma3n,
    Gemma4,
    Phi3,
    Glm4,
    Glm4Moe,
    GlmDsa,
    DeepSeek2,
    DeepSeek4,
    GptOss,
    /// NVIDIA Nemotron.
    Deci,
    Talkie,
    /// LiquidAI LFM2/LFM2.5.
    Lfm2,
    Laguna,
    Mamba,
    Jamba,
    /// A vision projector, which is not a model in its own right — read for
    /// its tensor bytes and never dispatched to a family estimator.
    Clip,
    /// A name no variant covers. The estimator refuses these; the operator
    /// declares the reservation explicitly instead.
    Unknown(SmolStr),
}

impl Architecture {
    /// The `general.architecture` string this variant was parsed from.
    ///
    /// Round-trips: `Architecture::from(a.as_str()) == a` for every value.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Llama => "llama",
            Self::Llama4 => "llama4",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3VlMoe => "qwen3vlmoe",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
            Self::Gemma3n => "gemma3n",
            Self::Gemma4 => "gemma4",
            Self::Phi3 => "phi3",
            Self::Glm4 => "glm4",
            Self::Glm4Moe => "glm4moe",
            Self::GlmDsa => "glm-dsa",
            Self::DeepSeek2 => "deepseek2",
            Self::DeepSeek4 => "deepseek4",
            Self::GptOss => "gpt-oss",
            Self::Deci => "deci",
            Self::Talkie => "talkie",
            Self::Lfm2 => "lfm2",
            Self::Laguna => "laguna",
            Self::Mamba => "mamba",
            Self::Jamba => "jamba",
            Self::Clip => "clip",
            Self::Unknown(name) => name,
        }
    }

    /// Whether this is a name no variant covers.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Every variant but [`Architecture::Unknown`], for callers that need to
    /// enumerate what is recognised — a diagnostic listing them, or a test
    /// holding a table of per-architecture data to the same set.
    pub fn known() -> &'static [Architecture] {
        use Architecture::*;
        &[
            Llama, Llama4, Qwen2, Qwen3, Qwen3Moe, Qwen3VlMoe, Qwen35, Qwen35Moe, Mistral, Mixtral,
            Gemma, Gemma2, Gemma3, Gemma3n, Gemma4, Phi3, Glm4, Glm4Moe, GlmDsa, DeepSeek2,
            DeepSeek4, GptOss, Deci, Talkie, Lfm2, Laguna, Mamba, Jamba, Clip,
        ]
    }
}

impl From<&str> for Architecture {
    fn from(name: &str) -> Self {
        Self::known()
            .iter()
            .find(|known| known.as_str() == name)
            .cloned()
            .unwrap_or_else(|| Self::Unknown(SmolStr::new(name)))
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips_through_its_name() {
        for arch in Architecture::known() {
            assert_eq!(&Architecture::from(arch.as_str()), arch);
        }
    }

    /// Two variants sharing a name would make dispatch depend on declaration
    /// order, and the loser would be unreachable.
    #[test]
    fn names_are_distinct() {
        let mut names: Vec<&str> = Architecture::known()
            .iter()
            .map(Architecture::as_str)
            .collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate architecture name");
    }

    #[test]
    fn an_unrecognised_name_is_unknown_and_keeps_it() {
        let arch = Architecture::from("no-such-arch");
        assert!(arch.is_unknown());
        assert_eq!(arch.as_str(), "no-such-arch");
        assert_ne!(arch, Architecture::Llama);
    }
}
