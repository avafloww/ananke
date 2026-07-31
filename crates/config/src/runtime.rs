//! Which llama.cpp fork serves — or served — a model.
//!
//! The two forks size the graph arena by different rules, so the fork is a
//! factor on both sides of the project: the daemon spawns one of two binaries
//! with two flag dialects, and the calibration records which one produced a
//! measured row. It lives here so those two sides name the same fact once.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A llama.cpp fork.
///
/// Guardrail: the serde spellings are a wire. They are the `runtime` column of
/// `calibration/data/measurements.ndjson` and part of the payload a cell's
/// identity is hashed over, so renaming a variant re-keys the whole campaign;
/// `fork_wire_spelling_is_pinned` asserts them. The daemon's TOML spellings are
/// *not* these — see `ananke::config::parse::RawRuntime`, which is tagged
/// `kind = "llama-cpp" | "ik-llama"` and is a separate vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    /// Upstream llama.cpp.
    #[default]
    Mainline,
    /// ikawrakow's ik_llama.cpp fork.
    Ik,
}

impl Runtime {
    /// How the fork is spelled in a measurement record, in a cell's label, and
    /// in every report keyed on it — deliberately the same word in all three.
    pub fn name(self) -> &'static str {
        match self {
            Runtime::Mainline => "mainline",
            Runtime::Ik => "ik",
        }
    }
}

impl fmt::Display for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_wire_spelling_is_pinned() {
        for variant in [Runtime::Mainline, Runtime::Ik] {
            // Completeness: a new variant is a compile error here until it is
            // listed above and given a name.
            match variant {
                Runtime::Mainline | Runtime::Ik => {}
            }
            assert_eq!(
                serde_json::to_string(&variant).expect("a unit variant serializes"),
                format!("\"{}\"", variant.name())
            );
        }
        assert_eq!(Runtime::Mainline.name(), "mainline");
        assert_eq!(Runtime::Ik.name(), "ik");
    }
}
