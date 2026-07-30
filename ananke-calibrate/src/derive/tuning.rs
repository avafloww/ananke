//! Reading the constants the estimator already compiles in.
//!
//! Several derivers need a constant they do not derive: the arena model charges
//! ik's MoE rate, the flash-attention residual subtracts the E-variant's
//! per-layer term, and the baseline offset subtracts the whole process-baseline
//! model. Those were copied by hand once, and the copy went stale the moment the
//! constant was re-derived — inflating every residual computed for an ik mixture
//! of experts until it was noticed.
//!
//! So they are read instead: `tuning.json` is the source of truth for the Rust
//! estimator already, and there is no reason for the analysis to hold its own
//! opinion. A default applies only before the constant exists, which is the
//! bootstrap case when a new term is being added.

use serde_json::Value;

/// The committed tuning document, as the derivers see it.
#[derive(Debug, Clone)]
pub struct Tuning {
    document: Value,
}

/// ik's per-architecture MoE rate to fall back on before the table exists.
const IK_MOE_DEFAULT: i64 = 54;

impl Tuning {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        Ok(Self {
            document: serde_json::from_str(text)?,
        })
    }

    /// The document itself, for `emit` to mutate and write back.
    pub fn document(&self) -> &Value {
        &self.document
    }

    /// A constant's value as an integer, or `default` if it is not there yet.
    pub fn constant(&self, name: &str, default: i64) -> i64 {
        self.constant_f64(name, default as f64) as i64
    }

    /// A constant's value as a float.
    pub fn constant_f64(&self, name: &str, default: f64) -> f64 {
        self.document
            .get("constants")
            .and_then(|c| c.get(name))
            .and_then(|e| e.get("value"))
            .and_then(Value::as_f64)
            .unwrap_or(default)
    }

    /// The per-architecture ik MoE rate, as the estimator resolves it.
    ///
    /// Note that the table is keyed `{arch}@{cards}` while the lookup is by
    /// architecture alone, so in practice every call lands on `default`. That is
    /// what the Python does and what the residuals in this file are computed
    /// against, so it is reproduced rather than corrected here — changing it
    /// would move the arena model out from under every constant fitted on it.
    pub fn ik_moe_rate(&self, arch: &str) -> i64 {
        let rates = self.document.get("ik_moe_rates");
        let default = rates
            .and_then(|r| r.get("default"))
            .and_then(Value::as_i64)
            .unwrap_or(IK_MOE_DEFAULT);
        rates
            .and_then(|r| r.get("by_arch"))
            .and_then(|b| b.get(arch))
            .and_then(Value::as_i64)
            .unwrap_or(default)
    }

    /// mainline's host-resident MoE rate under a tensor split, per unit of
    /// hidden size. The arena model charges it, and derives it too — so the
    /// value read here is the previous run's, exactly as the Python's
    /// module-level constant is.
    pub fn mainline_tensor_moe_per_nembd(&self) -> i64 {
        self.constant("MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD", 57)
    }
}
