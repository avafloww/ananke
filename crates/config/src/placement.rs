//! Types describing how a model is spread over devices.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
//!
//! These live beside the flag vocabulary rather than with the config validator
//! because the estimator reasons about them without knowing what a
//! `ServiceConfig` is: how the model is split, and which slot a tensor is pinned
//! to, are properties of the runtime invocation. `ananke::crate::validate`
//! re-exports both.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::flags;

/// Look up a variant's flag string in its `VARIANTS` table. Every variant is
/// registered (guarded by each enum's round-trip test), so the lookup is total
/// in practice.
pub fn variant_flag<T: Copy + PartialEq>(table: &[(T, &'static str)], value: T) -> &'static str {
    // Invariant: every variant is registered in its `VARIANTS` table, guarded
    // by each enum's round-trip test, so the lookup is total in practice.
    table
        .iter()
        .find_map(|&(v, flag)| (v == value).then_some(flag))
        .unwrap_or_else(|| unreachable!("enum variant is registered in its VARIANTS table"))
}

/// Inverse of [`variant_flag`]: resolve an accepted string to its variant.
pub fn flag_variant<T: Copy>(table: &[(T, &'static str)], s: &str) -> Option<T> {
    table.iter().find_map(|&(v, flag)| (flag == s).then_some(v))
}

/// How llama.cpp spreads the model across devices.
///
/// Guardrail: the serde spellings are a wire, not a convenience. They are the
/// `split` column of `calibration/data/measurements.ndjson` and part of the
/// payload a cell's identity is hashed over, so they must stay identical to
/// [`SplitMode::as_flag`] — which `split_mode_vocab_is_single_sourced_and_complete`
/// asserts.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    /// Whole layers pipelined across the cards, each layer living on one.
    #[default]
    Layer,
    /// Every layer's weights split by row across the cards. The older of the two
    /// sharding modes, kept for parity.
    Row,
    /// Every layer's weights split across the cards, which llama.cpp fuses into
    /// a single reported device.
    Tensor,
}

impl SplitMode {
    /// Variant ↔ flag binding. The strings come from
    /// [`crate::flags::split_mode`], so `as_flag`, `from_flag`, and
    /// `valid_values` all resolve to the same single-sourced vocabulary the
    /// schema docs use.
    const VARIANTS: &'static [(Self, &'static str)] = &[
        (Self::Layer, flags::split_mode::LAYER),
        (Self::Row, flags::split_mode::ROW),
        (Self::Tensor, flags::split_mode::TENSOR),
    ];

    /// The `--split-mode` flag value.
    pub fn as_flag(self) -> &'static str {
        variant_flag(Self::VARIANTS, self)
    }

    /// Parse an accepted `devices.split` string into a variant.
    pub fn from_flag(s: &str) -> Option<Self> {
        flag_variant(Self::VARIANTS, s)
    }

    /// Accepted values as a quoted list for operator-facing errors.
    pub fn valid_values() -> String {
        flags::quoted_list(flags::split_mode::ALL)
    }

    /// Whether this mode shards every layer across all spanned GPUs (as opposed
    /// to `Layer`'s whole-layer pipeline). Drives the packer's
    /// balanced-distribution path.
    pub fn is_sharded(self) -> bool {
        matches!(self, SplitMode::Row | SplitMode::Tensor)
    }
}

impl fmt::Display for SplitMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_flag())
    }
}

/// A device a tensor or a share of the model can be pinned to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceSlot {
    /// Host memory.
    Cpu,
    /// The GPU at this index.
    Gpu(u32),
}

/// Which device classes a service may be placed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlacementPolicy {
    /// GPUs only; a model that does not fit fails placement rather than spilling.
    #[default]
    GpuOnly,
    /// Host memory only.
    CpuOnly,
    /// GPUs first, spilling to the host.
    Hybrid,
}

/// Per-device VRAM/RAM the daemon keeps free, resolved from the global
/// `[devices]` config. Copied onto each service config so the (pure) packer
/// can read reserves without a separate config handle. The per-service
/// `gpu_headroom_mb` is layered on top of these by the packer.
#[derive(Debug, Clone, Default)]
pub struct DeviceReserves {
    /// VRAM (MiB) kept free on every GPU that lacks a `per_gpu_mb` entry.
    pub default_gpu_mb: u64,
    /// VRAM (MiB) kept free on specific GPUs, keyed by GPU id.
    pub per_gpu_mb: BTreeMap<u32, u64>,
    /// Host RAM (bytes) kept free; bounds the packer's CPU expert offload.
    pub cpu_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_mode_vocab_is_single_sourced_and_complete() {
        for &(variant, flag) in SplitMode::VARIANTS {
            assert_eq!(variant.as_flag(), flag);
            assert_eq!(SplitMode::from_flag(flag), Some(variant));
            // The dataset's `split` column and the flag are one spelling; a
            // rename here re-hashes every cell identity in the campaign.
            assert_eq!(
                serde_json::to_string(&variant).expect("a unit variant serializes"),
                format!("\"{flag}\"")
            );
        }
        // Completeness: the exhaustive match makes a newly added variant a
        // compile error until it is handled, and the assert then requires it
        // to be registered in VARIANTS (so `as_flag` never hits its expect).
        for variant in [SplitMode::Layer, SplitMode::Row, SplitMode::Tensor] {
            match variant {
                SplitMode::Layer | SplitMode::Row | SplitMode::Tensor => {}
            }
            assert!(
                SplitMode::VARIANTS.iter().any(|&(v, _)| v == variant),
                "{variant:?} missing from SplitMode::VARIANTS"
            );
        }
        assert_eq!(SplitMode::from_flag("bogus"), None);
        // Errors and docs draw from the same vocabulary.
        assert_eq!(
            SplitMode::valid_values(),
            flags::quoted_list(flags::split_mode::ALL)
        );
    }
}

/// What the packer reads about a service, distilled out of its config.
///
/// The packer needs eight fields, all of them either primitives or types declared
/// above, so — exactly as `EstimatorInputs` does for the estimator — it takes those
/// and nothing else rather than a whole `ServiceConfig`, which would tie it to the
/// config validator and everything behind it. Building one from a validated config
/// is the daemon's business; see `ananke::crate::service_inputs`.
#[derive(Debug, Clone, Default)]
pub struct PlacementInputs {
    /// Service name. Compared against the reservation table's keys, so it is the
    /// same small string the rest of the daemon uses.
    pub name: SmolStr,
    /// Which device classes this service may be placed on.
    pub policy: PlacementPolicy,
    /// Operator-pinned per-device byte counts, which override the walk entirely.
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    /// How the model is spread across the cards it spans.
    pub split_mode: SplitMode,
    /// GPU indices this service is restricted to. Empty means every present card.
    pub gpu_allow: Vec<u32>,
    /// Extra VRAM (MiB) to keep free on each spanned GPU, on top of `reserves`.
    pub gpu_headroom_mb: u64,
    /// Per-device memory the daemon keeps free.
    pub reserves: DeviceReserves,
    /// Whether the service runs on the ik_llama fork, which differs from
    /// mainline on which end of the expert layers `-ncmoe` moves and on how it
    /// counts.
    pub ik_llama: bool,
    /// Whether, and how far, expert tensors are moved to the host.
    pub expert_offload: OffloadMode,
    /// Explicit per-GPU shares for a sharded split, one per allowed GPU in
    /// ascending id order. `None` gives an equal split.
    pub tensor_split_weights: Option<Vec<f32>>,
    /// `override_tensor` rules the operator pinned by hand.
    pub override_tensor: Vec<String>,
}

impl PlacementInputs {
    /// A placement with everything at its default and only a name set.
    ///
    /// For tests in crates that cannot reach the config validator to build a
    /// real `ServiceConfig`. Production code goes through
    /// `ananke::crate::service_inputs::placement_inputs`, so the two cannot
    /// disagree about how a config is read.
    pub fn named(name: &str) -> Self {
        Self {
            name: SmolStr::new(name),
            ..Self::default()
        }
    }
}

/// MoE expert-offload policy for a llama-cpp service. Resolved from the
/// `expert_offload` config value. The packer reads this to decide whether and how
/// much expert weight to move off the GPU when the model doesn't fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadMode {
    /// No expert offload. The model packs whole layers, spilling entire layers to
    /// CPU only under a CPU-allowing placement.
    #[default]
    Off,
    /// The packer keeps each expert on its layer's home GPU while that GPU has
    /// room, then greedily spills the experts that don't fit — to the most-free
    /// other GPU first, then to CPU — so only the surplus over live VRAM moves.
    Auto,
    /// The packer offloads the experts of exactly the `N` tail-most
    /// expert-bearing layers, regardless of fit.
    Layers(u32),
}

impl OffloadMode {
    /// Whether any expert offload is requested (i.e. not [`OffloadMode::Off`]).
    pub fn is_enabled(self) -> bool {
        !matches!(self, OffloadMode::Off)
    }
}
