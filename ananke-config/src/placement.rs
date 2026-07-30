//! Types describing how a model is spread over devices.
//!
//! These live beside the flag vocabulary rather than with the config validator
//! because the estimator reasons about them without knowing what a
//! `ServiceConfig` is: how the model is split, and which slot a tensor is pinned
//! to, are properties of the runtime invocation. `ananke::config::validate`
//! re-exports both, so config-side paths are unchanged.

use crate::flags;

/// Look up a variant's flag string in its `VARIANTS` table. Every variant is
/// registered (guarded by each enum's round-trip test), so the lookup is total
/// in practice.
pub fn variant_flag<T: Copy + PartialEq>(table: &[(T, &'static str)], value: T) -> &'static str {
    table
        .iter()
        .find_map(|&(v, flag)| (v == value).then_some(flag))
        .expect("enum variant is registered in its VARIANTS table")
}

/// Inverse of [`variant_flag`]: resolve an accepted string to its variant.
pub fn flag_variant<T: Copy>(table: &[(T, &'static str)], s: &str) -> Option<T> {
    table.iter().find_map(|&(v, flag)| (flag == s).then_some(v))
}

/// How llama.cpp spreads the model across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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

/// A device a tensor or a share of the model can be pinned to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceSlot {
    /// Host memory.
    Cpu,
    /// The GPU at this index.
    Gpu(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_mode_vocab_is_single_sourced_and_complete() {
        for &(variant, flag) in SplitMode::VARIANTS {
            assert_eq!(variant.as_flag(), flag);
            assert_eq!(SplitMode::from_flag(flag), Some(variant));
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
