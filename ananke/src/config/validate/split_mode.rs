//! The `--split-mode` vocabulary: how a multi-GPU llama.cpp service divides
//! a model across the GPUs it spans.

use ananke_config::flags;

use crate::config::validate::{flag_variant, variant_flag};

/// How a multi-GPU llama.cpp service divides the model across the GPUs it
/// spans. Orthogonal to [`PlacementPolicy`], which decides CPU-vs-GPU and
/// whether CPU spill is allowed; this decides the *inter-GPU* strategy and
/// maps straight onto llama.cpp's `--split-mode`.
///
/// - `Layer` (default): pipeline — each GPU holds whole layers and the
///   first-fit packer fills one GPU before spilling to the next. Minimal
///   inter-GPU traffic, but only one GPU computes at a time for a single
///   request.
/// - `Row` / `Tensor`: tensor parallelism — every layer is sharded across
///   all spanned GPUs, which compute in parallel and reduce per layer.
///   `tensor` is llama.cpp's newer, faster implementation; `row` is the
///   older one, kept for parity. Both require [`PlacementPolicy::GpuOnly`]
///   (no CPU spill) and a llama-cpp service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitMode {
    #[default]
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    /// Variant ↔ flag binding. The strings come from
    /// [`ananke_config::flags::split_mode`], so `as_flag`, `from_flag`, and
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

    /// Whether this mode shards every layer across all spanned GPUs (as
    /// opposed to `Layer`'s whole-layer pipeline). Drives the packer's
    /// balanced-distribution path.
    pub fn is_sharded(self) -> bool {
        matches!(self, SplitMode::Row | SplitMode::Tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::{test_fixtures::parse_and_merge, validate};

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
    #[test]
    fn parses_tensor_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Tensor);
    }

    #[test]
    fn defaults_split_mode_to_layer() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Layer);
    }

    #[test]
    fn rejects_unknown_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.split = "diagonal"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("unknown devices.split"));
    }

    #[test]
    fn rejects_tensor_split_with_cpu_spill() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("requires placement=gpu-only"));
    }

    #[test]
    fn rejects_tensor_split_on_command_service() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "command"
command = ["/bin/true"]
port = 11435
allocation.mode = "static"
allocation.vram_gb = 4
devices.placement = "gpu-only"
devices.split = "row"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("only valid for llama-cpp"));
    }

    #[test]
    fn parses_tensor_split_weights() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(
            ec.services[0].tensor_split_weights.as_deref(),
            Some(&[2.6f32, 1.0f32][..])
        );
    }

    #[test]
    fn rejects_tensor_split_weights_wrong_count() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("tensor_split_weights has 3 entries but 2 allowed GPU(s)"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_tensor_split_weights_non_positive() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 0.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("tensor_split_weights[1] must be a positive finite number"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_tensor_split_weights_on_non_sharded_split() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}")
                .contains("tensor_split_weights is only valid with a sharded split mode"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_tensor_split_weights_on_hybrid_placement() {
        // Sharded splits already require gpu-only, so this fails on the split
        // constraint before it reaches the weight check.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("requires placement=gpu-only"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_tensor_split_weights_with_unsorted_gpu_allow() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [1, 0]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("devices.gpu_allow must be in ascending GPU-id order"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_tensor_split_weights_with_duplicate_gpu_allow() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 0]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("devices.gpu_allow must not contain duplicate GPU ids"),
            "got: {err}"
        );
    }
}
