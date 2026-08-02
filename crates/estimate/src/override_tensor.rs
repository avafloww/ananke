//! `override_tensor` rule application.
//!
//! The user declares `override_tensor = ["<regex>=<device>", ...]`; llama.cpp
//! takes the same rules via `-ot`. We must mirror the placement accounting so
//! the allocator and placement walker see the correct per-device budgets:
//! any tensor matching a rule is attributed to the declared device rather
//! than following layer placement.
//!
//! Rules apply in array order; first match wins. Matched tensor bytes are
//! subtracted from per-layer / non-layer accounting (so the layer walker
//! packs only the residual) and accumulated into `Estimate.layout.override_tensor_bytes`.

use std::collections::BTreeMap;

use ananke_config::placement::DeviceSlot;
use ananke_gguf::GgufSummary;
use regex::Regex;
use tracing::warn;

use crate::{llama::layer_index, moe::expert_kind, types::Estimate};

#[derive(Debug)]
pub struct OverrideRule {
    pub regex: Regex,
    pub target: DeviceSlot,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// Parse a user-supplied `override_tensor` array into compiled rules.
pub fn parse_rules(rules: &[String]) -> Result<Vec<OverrideRule>, ParseError> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let (pattern, device_str) = rule
            .rsplit_once('=')
            .ok_or_else(|| ParseError(format!("missing '=' in rule `{rule}`")))?;
        let regex =
            Regex::new(pattern).map_err(|e| ParseError(format!("regex `{pattern}`: {e}")))?;
        let target = parse_device(device_str.trim())
            .ok_or_else(|| ParseError(format!("unknown device `{device_str}` in rule `{rule}`")))?;
        out.push(OverrideRule { regex, target });
    }
    Ok(out)
}

fn parse_device(s: &str) -> Option<DeviceSlot> {
    let upper = s.to_ascii_uppercase();
    if upper == "CPU" {
        return Some(DeviceSlot::Cpu);
    }
    if let Some(tail) = upper.strip_prefix("GPU")
        && let Ok(n) = tail.parse::<u32>()
    {
        return Some(DeviceSlot::Gpu(n));
    }
    None
}

/// Apply `rules` to `summary`, moving matched tensor bytes out of
/// `estimate.layout.per_layer_bytes` / `non_layer` into `override_tensor_bytes`.
///
/// The placement walker will then use the reduced per-layer totals and the
/// pre-seeded per-device map (via `override_tensor_bytes`) to produce a
/// consistent allocation.
pub fn apply(estimate: &mut Estimate, summary: &GgufSummary, rules: &[OverrideRule]) {
    if rules.is_empty() {
        return;
    }

    let mut override_bytes: BTreeMap<DeviceSlot, u64> = BTreeMap::new();

    for tensor in summary.tensors.values() {
        let Some(rule) = rules
            .iter()
            .find(|r| r.regex.is_match(tensor.name.as_str()))
        else {
            continue;
        };
        *override_bytes.entry(rule.target.clone()).or_default() += tensor.byte_size;

        if let Some(idx) = layer_index(tensor.name.as_str()) {
            if let Some(per_layer) = estimate.layout.per_layer_bytes.as_mut()
                && (idx as usize) < per_layer.len()
            {
                per_layer[idx as usize] = per_layer[idx as usize].saturating_sub(tensor.byte_size);
            }
            // An operator rule that pins an expert tensor takes it out of the
            // packer's auto-offload pool: drop it from the itemised experts so
            // the packer doesn't double-place or re-offload it.
            if let Some(kind) = expert_kind(tensor.name.as_str())
                && let Some(experts) = estimate.layout.expert_tensors.as_mut()
            {
                experts.retain(|e| !(e.layer == idx && e.kind == kind));
            }
        } else {
            match tensor.name.as_str() {
                "output.weight" => {
                    estimate.layout.non_layer.output_head_bytes = estimate
                        .layout
                        .non_layer
                        .output_head_bytes
                        .saturating_sub(tensor.byte_size)
                }
                "token_embd.weight" => {
                    estimate.layout.non_layer.token_embd_bytes = estimate
                        .layout
                        .non_layer
                        .token_embd_bytes
                        .saturating_sub(tensor.byte_size)
                }
                _ => {
                    estimate.layout.non_layer.other_bytes = estimate
                        .layout
                        .non_layer
                        .other_bytes
                        .saturating_sub(tensor.byte_size)
                }
            }
        }
    }

    let override_total: u64 = override_bytes.values().sum();
    estimate.layout.override_tensor_bytes = override_bytes;

    // Recompute the weights total to reflect the redirected tensors.
    //
    // When the architecture-specific estimator supplied a per-layer
    // breakdown, the loop above mutated `per_layer_bytes` + `non_layer`
    // in-place and we sum those back up.
    //
    // An estimate with no per-layer breakdown sets `per_layer_bytes = None` and
    // leaves `non_layer` all-zero, populating only the coarse `weights_bytes`
    // total. Recomputing from zero would clobber the one sensible number it
    // does have, so subtract the redirected bytes from that total instead and
    // let the remainder account for what stays on the device.
    if estimate.layout.per_layer_bytes.is_some() {
        let per_layer_sum = estimate
            .layout
            .per_layer_bytes
            .as_ref()
            .map(|p| p.iter().sum::<u64>())
            .unwrap_or(0);
        estimate.weights_bytes = per_layer_sum
            + estimate.layout.non_layer.output_head_bytes
            + estimate.layout.non_layer.token_embd_bytes
            + estimate.layout.non_layer.other_bytes;
    } else {
        estimate.weights_bytes = estimate.weights_bytes.saturating_sub(override_total);
    }
}

/// Convenience: parse and apply in one call; errors are logged (not returned),
/// consistent with how the mmproj integration handles soft failures.
pub fn parse_and_apply(estimate: &mut Estimate, summary: &GgufSummary, rules: &[String]) {
    match parse_rules(rules) {
        Ok(parsed) => apply(estimate, summary, &parsed),
        Err(e) => warn!(error = %e, "override_tensor parse failed; running without overrides"),
    }
}

#[cfg(test)]
mod tests {
    use ananke_gguf::{
        Architecture,
        types::{GgufSummary, GgufTensor, GgufType},
    };
    use smol_str::SmolStr;

    use super::*;
    use crate::types::{Buffers, Estimate, Layout};

    fn tensor(name: &str, bytes: u64) -> GgufTensor {
        GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        }
    }

    fn base_estimate(per_layer: Vec<u64>) -> Estimate {
        let weights = per_layer.iter().sum::<u64>();
        Estimate {
            weights_bytes: weights,
            kv_per_token: 0,
            layout: Layout {
                per_layer_bytes: Some(per_layer),
                ..Layout::default()
            },
            buffers: Buffers {
                compute_mb: 400,
                ..Buffers::default()
            },
            ..Estimate::empty(Architecture::Qwen3Moe, 4096)
        }
    }

    fn summary_with(tensors: Vec<GgufTensor>) -> GgufSummary {
        let mut map = std::collections::BTreeMap::new();
        let mut total = 0;
        for t in tensors {
            total += t.byte_size;
            map.insert(t.name.clone(), t);
        }
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: total,
            tensors: map,
            metadata: Default::default(),
            block_count: Some(2),
            architecture: Architecture::Qwen3Moe,
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn parses_single_rule() {
        let rules = parse_rules(&[".ffn_(up|down)_exps.=CPU".into()]).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].target, DeviceSlot::Cpu);
    }

    #[test]
    fn rejects_unknown_device() {
        let err = parse_rules(&["foo=BLAH".into()]).unwrap_err();
        assert!(format!("{err}").contains("BLAH"));
    }

    #[test]
    fn rejects_bad_regex() {
        let err = parse_rules(&["(unclosed=CPU".into()]).unwrap_err();
        assert!(format!("{err}").contains("regex"));
    }

    #[test]
    fn parses_gpu_index() {
        let rules = parse_rules(&["output=GPU0".into(), "foo=GPU1".into()]).unwrap();
        assert_eq!(rules[0].target, DeviceSlot::Gpu(0));
        assert_eq!(rules[1].target, DeviceSlot::Gpu(1));
    }

    #[test]
    fn moves_expert_tensors_to_cpu() {
        // 2 layers, each with an attn tensor (1 MiB) and an expert tensor (10 MiB).
        let tensors = vec![
            tensor("blk.0.attn_q.weight", 1024 * 1024),
            tensor("blk.0.ffn_up_exps.weight", 10 * 1024 * 1024),
            tensor("blk.1.attn_q.weight", 1024 * 1024),
            tensor("blk.1.ffn_down_exps.weight", 10 * 1024 * 1024),
        ];
        let summary = summary_with(tensors);
        // Per-layer starts at 11 MiB each (sum of both tensors on the layer).
        let mut est = base_estimate(vec![11 * 1024 * 1024, 11 * 1024 * 1024]);
        let rules = parse_rules(&[".ffn_(up|down)_exps.=CPU".into()]).unwrap();

        apply(&mut est, &summary, &rules);

        // Each layer drops to 1 MiB (just the attn tensor).
        let per_layer = est.layout.per_layer_bytes.unwrap();
        assert_eq!(per_layer, vec![1024 * 1024, 1024 * 1024]);
        // CPU gets 20 MiB total.
        assert_eq!(
            est.layout
                .override_tensor_bytes
                .get(&DeviceSlot::Cpu)
                .copied(),
            Some(20 * 1024 * 1024)
        );
        // weights_bytes reflects the reduced per-layer + non-layer.
        assert_eq!(est.weights_bytes, 2 * 1024 * 1024);
    }

    /// Regression: when the architecture-specific estimator doesn't run —
    /// an architecture in no family's list — `apply` receives
    /// an estimate whose `per_layer_bytes` is `None` and
    /// whose non-layer fields are all zero. The recompute step must not
    /// zero `weights_bytes` in that case — that gives a 400 MiB prediction
    /// for a 26 GiB model — but subtract the redirected bytes from the
    /// coarse weights total, so the remaining on-device weights
    /// are still accounted for.
    #[test]
    fn preserves_the_coarse_weights_when_there_is_no_per_layer_breakdown() {
        let tensors = vec![
            tensor("blk.0.attn_q.weight", 1024 * 1024),
            tensor("blk.0.ffn_up_exps.weight", 10 * 1024 * 1024),
            tensor("blk.0.ffn_down_exps.weight", 10 * 1024 * 1024),
        ];
        let summary = summary_with(tensors);

        // An estimate with no per-layer breakdown: coarse `weights_bytes` with no
        // per-layer breakdown and empty non_layer.
        let total_on_disk: u64 = summary.total_tensor_bytes;
        let mut est = Estimate {
            weights_bytes: total_on_disk,
            kv_per_token: 0,
            buffers: Buffers {
                compute_mb: 400,
                ..Buffers::default()
            },
            ..Estimate::empty(Architecture::Glm4Moe, 4096)
        };

        let rules = parse_rules(&[".ffn_(up|down)_exps.=CPU".into()]).unwrap();
        apply(&mut est, &summary, &rules);

        // 20 MiB of experts moved to CPU; remaining on-GPU weights =
        // total − override = 1 MiB attn.
        assert_eq!(
            est.layout
                .override_tensor_bytes
                .get(&DeviceSlot::Cpu)
                .copied(),
            Some(20 * 1024 * 1024)
        );
        assert_eq!(est.weights_bytes, 1024 * 1024);
    }

    #[test]
    fn first_match_wins() {
        let tensors = vec![tensor("blk.0.ffn_up_exps.weight", 10 * 1024 * 1024)];
        let summary = summary_with(tensors);
        let mut est = base_estimate(vec![10 * 1024 * 1024, 0]);
        // GPU1 comes first; CPU second. Expert should land on GPU1.
        let rules =
            parse_rules(&["ffn_up=GPU1".into(), ".ffn_(up|down)_exps.=CPU".into()]).unwrap();

        apply(&mut est, &summary, &rules);

        assert_eq!(
            est.layout
                .override_tensor_bytes
                .get(&DeviceSlot::Gpu(1))
                .copied(),
            Some(10 * 1024 * 1024)
        );
        assert!(
            !est.layout
                .override_tensor_bytes
                .contains_key(&DeviceSlot::Cpu)
        );
    }
}
