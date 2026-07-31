//! Weights a tensor split holds on every card instead of dividing.
//!
//! `--split-mode tensor` shards the big matmuls along a row dimension, but not
//! everything in a graph is a big matmul. The gating and shared-path tensors are
//! narrow and consumed by every shard, so llama.cpp keeps a copy on each card
//! rather than paying a bus round-trip per token — and a packer that divides
//! them under-reserves every card by the difference.
//!
//! Measured against ananke's own `gpu_weight_bytes`, two-card tensor cells:
//!
//! | model | measured | modelled | short |
//! |---|---|---|---|
//! | Qwen3.6-35B-A3B | 2862 | 2641 | 221 |
//! | gemma-4-E4B | 4266 | 4002 | 264 |
//! | gemma-4-31B-QAT | 17232 | 17228 | 4 |
//! | talkie-13B | 10774 | 10775 | -1 |
//!
//! The two dense models are already exact, which is what makes this a real
//! effect rather than a fitted correction: only architectures that *have* these
//! tensors are short, and by very nearly what the tensors weigh.
//!
//! - Qwen3.6-35B-A3B: `ffn_*_shexp` 132 MiB + `ffn_gate_inp` 81 = 213 against
//!   221 measured. The remaining 8 MiB is unattributed; the architecture's other
//!   narrow tensors come to 283 MiB and would overshoot badly, so the replicated
//!   set is a specific subset rather than "everything small".
//! - gemma-4-E4B: `blk.*.inp_gate` and `blk.*.proj` at 2.5 MiB each over its
//!   blocks = 210, plus `per_layer_model_proj` 52.5, plus norms 2 — 264.5 against
//!   264 measured.
//!
//! Every figure here is read from the GGUF's own tensor table, so this scales
//! with the model rather than carrying a per-architecture constant. gemma-4-31B
//! ships none of these tensors, which is why it needs no entry to come out at
//! zero.

use ananke_gguf::GgufSummary;

/// Tensor name fragments llama.cpp replicates across a tensor split.
///
/// Matched against the part of the name after `blk.N.`, or against the whole
/// name for a non-layer tensor. Narrow gating and shared-expert paths only —
/// adding the main attention or feed-forward projections here would double-count
/// weights that genuinely are sharded.
const REPLICATED: &[&str] = &[
    // A mixture of experts' shared expert runs for every token on every shard.
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    // …and its router, which has to score every expert wherever they live.
    "ffn_gate_inp",
    "ffn_gate_inp_shexp",
    // Gemma 4's E-variant gating and laurel projections.
    "inp_gate",
    "proj",
    "per_layer_model_proj",
];

/// Bytes a tensor split holds on *each* spanned card rather than dividing.
pub(crate) fn tensor_split_replicated_bytes(summary: &GgufSummary) -> u64 {
    summary
        .tensors
        .values()
        .filter(|tensor| is_replicated(&tensor.name))
        .map(|tensor| tensor.byte_size)
        .sum()
}

/// Whether this tensor is one of the replicated set.
///
/// Compares the final component so that `blk.3.proj.weight` matches `proj` while
/// `blk.3.attn_output.weight` does not match anything — a substring test would
/// have caught `per_layer_model_proj` under `proj` and, worse, `attn_q_proj`-style
/// names on architectures that use them.
fn is_replicated(name: &str) -> bool {
    let without_suffix = name.strip_suffix(".weight").unwrap_or(name);
    let leaf = match without_suffix.strip_prefix("blk.") {
        Some(rest) => rest.split_once('.').map(|(_, tail)| tail),
        None => Some(without_suffix),
    };
    leaf.is_some_and(|leaf| REPLICATED.contains(&leaf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_gating_and_shared_paths() {
        for name in [
            "blk.0.ffn_gate_shexp.weight",
            "blk.12.ffn_down_shexp.weight",
            "blk.3.ffn_gate_inp.weight",
            "blk.3.inp_gate.weight",
            "blk.41.proj.weight",
            "per_layer_model_proj.weight",
        ] {
            assert!(is_replicated(name), "{name} should be replicated");
        }
    }

    #[test]
    fn leaves_the_sharded_matmuls_alone() {
        // Anything that is genuinely row-split must not be counted, or its
        // weight is charged twice.
        for name in [
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_output.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
            "output.weight",
        ] {
            assert!(!is_replicated(name), "{name} must not be replicated");
        }
    }
}
