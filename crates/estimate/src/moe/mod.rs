//! MoE estimator.
//!
//! Applies to: llama4, qwen3moe, qwen3vlmoe, deepseek2, mixtral, gpt-oss,
//! glm4moe, qwen35moe, deepseek4, glm-dsa, laguna.
//!
//! Identifies expert tensors by the `_exps` suffix on
//! `blk.N.ffn_{gate,up,down}_exps.weight` and itemises them into
//! `Estimate::expert_tensors`. The estimate keeps every expert's bytes in the
//! full `per_layer_bytes` total; the packer decides which experts to offload
//! (from live VRAM) and synthesises the matching `-ot` rules.

use ananke_gguf::Architecture;

mod deepseek4;
mod estimate;
mod mla;

pub use estimate::estimate;
pub(crate) use estimate::expert_kind;

pub const MOE_FAMILY: &[Architecture] = &[
    Architecture::Llama4,
    Architecture::Qwen3Moe,
    Architecture::Qwen3VlMoe,
    Architecture::DeepSeek2,
    Architecture::Mixtral,
    Architecture::GptOss,
    // GLM-4.5 series (including glm-4-5-air) uses the standard MoE tensor
    // layout: `blk.N.ffn_{gate,up,down}_exps.weight` + shared expert tensors
    // (`_shexp`). Registered here because the dispatcher would otherwise
    // refuse it outright, and because it was once estimated without a
    // per-layer breakdown: a CPU-offload `override_tensor` regex then zeroed
    // the weight estimate entirely, a 67x under-reservation.
    Architecture::Glm4Moe,
    // Qwen 3.5+ MoE is a hybrid: every `full_attention_interval`-th layer
    // runs full attention (with KV cache); the others run a linear-
    // attention / gated-delta-net SSM that carries constant per-layer
    // state instead of context-dependent KV. Every layer still has the
    // standard `blk.N.ffn_{gate,up,down}_exps.weight` tensors so the
    // MoE weight accounting applies as-is; the attention interval drops
    // KV by ~`1 / interval`. The SSM state bytes are small (<100 MiB
    // total across all recurrent layers for typical sizes) and are
    // absorbed by the compute-buffer headroom rather than modelled
    // explicitly.
    Architecture::Qwen35Moe,
    // DeepSeek-V4-Flash (deepseek4) uses the standard fused-expert tensor
    // layout — `blk.N.ffn_{gate,up,down}_exps.weight` plus a `_shexp`
    // shared expert — so the weight accounting and expert itemisation
    // apply unchanged. Its KV cache does *not*, though: only the ~half of
    // layers whose `attention.compress_ratios` entry is `4` keep a
    // key-only "CSA" (compressed sparse attention) cache of `n_ctx / 4`
    // cells; the rest use a far smaller HCA cache (ratio 128) plus a
    // fixed sliding-window cache. The generic `kv_for_hybrid` would price
    // it at `head_count_kv × (key_length + value_length) × n_layers` ≈ 88
    // KiB/token (11.5 GiB at 128k) versus the measured ~6.65 KiB/token
    // (0.84 GiB at 128k), a 13× over-reservation, so deepseek4 routes to
    // `deepseek4_kv_per_token` below instead.
    Architecture::DeepSeek4,
    // GLM-5 (glm-dsa) pairs the standard fused-expert layout —
    // `blk.N.ffn_{gate,up,down}_exps.weight` plus a `_shexp` shared
    // expert — with DeepSeek-style MLA attention, so the weight
    // accounting applies unchanged. Its KV cache does not: llama.cpp
    // stores no V cache at all for MLA architectures (`has_v =
    // !is_mla`), so the cache is a single K tensor of
    // `attention.key_length` (kv_lora_rank + rope dims, 576 for
    // GLM-5.2) elements per token per layer, and the trailing
    // `nextn_predict_layers` MTP block carries no main-context KV. The
    // generic `kv_for_hybrid` would add a phantom `value_length` V term
    // (a ~1.9× over-reservation), so glm-dsa routes to
    // `mla_kv_per_token` below. Despite the "dsa" in the name, the
    // pinned llama.cpp runs this arch as dense MLA (the deepseek2
    // graph, plain KV cache); the sparse-attention indexer tensors are
    // loaded but only the deepseek32 arch gets the DSA indexer cache.
    Architecture::GlmDsa,
    // Laguna MoE: fused-expert layout (`ffn_{gate,up,down}_exps` + `_shexp`
    // shared experts), plain GQA KV (scalar `head_count_kv`, constant
    // `key_length`/`value_length`). The per-layer `attention.head_count`
    // array only sizes Q projections and is irrelevant to KV, so the generic
    // `kv_for_hybrid` path is correct. Advertises `sliding_window` but
    // `kv_for_hybrid` doesn't model SWA eviction — safe over-estimation.
    Architecture::Laguna,
];

pub fn is_moe(arch: &Architecture) -> bool {
    MOE_FAMILY.contains(arch)
}
