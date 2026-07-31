//! Llama-family estimator.
//!
//! Applies to: llama, qwen2, qwen3, mistral, gemma(1/2/3), phi3, glm4, talkie.
//!
//! weights = Σ per-layer tensor bytes + non-layer bytes;
//! kv_per_token = n_layers × n_kv_heads ×
//!                (key_length × bytes(cache_k) + value_length × bytes(cache_v)).

mod estimate;
mod kv_per_token;
#[cfg(test)]
pub(crate) mod test_support;

pub use estimate::estimate;
pub(crate) use estimate::{collect_non_layer, collect_per_layer, layer_index};
pub(crate) use kv_per_token::compute_kv_per_token;

pub const LLAMA_FAMILY: &[&str] = &[
    "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3", "phi3", "glm4",
    // NVIDIA Nemotron ("deci") is a Llama derivative with a compressed attention
    // stack. Same `blk.N.*` tensor naming and same `{arch}.attention.*` metadata
    // keys so the llama-family estimator works unchanged; added here so
    // `dispatch` routes it away from the weights-only fallback that leaves
    // `per_layer_bytes = None` and breaks multi-GPU layer splits.
    "deci",
    // Gemma 4: attention.head_count_kv is a per-layer array (like deci),
    // attention.sliding_window_pattern is a per-layer bool mask, and
    // SWA layers use `*_length_swa` head dims distinct from the full-
    // attention layers' `*_length`. All of this is handled below; the
    // tensor layout is unchanged from llama-family.
    "gemma4",
    // Gemma 3n: MatFormer + PLE variant sharing gemma4's metadata schema
    // (per-layer `sliding_window_pattern` bool mask, `shared_kv_layers`).
    // head_count_kv is a scalar here rather than a per-layer array, but
    // `compute_kv_per_token` already handles the scalar case uniformly.
    // The architecture's 2+ GiB `per_layer_token_embd.weight` (PLE table)
    // is routed to CPU by the existing gemma4 non-layer special case, so
    // the packer's GPU pledge matches llama.cpp's actual placement.
    // MatFormer altup/laurel tensors live under `blk.N.*` and are picked
    // up by `collect_per_layer` automatically.
    "gemma3n",
    // Talkie: a dense transformer with the standard `blk.N.*` attention +
    // dense-FFN layout and `talkie.attention.*` metadata keys. It adds a
    // handful of per-tensor "gain" scalars (`attn_q_gain`, `ffn_output_gain`,
    // `token_embd_skip_gain`, `output_gain`, …) that are a few bytes each and
    // fall through `collect_per_layer` / `collect_non_layer` harmlessly. It
    // omits `attention.head_count_kv` entirely (full MHA, no GQA), which
    // `compute_kv_per_token` resolves by falling back to `head_count`.
    "talkie",
    // LiquidAI LFM2/LFM2.5: a hybrid where most blocks are gated short-
    // convolution layers (`blk.N.shortconv.*` tensors, no KV cache) and a
    // minority run GQA attention. `attention.head_count_kv` is a per-layer
    // array with zeros on the conv layers, so the existing array handling in
    // `compute_kv_per_token` prices the conv layers' KV at exactly zero.
    // The shortconv tensors live under `blk.N.*` and fall through
    // `collect_per_layer` like any other layer weight; the conv recurrent
    // state (`shortconv.l_cache` × embedding width per layer per sequence)
    // is a few hundred KiB total and is absorbed by the compute buffer.
    // No `attention.key_length`/`value_length` keys, so the head-dim
    // fallback below derives `embedding_length / head_count` (64 for the
    // 350M embedder) the same way llama.cpp does.
    "lfm2",
];

pub fn is_llama_family(arch: &str) -> bool {
    LLAMA_FAMILY.contains(&arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deci_is_llama_family() {
        // Regression: Nemotron-49B (arch = "deci") must be recognised as
        // llama-family so the per-layer walk runs. Falling through to the
        // fallback estimator returns `per_layer_bytes = None`, which breaks
        // multi-GPU layer splits.
        assert!(is_llama_family("deci"));
    }

    #[test]
    fn gemma4_is_llama_family() {
        // Gemma 4 reuses the llama-family tensor layout. It's handled via
        // compute_kv_per_token's per-layer bool mask + separate SWA head
        // dim paths, not a distinct estimator.
        assert!(is_llama_family("gemma4"));
    }

    #[test]
    fn talkie_is_llama_family() {
        // Talkie is a dense transformer with the standard llama-family tensor
        // layout; it must dispatch here rather than falling through to the
        // weights-only fallback (which leaves `per_layer_bytes = None`).
        assert!(is_llama_family("talkie"));
    }

    #[test]
    fn gemma3n_is_llama_family() {
        // Gemma 3n (MatFormer / PLE variant) shares gemma4's metadata
        // schema. The estimator needs to recognise it so that the service
        // doesn't flip to `Disabled { ConfigError }` on first-Ensure.
        assert!(is_llama_family("gemma3n"));
    }
}
