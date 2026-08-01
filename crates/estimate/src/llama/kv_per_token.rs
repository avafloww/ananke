//! KV cache sizing for llama-family models, including the SWA and shared-KV
//! variants (Gemma 2/3/4, LFM2).

use ananke_gguf::{GgufSummary, keys};

use crate::{kv, recurrent, types::EstimatorInputs};

/// Hardcoded sliding-window group length for architectures where llama.cpp
/// knows the pattern but the GGUF doesn't expose it. Gemma 2 / Gemma 3 use
/// 1 global-attention layer per every 6 layers (5 SWA + 1 global); the HF
/// config isn't round-tripped, so this constant mirrors the value baked
/// into llama.cpp's `LLM_ARCH_GEMMA*` handling.
///
/// Architectures that ship the pattern as a per-layer bool mask (e.g.
/// `gemma4.attention.sliding_window_pattern`) do *not* go through this
/// function — they use the mask directly.
pub fn hardcoded_swa_group_size(arch: &str) -> Option<u32> {
    match arch {
        "gemma2" | "gemma3" => Some(6),
        // Laguna-S: 1:3 global:SWA pattern across all layers (every 4th is
        // global), confirmed by measured KV at ctx 32768: 12 full-attention
        // layers + 36 SWA layers (window 512) reproduces the 1680 MiB
        // reading within 4%.
        "laguna" => Some(4),
        _ => None,
    }
}

/// Compute the `kv_per_token` term for a llama-family model, handling
/// three independent knobs that can all appear together:
///
/// - **Scalar vs per-layer `head_count_kv`.** Nvidia's `deci` and Gemma 4
///   both store `attention.head_count_kv` as a length-`n_layer` array;
///   every other family we've seen stores a scalar.
/// - **Sliding-window attention.** Gemma 2 / Gemma 3 use a hardcoded 1:5
///   global:SWA pattern (see `hardcoded_swa_group_size`); Gemma 4 stores
///   a per-layer bool mask in `attention.sliding_window_pattern` plus
///   separate K/V head dims for its SWA layers. On SWA layers the KV
///   cost caps at `min(context, sliding_window)` tokens.
/// - **Shared-KV layers.** Gemma 4 exposes `attention.shared_kv_layers`
///   N: the last N layers reuse the preceding layers' KV and therefore
///   don't contribute additional cache bytes.
///
/// The result is folded back into `kv_per_token × context = total KV
/// bytes` so the packer's downstream math stays identical.
pub(crate) fn compute_kv_per_token(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");
    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    // Head-count-KV can be a scalar (broadcast across all layers) or a
    // per-layer array. Materialise a vector of length `n_layers` so the
    // loop below treats both uniformly. When `head_count_kv` is absent the
    // model has no GQA, so llama.cpp defaults `n_head_kv` to `n_head`; we
    // mirror that by falling back to `attention.head_count` (e.g. talkie,
    // which omits the KV key entirely).
    let kv_heads_raw: Vec<u32> = summary
        .metadata
        .get(&keys::attention_head_count_kv(arch))
        .or_else(|| summary.metadata.get(&keys::attention_head_count(arch)))
        .and_then(|v| v.as_u32_array())
        .unwrap_or_default();
    let kv_heads_per_layer: Vec<u32> = if kv_heads_raw.len() == 1 {
        vec![kv_heads_raw[0]; n_layers as usize]
    } else {
        kv_heads_raw
    };
    if kv_heads_per_layer.is_empty() || kv_heads_per_layer.len() != n_layers as usize {
        return 0;
    }

    // Per-head byte widths. Gemma 4 uses different head dims for SWA
    // layers (`key_length_swa` / `value_length_swa`); everyone else
    // reuses the same pair for both. When the keys are absent llama.cpp
    // derives `n_embd_head = n_embd / n_head`, so mirror that before
    // falling back to the classic 128 (which is only correct for models
    // whose ratio happens to be 128 — e.g. lfm2's is 64).
    let derived_head_dim = summary
        .meta_u32(&keys::embedding_length(arch))
        .zip(
            summary
                .meta_u32(&keys::attention_head_count(arch))
                .filter(|&h| h > 0),
        )
        .map(|(embd, heads)| (embd / heads) as u64)
        .unwrap_or(128);
    let key_length = summary
        .meta_u32(&keys::attention_key_length(arch))
        .map(|v| v as u64)
        .unwrap_or(derived_head_dim);
    let value_length = summary
        .meta_u32(&keys::attention_value_length(arch))
        .map(|v| v as u64)
        .unwrap_or(derived_head_dim);
    let key_length_swa = summary
        .meta_u32(&keys::attention_key_length_swa(arch))
        .map(|v| v as u64)
        .unwrap_or(key_length);
    let value_length_swa = summary
        .meta_u32(&keys::attention_value_length_swa(arch))
        .map(|v| v as u64)
        .unwrap_or(value_length);
    let per_head_full = ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
    let per_head_swa =
        ((key_length_swa as f64 * bytes_k) + (value_length_swa as f64 * bytes_v)) as u64;

    // Build the per-layer SWA mask. Three sources, tried in order:
    //   1. An explicit per-layer bool array (gemma4).
    //   2. A hardcoded group size for architectures where llama.cpp
    //      bakes the pattern in (gemma2 / gemma3).
    //   3. No SWA — every layer is full attention.
    let sliding_window = summary
        .meta_u32(&keys::attention_sliding_window(arch))
        .map(|v| v as u64);
    let pattern_mask: Option<Vec<bool>> = summary
        .metadata
        .get(&keys::attention_sliding_window_pattern(arch))
        .and_then(|v| v.as_bool_array())
        .filter(|m| m.len() == n_layers as usize);
    let is_swa_layer: Vec<bool> = match (&pattern_mask, hardcoded_swa_group_size(arch)) {
        (Some(mask), _) => mask.clone(),
        (None, Some(group)) if group > 0 => (0..n_layers)
            // Pattern `g` means every `g`-th layer (1-indexed) is global
            // — matches both llama.cpp's gemma3 handling and the "1
            // global per 6" HF config.
            .map(|i| (i + 1) % group != 0)
            .collect(),
        _ => vec![false; n_layers as usize],
    };

    // `shared_kv_layers = N` means the last N layers reuse earlier
    // layers' KV and contribute no additional cache bytes. Absent key =
    // 0 shared = every layer unique.
    let shared_kv_layers = summary
        .meta_u32(&keys::attention_shared_kv_layers(arch))
        .unwrap_or(0);
    let unique_kv_count = (n_layers as u64).saturating_sub(shared_kv_layers as u64);

    // Walk the first `unique_kv_count` layers, summing the per-layer
    // cache bytes. SWA layers cap at the window size; full-attention
    // layers scale with the full context. We return `kv_per_token` so
    // the packer can multiply by context and recover the total.
    //
    // ik_llama does not cap SWA layers' KV at the window — it allocates
    // full-context KV for every layer regardless of the attention type.
    // Measured on laguna at ctx 131072: ik_llama uses 13056 MiB KV
    // (full-context) while mainline uses 1680 MiB (window-capped).
    let context = inputs.context as u64;
    if context == 0 {
        return 0;
    }
    let swa_capped = !inputs.ik_llama;
    // With `kv_unified=true`, the non-SWA (full-attention) layers share a
    // unified cache of `context` cells across all slots. However, the SWA
    // layers are NOT unified — each slot gets its own window-sized cache, so
    // the total SWA cells are `parallel × window`. Without `kv_unified`,
    // both caches are per-slot, and the totals are the same because
    // `parallel × (context/parallel) = context` for full-attention and
    // `parallel × window` for SWA. The only case that differs is
    // `kv_unified=true` with `parallel>1`, where the SWA cache uses
    // `parallel × window` instead of `window`.
    //
    // Measured on gemma-4-31b-qat at ctx=240000, np=4, kvu=true: the non-SWA
    // cache has 240128 cells (unified context), while the SWA cache has 4608
    // cells (4 slots × 1152 padded window of 1024).
    let parallel = inputs.parallel.unwrap_or(1).max(1) as u64;
    let kv_unified = inputs.kv_unified.unwrap_or(false);
    let swa_multiplier = if kv_unified { parallel } else { 1 };
    let mut total_kv_bytes = 0u64;
    for i in 0..unique_kv_count as usize {
        let kv_heads = kv_heads_per_layer[i] as u64;
        let (per_head, tokens) = if is_swa_layer[i] && swa_capped {
            let window = sliding_window.unwrap_or(context);
            (per_head_swa, context.min(window) * swa_multiplier)
        } else {
            (per_head_full, context)
        };
        total_kv_bytes += kv_heads * per_head * tokens;
    }

    // A layer declaring zero KV heads has no cache because it is not an
    // attention layer: LFM2 interleaves short-convolution blocks, which carry
    // recurrent state instead. llama.cpp allocates that in a separate module
    // and reports it in the same "context" bucket, so it is folded in here the
    // way `kv_for_hybrid` does for the `ssm.*` families.
    let recurrent_layers = kv_heads_per_layer
        .iter()
        .take(unique_kv_count as usize)
        .filter(|&&heads| heads == 0)
        .count() as u64;
    let state = recurrent::state_bytes(summary, arch, recurrent_layers, inputs);

    (total_kv_bytes + state) / context
}

#[cfg(test)]
mod tests {
    use ananke_gguf::{
        keys,
        types::{GgufSummary, GgufValue},
    };
    use smol_str::SmolStr;

    use crate::{
        llama::{
            estimate::estimate,
            test_support::{fake_summary, inputs, tensor},
        },
        types::EstimatorInputs,
    };

    #[test]
    fn kv_uses_arch_metadata() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // n_layers=2, n_kv=4, k=v=128, 2 bytes/element (f16).
        // per_layer_kv = 4 × (128*2 + 128*2) = 4 × 512 = 2048 bytes.
        // kv_per_token = 2 × 2048 = 4096 bytes.
        assert_eq!(e.kv_per_token, 4096);
    }

    #[test]
    fn lfm2_hybrid_kv_prices_only_attention_layers() {
        // LFM2.5-Embedding-350M shape: 16 blocks, head_count 16 (scalar),
        // head_count_kv a per-layer array with zeros on shortconv layers,
        // no key/value_length keys → head dim derives 1024 / 16 = 64.
        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..16u32 {
            let kind = if layer % 3 == 2 {
                "attn_q"
            } else {
                "shortconv.in_proj"
            };
            let name = format!("blk.{layer}.{kind}.weight");
            tensors.insert(SmolStr::new(&name), tensor(&name, 1024 * 1024));
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("lfm2".into()),
        );
        metadata.insert(SmolStr::new("lfm2.block_count"), GgufValue::U32(16));
        metadata.insert(SmolStr::new("lfm2.embedding_length"), GgufValue::U32(1024));
        metadata.insert(
            SmolStr::new("lfm2.attention.head_count"),
            GgufValue::U32(16),
        );
        let kv_array: Vec<GgufValue> = (0..16)
            .map(|i| GgufValue::U32(if i % 3 == 2 { 8 } else { 0 }))
            .collect();
        metadata.insert(
            SmolStr::new("lfm2.attention.head_count_kv"),
            GgufValue::Array(kv_array),
        );
        let s = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 16 * 1024 * 1024,
            tensors,
            metadata,
            block_count: Some(16),
            architecture: SmolStr::new("lfm2"),
            shards: vec!["/fake".into()],
        };

        assert!(crate::llama::is_llama_family("lfm2"));
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 16384, &empty));
        // 5 attention layers (indices 2,5,8,11,14) × 8 kv-heads ×
        // (64 + 64) head dims × 2 bytes (f16) = 5 × 2048 = 10240 B/token.
        // Shortconv layers contribute exactly zero.
        assert_eq!(e.kv_per_token, 10240);
        // All 16 layers still carry weights for the layer split.
        assert_eq!(e.per_layer_bytes.as_ref().unwrap().len(), 16);
    }

    #[test]
    fn kv_quantised_shrinks() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e_q8 = estimate(&s, &inputs("q8_0", "q8_0", 4096, &empty));
        let e_f16 = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        assert!(e_q8.kv_per_token < e_f16.kv_per_token);
    }

    #[test]
    fn missing_head_count_kv_falls_back_to_head_count() {
        // Talkie omits `attention.head_count_kv` (full MHA, no GQA). The KV
        // computation must fall back to `attention.head_count` rather than
        // returning zero, which would silently under-reserve the cache.
        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..2u32 {
            for kind in ["attn_q", "attn_k", "attn_v", "ffn_down"] {
                let name = format!("blk.{layer}.{kind}.weight");
                tensors.insert(SmolStr::new(&name), tensor(&name, 1024 * 1024));
            }
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("talkie".into()),
        );
        metadata.insert(SmolStr::new("talkie.block_count"), GgufValue::U32(2));
        metadata.insert(
            SmolStr::new("talkie.attention.head_count"),
            GgufValue::U32(8),
        );
        metadata.insert(
            SmolStr::new("talkie.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("talkie.attention.value_length"),
            GgufValue::U32(128),
        );
        let s = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(2),
            architecture: SmolStr::new("talkie"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // n_layers=2, n_kv=head_count=8, k=v=128, 2 bytes/element (f16).
        // per_layer_kv = 8 × (128*2 + 128*2) = 8 × 512 = 4096 bytes.
        // kv_per_token = 2 × 4096 = 8192 bytes.
        assert_eq!(e.kv_per_token, 8192);
    }

    /// Build a gemma4-shaped summary with a given per-layer SWA mask and
    /// head-count-KV array so the KV computation can be exercised end-to-end.
    fn gemma4_summary(is_swa: &[bool], kv_heads: &[u32], sliding_window: u32) -> GgufSummary {
        assert_eq!(is_swa.len(), kv_heads.len());
        let n_layers = is_swa.len() as u32;
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("gemma4".into()),
        );
        metadata.insert(SmolStr::new("gemma4.block_count"), GgufValue::U32(n_layers));
        metadata.insert(
            SmolStr::new("gemma4.attention.head_count_kv"),
            GgufValue::Array(kv_heads.iter().map(|h| GgufValue::U32(*h)).collect()),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.key_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.key_length_swa"),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.value_length_swa"),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.sliding_window"),
            GgufValue::U32(sliding_window),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.sliding_window_pattern"),
            GgufValue::Array(is_swa.iter().map(|b| GgufValue::Bool(*b)).collect()),
        );
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new("gemma4"),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn gemma4_swa_uses_per_layer_mask_and_swa_head_dims() {
        // 4 layers: layer 2 is full-attention, the rest are SWA. Each layer
        // has 8 KV heads. Context = 4096, window = 1024 — SWA layers cap
        // their cache at the window; full layer uses the entire context.
        let mask = [true, true, false, true];
        let heads = [8u32, 8, 8, 8];
        let s = gemma4_summary(&mask, &heads, 1024);
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));

        // Per-head bytes: f16 (2 b) × (512+512) = 2048 for full, × (256+256) = 1024 for SWA.
        // Layer cost (K+V bytes per layer at this context):
        //   SWA layer: 8 × 1024 × 1024 = 8_388_608 bytes
        //   Full layer: 8 × 2048 × 4096 = 67_108_864 bytes
        // Total: 3 × 8_388_608 + 1 × 67_108_864 = 92_274_688 bytes
        // kv_per_token × context = total bytes, so kv_per_token = 22_528.
        let total_kv = e.kv_per_token * e.context as u64;
        assert_eq!(total_kv, 92_274_688);
    }

    #[test]
    fn gemma4_shared_kv_layers_skip_cache() {
        // Same 4-layer model as above, but the last layer is marked as a
        // shared-KV slot — it must contribute zero KV bytes.
        let mask = [true, true, false, true];
        let heads = [8u32, 8, 8, 8];
        let mut s = gemma4_summary(&mask, &heads, 1024);
        s.metadata.insert(
            SmolStr::new("gemma4.attention.shared_kv_layers"),
            GgufValue::U32(1),
        );
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // Total must drop by one SWA layer's worth (8_388_608 bytes).
        let total_kv = e.kv_per_token * e.context as u64;
        assert_eq!(total_kv, 92_274_688 - 8_388_608);
    }

    #[test]
    fn gemma4_swa_kv_unified_multiplies_window_by_parallel() {
        // With kv_unified=true and parallel>1, llama.cpp does NOT unify the
        // SWA cache — each slot gets its own window-sized cache. The
        // full-attention layers share a unified cache of `context` cells.
        //
        // Measured on gemma-4-31b-qat (ctx=240000, np=4, kvu=true): the
        // non-SWA cache has 240128 cells (unified context) while the SWA
        // cache has 4608 cells (4 slots × padded window of 1024).
        //
        // Model: 4 layers, 3 SWA + 1 full, 8 kv-heads each.
        // key_length=512, key_length_swa=256, f16, window=1024, ctx=65536.
        //
        // Without kv_unified (parallel=1, the default):
        //   Full layer (1): 8 * 2048 * 65536 = 1_073_741_824 bytes
        //   SWA layers (3): 3 * 8 * 1024 * 1024 = 25_165_824 bytes (1x window)
        //   Total: 1_098_907_648
        //
        // With kv_unified=true, parallel=4:
        //   Full layer (1): 8 * 2048 * 65536 = 1_073_741_824 (unified, same)
        //   SWA layers (3): 3 * 8 * 1024 * (4*1024) = 100_663_296 (4x window)
        //   Total: 1_174_405_120
        let mask = [true, true, false, true];
        let heads = [8u32, 8, 8, 8];
        let s = gemma4_summary(&mask, &heads, 1024);
        let empty: Vec<String> = Vec::new();

        let no_kvu = inputs("f16", "f16", 65536, &empty);
        let e_no_kvu = estimate(&s, &no_kvu);
        assert_eq!(
            e_no_kvu.kv_per_token * 65536,
            1_098_907_648,
            "without kv_unified, SWA uses 1x window"
        );

        let with_kvu = EstimatorInputs {
            parallel: Some(4),
            kv_unified: Some(true),
            ..inputs("f16", "f16", 65536, &empty)
        };
        let e_with_kvu = estimate(&s, &with_kvu);
        assert_eq!(
            e_with_kvu.kv_per_token * 65536,
            1_174_405_120,
            "with kv_unified+parallel=4, SWA uses 4x window"
        );
    }
}
