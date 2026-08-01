//! Every GGUF metadata key the workspace reads.
//!
//! Nothing outside this module should spell a key literally: a misspelling
//! reads as an absent key rather than as an error, so the failure is a
//! silently-defaulted term rather than a panic.
//!
//! Most keys are prefixed with the model's architecture, and the function for
//! each builds the whole key. A caller holding the architecture separately —
//! the calibration dataset keys its own map that way, and `dump-gguf` prints
//! the suffixes — takes the matching constant from [`suffix`] instead.

use smol_str::SmolStr;

use crate::Architecture;

/// `general.architecture` — the value every [`scoped`] key is prefixed with.
pub const ARCHITECTURE: &str = "general.architecture";
pub const NAME: &str = "general.name";
pub const PARAMETER_COUNT: &str = "general.parameter_count";
pub const LICENSE: &str = "general.license";

/// Number of shards this file belongs to, absent on a single-file model.
pub const SPLIT_COUNT: &str = "split.count";
/// This file's zero-based index within its shard set.
pub const SPLIT_NO: &str = "split.no";

/// The architecture-scoped keys, without their `{arch}.` prefix.
pub mod suffix {
    pub const BLOCK_COUNT: &str = "block_count";
    pub const EMBEDDING_LENGTH: &str = "embedding_length";
    /// Blocks of embedded multi-token-prediction head, zero or absent on a
    /// model that ships none.
    pub const NEXTN_PREDICT_LAYERS: &str = "nextn_predict_layers";
    /// Period of full-attention layers on an architecture that interleaves
    /// them with sliding-window or recurrent ones.
    pub const FULL_ATTENTION_INTERVAL: &str = "full_attention_interval";
    pub const EXPERT_COUNT: &str = "expert_count";
    pub const EXPERT_USED_COUNT: &str = "expert_used_count";

    pub const ATTENTION_HEAD_COUNT: &str = "attention.head_count";
    pub const ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
    pub const ATTENTION_KEY_LENGTH: &str = "attention.key_length";
    pub const ATTENTION_VALUE_LENGTH: &str = "attention.value_length";
    /// Key width on the sliding-window layers, where they differ from the
    /// full ones.
    pub const ATTENTION_KEY_LENGTH_SWA: &str = "attention.key_length_swa";
    pub const ATTENTION_VALUE_LENGTH_SWA: &str = "attention.value_length_swa";
    pub const ATTENTION_SLIDING_WINDOW: &str = "attention.sliding_window";
    /// Period of the sliding-window/full interleave, where the model states
    /// it separately from [`FULL_ATTENTION_INTERVAL`].
    pub const ATTENTION_SLIDING_WINDOW_PATTERN: &str = "attention.sliding_window_pattern";
    /// Trailing layers that share the preceding layer's KV cache rather than
    /// holding one of their own.
    pub const ATTENTION_SHARED_KV_LAYERS: &str = "attention.shared_kv_layers";
    /// Per-layer latent compression ratios on a multi-head latent attention
    /// model.
    pub const ATTENTION_COMPRESS_RATIOS: &str = "attention.compress_ratios";
    /// Key width of the sparse-attention indexer, which is not the attention
    /// key width.
    pub const ATTENTION_INDEXER_KEY_LENGTH: &str = "attention.indexer.key_length";
    /// Per-layer attention kind, which some architectures publish in place of
    /// a pattern or interval.
    pub const ATTENTION_LAYER_TYPES: &str = "attention.layer_types";

    pub const SSM_CONV_KERNEL: &str = "ssm.conv_kernel";
    pub const SSM_INNER_SIZE: &str = "ssm.inner_size";
    pub const SSM_STATE_SIZE: &str = "ssm.state_size";
    pub const SSM_GROUP_COUNT: &str = "ssm.group_count";

    /// Short-convolution cache depth, the recurrent state of an LFM-style
    /// layer.
    pub const SHORTCONV_L_CACHE: &str = "shortconv.l_cache";
}

/// `{arch}.{suffix}`, the general form of every architecture-scoped key.
///
/// Prefer a named accessor below. This is for the caller whose suffix is
/// chosen at runtime.
pub fn scoped(arch: &Architecture, suffix: &str) -> SmolStr {
    SmolStr::new(format!("{}.{suffix}", arch.as_str()))
}

macro_rules! scoped_keys {
    ($($name:ident => $suffix:ident),* $(,)?) => {
        $(
            #[doc = concat!("[`suffix::", stringify!($suffix), "`], prefixed by `arch`.")]
            pub fn $name(arch: &Architecture) -> SmolStr {
                scoped(arch, suffix::$suffix)
            }
        )*

        #[cfg(test)]
        const ALL_SCOPED: &[(fn(&Architecture) -> SmolStr, &str)] =
            &[$(($name, suffix::$suffix)),*];
    };
}

scoped_keys! {
    block_count => BLOCK_COUNT,
    embedding_length => EMBEDDING_LENGTH,
    nextn_predict_layers => NEXTN_PREDICT_LAYERS,
    full_attention_interval => FULL_ATTENTION_INTERVAL,
    expert_count => EXPERT_COUNT,
    expert_used_count => EXPERT_USED_COUNT,
    attention_head_count => ATTENTION_HEAD_COUNT,
    attention_head_count_kv => ATTENTION_HEAD_COUNT_KV,
    attention_key_length => ATTENTION_KEY_LENGTH,
    attention_value_length => ATTENTION_VALUE_LENGTH,
    attention_key_length_swa => ATTENTION_KEY_LENGTH_SWA,
    attention_value_length_swa => ATTENTION_VALUE_LENGTH_SWA,
    attention_sliding_window => ATTENTION_SLIDING_WINDOW,
    attention_sliding_window_pattern => ATTENTION_SLIDING_WINDOW_PATTERN,
    attention_shared_kv_layers => ATTENTION_SHARED_KV_LAYERS,
    attention_compress_ratios => ATTENTION_COMPRESS_RATIOS,
    attention_indexer_key_length => ATTENTION_INDEXER_KEY_LENGTH,
    ssm_conv_kernel => SSM_CONV_KERNEL,
    ssm_inner_size => SSM_INNER_SIZE,
    ssm_state_size => SSM_STATE_SIZE,
    ssm_group_count => SSM_GROUP_COUNT,
    shortconv_l_cache => SHORTCONV_L_CACHE,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These spellings are llama.cpp's, not ours. A wrong one reads as an
    /// absent key and silently defaults the term it feeds, so pin the literals
    /// a GGUF actually carries.
    #[test]
    fn keys_spell_what_llama_cpp_writes() {
        assert_eq!(ARCHITECTURE, "general.architecture");
        assert_eq!(NAME, "general.name");
        assert_eq!(PARAMETER_COUNT, "general.parameter_count");
        assert_eq!(LICENSE, "general.license");
        assert_eq!(SPLIT_COUNT, "split.count");
        assert_eq!(SPLIT_NO, "split.no");
        assert_eq!(suffix::BLOCK_COUNT, "block_count");
        assert_eq!(suffix::ATTENTION_HEAD_COUNT_KV, "attention.head_count_kv");
        assert_eq!(
            suffix::ATTENTION_INDEXER_KEY_LENGTH,
            "attention.indexer.key_length"
        );
        assert_eq!(suffix::SSM_CONV_KERNEL, "ssm.conv_kernel");
        assert_eq!(suffix::SHORTCONV_L_CACHE, "shortconv.l_cache");
    }

    /// Every accessor prefixes its suffix and nothing else, so a caller that
    /// has to hold the two apart builds the same string the accessor does.
    #[test]
    fn every_accessor_is_its_suffix_prefixed_by_the_architecture() {
        for (build, suffix) in ALL_SCOPED {
            assert_eq!(build(&Architecture::Qwen3), format!("qwen3.{suffix}"));
        }
    }
}
