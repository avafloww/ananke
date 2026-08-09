//! Types returned by the GGUF reader.

use std::{collections::BTreeMap, path::PathBuf};

use smol_str::SmolStr;

use crate::Architecture;

/// Summary of a GGUF file (or aggregated shard set).
#[derive(Debug, Clone)]
pub struct GgufSummary {
    /// Canonical path (shard 0 for multi-file).
    pub path: PathBuf,
    /// Total tensor byte count across all shards.
    pub total_tensor_bytes: u64,
    /// Tensors keyed by name.
    pub tensors: BTreeMap<SmolStr, GgufTensor>,
    /// Metadata key-value map.
    pub metadata: BTreeMap<SmolStr, GgufValue>,
    /// Number of layers (`<arch>.block_count` typical). `None` if the
    /// architecture doesn't expose this.
    pub block_count: Option<u32>,
    /// The model's graph, parsed from `general.architecture`.
    pub architecture: Architecture,
    /// For sharded files: the discovered shard list. Single-file → len 1.
    pub shards: Vec<PathBuf>,
}

impl GgufSummary {
    /// A metadata value, by a key from [`crate::keys`].
    pub fn meta(&self, key: &SmolStr) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// A metadata value read as a u32, absent where the key is missing or
    /// holds something that does not fit one.
    pub fn meta_u32(&self, key: &SmolStr) -> Option<u32> {
        self.meta(key).and_then(GgufValue::as_u32)
    }

    /// A metadata value read as a u64, absent where the key is missing or
    /// holds something that does not fit one. Callers doing byte arithmetic
    /// want this rather than a `u32` they immediately cast.
    pub fn meta_u64(&self, key: &SmolStr) -> Option<u64> {
        self.meta_u32(key).map(u64::from)
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: SmolStr,
    pub dtype: GgufType,
    pub shape: Vec<u64>,
    pub byte_size: u64,
    /// 0-based shard index where this tensor lives.
    pub shard_idx: u16,
    /// Byte offset within the shard's tensor-data region.
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// GGUF type tags (Q4_0, IQ2_XXS, ...) are the file format's wire spelling;
// renaming them to camelCase would corrupt the reader.
#[expect(non_camel_case_types)]
pub enum GgufType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M,
    I8,
    I16,
    I32,
    I64,
    F64,
    /// Ternary quant 1.7 bpe (ggml id 34).
    TQ1_0,
    /// Ternary quant 2.0 bpe (ggml id 35).
    TQ2_0,
    /// OpenAI MXFP4 4-bit, used by gpt-oss experts (ggml id 39).
    MXFP4,
    // ik_llama.cpp fork types (ggml ids 133-158). The fork parks its
    // additions well above mainline's range to avoid collisions; sizes
    // taken from its ggml-common.h block structs. Seen in the wild in
    // ik-native quants of GLM-5.2 (muzzy/sokann/ubergarm recipes).
    /// ik q6_0: legacy-style 32-block, 26 bytes (6.5 bpw). ggml id 133.
    Q6_0,
    /// ik IQ2_K family, 256-superblock (ggml ids 137-141, 144-146).
    IQ2_K,
    IQ3_K,
    IQ4_K,
    IQ5_K,
    IQ6_K,
    IQ4_KS,
    IQ2_KS,
    IQ4_KSS,
    /// ik KV-cache-oriented q8 variant, 32-block, 1.0 bpe. ggml id 151.
    Q8_KV,
    /// ik KS/KT/KL additions, 256-superblock (ggml ids 152-158).
    IQ5_KS,
    IQ2_KT,
    IQ3_KT,
    IQ4_KT,
    IQ3_KS,
    IQ2_KL,
    IQ1_KT,
    Unknown(u32),
}

impl GgufType {
    pub fn from_u32(n: u32) -> Self {
        match n {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            133 => Self::Q6_0,
            137 => Self::IQ2_K,
            138 => Self::IQ3_K,
            139 => Self::IQ4_K,
            140 => Self::IQ5_K,
            141 => Self::IQ6_K,
            144 => Self::IQ4_KS,
            145 => Self::IQ2_KS,
            146 => Self::IQ4_KSS,
            151 => Self::Q8_KV,
            152 => Self::IQ5_KS,
            153 => Self::IQ2_KT,
            154 => Self::IQ3_KT,
            155 => Self::IQ4_KT,
            156 => Self::IQ3_KS,
            157 => Self::IQ2_KL,
            158 => Self::IQ1_KT,
            other => Self::Unknown(other),
        }
    }

    /// The ggml type name, lowercased — how llama.cpp spells the type in
    /// `--cache-type-k` / `-ctk` and in its own load log. `None` for a type id
    /// the reader did not recognise, which has no name to give.
    pub fn as_str(self) -> Option<&'static str> {
        NAMES
            .iter()
            .find(|(ty, _)| *ty == self)
            .map(|(_, name)| *name)
    }

    /// The inverse of [`Self::as_str`], case-insensitive so llama.cpp's own
    /// mixed-case K-quant spellings (`q4_K`) parse alongside the lowercase ones
    /// an operator is more likely to write.
    pub fn from_name(name: &str) -> Option<Self> {
        NAMES
            .iter()
            .find(|(_, candidate)| candidate.eq_ignore_ascii_case(name))
            .map(|(ty, _)| *ty)
    }
}

/// Every named ggml type, paired with its name exactly once so the two
/// directions cannot disagree. [`GgufType::Unknown`] is deliberately absent: it
/// stands for an id no name is known for.
const NAMES: &[(GgufType, &str)] = &[
    (GgufType::F32, "f32"),
    (GgufType::F16, "f16"),
    (GgufType::BF16, "bf16"),
    (GgufType::F64, "f64"),
    (GgufType::I8, "i8"),
    (GgufType::I16, "i16"),
    (GgufType::I32, "i32"),
    (GgufType::I64, "i64"),
    (GgufType::Q4_0, "q4_0"),
    (GgufType::Q4_1, "q4_1"),
    (GgufType::Q5_0, "q5_0"),
    (GgufType::Q5_1, "q5_1"),
    (GgufType::Q8_0, "q8_0"),
    (GgufType::Q8_1, "q8_1"),
    (GgufType::Q2K, "q2_k"),
    (GgufType::Q3K, "q3_k"),
    (GgufType::Q4K, "q4_k"),
    (GgufType::Q5K, "q5_k"),
    (GgufType::Q6K, "q6_k"),
    (GgufType::Q8K, "q8_k"),
    (GgufType::IQ1_S, "iq1_s"),
    (GgufType::IQ1_M, "iq1_m"),
    (GgufType::IQ2_XXS, "iq2_xxs"),
    (GgufType::IQ2_XS, "iq2_xs"),
    (GgufType::IQ2_S, "iq2_s"),
    (GgufType::IQ3_XXS, "iq3_xxs"),
    (GgufType::IQ3_S, "iq3_s"),
    (GgufType::IQ4_NL, "iq4_nl"),
    (GgufType::IQ4_XS, "iq4_xs"),
    (GgufType::TQ1_0, "tq1_0"),
    (GgufType::TQ2_0, "tq2_0"),
    (GgufType::MXFP4, "mxfp4"),
    (GgufType::Q6_0, "q6_0"),
    (GgufType::Q8_KV, "q8_kv"),
    (GgufType::IQ2_K, "iq2_k"),
    (GgufType::IQ3_K, "iq3_k"),
    (GgufType::IQ4_K, "iq4_k"),
    (GgufType::IQ5_K, "iq5_k"),
    (GgufType::IQ6_K, "iq6_k"),
    (GgufType::IQ2_KS, "iq2_ks"),
    (GgufType::IQ3_KS, "iq3_ks"),
    (GgufType::IQ4_KS, "iq4_ks"),
    (GgufType::IQ5_KS, "iq5_ks"),
    (GgufType::IQ4_KSS, "iq4_kss"),
    (GgufType::IQ1_KT, "iq1_kt"),
    (GgufType::IQ2_KT, "iq2_kt"),
    (GgufType::IQ3_KT, "iq3_kt"),
    (GgufType::IQ4_KT, "iq4_kt"),
    (GgufType::IQ2_KL, "iq2_kl"),
];

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U8(v) => Some(*v as u32),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u32),
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            GgufValue::I64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Interpret this value as a per-layer u32 series. Used for architectures
    /// (Nvidia's `deci`, among others) that store metadata like
    /// `{arch}.attention.head_count_kv` as an array with one entry per
    /// transformer block rather than a single scalar. A scalar coerces to
    /// a one-element vector so the caller can treat both shapes uniformly.
    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            GgufValue::Array(items) => items.iter().map(|v| v.as_u32()).collect(),
            other => other.as_u32().map(|v| vec![v]),
        }
    }

    /// Interpret this value as a per-layer bool mask. Used by Gemma 4's
    /// `{arch}.attention.sliding_window_pattern`, which is `true` for
    /// SWA layers and `false` for full-attention layers. A scalar bool
    /// coerces to a one-element vector; anything else returns `None`.
    pub fn as_bool_array(&self) -> Option<Vec<bool>> {
        match self {
            GgufValue::Array(items) => items
                .iter()
                .map(|v| match v {
                    GgufValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect(),
            GgufValue::Bool(b) => Some(vec![*b]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Every type the reader can name from an id must also be nameable as a
    /// string and parse back — a variant added to `from_u32` but forgotten in
    /// `NAMES` would otherwise be silently unspellable in a config.
    #[test]
    fn every_known_type_round_trips_through_its_name() {
        for id in 0..256 {
            let ty = GgufType::from_u32(id);
            if matches!(ty, GgufType::Unknown(_)) {
                continue;
            }
            let name = ty.as_str().unwrap_or_else(|| panic!("{ty:?} has no name"));
            assert_eq!(GgufType::from_name(name), Some(ty));
        }
    }

    #[test]
    fn names_are_distinct() {
        let names: HashSet<&str> = NAMES.iter().map(|(_, name)| *name).collect();
        assert_eq!(names.len(), NAMES.len(), "two types share a name");
    }

    #[test]
    fn a_name_parses_regardless_of_case() {
        assert_eq!(GgufType::from_name("Q4_K"), Some(GgufType::Q4K));
        assert_eq!(GgufType::from_name("q4_k"), Some(GgufType::Q4K));
        assert_eq!(GgufType::from_name("not-a-type"), None);
    }
}
