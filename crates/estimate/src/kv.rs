//! KV cache bytes-per-element table.

use ananke_gguf::GgufType;

/// Approximate bytes per element for a KV cache stored as `cache_type`.
///
/// Every type ggml names gets a number, not just the ones llama.cpp accepts as
/// a cache type: this table prices whatever it is handed, and refusing an
/// impossible cache is config validation's job.
pub fn kv_bytes_per_element(cache_type: GgufType) -> f64 {
    match cache_type {
        GgufType::F32 => 4.0,
        GgufType::F16 | GgufType::BF16 => 2.0,
        GgufType::Q8_0 => 1.0625, // 34 bytes / 32 elements
        GgufType::Q5_1 => 0.75,   // 24/32
        GgufType::Q5_0 => 0.6875, // 22/32
        GgufType::Q4_1 => 0.625,  // 20/32
        GgufType::Q4_0 | GgufType::IQ4_NL => 0.5625, // 18/32
        GgufType::Q6_0 => 0.8125, // 26/32 (ik_llama.cpp fork)
        GgufType::Q8_KV => 1.0,   // ik_llama.cpp fork
        _ => 2.0,                 // unpriced → f16 equivalent
    }
}

/// Whether a cache type stores the cache quantised.
///
/// Guardrail: `f16` and `f32` are the unquantised forms and *everything* else —
/// `bf16` included — is charged the quantised rate. That rate was fitted
/// against this exact partition, so narrowing it means refitting
/// `quantised_cache_rates`, not just editing this line. The string-keyed twin
/// is [`ananke_config::flags::cache_type::is_quantised`], which the
/// calibration's `KvType` uses; the test below holds the two to one verdict.
pub fn is_quantised(cache_type: GgufType) -> bool {
    !matches!(cache_type, GgufType::F16 | GgufType::F32)
}

#[cfg(test)]
mod tests {
    use ananke_config::flags::cache_type;

    use super::*;

    #[test]
    fn q8_0_is_around_1_point_06() {
        assert!((kv_bytes_per_element(GgufType::Q8_0) - 1.0625).abs() < 1e-6);
    }

    #[test]
    fn an_unpriced_type_falls_back_to_f16() {
        assert_eq!(kv_bytes_per_element(GgufType::Q4K), 2.0);
        assert_eq!(kv_bytes_per_element(GgufType::Unknown(999)), 2.0);
    }

    #[test]
    fn q6_0_and_q8_kv_have_correct_bpe() {
        assert_eq!(kv_bytes_per_element(GgufType::Q6_0), 0.8125);
        assert_eq!(kv_bytes_per_element(GgufType::Q8_KV), 1.0);
    }

    /// The estimator's partition and the calibration's must agree on every
    /// cache type an operator can write, or a row is fitted under one rate and
    /// estimated under the other.
    #[test]
    fn the_two_quantised_partitions_agree() {
        for name in cache_type::ALL {
            let ty = GgufType::from_name(name).unwrap_or_else(|| panic!("{name} has no ggml type"));
            assert_eq!(
                is_quantised(ty),
                cache_type::is_quantised(name),
                "{name} is classified differently by the two partitions"
            );
        }
    }
}
