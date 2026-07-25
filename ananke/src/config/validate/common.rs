//! Small helpers shared across the validation passes: unit conversion,
//! duration parsing, the vocabulary-table lookups, and the config-error
//! constructor.

use std::path::PathBuf;

use crate::errors::ExpectedError;

/// Convert GiB (as declared by users in config) to MiB using the same
/// truncating cast the validator has always used. Centralised so the oneshot
/// API path and the TOML path agree on rounding.
pub fn gib_to_mib(gib: f32) -> u64 {
    (gib * 1024.0) as u64
}

/// Look up a variant's flag string in its `VARIANTS` table. Every variant
/// is registered (guarded by each enum's `*_variants_round_trip` test), so
/// the lookup is total in practice.
pub(crate) fn variant_flag<T: Copy + PartialEq>(
    table: &[(T, &'static str)],
    value: T,
) -> &'static str {
    table
        .iter()
        .find_map(|&(v, flag)| (v == value).then_some(flag))
        .expect("enum variant is registered in its VARIANTS table")
}

/// Inverse of [`variant_flag`]: resolve an accepted string to its variant.
pub(crate) fn flag_variant<T: Copy>(table: &[(T, &'static str)], s: &str) -> Option<T> {
    table.iter().find_map(|&(v, flag)| (flag == s).then_some(v))
}

pub(crate) fn fail(msg: String) -> ExpectedError {
    ExpectedError::config_unparseable(PathBuf::from("<config>"), msg)
}

pub(crate) fn parse_duration_ms(s: &str) -> Result<u64, String> {
    // Accepts "10m", "30s", "500ms", "2h". Returns milliseconds.
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        return rest.parse::<u64>().map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('s') {
        return rest
            .parse::<u64>()
            .map(|n| n * 1000)
            .map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('m') {
        return rest
            .parse::<u64>()
            .map(|n| n * 60_000)
            .map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('h') {
        return rest
            .parse::<u64>()
            .map(|n| n * 3_600_000)
            .map_err(|e| e.to_string());
    }
    Err(format!("unrecognised duration: {s}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_parser() {
        assert_eq!(parse_duration_ms("500ms").unwrap(), 500);
        assert_eq!(parse_duration_ms("30s").unwrap(), 30_000);
        assert_eq!(parse_duration_ms("10m").unwrap(), 600_000);
        assert_eq!(parse_duration_ms("2h").unwrap(), 7_200_000);
        assert!(parse_duration_ms("bogus").is_err());
    }
}
