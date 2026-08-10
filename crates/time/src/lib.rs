//! Wall-clock helpers shared across the daemon's subsystems.
//!
//! Timestamps are captured once here so every consumer uses the same
//! epoch convention (`i64` millis for log timestamps and DB rows, `u64`
//! millis for APIs that carry them unsigned).

use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock milliseconds since the Unix epoch.
///
/// Returned as `i64` because that is the width we use for log timestamps,
/// DB rows (`service_logs.timestamp_ms`), and `ServiceConfig` durations.
pub fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Unsigned variant of [`now_unix_ms`] for APIs that carry millis as `u64`.
pub fn now_unix_ms_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_unix_ms_is_recent() {
        let now = now_unix_ms();
        // Sanity bound: within 10 minutes of the test running.
        let lower = now_unix_ms() - 600_000;
        assert!(now >= lower, "clock went backwards: {now} < {lower}");
    }

    #[test]
    fn u64_variant_matches_i64() {
        assert_eq!(now_unix_ms_u64(), now_unix_ms() as u64);
    }
}
