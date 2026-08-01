//! Byte-unit conversions.
//!
//! Here because every crate in the workspace converts between bytes and MiB,
//! and five of them had grown their own `const MIB` to do it.

/// Bytes in a mebibyte.
pub const MIB: u64 = 1024 * 1024;
/// Bytes in a gibibyte.
pub const GIB: u64 = 1024 * MIB;

/// [`MIB`], for the fractional arithmetic reports and tolerances do.
pub const MIB_F64: f64 = MIB as f64;
/// [`MIB`], for the signed arithmetic a correction that can go either way does.
pub const MIB_I64: i64 = MIB as i64;

/// Whole MiB in `bytes`, rounded down.
pub fn to_mib(bytes: u64) -> u64 {
    bytes / MIB
}

/// MiB in `bytes`, keeping the fraction. For reports and tolerances, where
/// rounding to a whole MiB would hide the difference being measured.
pub fn to_mib_f64(bytes: u64) -> f64 {
    bytes as f64 / MIB_F64
}

/// `mib` MiB as bytes, saturating rather than wrapping — the inputs are
/// operator-supplied and a wrap would read as a tiny reservation.
pub fn from_mib(mib: u64) -> u64 {
    mib.saturating_mul(MIB)
}

/// `gib` GiB as whole MiB, rounded down. Config states reservations in GiB and
/// every consumer wants MiB.
pub fn gib_to_mib(gib: f32) -> u64 {
    (gib.max(0.0) * 1024.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip_through_whole_mib() {
        assert_eq!(to_mib(from_mib(37)), 37);
        assert_eq!(to_mib(MIB - 1), 0);
        assert_eq!(to_mib_f64(MIB / 2), 0.5);
    }

    /// A reservation stated in GiB must never wrap into a small one.
    #[test]
    fn oversized_and_negative_inputs_saturate_rather_than_wrap() {
        assert_eq!(from_mib(u64::MAX), u64::MAX);
        assert_eq!(gib_to_mib(-1.0), 0);
        assert_eq!(gib_to_mib(24.0), 24 * 1024);
    }
}
