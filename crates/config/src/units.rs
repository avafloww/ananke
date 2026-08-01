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

/// `gib` GiB as whole MiB, rounded down. Config states reservations in GiB and
/// every consumer wants MiB.
pub fn gib_to_mib(gib: f32) -> u64 {
    (gib.max(0.0) * 1024.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reservation stated in GiB must never wrap into a small one.
    #[test]
    fn a_negative_reservation_clamps_rather_than_wrapping() {
        assert_eq!(gib_to_mib(-1.0), 0);
        assert_eq!(gib_to_mib(24.0), 24 * 1024);
    }
}
