//! Growth detection over a rolling sample window.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;

use crate::allocator::balloon::WINDOW_SIZE;

/// Detect growth in a sample window using linear-regression slope and a
/// majority-non-decreasing jitter check.
///
/// Returns `true` when:
/// 1. The window has at least `WINDOW_SIZE / 2 + 1` samples.
/// 2. The OLS slope over the window is strictly positive.
/// 3. A majority of adjacent-sample deltas are non-negative (jitter tolerance).
/// 4. The slope-projected next sample would exceed `floor_bytes`.
pub fn detect_growth(window: &VecDeque<u64>, floor_bytes: u64) -> bool {
    let min_samples = WINDOW_SIZE / 2 + 1;
    if window.len() < min_samples {
        return false;
    }

    let n = window.len() as i64;
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;
    let mut sum_xy: i64 = 0;
    let mut sum_xx: i64 = 0;
    for (i, v) in window.iter().enumerate() {
        let x = i as i64;
        let y = *v as i64;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom == 0 {
        return false;
    }
    // Integer slope; positive means the fit line is rising.
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    if slope <= 0 {
        return false;
    }

    // Majority of consecutive pairs must be non-decreasing.
    let samples: Vec<u64> = window.iter().copied().collect();
    let total = samples.len().saturating_sub(1);
    if total == 0 {
        return false;
    }
    let non_neg = samples.windows(2).filter(|pair| pair[1] >= pair[0]).count();
    if non_neg * 2 <= total {
        return false;
    }

    // The slope-projected next value must exceed the floor.
    // Invariant: `window` is non-empty (checked above against the minimum
    // sample count), so `back()` always yields a sample.
    let Some(&last) = window.back() else {
        unreachable!("window passed the non-empty guard");
    };
    let last = last as i64;
    let projected = last + slope;
    projected as u64 > floor_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::balloon::test_support::mk_window;

    #[test]
    fn flat_window_no_growth() {
        let w = mk_window(&[10, 10, 10, 10, 10, 10]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn monotonic_growth_detected() {
        let w = mk_window(&[10, 12, 14, 16, 18, 20]);
        assert!(detect_growth(&w, 0));
    }

    #[test]
    fn noisy_but_growing_detected() {
        let w = mk_window(&[10, 13, 12, 17, 16, 20]);
        assert!(detect_growth(&w, 0));
    }

    #[test]
    fn declining_rejected() {
        let w = mk_window(&[20, 18, 16, 14, 12, 10]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn insufficient_samples_rejected() {
        let w = mk_window(&[10, 20]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn floor_gate_applied() {
        // Growing, but projected stays below floor.
        let w = mk_window(&[10, 11, 12, 13, 14, 15]);
        assert!(!detect_growth(&w, 1000));
    }
}
