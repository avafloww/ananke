//! Reducing many measurements to one number, and refusing to when they disagree.

use crate::derive::error::{DeriveError, Result};

/// The default relative spread `consensus` will tolerate before refusing.
pub const DEFAULT_TOLERANCE: f64 = 0.15;

/// Round a length up to a multiple of `to`. llama.cpp pads the KV cache this
/// way, so the mask the graph builds is sized against the padded figure.
pub fn pad(value: u64, to: u64) -> u64 {
    value.div_ceil(to) * to
}

/// The median, matching Python's `statistics.median`: the mean of the two middle
/// values on an even-length sample rather than the lower of them.
pub fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("measurements are never NaN"));
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
}

/// Reduce measurements to one number, refusing when they disagree.
///
/// Every deriver in the campaign used to take a median and say nothing about the
/// spread behind it. That is how a constant absorbs a factor nobody thought to
/// group by: the median of a bimodal set is a number that describes none of its
/// members, and it looks exactly like the median of a tight one.
///
/// Ten conclusions were drawn that way and were wrong — card count, split mode,
/// cell label, runtime, flash-attention state, placement, architecture,
/// measurement time, serving state, slot count. Each produced a plausible law.
/// So a spread wider than the tolerance is treated as a *failure to have grouped
/// properly*, not as noise to average over, and it stops the derivation rather
/// than quietly widening the constant.
///
/// `absolute_floor` exists because a relative tolerance is meaningless around
/// zero: a term whose median is 0.1 and whose values span 21 reads as 218%
/// disagreement while being 21 units wide. Where the caller knows what "small"
/// means in its own units, it says so.
pub fn consensus(
    values: &[f64],
    name: &str,
    tolerance: f64,
    absolute_floor: f64,
) -> Result<f64> {
    if values.is_empty() {
        return Err(DeriveError::no_data(format!("{name}: no measurements")));
    }
    let middle = median(values);
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread_abs = hi - lo;
    if spread_abs <= absolute_floor || middle == 0.0 {
        return Ok(middle);
    }
    let spread = spread_abs / middle.abs();
    if spread > tolerance {
        return Err(DeriveError::disagreement(format!(
            "{name}: {} measurements span {} to {}, which is {} of the median — \
             they do not agree. A single value would collapse a real difference; \
             find the factor that separates them and group by it.",
            values.len(),
            format_g(lo, 4),
            format_g(hi, 4),
            format_percent(spread),
        )));
    }
    Ok(middle)
}

/// `consensus` at its default tolerance and with no absolute floor.
pub fn consensus_default(values: &[f64], name: &str) -> Result<f64> {
    consensus(values, name, DEFAULT_TOLERANCE, 0.0)
}

/// Refuse a `max` reduction that one cell alone decides.
///
/// `consensus` protects constants reduced by median: it refuses to average away
/// a disagreement. Constants reduced by `max` bypass it by design, since a
/// maximum bounds a spread rather than hiding it — but that makes a single bad
/// cell able to set the value outright, which is not a bound, it is an artifact.
///
/// It has happened: an MTP pair matched an idle cell against a served one and
/// drove a fitted base to 632 MiB where every honest pair of the same
/// configuration gives 239 to 243. The absurdity is what exposed it. This
/// catches the same shape before it needs to be absurd, by asking whether the
/// winner stands far apart from the rest rather than merely above them.
pub fn check_no_outlier_dominates(values: &[f64], name: &str, tolerance: f64) -> Result<()> {
    if values.len() < 3 {
        return Ok(());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("measurements are never NaN"));
    let top = sorted[sorted.len() - 1];
    let runner_up = sorted[sorted.len() - 2];
    if runner_up <= 0.0 || top <= 0.0 {
        return Ok(());
    }
    if top / runner_up > tolerance {
        return Err(DeriveError::disagreement(format!(
            "{name}: the largest of {} measurements is {top:.0}, {:.1}x the next \
             largest at {runner_up:.0}. A maximum is a bound when the values crowd \
             it and an artifact when one stands alone — check that cell against its \
             neighbours before letting it set the constant.",
            values.len(),
            top / runner_up,
        )));
    }
    Ok(())
}

/// The default ratio between the winner and the runner-up above which a `max`
/// reduction is treated as an artifact.
pub const OUTLIER_TOLERANCE: f64 = 4.0;

/// Python's `round` — half-to-even, where Rust's `f64::round` is half-away.
///
/// The difference only shows on an exact tie, but a tie is exactly what a
/// measurement in whole MiB divided by two produces, and a constant that differs
/// by one from the committed value is a failed `emit --check`.
pub fn py_round(value: f64) -> i64 {
    let fractional = value - value.trunc();
    if fractional.abs() != 0.5 {
        return value.round() as i64;
    }
    let floor = value.floor() as i64;
    if floor % 2 == 0 { floor } else { floor + 1 }
}

/// Python's `round(value, 1)`, as tenths, so a rate can be deduplicated and
/// sorted on the same figure it is printed as.
pub fn py_round_tenths(value: f64) -> i64 {
    py_round(value * 10.0)
}

/// Python's `%g` — significant figures, switching to exponent form outside the
/// range where a plain decimal is shorter, with trailing zeros stripped.
///
/// Reimplemented because the disagreement messages are part of the contract
/// this port has to reproduce, and Rust has no `g` formatter.
pub fn format_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let significant = precision.max(1);
    // The exponent has to be read off the *rounded* value, or a figure that
    // rounds up across a power of ten picks the wrong branch.
    let scientific = format!("{:.*e}", significant - 1, value);
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust's LowerExp always emits an exponent");
    let exponent: i32 = exponent.parse().expect("the exponent is an integer");
    if exponent < -4 || exponent >= significant as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip_trailing_zeros(mantissa), exponent.abs())
    } else {
        let decimals = (significant as i32 - 1 - exponent).max(0) as usize;
        strip_trailing_zeros(&format!("{value:.decimals$}"))
    }
}

/// Python's `{:.0%}`: a ratio as a whole-number percentage.
pub fn format_percent(ratio: f64) -> String {
    format!("{:.0}%", ratio * 100.0)
}

fn strip_trailing_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_averages_the_middle_pair() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 6.0]), 2.5);
    }

    #[test]
    fn pad_rounds_up_to_the_cache_granularity() {
        assert_eq!(pad(1, 256), 256);
        assert_eq!(pad(256, 256), 256);
        assert_eq!(pad(257, 256), 512);
    }

    #[test]
    fn consensus_accepts_a_tight_group_and_refuses_a_split_one() {
        assert_eq!(consensus_default(&[100.0, 101.0, 102.0], "tight").unwrap(), 101.0);
        let error = consensus_default(&[28.0, 42.8], "split").unwrap_err();
        assert!(error.to_string().contains("they do not agree"), "{error}");
        assert!(error.to_string().contains("28 to 42.8"), "{error}");
    }

    #[test]
    fn consensus_lets_a_small_absolute_spread_through_near_zero() {
        // The relative spread here is 2000%, but the group is two bytes wide.
        assert!(consensus(&[0.1, 2.1], "near zero", 0.15, 4.0).is_ok());
    }

    #[test]
    fn format_g_matches_python() {
        assert_eq!(format_g(28.0, 4), "28");
        assert_eq!(format_g(42.83, 4), "42.83");
        assert_eq!(format_g(1234567.0, 4), "1.235e+06");
        assert_eq!(format_g(0.0000123, 4), "1.23e-05");
        assert_eq!(format_g(0.0, 4), "0");
    }

    #[test]
    fn outlier_check_refuses_a_lone_winner() {
        assert!(check_no_outlier_dominates(&[239.0, 243.0, 240.0], "ok", OUTLIER_TOLERANCE).is_ok());
        assert!(
            check_no_outlier_dominates(&[239.0, 243.0, 1632.0], "bad", OUTLIER_TOLERANCE).is_err()
        );
    }
}
