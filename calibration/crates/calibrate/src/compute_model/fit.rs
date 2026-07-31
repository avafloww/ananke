//! Greedy column selection over the non-negative weighted least-squares solve.
//!
//! Split from the collection half because the two answer different questions: what
//! observations exist and belong to which group, against which columns a given
//! group can honestly support. The linear algebra both of these stand on is in
//! `solve`.

use std::collections::BTreeSet;

use crate::compute_model::{Row, column_value, evaluate};

/// Coefficients for one group and its worst residual as a fraction.
///
/// Weighted by 1/y, because the criterion is relative: a 200 MiB miss on a 5 GiB
/// card is the failure and the same miss on a 40 GiB card is not. Unweighted, the
/// largest cells dominate the sum of squares and the small ones fit terribly.
///
/// Constrained to non-negative coefficients, which is not a regularisation
/// convenience but the physics: every column counts bytes of a buffer that either
/// exists or does not, so a negative coefficient is always the fit paying for one
/// column's error with another's. Unconstrained, `logits` came out negative for
/// eleven of nineteen groups — it is near-collinear with `hidden`, both being
/// proportional to the batch, and separated only by which card is the head — and
/// `quant` went negative wherever a quantised cache happened to correlate with
/// something else. The active set is found by dropping the most negative column
/// and re-solving until none remain, which is Lawson-Hanson's elimination step
/// without its exchange step: enough here, because the columns are few and a
/// dropped one is genuinely unidentifiable rather than merely awkward.
pub fn fit(points: &[Row]) -> Option<(Coefficients, f64)> {
    let mut live: Vec<&'static str> = Vec::new();
    for name in PRIORITY {
        // A column that does not vary within the group is perfectly collinear
        // with `flat`, which makes the system singular and its coefficient
        // meaningless. `flat` itself is kept, being the intercept.
        if name != "flat" && !varies(points, name) {
            continue;
        }
        if points.len() < live.len() + 1 + MIN_SURPLUS_ROWS {
            break;
        }
        live.push(name);
        if crate::compute_model::solve::weighted(points, &live).is_none() {
            live.pop();
        }
    }
    if live.is_empty() {
        return None;
    }
    let coefficients = crate::compute_model::solve::non_negative(points, &live)?;
    let worst = points
        .iter()
        .map(|p| (evaluate(&coefficients, &p.columns) - p.target).abs() / p.target)
        .fold(f64::NEG_INFINITY, f64::max);
    Some((coefficients, worst))
}

/// One group's fitted coefficients, in the order the greedy selection admitted
/// them. The order is part of the value: [`evaluate`] sums in it, so a report and
/// a re-evaluation agree to the last bit.
pub type Coefficients = Vec<(&'static str, f64)>;

/// Greedy selection order. A column joins the design only if the system stays
/// solvable with it, so the order decides which of two collinear columns wins.
/// Terms that generalise across architectures come first; `doubling` comes last
/// because it is a pure function of the batch and so is collinear with `hidden`
/// in any group holding a single model width and only two batch sizes.
const PRIORITY: [&str; 8] = [
    "flat",
    "mask",
    "quant",
    "hidden",
    "head_flat",
    "logits",
    "offload_head",
    "doubling",
];

/// How many rows a group must hold beyond the columns it is fitting. Checked
/// against the *selected* columns, not all of them: a group with few cells fits a
/// short design honestly, and testing against the full column list benched four
/// groups that had plenty of rows for the three or four columns they varied.
const MIN_SURPLUS_ROWS: usize = 3;

/// The scale column values are compared as integers at, nine decimal places.
const VARIES_SCALE: f64 = 1e9;

/// Whether a column takes more than one value across the group.
///
/// Compared at nine decimals, matching the fitter this replaces: two rows whose
/// column differs only below that are the same point as far as identifiability
/// goes.
fn varies(points: &[Row], name: &str) -> bool {
    let distinct: BTreeSet<i64> = points
        .iter()
        .map(|p| (column_value(&p.columns, name) * VARIES_SCALE).round() as i64)
        .collect();
    distinct.len() > 1
}
