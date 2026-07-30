//! The linear algebra behind the fit: weighted normal equations, Gaussian
//! elimination, and the active-set search that keeps every coefficient
//! non-negative.
//!
//! Deliberately a few dozen lines of dense arithmetic rather than a linear-algebra
//! dependency. The largest system is eight columns over a few hundred rows, the
//! conditioning is checked by the pivot test rather than assumed, and the routine
//! has to reproduce the Python it replaces bit for bit — which means the summation
//! order is part of the contract.

use ananke_estimate::compute_model::Columns;

use crate::compute_model::{Coefficients, Row, column_value};

/// The modelled per-device compute for one design row, in MiB.
///
/// Sums in the coefficient list's own order, which is the order the greedy
/// selection admitted the columns in.
pub fn evaluate(coefficients: &Coefficients, columns: &Columns) -> f64 {
    coefficients
        .iter()
        .map(|(name, value)| value * column_value(columns, name))
        .sum()
}

/// The least-squares solution over `live`, weighted by `1 / target`, or `None` if
/// the normal equations are singular.
pub(crate) fn weighted(points: &[Row], live: &[&'static str]) -> Option<Vec<f64>> {
    let weights: Vec<f64> = points.iter().map(|p| 1.0 / p.target).collect();
    let normal: Vec<Vec<f64>> = live
        .iter()
        .map(|a| {
            live.iter()
                .map(|b| {
                    points
                        .iter()
                        .zip(&weights)
                        .map(|(p, w)| w * column_value(&p.columns, a) * column_value(&p.columns, b))
                        .sum()
                })
                .collect()
        })
        .collect();
    let right: Vec<f64> = live
        .iter()
        .map(|a| {
            points
                .iter()
                .zip(&weights)
                .map(|(p, w)| w * column_value(&p.columns, a) * p.target)
                .sum()
        })
        .collect();
    solve(&normal, &right)
}

/// The non-negative solution, found by dropping the most negative column and
/// re-solving until none remain.
///
/// Lawson-Hanson's elimination step without its exchange step. That is enough
/// here: the columns are few and a column the constrained fit wants to push below
/// zero is genuinely unidentifiable in the group rather than merely awkward, so
/// re-admitting it later would not help.
pub(crate) fn non_negative(points: &[Row], live: &[&'static str]) -> Option<Coefficients> {
    let mut active: Vec<&'static str> = live.to_vec();
    while !active.is_empty() {
        let solution = weighted(points, &active)?;
        // The first minimum wins a tie, so the earlier column in the priority
        // order is the one kept.
        let mut worst = 0;
        for index in 1..active.len() {
            if solution[index] < solution[worst] {
                worst = index;
            }
        }
        if solution[worst] >= 0.0 {
            return Some(active.into_iter().zip(solution).collect());
        }
        active.remove(worst);
    }
    None
}

/// Gaussian elimination with partial pivoting, or `None` if singular.
fn solve(matrix: &[Vec<f64>], right: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    let mut augmented: Vec<Vec<f64>> = matrix
        .iter()
        .zip(right)
        .map(|(row, &value)| {
            let mut row = row.clone();
            row.push(value);
            row
        })
        .collect();
    for i in 0..n {
        let mut pivot = i;
        for r in i + 1..n {
            if augmented[r][i].abs() > augmented[pivot][i].abs() {
                pivot = r;
            }
        }
        augmented.swap(i, pivot);
        if augmented[i][i].abs() < 1e-9 {
            return None;
        }
        for r in i + 1..n {
            let (above, below) = augmented.split_at_mut(r);
            let pivot_row = &above[i];
            let row = &mut below[0];
            let factor = row[i] / pivot_row[i];
            for c in i..=n {
                row[c] -= factor * pivot_row[c];
            }
        }
    }
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        let known: f64 = (i + 1..n).map(|j| augmented[i][j] * solution[j]).sum();
        solution[i] = (augmented[i][n] - known) / augmented[i][i];
    }
    Some(solution)
}
