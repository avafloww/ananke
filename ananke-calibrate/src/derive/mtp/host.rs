//! What an MTP context costs in host memory.
//!
//! Both constants here were held under review on one or two models at two contexts.
//! The slot sweep added the axis they lacked, and the answer is that slots are not it:
//! the host cost is flat in the slot count and scales with context instead.

use std::collections::BTreeMap;

use crate::{
    derive::{
        Scalar,
        error::{DeriveError, Result},
        mtp::pairs::mtp_pairs,
        ordered::OrderedMap,
    },
    record::Record,
};

/// One MTP shape's *host* cost, fitted as `base + slope x (ctx / 1024)`.
///
/// Both host constants were held under review on one or two models at two contexts.
/// The slot sweep adds the axis they lacked, and the answer is that slots are not
/// it: the host cost is flat in slot count — 239, 243, and 240 MiB for Qwen3.6-27B
/// at one, two, and four — and scales with context instead, at about 1.1 MiB per
/// 1024 on every model of both shapes.
///
/// Which makes both flat constants wrong in opposite directions: the embedded figure
/// under-reserves by 117 MiB at ctx 131072 and the separate-draft one over-reserves
/// by 128.
pub fn mtp_host_fit(rows: &[Record], draft: bool) -> Result<HostFit> {
    let mut by_model: OrderedMap<String, BTreeMap<u32, i64>> = OrderedMap::new();
    for pair in mtp_pairs(rows, draft) {
        if pair.host_delta <= 0 {
            continue;
        }
        // The worst at each (model, context): a pair taken across sittings reads
        // high, and `mtp_pairs` already drops the far-apart ones.
        by_model
            .or_insert_with(pair.model.clone(), BTreeMap::new)
            .entry(pair.ctx)
            .and_modify(|held| *held = (*held).max(pair.host_delta))
            .or_insert(pair.host_delta);
    }
    let mut slopes: Vec<f64> = Vec::new();
    let mut points: Vec<(u32, i64)> = Vec::new();
    for (_model, series) in by_model.iter() {
        points.extend(series.iter().map(|(ctx, value)| (*ctx, *value)));
        if series.len() > 1 {
            let lo = *series.keys().next().expect("non-empty");
            let hi = *series.keys().next_back().expect("non-empty");
            slopes.push((series[&hi] - series[&lo]) as f64 / (f64::from(hi - lo) / 1024.0));
        }
    }
    if points.is_empty() {
        return Err(DeriveError::no_data("no MTP pairs with a host delta"));
    }
    let worst_slope = slopes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let slope = if slopes.is_empty() {
        0
    } else {
        worst_slope.ceil() as i64
    };
    // Plus a margin, for the reason the GPU fit needs one: taking the worst residual
    // reproduces some model's point exactly — the separate draft lands on 287 MiB
    // against 287 measured — and an exact fit under-reserves on any variation at all.
    let residual = points
        .iter()
        .map(|(ctx, value)| *value as f64 - slope as f64 * (f64::from(*ctx) / 1024.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let base = (residual * 1.10).ceil() as i64;
    Ok(HostFit {
        base,
        slope,
        evidence: format!(
            "{} paired with/without cells across {} model(s), host owned memory rather \
             than driver VRAM. Flat in the slot count and linear in context at {:.1} \
             MiB per 1024, so the flat constant this replaces was wrong in shape. Base \
             covers the worst residual over every cell.",
            points.len(),
            by_model.len(),
            if slopes.is_empty() { 0.0 } else { worst_slope },
        ),
    })
}

/// The base and context slope of an MTP shape's host cost, in MiB.
#[derive(Debug, Clone)]
pub struct HostFit {
    pub base: i64,
    pub slope: i64,
    pub evidence: String,
}

/// Host cost of an embedded MTP head.
pub fn mtp_host_embedded(rows: &[Record]) -> Result<Scalar> {
    let fit = mtp_host_fit(rows, false)?;
    Ok(Scalar {
        value: fit.base * 1024 * 1024,
        evidence: fit.evidence,
    })
}

/// Host cost of a separate MTP draft model.
pub fn mtp_host_separate(rows: &[Record]) -> Result<Scalar> {
    let fit = mtp_host_fit(rows, true)?;
    Ok(Scalar {
        value: fit.base * 1024 * 1024,
        evidence: fit.evidence,
    })
}

/// The context slope both MTP shapes share, in MiB per 1024 tokens.
pub fn mtp_host_slope(rows: &[Record]) -> Result<Scalar> {
    let embedded = mtp_host_fit(rows, false)?.slope;
    let separate = mtp_host_fit(rows, true)?.slope;
    Ok(Scalar {
        value: embedded.max(separate),
        evidence: format!(
            "the larger of the two MTP shapes' host context slopes — embedded \
             {embedded}, separate draft {separate} MiB per 1024 tokens. One value for \
             both, since they measure within a megabyte of each other and a second \
             constant would claim a distinction the data does not show."
        ),
    })
}
