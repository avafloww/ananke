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
        mtp::pairs::{MtpShape, mtp_pairs},
        ordered::OrderedMap,
        units::MIB_I64,
    },
    record::Record,
};

/// How far above the worst measured residual the fitted base is lifted. Taking
/// the worst reproduces some model's point exactly — the separate draft lands on
/// 287 MiB against 287 measured — and an exact fit under-reserves on any variation
/// at all.
const HOST_BASE_MARGIN: f64 = 1.10;

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
pub fn mtp_host_fit(rows: &[Record], shape: MtpShape) -> Result<HostFit> {
    let mut by_model: OrderedMap<String, BTreeMap<u32, i64>> = OrderedMap::new();
    for pair in mtp_pairs(rows, shape) {
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
    let residual = points
        .iter()
        .map(|(ctx, value)| *value as f64 - slope as f64 * (f64::from(*ctx) / 1024.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let base = (residual * HOST_BASE_MARGIN).ceil() as i64;
    Ok(HostFit {
        base,
        slope,
        // The slope reported is the one charged, not the one measured: `slope`
        // is `worst_slope` rounded up, and a measured 1.04 printed as "1.0"
        // beside a charged 2 makes this document contradict itself — which is
        // the failure the evidence field exists to prevent. The measured figure
        // rides alongside so the rounding is visible rather than hidden.
        evidence: format!(
            "{} paired with/without cells across {} model(s), host owned memory rather \
             than driver VRAM. Flat in the slot count and linear in context at {} MiB \
             per 1024 (worst measured {:.2}, rounded up), so the flat constant this \
             replaces was wrong in shape. Base covers the worst residual over every cell.",
            points.len(),
            by_model.len(),
            slope,
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
    let fit = mtp_host_fit(rows, MtpShape::Embedded)?;
    Ok(Scalar {
        value: fit.base * MIB_I64,
        evidence: fit.evidence,
    })
}

/// Host cost of a separate MTP draft model.
pub fn mtp_host_separate(rows: &[Record]) -> Result<Scalar> {
    let fit = mtp_host_fit(rows, MtpShape::SeparateDraft)?;
    Ok(Scalar {
        value: fit.base * MIB_I64,
        evidence: fit.evidence,
    })
}

/// The context slope both MTP shapes share, in MiB per 1024 tokens.
pub fn mtp_host_slope(rows: &[Record]) -> Result<Scalar> {
    let embedded = mtp_host_fit(rows, MtpShape::Embedded)?.slope;
    let separate = mtp_host_fit(rows, MtpShape::SeparateDraft)?.slope;
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
