//! What an MTP context costs in VRAM.
//!
//! Three terms, taken three different ways: the draft context's own compute buffer is
//! *read* from the runtime's buffer lines and cross-checked against its `[spec]`
//! summary; a separate draft's context slope is taken from differences between
//! contexts, which cancel its weight term exactly; and what is left over after every
//! buffer llama.cpp names is the residual that turns the whole model from a fitted
//! constant into an accounting identity.

use std::collections::BTreeMap;

use ananke_dataset::BufferRole;

use crate::{
    derive::{
        Scalar,
        error::{DeriveError, Result},
        mtp::pairs::mtp_pairs,
        shape::device_context_sums,
        stats::round_half_even,
    },
    record::Record,
};

/// How a separate MTP draft's compute grows with context, MiB per 1024.
///
/// The draft shares the target's KV cache — the load log shows `layer 3: sharing with
/// layer 59` — so it has no context-scaling *cache* term, and the constant covering
/// it was flat. Its compute buffer scales anyway: gemma-4-31B-QAT measures a driver
/// delta of 724, 788, and 920 MiB at ctx 32768, 65536, and 131072, against a modelled
/// 407 at every one.
///
/// Taken from differences between contexts, which cancel the draft's weight term
/// exactly and so need no figure for it. Flat in the slot count — 724, 724, and 728
/// MiB at one, two, and four slots — which is the control that confirms the embedded
/// head's slot scaling is its own per-slot cache.
pub fn draft_compute_slope(rows: &[Record]) -> Result<Scalar> {
    let mut by_ctx: BTreeMap<u32, i64> = BTreeMap::new();
    for pair in mtp_pairs(rows, true) {
        by_ctx
            .entry(pair.ctx)
            .and_modify(|held| *held = (*held).max(pair.delta))
            .or_insert(pair.delta);
    }
    if by_ctx.len() < 2 {
        return Err(DeriveError::no_data(
            "fewer than two contexts for a separate draft",
        ));
    }
    let lo = *by_ctx.keys().next().expect("non-empty");
    let hi = *by_ctx.keys().next_back().expect("non-empty");
    let slope = (by_ctx[&hi] - by_ctx[&lo]) as f64 / (f64::from(hi - lo) / 1024.0);
    let series = by_ctx
        .iter()
        .map(|(ctx, value)| format!("{ctx} -> {value} MiB"))
        .collect::<Vec<_>>()
        .join(", ");
    // Rounded up, since the term is charged against an under-reservation that OOMs
    // and the whole range it spans here is 196 MiB.
    Ok(Scalar {
        value: slope.ceil() as i64,
        evidence: format!(
            "driver delta across ctx {lo}-{hi} on the one model with a separate draft: \
             {series}. Taken from differences between contexts, which cancel the \
             draft's weight term. Flat in the slot count, which is the control \
             confirming the embedded head's slot scaling is its own per-slot cache."
        ),
    })
}

/// What an MTP process takes per device beyond every buffer llama.cpp names.
///
/// The driver delta between a with-MTP cell and its without-MTP twin decomposes
/// almost entirely into buffers the runtime reports: the draft context's cache and
/// graph, the extra recurrent rollback copies in the main context, and a small growth
/// in the main context's own graph. What is left over is constant in the slot count —
/// 52 MiB at one slot and at four alike on one card, 172 and 174 on two — so it is a
/// per-device cost of standing up a second context that the runtime's own books do
/// not carry.
///
/// Deriving it is what turns the MTP model from a fitted constant into an accounting
/// identity: every term above is measured or modelled, and this is the residual.
pub fn mtp_unaccounted(rows: &[Record]) -> Result<Scalar> {
    let mut per_device = Vec::new();
    let mut detail = Vec::new();
    for pair in mtp_pairs(rows, false) {
        let cards = pair.on.factors.cards_nonempty();
        let (on, off) = (
            device_context_sums(pair.on),
            device_context_sums(pair.without),
        );
        if on.len() < 2 || off.is_empty() {
            continue;
        }
        let draft = on[0];
        let main_on = on[on.len() - 1];
        let main_off = off[off.len() - 1];
        let named = draft.kv
            + cards as f64 * draft.compute
            + cards as f64 * (main_on.rs - main_off.rs)
            + cards as f64 * (main_on.compute - main_off.compute);
        let gap = (pair.delta as f64 - named) / cards as f64;
        if gap <= 0.0 {
            continue;
        }
        per_device.push(gap);
        detail.push(format!(
            "{} {cards}card np{} ctx{} {gap:.0}",
            pair.on.parsed.arch.as_str(),
            pair.on.factors.parallel,
            pair.ctx,
        ));
    }
    if per_device.is_empty() {
        return Err(DeriveError::no_data(
            "no embedded-MTP pair with a full context breakdown",
        ));
    }
    // The largest, so a model measured on one card is not under-reserved when spread
    // across more.
    let worst = per_device.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Scalar {
        value: round_half_even(worst),
        evidence: format!(
            "{} paired with/without cell(s), as the driver delta less every buffer \
             llama.cpp names — the draft context's cache and graph, the extra recurrent \
             rollback copies, and the main graph's growth — over the card count: {} \
             MiB/device. Flat in the slot count, which is what makes it a per-context \
             cost rather than a share of one of the terms above. The largest is taken.",
            per_device.len(),
            detail.join("; "),
        ),
    })
}

/// The MTP draft context's own compute buffer, per device, by architecture.
///
/// Taken from what the runtime reports for that context and nothing else. The
/// `[spec] estimated memory usage of MTP context is N MiB` line is exact for the
/// draft context — 258 MiB for Qwen3.6-27B at ctx 32768 is precisely the 128 MiB
/// cache plus the 130 MiB the load log then shows as that context's device compute —
/// so the quantity wanted here is `reported - cache`, read straight off.
///
/// What the old model got wrong was not the magnitude but the *shape*. It built the
/// share from an f16 KQ mask over one stream's share of the context, which is wrong
/// twice over: the draft context keeps a single cache spanning every cell, so
/// `--parallel` does not narrow it (130.02 MiB at one, two, and four slots alike), and
/// the width itself is far below a full mask — the share grows 0.36 MiB per 1024 tokens
/// on qwen35 and 0.45 on qwen35moe, against the 1.00 a full `ubatch x n_kv x 2` mask
/// would cost at ubatch 512.
///
/// So it is fitted as `base + slope x (ctx / 1024)` per architecture, which the
/// measured points support to within 26 MiB across a 32768-to-524288 range.
pub fn mtp_draft_compute(rows: &[Record]) -> Result<DraftComputeFit> {
    let mut by_arch: BTreeMap<String, BTreeMap<u32, f64>> = BTreeMap::new();
    for record in rows {
        let (factors, parsed) = (&record.factors, &record.parsed);
        if !factors.has_spec() || factors.draft.as_deref().is_some_and(|d| !d.is_empty()) {
            continue;
        }
        if parsed.contexts.len() < 2 {
            continue;
        }
        let draft = &parsed.contexts[0];
        let cache: f64 = draft.kv_pools.iter().map(|p| p.total_mib).sum();
        let share: f64 = draft
            .buffers
            .iter()
            .filter(|(name, _)| !name.ends_with("_Host") && !name.starts_with("CPU"))
            .map(|(_, buffers)| buffers.get(&BufferRole::Compute).copied().unwrap_or(0.0))
            .sum();
        if share <= 0.0 {
            continue;
        }
        // Cross-check the runtime's own summary line against the per-buffer lines, so
        // a parse that picked up the wrong context is caught here rather than becoming
        // a coefficient.
        let reported = parsed.mtp_context_mib;
        if reported != 0.0 && (reported - (cache + share)).abs() > 1.0 {
            return Err(DeriveError::disagreement(format!(
                "{} ctx {}: [spec] reports {reported} MiB but the buffer lines sum \
                     to {}",
                parsed.arch.as_str(),
                factors.ctx,
                cache + share,
            )));
        }
        by_arch
            .entry(parsed.arch.clone())
            .or_default()
            .insert(factors.ctx, share);
    }

    let mut bases: BTreeMap<String, i64> = BTreeMap::new();
    let mut slopes: BTreeMap<String, i64> = BTreeMap::new();
    let mut detail = Vec::new();
    for (arch, points) in &by_arch {
        if points.len() < 2 {
            continue;
        }
        let xs: Vec<f64> = points.keys().map(|ctx| f64::from(*ctx) / 1024.0).collect();
        let ys: Vec<f64> = points.values().copied().collect();
        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let sxy: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
        let slope = if sxx != 0.0 { sxy / sxx } else { 0.0 };
        let mut base = mean_y - slope * mean_x;
        // Lift the base so the line covers every measured point: this term is a
        // reservation, and the draft context is small enough that a few tens of MiB of
        // headroom costs less than an OOM.
        let deficit = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| y - (base + slope * x))
            .fold(f64::NEG_INFINITY, f64::max);
        base += deficit.max(0.0);
        bases.insert(arch.clone(), round_half_even(base));
        slopes.insert(arch.clone(), round_half_even(slope * 1000.0));
        detail.push(format!(
            "{arch} {} context(s) {base:.0}+{slope:.3}/1k",
            points.len()
        ));
    }
    if bases.is_empty() {
        return Err(DeriveError::no_data(
            "no embedded-MTP architecture measured at two contexts",
        ));
    }
    Ok(DraftComputeFit {
        bases,
        slopes,
        evidence: format!(
            "read from the runtime's own draft-context buffers, cross-checked against \
             its [spec] line: {}. The slope is stored per 1000 of `ctx / 1024`. Flat in \
             the slot count, the draft cache spanning every cell rather than one \
             stream's share.",
            detail.join("; "),
        ),
    })
}

/// The MTP draft context's compute buffer as a per-architecture line.
#[derive(Debug, Clone)]
pub struct DraftComputeFit {
    /// The buffer at zero context, in MiB.
    pub bases: BTreeMap<String, i64>,
    /// The slope, in thousandths of a MiB per 1024 context tokens.
    pub slopes: BTreeMap<String, i64>,
    pub evidence: String,
}
