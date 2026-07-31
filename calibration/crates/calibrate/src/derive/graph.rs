//! Terms that describe the *shape* of the graph llama.cpp builds: how many times
//! it replicates a mask, and when it keeps a mixture of experts' intermediates on
//! the host.

use std::collections::{BTreeMap, BTreeSet};

use ananke_config::placement::SplitMode;

use crate::{
    derive::{
        Scalar, Table,
        arena::arena_terms,
        error::{DeriveError, Result},
        keys::{ArchCardsKey, ArchKey},
        stats::{consensus_default, median, round_half_even, round_tenths_half_even},
        tuning::Tuning,
    },
    record::Record,
};

/// How many times mainline replicates the masks under layer split.
pub fn layer_split_copies(rows: &[Record], tuning: &Tuning) -> Result<Scalar> {
    let mut multiples = Vec::new();
    let mut singles = Vec::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.runtime_is_ik() {
            continue;
        }
        let arena = parsed.arena_mib;
        if arena == 0.0 {
            continue;
        }
        if !factors.fully_offloaded() || factors.gpus.is_empty() || !factors.flash_attn_on() {
            continue;
        }
        // gemma4 carries its own per-layer term; excluded to keep this clean.
        if parsed.arch == "gemma4" {
            continue;
        }
        // A hybrid does not replicate the masks — that is the point.
        if factors.is_hybrid() {
            continue;
        }
        let terms = arena_terms(record, true, tuning);
        if terms.masks() <= 0.0 {
            continue;
        }
        let k = (arena - terms.hidden) / terms.masks();
        if factors.cards_or(1) > 1 && factors.split_or_layer() == SplitMode::Layer {
            multiples.push(k);
        } else {
            singles.push(k);
        }
    }
    if multiples.is_empty() {
        return Err(DeriveError::no_data("no mainline layer-split cells"));
    }
    consensus_default(&multiples, "layer-split mask multiple")?;
    let single_median = if singles.is_empty() {
        0.0
    } else {
        median(&singles)
    };
    Ok(Scalar {
        value: round_half_even(median(&multiples)),
        evidence: format!(
            "{} mainline layer-split cells at {:.2}, against {} single-card and \
             tensor-split cells at {single_median:.2}. Flat across context, batch, \
             slot count and cache mode; ik is 1.00 at either card count.",
            multiples.len(),
            median(&multiples),
            singles.len(),
        ),
    })
}

/// The batch threshold at which ik moves its MoE ops off the CPU.
pub fn offload_min_batch(rows: &[Record], tuning: &Tuning) -> Result<Scalar> {
    let mut on: Vec<f64> = Vec::new();
    let mut off: Vec<f64> = Vec::new();
    let mut cells = 0usize;
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if !factors.runtime_is_ik() {
            continue;
        }
        let arena = parsed.arena_mib;
        if arena == 0.0 {
            continue;
        }
        let experts = parsed.n_expert;
        let used = parsed.n_expert_used;
        if experts == 0 || used == 0 {
            continue;
        }
        let tokens = factors.tokens();
        if !factors.flash_attn_on() || !factors.fully_offloaded() {
            continue;
        }
        let terms = arena_terms(record, false, tuning);
        // Whether the term is there at all, judged from the measurement rather
        // than from the threshold being solved for. A present term is tens of
        // MiB; an absent one leaves hundredths.
        let excess_per_token = (arena - terms.total()) * 1048576.0 / tokens as f64;
        let ratio = (tokens * used) as f64 / experts as f64;
        cells += 1;
        if excess_per_token > 1024.0 {
            on.push(ratio)
        } else {
            off.push(ratio)
        }
    }
    if on.is_empty() || off.is_empty() {
        return Err(DeriveError::no_data(
            "the threshold is not bracketed by the dataset",
        ));
    }
    let worst_on = on.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let first_off = off.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(Scalar {
        value: round_half_even(first_off),
        evidence: format!(
            "bracketed to ({worst_on:.0}, {first_off:.0}] by {cells} ik MoE cells \
             across two models with different expert counts; the term collapses from \
             tens of MiB to hundredths exactly at the predicted crossing."
        ),
    })
}

/// mainline's host-resident MoE buffers under tensor split.
///
/// A hybrid served with `--split-mode tensor` keeps per-token MoE intermediates on
/// the host, the same shape as ik's term and at a higher rate. Under layer split
/// the same models show none of it.
pub fn mainline_tensor_moe(rows: &[Record], tuning: &Tuning) -> Result<Scalar> {
    let mut points: Vec<(String, f64)> = Vec::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.runtime_is_ik() || !factors.is_hybrid() {
            continue;
        }
        if factors.split_or_layer() != SplitMode::Tensor || !factors.flash_attn_on() {
            continue;
        }
        // An MTP run measures without this term entirely.
        if factors.has_spec() {
            continue;
        }
        let arena = parsed.arena_mib;
        if arena == 0.0 {
            continue;
        }
        if !factors.fully_offloaded() {
            continue;
        }
        let terms = arena_terms(record, false, tuning);
        let tokens = factors.tokens() as f64;
        let excess = (arena - terms.total()) * 1048576.0;
        if excess <= 0.0 {
            continue;
        }
        let n_embd = parsed.n_embd as f64;
        points.push((parsed.arch.clone(), excess / tokens / n_embd));
    }
    if points.is_empty() {
        return Err(DeriveError::no_data(
            "no mainline tensor-split hybrid cells",
        ));
    }
    let rates: Vec<f64> = points.iter().map(|(_, r)| *r).collect();
    let shapes: BTreeSet<(String, i64)> = points
        .iter()
        .map(|(a, r)| (a.clone(), round_tenths_half_even(*r)))
        .collect();
    let detail = shapes
        .iter()
        .map(|(arch, tenths)| format!("{arch} {:.1}/unit", *tenths as f64 / 10.0))
        .collect::<Vec<_>>()
        .join(", ");
    // Named for the constant it was first written against. Renaming it would
    // change a message the check compares.
    consensus_default(&rates, "MTP embedded compute")?;
    Ok(Scalar {
        value: round_half_even(median(&rates)),
        evidence: format!(
            "{} mainline tensor-split hybrid cells: {detail}. Linear in batch — \
             qwen35moe measures 28.3, 56.5, 113.0 and 226.1 MiB at ub 256, 512, 1024 \
             and 2048, a constant rate. The same models under layer split show 0.02 \
             MiB, so this belongs to the split mode rather than to the hybrid \
             placement alone.",
            points.len(),
        ),
    })
}

/// Bytes per batch token per unit of hidden size for ik's CPU-MoE buffers.
///
/// The *worst* rate across architectures, not the median. They differ — qwen35moe
/// 40.5, glm-dsa 42.8, laguna 53.7 — and a median under-reserves every
/// architecture above it, which is the direction that OOMs. Taking the maximum
/// over-reserves the others by at most a third.
///
/// Within an architecture the rate is exact: glm-dsa measures 42.8 on nine cells
/// spanning ctx 8192-131072 and ub 256-512 without deviating. It does vary with
/// card count on two of the three — glm-dsa 28.0 on one card against 42.8 on two,
/// laguna 36.0 against 53.7 — and qwen35moe not at all. That is unexplained, and
/// it is bounded by taking the maximum.
///
/// The table is keyed `{arch}@{cards}` for that reason.
pub fn ik_moe_per_nembd(rows: &[Record], tuning: &Tuning) -> Result<(Scalar, Table<ArchCardsKey>)> {
    struct Point {
        n_embd: u64,
        rate: f64,
        arch: ArchKey,
        cards: usize,
    }
    let mut points: Vec<Point> = Vec::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if !factors.runtime_is_ik() {
            continue;
        }
        let arena = parsed.arena_mib;
        if arena == 0.0 {
            continue;
        }
        let experts = parsed.n_expert;
        let used = parsed.n_expert_used;
        let tokens = factors.tokens();
        if experts == 0 || used == 0 || tokens * used >= 32 * experts {
            continue;
        }
        if !factors.flash_attn_on() || !factors.fully_offloaded() {
            continue;
        }
        let terms = arena_terms(record, false, tuning);
        let excess = (arena - terms.total()) * 1048576.0 / tokens as f64;
        let n_embd = parsed.n_embd;
        points.push(Point {
            n_embd,
            rate: excess / n_embd as f64,
            arch: ArchKey::recorded(record),
            cards: factors.cards_or(1),
        });
    }
    if points.is_empty() {
        return Err(DeriveError::no_data(
            "no ik MoE cells below the offload threshold",
        ));
    }
    // Keyed by card count as well as architecture. glm-dsa measures 28.0 per unit
    // on one card and 42.8 on two at *identical* placement — both its offload
    // values exceed its layer count, and both logs show 80 of 80 layers on the GPU
    // — and the difference scales with tokens, so it is a rate that depends on the
    // device count rather than a separate term. Above the MoE threshold the two
    // card counts agree exactly, which is what localises it here rather than in
    // the arena.
    let by_arch_cards: BTreeSet<(ArchKey, usize)> =
        points.iter().map(|p| (p.arch.clone(), p.cards)).collect();
    for (arch, cards) in &by_arch_cards {
        let group: Vec<f64> = points
            .iter()
            .filter(|p| p.arch == *arch && p.cards == *cards)
            .map(|p| p.rate)
            .collect();
        consensus_default(
            &group,
            &format!("ik MoE rate for {arch} on {cards} card(s)"),
        )?;
    }
    let archs: BTreeSet<&ArchKey> = points.iter().map(|p| &p.arch).collect();
    for arch in &archs {
        let group: Vec<f64> = points
            .iter()
            .filter(|p| p.arch == **arch)
            .map(|p| p.rate)
            .collect();
        if let Err(error) = consensus_default(&group, &format!("ik MoE rate for {arch}")) {
            // glm-dsa measures 28.0 per unit on one card and 42.8 on two, at
            // identical expert placement, each figure exact within its group. The
            // cause is not in the dataset. Taking the larger over-reserves the
            // single-card case rather than under-reserving the other, which is the
            // direction that does not OOM — so the disagreement is carried
            // deliberately rather than papered over.
            //
            // laguna shows the same shape — 36.0 on one card against 53.7 on two —
            // though its expert offload differs between them too, so its cause is
            // confounded where glm's is not.
            if !matches!(arch.as_str(), "glm-dsa" | "laguna") {
                return Err(error);
            }
        }
    }
    let per_unit = points
        .iter()
        .map(|p| p.rate)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut by_key: BTreeMap<ArchCardsKey, f64> = BTreeMap::new();
    for point in &points {
        let key = ArchCardsKey::new(&point.arch, point.cards);
        let held = by_key.entry(key).or_insert(0.0);
        *held = held.max(point.rate);
    }
    let shapes: BTreeSet<(u64, i64, ArchKey)> = points
        .iter()
        .map(|p| (p.n_embd, round_tenths_half_even(p.rate), p.arch.clone()))
        .collect();
    let detail = shapes
        .iter()
        .map(|(embd, tenths, arch)| {
            let rate = *tenths as f64 / 10.0;
            format!(
                "{arch} {:.0} B/token at n_embd {embd} ({rate:.1}/unit)",
                rate * *embd as f64
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let table = Table {
        by_key: by_key
            .into_iter()
            .map(|(k, v)| (k, round_half_even(v)))
            .collect(),
        evidence: String::new(),
    };
    Ok((
        Scalar {
            value: round_half_even(per_unit),
            evidence: format!(
                "{} cells below the offload threshold: {detail}. The worst rate is \
                 taken rather than the median: the architectures differ, and a median \
                 under-reserves every one above it. Replaces a flat 81 KiB, which was \
                 this term evaluated at qwen35moe's hidden size and frozen.",
                points.len(),
            ),
        },
        table,
    ))
}
