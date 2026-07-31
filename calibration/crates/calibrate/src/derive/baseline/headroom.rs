//! Two terms that are *reserved* rather than charged.
//!
//! `host_overhead_bytes` is what the rolling correction divides an observation by, so
//! it has to model what a process actually holds. A service that never sees a long
//! prompt, or never has four slots busy at once, would otherwise read as a large
//! over-reservation and clamp unreachably — so these belong in the packer's slop,
//! beside the prompt cache, and are recorded here rather than folded into the
//! baseline.

use std::collections::BTreeMap;

use ananke_config::placement::SplitMode;
use ananke_dataset::{FlashAttn, KvType};
use ananke_measure::record::Status;

use crate::{
    derive::{
        Table,
        error::{DeriveError, Result},
        keys::{ArchKey, VariantKey},
        pair::Pair,
        shape::CHECKPOINT_MIN_STEP,
        stats::{OUTLIER_TOLERANCE, check_no_outlier_dominates, round_half_even},
        units::{MIB_F64, MIB_I64},
    },
    record::Record,
};

/// Host memory each *concurrently active* slot costs, by architecture.
///
/// Idle slots are free. The same model at `parallel` 1, 2, and 4 with a single
/// sequential probe holds the same anonymous memory at every one; it is a slot that
/// actually serves a request that allocates, which fits the first-use prefill step
/// being per-slot rather than per-process. A reservation cannot know how many will
/// be busy, so it charges all of them.
///
/// Measured per architecture because they disagree by two orders of magnitude —
/// about 165 MiB per slot on qwen35, 89 on gemma3, and 3 on llama — and a single
/// value taken from the one architecture that has a series would over-reserve
/// llama by half a gigabyte at four slots. Nothing the estimator already reads
/// predicts the spread: it is monotonic in layer count over these three but far too
/// steep to be a layer term, and unrelated to hidden size, KV-head count, or
/// vocabulary.
///
/// The arena does not move at all across the series, so this is a baseline term and
/// not an arena one.
pub fn per_slot_bytes(rows: &[Record]) -> Result<Table<ArchKey>> {
    let mut groups: BTreeMap<SlotKey, BTreeMap<u32, i64>> = BTreeMap::new();
    for record in rows {
        let factors = &record.factors;
        if !factors.served || factors.bench || factors.has_spec() {
            continue;
        }
        if factors.is_hybrid() {
            continue;
        }
        let Some(arch) = ArchKey::named(record) else {
            continue;
        };
        let key = SlotKey {
            arch: arch.clone(),
            runtime: factors.runtime.name().to_owned(),
            ctx: factors.ctx,
            ubatch: factors.ubatch,
            gpus: factors.gpus.clone(),
            split: factors.split_or_layer(),
            kv_type: factors.kv_type,
            kv_unified: factors.kv_unified,
            soak: factors.soak,
        };
        let owned = record.rss.rss_anon_kb * 1024;
        let concurrency = factors.concurrency.max(1);
        // The lowest reading at each point: a higher one means the process had done
        // more, and the difference being measured is the slot count.
        groups
            .entry(key)
            .or_default()
            .entry(concurrency)
            .and_modify(|held| *held = (*held).min(owned))
            .or_insert(owned);
    }

    let mut by_arch: BTreeMap<ArchKey, Vec<f64>> = BTreeMap::new();
    for (key, points) in &groups {
        if points.len() < 2 {
            continue;
        }
        // Taken against the lowest point and maximised, so the rate covers *every*
        // measured concurrency rather than just the largest. gemma3 holds 378, 467,
        // and 568 MiB at one, two, and four: the endpoint average is 63 MiB per
        // slot, which under-reserves the two-slot point by 26, while the worst
        // interval gives 89 and covers both.
        let lo = *points.keys().next().expect("non-empty");
        let worst = points
            .iter()
            .filter(|(c, _)| **c > lo)
            .map(|(c, value)| (value - points[&lo]) as f64 / f64::from(c - lo))
            .fold(f64::NEG_INFINITY, f64::max);
        by_arch.entry(key.arch.clone()).or_default().push(worst);
    }
    if by_arch.is_empty() {
        return Err(DeriveError::no_data(
            "no architecture measured at two concurrency levels",
        ));
    }
    for (arch, group) in &by_arch {
        check_no_outlier_dominates(
            group,
            &format!("per-slot host bytes for {arch}"),
            OUTLIER_TOLERANCE,
        )?;
    }
    let table: BTreeMap<ArchKey, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            let worst = group.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (arch.clone(), round_half_even(worst).max(0))
        })
        .collect();
    let detail = table
        .iter()
        .map(|(arch, value)| format!("{arch} {:.0} MiB/slot", *value as f64 / MIB_F64))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_key: table,
        evidence: format!(
            "anonymous memory against the number of concurrent requests, all else \
             equal: {detail}. Idle slots cost nothing — the same models at parallel 1, \
             2, and 4 with one sequential probe read the same — so it is the active \
             slot that allocates. Charged for every slot, since a reservation cannot \
             know how many will be busy."
        ),
    })
}

/// What a real prompt adds over the campaign's probe, by architecture.
///
/// llama.cpp's server checkpoints a slot's state so a prompt can be rewound, spaced
/// by `--checkpoint-min-step` (8192 tokens). The campaign's probe is four tokens, so
/// every baseline cell holds one checkpoint at most; a service serving real prompts
/// holds more.
///
/// How many more depends on the attention. The flag's other name is
/// `--swa-checkpoints`, and a sliding-window model needs checkpoints to rewind its
/// window, so it keeps far more of them: gemma-4-31B-QAT measures 524 MiB at the
/// probe and 2138 at a 16384-token prompt, against Qwen3.6-27B's 778 and 928.
///
/// Reserved rather than charged, like the prompt cache and the per-slot cost.
/// `host_overhead_bytes` is what the rolling correction divides an observation by,
/// so it has to model what a process holds; a service that never sees a long prompt
/// would otherwise read as a 1.6 GiB over-reservation and clamp unreachably.
pub fn checkpoint_headroom(rows: &[Record]) -> Result<Table<VariantKey>> {
    let mut matched: BTreeMap<CheckpointKey, Pair<i64>> = BTreeMap::new();
    for record in rows {
        let factors = &record.factors;
        if record.status != Status::Ok || !factors.served {
            continue;
        }
        if factors.bench || factors.has_spec() || factors.is_hybrid() {
            continue;
        }
        if !factors.fully_offloaded() {
            continue;
        }
        let key = CheckpointKey {
            variant: VariantKey::of(record),
            runtime: factors.runtime.name().to_owned(),
            ctx: factors.ctx,
            ubatch: factors.ubatch,
            gpus: factors.gpus.clone(),
            split: factors.split_or_layer(),
            kv_type: factors.kv_type,
            parallel: factors.parallel,
            flash_attn: factors.flash_attn,
            soak: factors.soak,
            concurrency: factors.concurrency,
            cram: factors.cram,
            no_mmap: factors.no_mmap,
        };
        // The threshold is the checkpoint spacing, not any prompt longer than the
        // default: a 256- or 1024-token prompt still makes one checkpoint.
        let steady = factors.probe_prompt_tokens >= CHECKPOINT_MIN_STEP;
        let owned = record.owned_mib();
        let held = matched.entry(key).or_default().half_mut(steady);
        *held = Some(held.map_or(owned, |seen| seen.max(owned)));
    }

    let mut by_variant: BTreeMap<VariantKey, Vec<i64>> = BTreeMap::new();
    for (key, pair) in &matched {
        let Some((steady, probe)) = pair.both() else {
            continue;
        };
        by_variant
            .entry(key.variant.clone())
            .or_default()
            .push((steady - probe).max(0));
    }
    if by_variant.is_empty() {
        return Err(DeriveError::no_data("no probe/steady-state pairs"));
    }
    for (variant, group) in &by_variant {
        let values: Vec<f64> = group.iter().map(|v| *v as f64).collect();
        check_no_outlier_dominates(
            &values,
            &format!("checkpoint headroom for {variant}"),
            OUTLIER_TOLERANCE,
        )?;
    }
    let table: BTreeMap<VariantKey, i64> = by_variant
        .iter()
        .map(|(variant, group)| {
            (
                variant.clone(),
                group.iter().copied().max().unwrap_or(0) * MIB_I64,
            )
        })
        .collect();
    let detail = table
        .iter()
        .map(|(variant, value)| format!("{variant} {} MiB", value / MIB_I64))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_key: table,
        evidence: format!(
            "measured as the difference between a cell with a prompt past the \
             8192-token checkpoint spacing and its short-probe twin, matched on every \
             other factor: {detail}. Sliding-window architectures dominate, the flag \
             being named --swa-checkpoints for that reason."
        ),
    })
}

/// Every factor that would otherwise be measured alongside the slot count.
///
/// `soak` belongs here: it grows the prompt over six rounds and costs memory of its
/// own, so pooling a soak-free single-request cell with a soaked concurrent one
/// measures both at once — which puts gemma3 at 211 MiB per slot against the 89 its
/// own series shows. `parallel` deliberately does not: an idle slot costs nothing,
/// so a cell with four slots and one request is a valid one-slot reading, and
/// excluding it would drop every series whose slot count moves alongside its
/// concurrency.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SlotKey {
    arch: ArchKey,
    runtime: String,
    ctx: u32,
    ubatch: u32,
    gpus: String,
    split: SplitMode,
    kv_type: KvType,
    kv_unified: bool,
    soak: u32,
}

/// Every factor that moves host memory, so the only difference left between a pair
/// is the prompt length.
///
/// Omitting `soak`, `concurrency`, and `cram` pooled a soaked, cache-enabled cell at
/// 3467 MiB with the plain ones and drove gemma3's difference negative.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CheckpointKey {
    variant: VariantKey,
    runtime: String,
    ctx: u32,
    ubatch: u32,
    gpus: String,
    split: SplitMode,
    kv_type: KvType,
    parallel: u32,
    flash_attn: FlashAttn,
    soak: u32,
    concurrency: u32,
    cram: u32,
    no_mmap: bool,
}
