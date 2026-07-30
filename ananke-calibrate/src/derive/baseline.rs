//! What the process holds beyond its buffers: the per-architecture baseline
//! correction, what a tensor split costs, what a busy slot costs, what a real
//! prompt costs, and what a second CUDA device costs.

use std::collections::BTreeMap;

use crate::{
    derive::{
        Scalar, Table,
        arena::arena_terms,
        error::{DeriveError, Result},
        ordered::OrderedMap,
        shape::{CHECKPOINT_MIN_STEP, variant_key},
        stats::{
            OUTLIER_TOLERANCE, check_no_outlier_dominates, consensus, consensus_default, median,
            py_round,
        },
        tuning::Tuning,
    },
    record::Record,
};

/// Per-architecture correction to the process baseline.
///
/// `PROCESS_BASE_BYTES` plus a per-layer term plus a flat MoE allowance is the
/// whole model, and it leaves a residual that is *architecture-shaped*: qwen35
/// holds 297 MiB more than it predicts and qwen35moe 75, while gemma3 holds 47
/// less. Two models of the same family show it and two of near-identical shape do
/// not, so it is not size, and a long list of other causes has been ruled out by
/// measurement — see FINDINGS.md.
///
/// Modelled as an offset per architecture rather than explained. That is honest
/// about what is known: the residual is reproducible, it is keyed on something the
/// architecture string captures, and leaving it uncharged under-reserves qwen35 by
/// nearly twice.
///
/// `no_fa_rates` is a dependency rather than shared state, deliberately. The
/// Python passes it through a module global and spells the ordering out in a
/// comment in `emit`, because subtracting zeros here folds a per-token arena term
/// into a flat baseline and leaves every flash-attention-off cell over-predicted at
/// 0.71 to 0.78.
pub fn baseline_offset(
    rows: &[Record],
    tuning: &Tuning,
    no_fa_rates: &BTreeMap<String, i64>,
) -> Result<(Scalar, Table)> {
    if no_fa_rates.is_empty() {
        return Err(DeriveError::disagreement(
            "baseline offset needs the flash-attention rates and they are empty, \
             which means derive_no_flash_attn_rates has not run yet. Without them \
             this silently subtracts zero and folds a per-token arena term into a \
             flat baseline.",
        ));
    }
    let per_layer = tuning.constant_f64("PROCESS_BASE_BYTES_PER_LAYER", 0.0);
    let flat = tuning.constant_f64("PROCESS_BASE_BYTES", 0.0);
    let moe = tuning.constant_f64("PROCESS_BASE_BYTES_MOE", 0.0);
    let dev = tuning.constant_f64("PROCESS_BASE_BYTES_PER_DEVICE", 0.0);
    let pinned = tuning.constant_f64("PINNED_EXTRA_BYTES", 0.0);
    let worst_no_fa = no_fa_rates.values().copied().max().unwrap_or(0);

    let mut by_arch: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.ngl != Some(99) || factors.gpus.is_empty() || factors.has_spec() {
            continue;
        }
        if factors.is_hybrid() || !factors.served || factors.bench {
            continue;
        }
        if factors.parallel != Some(1) {
            continue;
        }
        let Some(_arena) = parsed.arena_mib.filter(|v| *v != 0.0) else {
            continue;
        };
        // `--no-mmap` reads the weights into anonymous memory instead of mapping
        // them, and `host_overhead_bytes` models overhead rather than weights. One
        // such cell put qwen3@ik's offset at 704 MiB and over-predicted every other
        // ik cell of that model to 0.57.
        if factors.no_mmap {
            continue;
        }
        // Short-probe cells only. A prompt past the checkpoint spacing reaches a
        // steady state up to 1.6 GiB higher on a sliding-window model, and folding
        // that in here would charge it to every service of that architecture —
        // including one that never sees a long prompt, which the correction could
        // then never pull back. It is reserved as slop instead; see
        // `checkpoint_headroom`.
        if factors.probe_prompt_tokens.unwrap_or(4) >= CHECKPOINT_MIN_STEP {
            continue;
        }
        if factors.split_or_layer() != "layer" || parsed.n_layer.unwrap_or(0) == 0 {
            continue;
        }
        // Flash attention off is kept, under its own key. Pooling it with flash
        // attention on is what put lfm2's offset at both 35 and 169 MiB, but
        // excluding it left every such cell uncorrected — and while the bulk of the
        // effect is the per-token arena rate, a flat baseline shift remains
        // underneath it, small everywhere but lfm2 at +131 MiB.
        //
        // mainline and ik are separated by the key rather than by excluding one:
        // ik's residual against the same model runs from -264 to +120 MiB where
        // mainline's is -0 to +24, so the two binaries do not share a baseline.
        // Grouped per runtime, ik's resident cells are as consistent as mainline's
        // — spreads of 62 to 100 MiB against the same architectures — and simply
        // sit 24 to 192 MiB higher. Excluding them left every ik configuration with
        // no correction at all.
        let owned = record.owned_bytes() as f64;
        let terms = arena_terms(record, true, tuning);
        let cards = factors.cards_or(1);
        // ik does not replicate masks across cards at any count, so including its
        // cells means the multiplier can no longer be read off the card count alone.
        let copies = if cards > 1 && !factors.runtime_is_ik() { 4.0 } else { 1.0 };
        // The per-token flash-attention term is a separate constant. Leaving it in
        // the residual would charge it twice: once in the arena and again in the
        // baseline, where it would also stop being flat and break the group.
        let no_fa = if factors.flash_attn_on() {
            0.0
        } else {
            let rate =
                no_fa_rates.get(&variant_key(record, false)).copied().unwrap_or(worst_no_fa);
            // Times `copies`, matching `pinned_graph_bytes`: the term is replicated
            // per device under a layer split like the masks it sits beside.
            // Subtracting the unreplicated figure left every flash-attention-off
            // cell over-predicted, at 0.71 to 0.78.
            rate as f64 * factors.tokens() as f64 * copies
        };
        let modelled = flat
            + f64::from(parsed.n_layer.unwrap_or(0)) * per_layer
            + if parsed.n_expert.unwrap_or(0) != 0 { moe } else { 0.0 };
        let residual = owned
            - (copies * terms.masks() + terms.hidden) * 1048576.0
            - no_fa
            - pinned
            - dev * (cards as f64 - 1.0)
            - modelled;
        by_arch.entry(variant_key(record, true)).or_default().push(residual);
    }
    if by_arch.is_empty() {
        return Err(DeriveError::no_data("no resident served cells"));
    }
    // No `consensus` call here, deliberately. That guard exists to stop a *median*
    // from hiding a disagreement, and this reduces by `max`: a maximum bounds a
    // spread rather than concealing it, and erring high on a baseline is the safe
    // direction. The spread is reported in the evidence instead, so a wide one is
    // visible rather than silently averaged.
    //
    // Negative offsets are charged too. The earlier rule kept only positive ones,
    // reasoning that a negative residual means the baseline already over-covers and
    // shaving it trades a safe over-prediction for a risk. That does not survive two
    // objections. The reduction is `max` — the *least* negative residual — so
    // subtracting it leaves every measured cell still over-predicted; and an
    // over-prediction is only safe while it stays inside the band the rolling
    // correction can travel. gemma3 sat at 0.78 against a floor of 0.8, which no
    // amount of observation can pull back, so the "safe" direction had become the
    // unreachable one.
    let table: BTreeMap<String, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            (arch.clone(), py_round(group.iter().copied().fold(f64::NEG_INFINITY, f64::max)))
        })
        .collect();
    let detail = by_arch
        .iter()
        .map(|(arch, group)| {
            let lo = group.iter().copied().fold(f64::INFINITY, f64::min) / 1048576.0;
            let hi = group.iter().copied().fold(f64::NEG_INFINITY, f64::max) / 1048576.0;
            if hi - lo > 32.0 {
                format!("{arch} {hi:+.0} (spans {lo:+.0})")
            } else {
                format!("{arch} {hi:+.0}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let cells: usize = by_arch.values().map(Vec::len).sum();
    let worst = by_arch
        .values()
        .map(|group| group.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((
        Scalar {
            value: py_round(worst),
            evidence: format!(
                "residual over the layer-count baseline, per architecture, across \
                 {cells} resident served cells: {detail} MiB. Negative offsets are \
                 charged as well as positive ones: the reduction is the maximum, so \
                 subtracting it leaves every measured cell still over-predicted, and \
                 an over-prediction past the rolling correction's floor is \
                 unreachable rather than safe."
            ),
        },
        Table { by_arch: table, evidence: String::new() },
    ))
}

/// Host baseline a tensor split costs beyond a layer split.
///
/// Measured on every model that ran both, at the same context, batch, slot count
/// and card count: between 96 and 184 MiB more. The estimator had no term for it,
/// so every tensor-split service was under-predicted by that much — and tensor
/// split is what the operator runs for several of them.
pub fn tensor_split_baseline(rows: &[Record], tuning: &Tuning) -> Result<(Scalar, Table)> {
    type Key = (String, u32, u32, String);
    let mut pairs: BTreeMap<Key, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.ngl != Some(99) || factors.gpus != "0,1" || factors.has_spec() {
            continue;
        }
        if factors.is_hybrid() || !factors.served || factors.bench {
            continue;
        }
        if factors.parallel != Some(1) || parsed.arena_mib.unwrap_or(0.0) == 0.0 {
            continue;
        }
        // The same `--no-mmap` exclusion the baseline offset needs, for the same
        // reason: it moves weights into the counter this reads.
        if factors.no_mmap {
            continue;
        }
        if factors.probe_prompt_tokens.unwrap_or(4) >= CHECKPOINT_MIN_STEP {
            continue;
        }
        let owned = record.owned_bytes() as f64;
        let terms = arena_terms(record, true, tuning);
        let split = factors.split_or_layer().to_string();
        let copies = if split == "layer" { 4.0 } else { 1.0 };
        let base = (owned - (copies * terms.masks() + terms.hidden) * 1048576.0) / 1048576.0;
        let key = (
            record.provenance.model_key.clone(),
            factors.ctx,
            factors.ubatch.unwrap_or(0),
            parsed.arch.clone().unwrap_or_else(|| "None".to_string()),
        );
        pairs.entry(key).or_default().entry(split).or_default().push(base);
    }

    let mut deltas = Vec::new();
    let mut detail = Vec::new();
    let mut by_arch: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (key, group) in &pairs {
        let (Some(layer), Some(tensor)) = (group.get("layer"), group.get("tensor")) else {
            continue;
        };
        let delta = median(tensor) - median(layer);
        deltas.push(delta);
        by_arch.entry(key.3.clone()).or_default().push(delta);
        let name: String =
            key.0.rsplit('/').next().unwrap_or(&key.0).chars().take(18).collect();
        detail.push(format!("{name} {delta:+.0}"));
    }
    if deltas.is_empty() {
        return Err(DeriveError::no_data("no model ran both split modes at matching settings"));
    }
    for (arch, group) in &by_arch {
        consensus(group, &format!("tensor-split baseline for {arch}"), 0.20, 0.0)?;
    }
    detail.sort();
    let table: BTreeMap<String, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            let worst = group.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (arch.clone(), py_round(worst * 1048576.0))
        })
        .collect();
    let worst = deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((
        Scalar {
            value: py_round(worst * 1048576.0),
            evidence: format!(
                "{} models measured under both split modes at matching context, batch, \
                 slots and cards: {} MiB. Per architecture, since the spread across all \
                 of them is wider than any one of them is internally.",
                deltas.len(),
                detail.join("; "),
            ),
        },
        Table { by_arch: table, evidence: String::new() },
    ))
}

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
/// value taken from the one architecture that had a series would have over-reserved
/// llama by half a gigabyte at four slots. Nothing the estimator already reads
/// predicts the spread: it is monotonic in layer count over these three but far too
/// steep to be a layer term, and unrelated to hidden size, KV-head count, or
/// vocabulary.
///
/// The arena does not move at all across the series, so this is a baseline term and
/// not an arena one.
pub fn per_slot_bytes(rows: &[Record]) -> Result<Table> {
    // Every factor that would otherwise be measured alongside the slot count.
    // `soak` belongs here: it grows the prompt over six rounds and costs memory of
    // its own, so pooling a soak-free single-request cell with a soaked concurrent
    // one measures both at once — which put gemma3 at 211 MiB per slot against the
    // 89 its own series shows. `parallel` deliberately does not: an idle slot costs
    // nothing, so a cell with four slots and one request is a valid one-slot
    // reading, and excluding it would drop every series whose slot count moves
    // alongside its concurrency.
    type Key = (String, String, u32, u32, String, String, String, bool, u32);
    let mut groups: BTreeMap<Key, BTreeMap<u32, i64>> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if !factors.served || factors.bench || factors.has_spec() {
            continue;
        }
        if factors.is_hybrid() {
            continue;
        }
        let Some(arch) = parsed.arch.clone().filter(|a| !a.is_empty()) else {
            continue;
        };
        let key: Key = (
            arch.clone(),
            factors.runtime.clone(),
            factors.ctx,
            factors.ubatch.unwrap_or(0),
            factors.gpus.clone(),
            factors.split_or_layer().to_string(),
            factors.kv_type.clone().unwrap_or_default(),
            factors.kv_unified,
            factors.soak.unwrap_or(0),
        );
        let owned = record.rss_kb("rss_anon_kb") * 1024;
        let concurrency = factors.concurrency.unwrap_or(1).max(1);
        // The lowest reading at each point: a higher one means the process had done
        // more, and the difference being measured is the slot count.
        groups
            .entry(key)
            .or_default()
            .entry(concurrency)
            .and_modify(|held| *held = (*held).min(owned))
            .or_insert(owned);
    }

    let mut by_arch: BTreeMap<String, Vec<f64>> = BTreeMap::new();
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
        by_arch.entry(key.0.clone()).or_default().push(worst);
    }
    if by_arch.is_empty() {
        return Err(DeriveError::no_data("no architecture measured at two concurrency levels"));
    }
    for (arch, group) in &by_arch {
        check_no_outlier_dominates(
            group,
            &format!("per-slot host bytes for {arch}"),
            OUTLIER_TOLERANCE,
        )?;
    }
    let table: BTreeMap<String, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            let worst = group.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (arch.clone(), py_round(worst).max(0))
        })
        .collect();
    let detail = table
        .iter()
        .map(|(arch, value)| format!("{arch} {:.0} MiB/slot", *value as f64 / 1048576.0))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_arch: table,
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
pub fn checkpoint_headroom(rows: &[Record]) -> Result<Table> {
    // Every factor that moves host memory. Omitting `soak`, `concurrency`, and
    // `cram` pooled a soaked, cache-enabled cell at 3467 MiB with the plain ones and
    // drove gemma3's difference negative.
    type Key = (String, String, u32, u32, String, String, String, u32, String, u32, u32, u32, bool);
    let mut matched: BTreeMap<Key, BTreeMap<bool, i64>> = BTreeMap::new();
    for record in rows {
        let factors = &record.factors;
        if record.status != "ok" || !factors.served {
            continue;
        }
        if factors.bench || factors.has_spec() || factors.is_hybrid() {
            continue;
        }
        if factors.ngl != Some(99) {
            continue;
        }
        let key: Key = (
            variant_key(record, false),
            factors.runtime.clone(),
            factors.ctx,
            factors.ubatch.unwrap_or(0),
            factors.gpus.clone(),
            factors.split_or_layer().to_string(),
            factors.kv_type.clone().unwrap_or_default(),
            factors.parallel.unwrap_or(0),
            factors.flash_attn.clone().unwrap_or_default(),
            factors.soak.unwrap_or(0),
            factors.concurrency.unwrap_or(0),
            factors.cram.unwrap_or(0),
            factors.no_mmap,
        );
        // The threshold is the checkpoint spacing, not any prompt longer than the
        // default: a 256- or 1024-token prompt still makes one checkpoint.
        let steady = factors.probe_prompt_tokens.unwrap_or(4) >= CHECKPOINT_MIN_STEP;
        let owned = record.owned_mib();
        matched
            .entry(key)
            .or_default()
            .entry(steady)
            .and_modify(|held| *held = (*held).max(owned))
            .or_insert(owned);
    }

    let mut by_arch: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for (key, pair) in &matched {
        if pair.len() == 2 {
            let steady = pair[&true];
            let probe = pair[&false];
            by_arch.entry(key.0.clone()).or_default().push((steady - probe).max(0));
        }
    }
    if by_arch.is_empty() {
        return Err(DeriveError::no_data("no probe/steady-state pairs"));
    }
    for (arch, group) in &by_arch {
        let values: Vec<f64> = group.iter().map(|v| *v as f64).collect();
        check_no_outlier_dominates(
            &values,
            &format!("checkpoint headroom for {arch}"),
            OUTLIER_TOLERANCE,
        )?;
    }
    let table: BTreeMap<String, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            (arch.clone(), group.iter().copied().max().unwrap_or(0) * 1024 * 1024)
        })
        .collect();
    let detail = table
        .iter()
        .map(|(arch, value)| format!("{arch} {} MiB", value / 1024 / 1024))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_arch: table,
        evidence: format!(
            "measured as the difference between a cell with a prompt past the \
             8192-token checkpoint spacing and its short-probe twin, matched on every \
             other factor: {detail}. Sliding-window architectures dominate, the flag \
             being named --swa-checkpoints for that reason."
        ),
    })
}

/// Host cost of each visible CUDA device beyond the first.
pub fn per_device_bytes(rows: &[Record]) -> Result<Scalar> {
    let mut by_model: OrderedMap<String, BTreeMap<usize, i64>> = OrderedMap::new();
    for record in rows {
        let factors = &record.factors;
        if !(factors.label.starts_with("devices-") || factors.label.starts_with("offload-ngl0")) {
            continue;
        }
        if factors.ngl != Some(0) {
            continue;
        }
        let cards = factors.cards_or(0);
        let owned = record.rss_kb("rss_anon_kb") + record.rss_kb("rss_shmem_kb");
        by_model
            .or_insert_with(record.provenance.model_key.clone(), BTreeMap::new)
            .insert(cards, owned);
    }
    let mut deltas = Vec::new();
    let mut detail = Vec::new();
    for (model, points) in by_model.iter() {
        if let (Some(one), Some(two)) = (points.get(&1), points.get(&2)) {
            let delta = ((two - one) * 1024) as f64;
            deltas.push(delta);
            let name: String =
                model.rsplit('/').next().unwrap_or(model).chars().take(24).collect();
            detail.push(format!("{name} {:.0} MiB", delta / 1048576.0));
        }
    }
    if deltas.is_empty() {
        return Err(DeriveError::no_data("no paired one-card/two-card device-scaling cells"));
    }
    consensus_default(&deltas, "per-device host cost")?;
    Ok(Scalar {
        value: py_round(median(&deltas)),
        evidence: format!(
            "measured with placement pinned to the CPU so only the CUDA context count \
             varies: {} going from one card to two.",
            detail.join("; "),
        ),
    })
}
