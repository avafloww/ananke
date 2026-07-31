//! Terms the arena model leaves to its callers: the E-variant's per-layer input,
//! the quantised cache's extra, and what an unfused attention pass costs the
//! pinned buffer.

use std::collections::BTreeMap;

use crate::{
    derive::{
        Scalar, Table,
        arena::arena_terms,
        error::{DeriveError, Result},
        shape::variant_key,
        stats::{consensus, consensus_default, median, round_half_even},
        tuning::Tuning,
    },
    record::Record,
};

/// The E-variant's per-layer embedding input, in bytes per layer per token.
pub fn gemma_e_per_layer_token(rows: &[Record], tuning: &Tuning) -> Result<Scalar> {
    let mut residuals = Vec::new();
    let mut controls = Vec::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if parsed.arch.as_deref() != Some("gemma4") {
            continue;
        }
        let Some(arena) = parsed.arena_mib.filter(|v| *v != 0.0) else {
            continue;
        };
        if factors.ngl != Some(99) || factors.gpus.is_empty() || !factors.flash_attn_on() {
            continue;
        }
        if factors.has_spec() {
            continue;
        }
        // f16 only. A quantised cache shifts this residual — 1087 and 1278 against
        // f16's steady 1025-1028 — so the arena model is missing a term that
        // depends on the cache type, and pooling the two would attribute that term
        // to the E-variant.
        if factors.kv_type.as_deref() != Some("f16") {
            continue;
        }
        let terms = arena_terms(record, true, tuning);
        let copies = if factors.cards_or(1) > 1 && factors.split_or_layer() == "layer" {
            4.0
        } else {
            1.0
        };
        let residual = arena - (copies * terms.masks() + terms.hidden);
        let tokens = factors.tokens() as f64;
        let per = residual * 1048576.0 / (f64::from(parsed.n_layer.unwrap_or(0)) * tokens);
        // The two populations are ~1028 and ~3 bytes per layer per token, so the
        // boundary is nowhere near either. A cell between them is not an E-variant
        // reading and not a control; it is a sign the filter is wrong, and
        // `consensus` will say so rather than averaging it in.
        if per > 500.0 {
            residuals.push(per)
        } else {
            controls.push(per)
        }
    }
    if residuals.is_empty() {
        return Err(DeriveError::no_data("no gemma4 E-variant cells"));
    }
    consensus_default(&residuals, "gemma E-variant per-layer term")?;
    let middle = median(&residuals);
    let control = if controls.is_empty() {
        0.0
    } else {
        median(&controls)
    };
    Ok(Scalar {
        value: round_half_even(middle),
        evidence: format!(
            "{} E-variant cells at {middle:.0} B/layer/token, against {} \
             same-architecture control cells at {control:.0}. {} is {} f32 elements.",
            residuals.len(),
            controls.len(),
            round_half_even(middle),
            round_half_even(middle / 4.0),
        ),
    })
}

/// Extra pinned bytes per batch token when the KV cache is quantised.
///
/// Paired cells differing in nothing but `cache_type_*` show the arena larger with
/// a quantised cache, in all 117 pairs measured, always positive and scaling
/// exactly with batch — 1.28 MiB at ub 512 against 5.12 at 2048 on the same model.
///
/// The per-copy rate varies by architecture and is not predicted by head count,
/// head width, or layer count: 160 bytes per token on non-sliding-window models,
/// 328 on sliding-window ones, 532, 2621, and 6144 on deepseek4. Since the
/// mechanism is not identified, the worst observed rate is charged to all of them.
/// Doing so costs 12 MiB at the largest batch measured, which is cheap insurance
/// against an under-prediction whose size is not understood.
pub fn quantised_cache_bytes(rows: &[Record]) -> Result<(Scalar, Table)> {
    let mut paired: BTreeMap<CacheTypePairKey, BTreeMap<String, f64>> = BTreeMap::new();
    let mut archs: BTreeMap<String, String> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.ngl != Some(99) || factors.gpus.is_empty() || factors.has_spec() {
            continue;
        }
        let Some(arena) = parsed.arena_mib.filter(|v| *v != 0.0) else {
            continue;
        };
        let key = CacheTypePairKey {
            model_key: record.provenance.model_key.clone(),
            ctx: factors.ctx,
            ubatch: factors.ubatch.unwrap_or(0),
            parallel: factors.parallel.unwrap_or(0),
            kv_unified: factors.kv_unified,
            split: factors.split.clone().unwrap_or_else(|| "-".to_string()),
            gpus: factors.gpus.clone(),
            flash_attn: factors.flash_attn.clone().unwrap_or_default(),
            served: factors.served,
            n_cpu_moe: factors.n_cpu_moe.unwrap_or(0),
            no_mmap: factors.no_mmap,
            rtr: factors.rtr,
            numa: factors.numa.clone().unwrap_or_else(|| "-".to_string()),
        };
        paired
            .entry(key)
            .or_default()
            .insert(factors.kv_type.clone().unwrap_or_default(), arena);
        archs.insert(
            record.provenance.model_key.clone(),
            parsed.arch.clone().unwrap_or_else(|| "?".to_string()),
        );
    }

    let mut rates = Vec::new();
    let mut by_arch: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (key, pair) in &paired {
        if pair.len() != 2 {
            continue;
        }
        let (Some(quantised), Some(f16)) = (pair.get("q8_0"), pair.get("f16")) else {
            continue;
        };
        let tokens = u64::from(key.ctx).min(u64::from(key.ubatch)) as f64;
        let cards = key.gpus.split(',').count();
        // Divide out the layer-split replication, so the rate is per copy — but a
        // hybrid does not replicate, and treating one as though it did put
        // qwen35moe's rate at both 133 and 532.
        let hybrid = key.n_cpu_moe != 0;
        let copies = if cards > 1 && key.split == "layer" && !hybrid {
            4.0
        } else {
            1.0
        };
        let rate = (quantised - f16) * 1048576.0 / tokens / copies;
        rates.push(rate);
        by_arch
            .entry(archs[&key.model_key].clone())
            .or_default()
            .push(rate);
    }
    if rates.is_empty() {
        return Err(DeriveError::no_data(
            "no cells differing only in cache type",
        ));
    }
    // Per architecture, because they differ by a factor of forty and charging the
    // worst to all costs ~3 MiB of over-prediction on every quantised cell.
    for (arch, group) in &by_arch {
        consensus(
            group,
            &format!("quantised-cache rate for {arch}"),
            0.05,
            0.0,
        )?;
    }
    let table = Table {
        by_arch: by_arch
            .iter()
            .map(|(arch, group)| {
                (
                    arch.clone(),
                    round_half_even(group.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                )
            })
            .collect(),
        evidence: String::new(),
    };
    let worst = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let least = rates.iter().copied().fold(f64::INFINITY, f64::min);
    Ok((
        Scalar {
            value: round_half_even(worst),
            evidence: format!(
                "{} pairs differing in nothing but the cache type, every one showing \
                 the arena larger when it is quantised. Per-copy rates run {least:.0} \
                 to {worst:.0} bytes per batch token and are not predicted by head \
                 count, head width or layer count, so the worst is charged to all — 12 \
                 MiB at the largest batch measured. Scaling with batch is exact, and \
                 dividing out the layer-split replication is what makes the two rates \
                 per model collapse to one.",
                rates.len(),
            ),
        },
        table,
    ))
}

/// Everything that could make two cells differ other than the cache type.
///
/// `no_mmap`, `rtr`, and `numa` belong here like everything else: without them a
/// `--no-mmap` cell pairs with a mapped one and the difference in *weights* is read
/// as a cache-type effect, which put qwen3's rate across a 6253% spread. And the
/// key carries the offload *count*, not merely whether there is one — coarsened to
/// a boolean it paired a `--n-cpu-moe 20` cell with a `--n-cpu-moe 40` one and read
/// the difference in resident weights as a cache-type effect, spreading qwen35moe's
/// rate across 13935%.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CacheTypePairKey {
    model_key: String,
    ctx: u32,
    ubatch: u32,
    parallel: u32,
    kv_unified: bool,
    split: String,
    gpus: String,
    flash_attn: String,
    served: bool,
    n_cpu_moe: u32,
    no_mmap: bool,
    rtr: bool,
    numa: String,
}

/// Extra pinned bytes per batch token when flash attention is off.
///
/// The single constant this replaces was chosen as a representative value because
/// the excess "is not uniform across architectures", and that is right — but the
/// non-uniformity is a clean per-architecture rate rather than noise, which a sweep
/// across context makes visible and a single context cannot.
///
/// What the residual over the modelled arena does *not* do is scale with context:
/// gemma-3-27B is 64 MiB out at ctx 8192, 32768, and 131072 alike, and 256 MiB out
/// at ubatch 2048 in every one of them. So it is a per-batch-token term, the same
/// shape as the quantised-cache rate, at 128 KiB per token on the sliding-window
/// models against 32 KiB on the rest.
///
/// The rate is per token and per *device copy*, replicated under a layer split the
/// same way the masks are. It does not depend on the slot count: cells run to
/// settle that measure gemma3 at 128.1 KiB per token and qwen35 at 32.1 at both one
/// slot and four, identical to the decimal.
///
/// ik_llama is excluded and keeps the small default: its fa-off arena is already
/// modelled to within a megabyte, since it sizes masks against the whole cache and
/// the widened element is the whole story there.
///
/// `quant_rates` is `None` on the emit path, and deliberately so: the quantised-cache
/// table is produced *after* this deriver runs, so that term is absent from the
/// residual the committed constants were fitted against. Passing the table would
/// move every rate in this table.
pub fn no_flash_attn_rates(
    rows: &[Record],
    tuning: &Tuning,
    quant_rates: Option<&BTreeMap<String, i64>>,
) -> Result<Table> {
    // Required, not defaulted: the E variant's per-layer term is subtracted out of
    // this residual, and reading it as zero would fold it into every rate below.
    let e_variant_rate = tuning.required_f64("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN")?;
    let mut by_arch: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if factors.flash_attn_on() {
            continue;
        }
        let Some(arena) = parsed.arena_mib.filter(|v| *v != 0.0) else {
            continue;
        };
        if factors.ngl != Some(99) || factors.gpus.is_empty() || factors.has_spec() {
            continue;
        }
        if factors.runtime_is_ik() {
            continue;
        }
        let terms = arena_terms(record, true, tuning);
        let cards = if factors.gpus.is_empty() {
            1
        } else {
            factors.gpus.split(',').count()
        };
        // A hybrid does not replicate the mask across cards — the same exception
        // the quantised-cache rate needs, and leaving it out here would put every
        // hybrid architecture's rate wildly negative and drop it from the table,
        // where it would then inherit the *worst* rate as its default.
        let copies = if cards > 1 && factors.split_or_layer() == "layer" && !factors.is_hybrid() {
            4.0
        } else {
            1.0
        };
        let tokens = factors.tokens() as f64;
        // The other two terms that live in the arena beside the masks. Leaving them
        // in the residual makes this rate absorb them, and then the model adds them
        // again: an E-variant cell came out 21 MiB over, which is exactly its
        // per-layer term counted twice.
        let mut extra = 0.0;
        if parsed.per_layer_token_embd.unwrap_or(false) {
            extra += e_variant_rate * f64::from(parsed.n_layer.unwrap_or(0)) * tokens / 1048576.0;
        }
        if factors.kv_type.as_deref() != Some("f16")
            && let Some(quant) = quant_rates.filter(|table| !table.is_empty())
        {
            let arch = parsed.arch.as_deref().unwrap_or("None");
            let rate = quant
                .get(arch)
                .copied()
                .unwrap_or_else(|| quant.values().copied().max().unwrap_or(0));
            extra += rate as f64 * tokens / 1048576.0;
        }
        let residual = arena - (copies * terms.masks() + terms.hidden) - extra;
        // Per device copy: the term is replicated under a layer split the same way
        // the masks are. Single-card cells measure a quarter of what two-card ones
        // do, which is the copy factor and not the slot count — gemma3 and qwen35
        // both measure the same rate at one slot and four.
        by_arch
            .entry(variant_key(record, false))
            .or_default()
            .push(residual * 1048576.0 / tokens / copies);
    }
    if by_arch.is_empty() {
        return Err(DeriveError::no_data(
            "no mainline cells with flash attention off",
        ));
    }
    for (arch, group) in &by_arch {
        // 4 KiB per token: below that the term is under a megabyte at the largest
        // batch measured, which is not worth splitting a group over.
        consensus(
            group,
            &format!("no-flash-attention rate for {arch}"),
            0.10,
            4096.0,
        )?;
    }
    // Negative rates mean the current constant over-charges that architecture;
    // clamping at zero keeps the term from *subtracting* pinned memory, which no
    // mechanism supports.
    let table: BTreeMap<String, i64> = by_arch
        .iter()
        .map(|(arch, group)| {
            let worst = group.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (arch.clone(), round_half_even(worst).max(0))
        })
        .collect();
    let cells: usize = by_arch.values().map(Vec::len).sum();
    let rates = table
        .iter()
        .map(|(arch, rate)| format!("{arch} {:.0} KiB", *rate as f64 / 1024.0))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_arch: table,
        evidence: format!(
            "{cells} mainline cells with flash attention off across {} \
             architectures. The residual over the modelled arena is flat in context \
             and proportional to batch tokens, so it is a per-token rate: {rates}. \
             ik_llama is excluded — its fa-off arena is already modelled to within a \
             megabyte.",
            by_arch.len(),
        ),
    })
}
