//! Device-side terms: what an unfused attention pass costs, what a vision
//! projector's graph costs, and what the runtimes that print no breakdown table
//! were measured holding.

use std::collections::{BTreeMap, BTreeSet};

use ananke_config::placement::SplitMode;
use ananke_dataset::{FlashAttn, KvType};

use crate::{
    derive::{
        NestedTable, Scalar, Table,
        error::{DeriveError, Result},
        keys::ArchKey,
        shape::{WEIGHT_TOLERANCE, query_head_count, same_resident_weights, table_less_compute},
        stats::round_half_even,
        units::MIB_F64,
    },
    record::Record,
};

/// Every factor that should decide a cell's VRAM, so the only thing left between a
/// pair is the flash-attention state itself.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ScorePairKey {
    arch: ArchKey,
    model: String,
    ctx: u32,
    ubatch: u32,
    kv_type: KvType,
    split: SplitMode,
    gpus: String,
    parallel: u32,
    kv_unified: bool,
    runtime: String,
    n_cpu_moe: u32,
    spec: bool,
    ngl: u32,
    embeddings: bool,
    mmproj: bool,
    served: bool,
    flash_attn: FlashAttn,
}

/// What an unfused attention pass costs per head, per cache token, per batch token,
/// by architecture.
///
/// With flash attention off the graph materialises the score matrix instead of
/// consuming it tile by tile. Paired against each cell's own flash-attention-on
/// sibling — same model, context, batch, split, cards, and slot count — the extra
/// comes to about one f32 per (head, cache token, batch token) for every dense and
/// MoE architecture measured.
///
/// deepseek4 measures far less. MLA shares one latent across heads, so there is no
/// per-head score row to materialise, and the near-zero is the real answer rather
/// than an outlier — which is why this is a table.
///
/// Paired on the **driver total**, not on the breakdown's `compute` column. The
/// column misses the `unaccounted` remainder that grows with it, which understated
/// gemma4 by 20 to 40%, and it does not exist at all under ik — so every ik cell was
/// silently skipped, which is how laguna came to have no entry despite being the
/// largest single miss in the set.
pub fn no_flash_attn_score(rows: &[Record]) -> Result<Table<ArchKey>> {
    let mut paired: BTreeMap<ScorePairKey, (u64, &Record)> = BTreeMap::new();
    for record in rows {
        let factors = &record.factors;
        let Some(total) = record.rss.gpu_used_mib.filter(|v| *v != 0) else {
            continue;
        };
        let Some(arch) = ArchKey::named(record) else {
            continue;
        };
        let key = ScorePairKey {
            arch,
            model: factors.model.clone(),
            ctx: factors.ctx,
            ubatch: factors.ubatch_or_default(),
            kv_type: factors.kv_type,
            split: factors.split_or_layer(),
            gpus: factors.gpus.clone(),
            parallel: factors.parallel,
            kv_unified: factors.kv_unified,
            runtime: factors.runtime.name().to_owned(),
            n_cpu_moe: factors.n_cpu_moe.unwrap_or(0),
            spec: factors.has_spec(),
            ngl: factors.ngl,
            embeddings: factors.embeddings,
            mmproj: factors.mmproj.as_deref().is_some_and(|m| !m.is_empty()),
            served: factors.served,
            flash_attn: factors.flash_attn,
        };
        paired.entry(key).or_insert((total, record));
    }
    let mut per_arch: BTreeMap<ArchKey, Vec<f64>> = BTreeMap::new();
    for (key, (total, record)) in &paired {
        if key.flash_attn != FlashAttn::Off {
            continue;
        }
        let sibling_key = ScorePairKey {
            flash_attn: FlashAttn::On,
            ..key.clone()
        };
        let Some((sibling_total, sibling)) = paired.get(&sibling_key) else {
            continue;
        };
        let heads = query_head_count(&record.parsed);
        if heads == 0 {
            continue;
        }
        let streams = if key.kv_unified {
            1
        } else {
            key.parallel.max(1)
        };
        let n_kv = u64::from(key.ctx / streams);
        let cards = key.gpus.split(',').filter(|g| !g.is_empty()).count().max(1);
        // Both halves must have loaded the same weights onto the same devices. The
        // key pins every factor that should decide placement, but placement is an
        // outcome rather than a factor — an `auto` expert offload can land
        // differently between two runs — and a mismatch there lands entirely in this
        // delta. Two cells disagreed by more than 8 GiB of resident weight and read
        // as 23.1 and 11.6 bytes an entry against the 3.7 to 4.2 their own siblings
        // at other batches show.
        if !same_resident_weights(record, sibling, WEIGHT_TOLERANCE) {
            continue;
        }
        // Charged to every spanned card, as the graph is built on each.
        let per_device = (*total as f64 - *sibling_total as f64) / cards as f64;
        if per_device <= 0.0 {
            continue;
        }
        let entries = heads * n_kv * n_kv.min(u64::from(key.ubatch));
        per_arch
            .entry(key.arch.clone())
            .or_default()
            .push(per_device * MIB_F64 / entries as f64);
    }
    if per_arch.is_empty() {
        return Err(DeriveError::no_data(
            "no flash-attention-off cell has an on sibling",
        ));
    }
    // The largest, so no configuration of a measured architecture is left short:
    // this term is worth thousands of MiB and under-reserving it OOMs.
    //
    // Stored in *hundredths* of a byte. Rounding up to a whole byte was the last
    // source of systematic over-reservation in the set: qwen35's pairs sit between
    // 3.3 and 4.2, and charging the 5 that `ceil` gives inflates a term worth
    // thousands of MiB by a fifth, which is the whole of the +6.7% its worst cell
    // showed.
    let table: BTreeMap<ArchKey, i64> = per_arch
        .iter()
        .map(|(arch, values)| {
            let worst = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (arch.clone(), ((worst * 100.0).ceil() as i64).max(1))
        })
        .collect();
    let detail = table
        .iter()
        .map(|(arch, value)| {
            format!(
                "{arch} {:.2} ({} pair(s))",
                *value as f64 / 100.0,
                per_arch[arch].len()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Table {
        by_key: table,
        evidence: format!(
            "paired against each cell's own flash-attention-on sibling on the driver \
             total: {detail} bytes per (head x cache token x batch token), stored in \
             hundredths and charged per spanned card. About an f32 for every dense and \
             MoE architecture; deepseek4's MLA shares one latent across heads and needs \
             far less. Proportionality in batch and in context is exact, and it follows \
             one stream's share of the cache rather than the whole budget: gemma3 at \
             four slots measures what one slot measures at a quarter the context. \
             laguna's GGUF omits `attention.head_count`, so its head count is inferred \
             from `embedding_length / key_length` and its rate absorbs any error in \
             that."
        ),
    })
}

/// The CLIP graph buffer a vision projector needs beyond its weights.
///
/// llama.cpp reserves both together and reports the sum per device:
///
/// ```text
/// [mtmd] adding 1283.89 MiB to fit_params_target for device CUDA0
/// ```
///
/// Subtracting the summed `clip_model_loader` tensor sizes isolates the graph term.
/// It was modelled as zero, which cost 140 to 248 MiB on every vision-capable
/// service.
///
/// Two configurations were measured, and the term is a property of the *vision*
/// settings rather than of the language model: Qwen3.6-27B and Qwen3.6-35B-A3B have
/// different hidden sizes and different projector files and land on 248.09 and
/// 248.10 MiB, sharing an image size of 1472 and a patch merge of 2, while
/// gemma-4-31B's 768/3 projector takes 140.50.
///
/// So it is charged as the worst of the two rather than as a rate. Both candidate
/// scalings — image size and merge factor — have exactly one degree of freedom
/// against two points, so either would fit and neither would be evidence. A third
/// vision configuration is what would settle it; until then a flat maximum
/// over-reserves gemma-4 by 108 MiB and predicts the Qwen projectors exactly, which
/// is the right way round given the Qwen shape is the one two of the three cells use.
pub fn mmproj_graph(rows: &[Record]) -> Result<Scalar> {
    let mut seen: Vec<MmprojCell> = Vec::new();
    for record in rows {
        let parsed = &record.parsed;
        let Some(reserved) = parsed
            .mmproj_reserved_mib
            .as_ref()
            .and_then(|per_device| per_device.get("CUDA0"))
            .copied()
            .filter(|v| *v != 0.0)
        else {
            continue;
        };
        let Some(tensors) = parsed.mmproj_tensor_bytes.filter(|v| *v != 0) else {
            continue;
        };
        seen.push(MmprojCell {
            graph: reserved * MIB_F64 - tensors as f64,
            image: parsed.clip_image_size.unwrap_or(0),
            merge: parsed.clip_n_merge.unwrap_or(0),
        });
    }
    if seen.is_empty() {
        return Err(DeriveError::no_data(
            "no cell reports an mmproj reservation",
        ));
    }
    let worst = seen
        .iter()
        .map(|cell| cell.graph)
        .fold(f64::NEG_INFINITY, f64::max);
    let shapes: BTreeSet<VisionShape> = seen
        .iter()
        .map(|cell| VisionShape {
            image: cell.image,
            merge: cell.merge,
            mib: round_half_even(cell.graph / MIB_F64),
        })
        .collect();
    let detail = shapes
        .iter()
        .map(|shape| {
            format!(
                "image {}/merge {} takes {} MiB",
                shape.image, shape.merge, shape.mib
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Scalar {
        value: round_half_even(worst),
        evidence: format!(
            "llama.cpp's own `adding N MiB to fit_params_target` figure net of the \
             summed clip tensor sizes, across {} cells and {} vision configurations: \
             {detail}. Charged flat at the worst rather than as a rate: two \
             configurations cannot distinguish a scaling in image size from one in the \
             merge factor, since either has a single degree of freedom against two \
             points. Identical across two language models sharing a vision \
             configuration, so it is a property of the projector.",
            seen.len(),
            shapes.len(),
        ),
    })
}

/// Record the per-device compute of architectures the curve fit cannot see.
///
/// ik_llama prints no memory-breakdown table, so an ik-only architecture's cells
/// never reach the curve fit and its entry has to be held by hand. That is a reason
/// to write the numbers down where they cannot go stale, not a reason to leave them
/// in a comment: `table_less_compute` recovers them, and this puts them in
/// `tuning.json` beside the held value they justify.
///
/// Pooling them into the mainline curves is the thing not to do. The two runtimes
/// build different graphs for the same architecture, and trying it moved four curves
/// at once and took GLM-5.2 from -3.2% to +7.8%.
pub fn table_less_observations(rows: &[Record]) -> Result<NestedTable> {
    let mut observed: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        if !parsed.devices.is_empty() {
            continue;
        }
        let Some(arch) = parsed.architecture().map(str::to_owned) else {
            continue;
        };
        if !factors.flash_attn_on() || factors.has_spec() {
            continue;
        }
        let Some(per_device) = table_less_compute(record) else {
            continue;
        };
        let key = format!("ctx{}/ub{}", factors.ctx, factors.ubatch_or_default());
        let entry = observed.entry(arch).or_default().entry(key).or_insert(0);
        *entry = (*entry).max(round_half_even(per_device));
    }
    if observed.is_empty() {
        return Err(DeriveError::no_data(
            "every architecture has a memory-breakdown table",
        ));
    }
    let evidence = observed
        .iter()
        .map(|(arch, points)| format!("{arch} at {} points", points.len()))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(NestedTable {
        by_arch: observed,
        evidence,
    })
}

/// One cell's vision reservation net of its projector tensors, with the vision
/// settings that shaped it.
#[derive(Debug, Clone, Copy)]
struct MmprojCell {
    graph: f64,
    image: u64,
    merge: u64,
}

/// A distinct vision configuration and what it takes.
///
/// The field order is the evidence string's order and a `BTreeSet` sorts by it, so
/// it is also what `tuning.json` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct VisionShape {
    image: u64,
    merge: u64,
    mib: i64,
}
