//! One per-device compute model, fitted per (runtime, split, architecture).
//!
//! This replaces three separate mechanisms — `compute_buffer_curves` for a
//! mainline layer split, `tensor_compute_curves` plus its intermediates/quantised/
//! shadow tables for a tensor split, and the four `ik_compute_*` rate tables —
//! with a single design matrix. They were three because each was derived against
//! a different target: llama.cpp's `compute` column plus `unaccounted` for one,
//! the fused `Meta()` row's `compute` column alone for another, and the driver
//! total minus a modelled remainder for ik, which prints no breakdown table at
//! all. Three targets meant three shapes, and a cell that fell between them was
//! covered by whichever mechanism claimed it rather than by the one that
//! described it.
//!
//! The target here is uniform and per *device*:
//!
//! ```text
//! target = gpu{card}_used_mib - model_share - kv_share
//! ```
//!
//! It is available for every cell, ik's included. ik prints no breakdown table at
//! all, but the same quantity is recoverable from the driver total less the
//! weights and context its own buffer lines name, over the number of cards —
//! which is what [`dataset::table_less_compute`] does. Those cells enter as one
//! averaged observation each, exactly as a fused tensor cell does.
//!
//! A layer split reports one breakdown row per card, so `model_share` and
//! `kv_share` are that card's own and each card contributes an observation.
//!
//! A tensor split reports a single fused row whose `total`, `free`, and `self` are
//! summed across cards but whose `model`, `kv`, and `compute` columns are one
//! card's share. Both cards therefore get charged the *same* share, and where the
//! real split is uneven — a mixture of experts under a tensor split need not
//! divide evenly — the whole difference lands in the residual with opposite signs
//! on the two cards. gemma-4-26B-A4B reads 184 MiB on device 0 against 490 on
//! device 1 at identical settings, which is not 306 MiB of extra compute on one
//! card but the fused share being wrong for both. So a fused cell contributes one
//! observation at the per-card *average*, which is the quantity the fused row's
//! own semantics support and the quantity the packer charges per spanned GPU.
//!
//! ## Columns
//!
//! Every column is dimensionally normalised, so models of different width share
//! coefficients instead of each needing its own absolute flat term. `hidden` is
//! the one that matters most: graph intermediates are some count of hidden-width
//! f32 buffers per batch token, so the column is `n_embd * ubatch` and the
//! coefficient is that count times four bytes. Fitting three gemma4 widths
//! against a shared *absolute* term is what previously left that group at 149%
//! error.
//!
//! ```text
//! flat          1                            CUDA context, workspaces, graph metadata
//! head_flat     head_share                   the head card's output graph and scratch
//! hidden        n_embd * ub                  per-token hidden-width intermediates
//! doubling      log2(max(ub, CHUNK)/CHUNK)   ik's step per batch doubling above -amb
//! mask          copies * ub * n_kv           the attention mask and its copies
//! quant         ub * ctx                     dequantisation of a quantised cache
//! logits        head_share * n_vocab * ub    the head card also materialises logits
//! offload_head  head_share when offloading   expert staging, which the head card holds
//! ```
//!
//! `head_share` is 1 on the primary card and 0 elsewhere for a layer split, and
//! `1 / cards` for a fused tensor row, whose target is already an average over
//! the cards it spans.
//!
//! `flat`, `head_flat`, and `doubling` carry MiB; the rest carry bytes per
//! element, the columns being pre-divided by 2^20.
//!
//! The arithmetic itself is not here. It lives in
//! [`ananke_estimate::compute_model::Columns::from_scalars`], which the estimator
//! evaluates its dot product against, so the fitter and the evaluator cannot
//! compute the same column differently — a coefficient fitted against one
//! definition would otherwise multiply the other.
//!
//! `head_flat` is not bookkeeping for `logits`, which is already per-head and
//! scales with the batch. It is there because the head card's cost is measurably
//! *flat*: deepseek4 on two cards holds 2359 MiB on device 0 at ctx 8192 ub 512
//! and 2381 at ctx 32768 ub 2048, while device 1 moves from 585 to 1287 across
//! the same pair. Without the column that asymmetry has nowhere to go and is paid
//! for by inflating terms that do scale, which is what left the group at 65%.
//!
//! The offload axis enters as that one boolean interaction rather than through the
//! `n_cpu_moe` count, because the count carries no additional signal. Columns
//! proportional to it — a flat per-offloaded-layer cost and a per-layer per-token
//! one — were fitted and changed nothing: 151 of 633 observations outside +/-5%
//! with them and 151 without, at a marginally better median. Dropping
//! `offload_head` instead takes qwen35moe from 11% to 71%. So the effect is real
//! and the head card is where it lands, but it does not scale with how many layers
//! are offloaded, and carrying the count would mean threading a placement outcome
//! back into the estimate that feeds placement for no measured gain.
//!
//! `offload_head` exists because expert offload, not the output head, is what
//! makes a layer split asymmetric. Qwen3.6-35B-A3B fully resident on two cards
//! holds 428 MiB on each, while the same model under `--n-cpu-moe 40` holds
//! visibly more on the primary — so a plain `head_flat` had to average the two and
//! missed both by 41%. Gathering and scattering CPU-resident expert activations
//! stages through the primary device, which is where the buffers land.
//!
//! `mask` carries its replication count rather than leaving the fit to discover
//! it. Mainline replicates the graph's masks a fixed number of times when layers
//! are split across more than one device — separately derived as
//! `MAINLINE_LAYER_SPLIT_MASK_COPIES`, 99 layer-split cells at 4.00 against 147
//! single-card and tensor-split cells at 1.00, flat across context, batch, slot
//! count, and cache mode — and the caller passes that count in. Pooling the two
//! card counts into one unreplicated column instead had the fit report 7.65 bytes
//! per token-pair for a buffer whose element is an f16. With the count supplied,
//! the coefficient is free to land on 2, which is then a check on the model rather
//! than a parameter of it.
//!
//! ## What is deliberately not in it
//!
//! Flash-attention-off cells are excluded and keep their own paired deriver. The
//! unfused score matrix is worth thousands of MiB against cells whose other terms
//! are hundreds, so fitting it here lets one term dominate a model it is not part
//! of. The estimator adds it on top.
//!
//! Speculative-decoding cells are excluded because the MTP overhead has its own
//! derived model and would otherwise be counted twice. Vision cells are excluded
//! for the same reason: the mmproj weights and CLIP graph buffer are charged
//! separately as `MMPROJ_GRAPH_BYTES`, and leaving them in put
//! gemma-4-31B-it-qat at 1870 MiB on its primary card against the 26B's 184 under
//! otherwise identical settings, which the fit could only split the difference on.
//!
//! Rows whose card holds no layers are excluded. A card holding no layers is not
//! doing compute — its whole cost is the bare CUDA context, which several groups
//! show as exactly 256 MiB — and those rows inform the per-device shadow instead.
//! Leaving them in made the fit predict 484 MiB where the card held 256.

use std::collections::HashMap;

use ananke_estimate::compute_model::{Columns, Scalars};

use crate::record::Record;

pub mod dataset;
mod document;
mod fit;
mod solve;

pub use document::document_section;
pub use fit::{Coefficients, fit};
pub use solve::evaluate;

/// One design row's value for a named column.
///
/// The fit works over named columns because the greedy selection admits them one
/// at a time and drops them again; the estimator's struct is positional because it
/// evaluates a fixed dot product. This is the join between the two.
pub fn column_value(columns: &Columns, name: &str) -> f64 {
    columns
        .by_name()
        .into_iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, value)| value)
        .unwrap_or(0.0)
}

/// llama.cpp's default micro-batch, which the harness records as null when it did
/// not pass `-ub`.
pub const DEFAULT_UBATCH: u32 = 512;

/// What a single set of coefficients is fitted against.
///
/// The variant is carried separately rather than folded into the architecture
/// string. Concatenating them made `gemma4` + `e` indistinguishable from an
/// architecture literally named `gemma4e`, and splitting it back off cost the arch
/// its trailing letter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Group {
    pub runtime: String,
    pub split: String,
    pub arch: String,
    pub variant: Option<&'static str>,
}

/// One measured device, with its design row and what it actually held.
#[derive(Debug, Clone)]
pub struct Row {
    pub columns: Columns,
    /// MiB the card held beyond its own weights and context.
    pub target: f64,
    pub tag: Tag,
}

/// Where a row came from. Carried for reporting; the fit itself never reads it.
#[derive(Debug, Clone)]
pub struct Tag {
    pub ctx: u32,
    pub ubatch: u32,
    pub kv_type: Option<String>,
    pub cards: usize,
    pub model: String,
    pub parallel: Option<u32>,
    pub n_cpu_moe: Option<u32>,
    pub mmproj: bool,
    pub label: String,
    /// The visible device index, or `-1` where the observation is an average over
    /// every card the cell spanned.
    pub device: i32,
}

/// Every usable per-device observation, grouped, in the order first seen.
///
/// Insertion order is preserved rather than sorted because the pooled default is
/// fitted over the concatenation of the mainline layer-split groups, and a
/// deterministic order there keeps the summation — and so the coefficients —
/// reproducible.
#[derive(Debug, Default)]
pub struct Groups {
    entries: Vec<(Group, Vec<Row>)>,
    index: HashMap<Group, usize>,
}

impl Groups {
    pub fn iter(&self) -> impl Iterator<Item = (&Group, &[Row])> {
        self.entries
            .iter()
            .map(|(key, rows)| (key, rows.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn push(&mut self, key: Group, row: Row) {
        match self.index.get(&key) {
            Some(&at) => self.entries[at].1.push(row),
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, vec![row]));
            }
        }
    }
}

/// Group every usable per-device observation by runtime, split, arch, and variant.
///
/// `split_mask_copies` is how many copies of the attention mask mainline holds
/// when layers span more than one device, derived separately as
/// `MAINLINE_LAYER_SPLIT_MASK_COPIES`.
///
/// A cell whose runtime prints no memory breakdown has its per-device target
/// recovered by [`dataset::table_less_compute`], so ik's cells and the reporting
/// that predates this model agree by construction.
pub fn collect(rows: &[&Record], split_mask_copies: u32, include_spec: bool) -> Groups {
    let mut groups = Groups::default();
    for record in rows {
        let factors = &record.factors;
        let parsed = &record.parsed;
        let Some(arch) = parsed.arch.as_deref().filter(|a| !a.is_empty()) else {
            continue;
        };
        let Some(n_embd) = parsed.n_embd.filter(|n| *n > 0) else {
            continue;
        };
        if factors.flash_attn.as_deref() == Some("off") {
            continue;
        }
        if !include_spec && present(factors.spec_type.as_deref()) {
            continue;
        }
        if present(factors.mmproj.as_deref()) {
            continue;
        }
        let cards: Vec<&str> = factors.gpus.split(',').filter(|c| !c.is_empty()).collect();
        let devices = &parsed.devices;
        let fused = devices
            .first()
            .is_some_and(|d| d.device.starts_with("Meta"));
        let ubatch = factors.ubatch.filter(|u| *u > 0).unwrap_or(DEFAULT_UBATCH);
        let streams = if factors.kv_unified {
            1
        } else {
            factors.parallel.unwrap_or(1).max(1)
        };
        // The E-variants keep their embeddings per layer on the host, which is a
        // different graph under the same architecture string.
        let variant = parsed
            .per_layer_token_embd
            .unwrap_or(false)
            .then_some("gemma_e");
        let split = factors
            .split
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("layer");
        // A tensor split reports one card's share, so its mask is unreplicated in
        // the figure regardless of card count; only a layer split spanning more
        // than one device pays for copies. A hybrid does not replicate them
        // either — measured at 1.00 against 4.00 — so the replication follows the
        // placement, not the model.
        let hybrid = factors.n_cpu_moe.is_some_and(|n| n > 0);
        let copies = if split == "layer" && cards.len() > 1 && !hybrid {
            split_mask_copies
        } else {
            1
        };
        let key = Group {
            runtime: factors.runtime.clone(),
            split: split.to_string(),
            arch: arch.to_string(),
            variant,
        };
        let readings: Vec<(usize, f64)> = cards
            .iter()
            .enumerate()
            .filter_map(|(index, card)| {
                card.parse::<u32>()
                    .ok()
                    .and_then(|id| record.gpu_card_used_mib(id))
                    .map(|used| (index, used))
            })
            .collect();
        if readings.is_empty() {
            continue;
        }
        let scalars = |head_share: f64| Scalars {
            ubatch: f64::from(ubatch),
            n_kv: f64::from(factors.ctx / streams),
            ctx: f64::from(factors.ctx),
            quantised: factors.kv_type.as_deref() != Some("f16"),
            head_share,
            n_vocab: parsed.n_vocab.unwrap_or(0) as f64,
            n_embd: f64::from(n_embd),
            offloading: factors.n_cpu_moe.is_some_and(|n| n > 0),
            mask_copies: f64::from(copies),
        };
        let tag = |device: i32| Tag {
            ctx: factors.ctx,
            ubatch,
            kv_type: factors.kv_type.clone(),
            cards: cards.len(),
            model: record.provenance.model_key.clone(),
            parallel: factors.parallel,
            n_cpu_moe: factors.n_cpu_moe,
            mmproj: present(factors.mmproj.as_deref()),
            label: factors.label.clone(),
            device,
        };
        if devices.is_empty() {
            // No breakdown table — ik. One averaged observation, recovered from
            // the driver total and the runtime's own buffer lines.
            let Some(target) = dataset::table_less_compute(record).filter(|t| *t > 0.0) else {
                continue;
            };
            let share = 1.0 / readings.len() as f64;
            groups.push(
                key,
                Row {
                    columns: Columns::from_scalars(scalars(share)),
                    target,
                    tag: tag(-1),
                },
            );
            continue;
        }
        if fused {
            let device = &devices[0];
            if device.model_mib == 0.0 {
                continue;
            }
            let targets: Vec<f64> = readings
                .iter()
                .map(|(_, used)| used - device.model_mib - device.kv_mib)
                .collect();
            if targets.iter().any(|t| *t <= 0.0) {
                continue;
            }
            let share = 1.0 / targets.len() as f64;
            let mean = targets.iter().sum::<f64>() / targets.len() as f64;
            groups.push(
                key,
                Row {
                    columns: Columns::from_scalars(scalars(share)),
                    target: mean,
                    tag: tag(-1),
                },
            );
            continue;
        }
        for (index, used) in readings {
            // A card holding no layers is not doing compute: its whole cost is the
            // bare CUDA context, and those rows inform the per-device shadow
            // rather than this model.
            let Some(device) = devices.get(index).filter(|d| d.model_mib != 0.0) else {
                continue;
            };
            let target = used - device.model_mib - device.kv_mib;
            if target <= 0.0 {
                continue;
            }
            let share = if index == 0 { 1.0 } else { 0.0 };
            groups.push(
                key.clone(),
                Row {
                    columns: Columns::from_scalars(scalars(share)),
                    target,
                    tag: tag(index as i32),
                },
            );
        }
    }
    groups
}

/// Whether an optional string factor was set to anything.
fn present(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.is_empty())
}
