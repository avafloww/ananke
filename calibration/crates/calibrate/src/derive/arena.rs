//! The pinned graph arena, *modelled* rather than fitted.
//!
//! This is arithmetic over the graph llama.cpp builds, so it does not get a
//! best-fit line — it either reproduces the measured `CUDA_Host compute buffer
//! size` or the model is wrong, and the interesting output is the residual. Every
//! rate derived from a residual over this model inherits its correctness, which is
//! why `check_arena_model` holds it to the measurements on every run.

use std::collections::BTreeMap;

use ananke_config::{placement::SplitMode, units::MIB_F64};
use ananke_estimate::host_buffer::{pad_to_kv_cache, swa_mask_copies};

use crate::{
    derive::{
        error::{DeriveError, Result},
        keys::{ArchKey, VariantKey},
        tuning::Tuning,
    },
    record::Record,
};

/// Below this many activated experts per expert — `tokens * n_expert_used /
/// n_expert` — the MoE op intermediates stay on the host. [`crate::derive::graph::offload_min_batch`]
/// derives the same threshold from the dataset and lands on this number, so the
/// modelled arena and the fitted constant are one fact stated twice.
pub const MOE_OFFLOAD_MIN_BATCH_RATIO: u64 = 32;

/// The modelled arena, split into its three terms, in MiB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArenaTerms {
    /// The KQ mask.
    pub mask: f64,
    /// The second, window-sized mask an interleaved-SWA model carries.
    pub swa_mask: f64,
    /// The f32 hidden-state graph inputs, plus any CPU-resident MoE
    /// intermediates.
    pub hidden: f64,
}

impl ArenaTerms {
    /// The two mask terms, which replicate together across devices.
    pub fn masks(&self) -> f64 {
        self.mask + self.swa_mask
    }

    /// Every term, on one device.
    pub fn total(&self) -> f64 {
        self.mask + self.swa_mask + self.hidden
    }
}

/// Whether the modelled arena includes the CPU-resident MoE intermediates.
///
/// [`MoeCharge::Off`] belongs to the derivers that are *fitting* that term, so it
/// shows up in their residual instead of being subtracted twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeCharge {
    On,
    Off,
}

/// The modelled arena for one cell.
///
/// mainline sizes the KQ mask against one slot's share of the context unless the
/// cache is unified; ik does not divide by slots at all. An interleaved SWA model
/// carries a second mask sized to the window plus the batch, not to the window
/// alone.
pub fn arena_terms(record: &Record, charge_moe: MoeCharge, tuning: &Tuning) -> ArenaTerms {
    let (factors, parsed) = (&record.factors, &record.parsed);
    let arch = parsed.arch.as_str();
    let ctx = u64::from(factors.ctx);
    let slots = u64::from(factors.parallel);
    let unified = factors.kv_unified;
    let ik = factors.runtime_is_ik();

    let n_kv = if ik || unified || slots == 1 {
        ctx
    } else {
        ctx / slots
    };
    let n_kv = pad_to_kv_cache(n_kv);
    let tokens = factors.tokens();
    let width = factors.flash_attn.mask_element_bytes();

    // MLA compresses K and V into a shared latent, so the mask is half width.
    let mla = matches!(arch, "deepseek4" | "deepseek2" | "glm-dsa");
    let dsa = ik && factors.extra.iter().any(|a| a == "-dsa");
    let mask = if dsa {
        // The sparse path allocates three masks and does *not* halve them:
        // measured at exactly 6.00 half-width units across two contexts and two
        // batch sizes, above the MoE threshold where nothing else contaminates
        // the figure.
        n_kv * tokens * width * 3
    } else {
        n_kv * tokens * width / if mla { 2 } else { 1 }
    };

    let swa = parsed.n_swa;
    // mainline sizes the second mask to the window plus the batch; ik sizes it to
    // the whole context, which is why an SWA model costs it so much more.
    let swa_rows = if ik {
        n_kv
    } else {
        pad_to_kv_cache(swa + tokens)
    };
    let swa_copies = swa_mask_copies(swa, tokens, slots > 1 && unified && !ik);
    let swa_mask = if swa != 0 {
        swa_copies * swa_rows * tokens * width
    } else {
        0
    };

    // Two f32 hidden-state buffers on mainline, one on ik.
    let n_embd = parsed.n_embd;
    let mut hidden = if ik { 1 } else { 2 } * n_embd * tokens * 4;

    // ik keeps its MoE op intermediates on the CPU below a batch threshold; see
    // FINDINGS.md. mainline shows the same shape under a tensor split alone.
    let experts = parsed.n_expert;
    let used = parsed.n_expert_used;
    if charge_moe == MoeCharge::On
        && experts != 0
        && used != 0
        && tokens * used < MOE_OFFLOAD_MIN_BATCH_RATIO * experts
    {
        if ik {
            hidden += tuning
                .ik_moe_rate_frozen_arch_miss(&ArchKey::recorded(record))
                .max(0) as u64
                * n_embd
                * tokens;
        } else if factors.is_hybrid() && factors.split_or_layer() == SplitMode::Tensor {
            hidden += tuning.mainline_tensor_moe_per_nembd().max(0) as u64 * n_embd * tokens;
        }
    }

    ArenaTerms {
        mask: mask as f64 / MIB_F64,
        swa_mask: swa_mask as f64 / MIB_F64,
        hidden: hidden as f64 / MIB_F64,
    }
}

/// Hold this model to the same measurements the estimator is held to.
///
/// `arena_terms` here and `pinned_graph_bytes` in `host_buffer.rs` are two
/// implementations of one model, and nothing in either stops them drifting apart:
/// an analysis modelling a single window mask against an estimator charging three
/// surfaces only as `consensus` reading a 5.27 multiple among cells that are
/// otherwise 4.00.
///
/// Neither can be checked against the other directly across languages, but both
/// can be checked against the measurement, and a model that reproduces the
/// hardware cannot have drifted from another model that also does. The Rust
/// estimator asserts this in `arena_reproduces_the_measured_pinned_buffer`; this
/// is the same assertion on this side, so drift fails on whichever side
/// introduces it rather than silently shifting a derived constant.
pub fn check_arena_model(
    rows: &[Record],
    tuning: &Tuning,
    no_fa_rates: &BTreeMap<VariantKey, i64>,
    quant_rates: &BTreeMap<ArchKey, i64>,
    tolerance_mib: f64,
) -> Result<()> {
    // Required, not defaulted: this compares the model against the measurement, and
    // a zeroed per-layer term would report the E variant as drifted rather than as
    // unread.
    let e_variant_rate = tuning.required_f64("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN")?;
    let mut worst: BTreeMap<String, WorstArenaCell> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        let arena = parsed.arena_mib;
        if arena == 0.0 {
            continue;
        }
        if factors.has_spec()
            || factors.split_or_layer() == SplitMode::Tensor
            || factors.is_hybrid()
        {
            continue;
        }
        // Layers on the GPU only. With `-ngl 0` nothing is pinned and the CPU
        // backend holds op intermediates a GPU run offloads, which is a different
        // graph and not what this models.
        if !factors.fully_offloaded() {
            continue;
        }
        let terms = arena_terms(record, MoeCharge::On, tuning);
        let cards = factors.cards_or(1);
        let copies = if cards > 1 && !factors.runtime_is_ik() {
            4.0
        } else {
            1.0
        };
        let tokens = factors.tokens() as f64;
        // Never on ik: `pinned_graph_bytes` returns before this term, so letting
        // the table's default apply here would compare against a model the
        // estimator does not use.
        let no_fa = if !factors.runtime_is_ik() && factors.flash_attn.charged_unfused() {
            table_rate(no_fa_rates, &VariantKey::of(record)) as f64 * tokens / MIB_F64
        } else {
            0.0
        };
        let quantised = if factors.kv_quantised() {
            table_rate(quant_rates, &ArchKey::recorded(record)) as f64 * tokens / MIB_F64
        } else {
            0.0
        };
        let e_variant = if parsed.per_layer_token_embd {
            e_variant_rate * parsed.n_layer as f64 * tokens / MIB_F64
        } else {
            0.0
        };
        let modelled = copies * (terms.masks() + no_fa) + terms.hidden + quantised + e_variant;
        let error = (arena - modelled).abs();
        let arch = parsed.arch.clone();
        let entry = worst.entry(arch).or_default();
        if error > entry.error {
            *entry = WorstArenaCell {
                error,
                log: record.log.clone(),
            };
        }
    }
    let bad: Vec<String> = worst
        .iter()
        .filter(|(_, cell)| cell.error > tolerance_mib)
        .map(|(arch, cell)| {
            let head: String = cell.log.chars().take(40).collect();
            format!("{arch} off by {:.1} MiB on {head}", cell.error)
        })
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    Err(DeriveError::disagreement(format!(
        "this file's arena model no longer reproduces the measurements: {}. It and \
         `pinned_graph_bytes` are two implementations of one model; whichever has \
         drifted, a derived constant is being fitted against the wrong residual.",
        bad.join("; ")
    )))
}

/// The tolerance `check_arena_model` allows before calling it drift.
pub const ARENA_TOLERANCE_MIB: f64 = 5.0;

/// The cell an architecture's model reproduces worst, and the log line naming it.
#[derive(Debug, Default, Clone)]
struct WorstArenaCell {
    error: f64,
    log: String,
}

/// A rate, falling back to the table's worst where the key has no row — the same
/// fallback the estimator applies.
fn table_rate<K: Ord>(table: &BTreeMap<K, i64>, key: &K) -> i64 {
    match table.get(key) {
        Some(rate) => *rate,
        None => table.values().copied().max().unwrap_or(0),
    }
}
