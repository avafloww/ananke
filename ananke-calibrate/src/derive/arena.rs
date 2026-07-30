//! The pinned graph arena, *modelled* rather than fitted.
//!
//! This is arithmetic over the graph llama.cpp builds, so it does not get a
//! best-fit line — it either reproduces the measured `CUDA_Host compute buffer
//! size` or the model is wrong, and the interesting output is the residual. Every
//! rate derived from a residual over this model inherits its correctness, which is
//! why `check_arena_model` holds it to the measurements on every run.

use std::collections::BTreeMap;

use crate::{
    derive::{
        error::{DeriveError, Result},
        shape::variant_key,
        stats::pad,
        tuning::Tuning,
    },
    record::Record,
};

/// llama.cpp pads the KV cache length to a multiple of this.
pub const KV_CACHE_PAD: u64 = 256;

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

/// The modelled arena for one cell.
///
/// mainline sizes the KQ mask against one slot's share of the context unless the
/// cache is unified; ik does not divide by slots at all. An interleaved SWA model
/// carries a second mask sized to the window plus the batch, not to the window
/// alone.
///
/// `charge_moe` is turned off by the derivers that are *fitting* the MoE term, so
/// it shows up in their residual instead of being subtracted twice.
pub fn arena_terms(record: &Record, charge_moe: bool, tuning: &Tuning) -> ArenaTerms {
    let (factors, parsed) = (&record.factors, &record.parsed);
    let arch = parsed.arch.as_deref().unwrap_or("");
    let ctx = u64::from(factors.ctx);
    let slots = u64::from(factors.parallel.unwrap_or(0));
    let unified = factors.kv_unified;
    let ik = factors.runtime_is_ik();

    let n_kv = if ik || unified || slots == 1 { ctx } else { ctx / slots };
    let n_kv = pad(n_kv, KV_CACHE_PAD);
    let tokens = factors.tokens();
    let width = if factors.flash_attn_on() { 2 } else { 4 };

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

    let swa = u64::from(parsed.n_swa.unwrap_or(0));
    // mainline sizes the second mask to the window plus the batch; ik sizes it to
    // the whole context, which is why an SWA model costs it so much more.
    let swa_rows = if ik { n_kv } else { pad(swa + tokens, KV_CACHE_PAD) };
    // Three window masks when several slots share one cache, matching
    // `host_buffer::pinned_graph_bytes`. This model went on charging one after
    // the estimator was changed, which is the same drift that left the ik MoE
    // rate stale — and it is why `consensus` saw a 5.27 multiple among cells that
    // are otherwise 4.00.
    //
    // One mask per batch the window spans, plus the batch's own. The constant 3
    // this replaces came from a sweep taken entirely at ubatch 512, where a
    // 1024-token window spans two batches; at 2048 the same configuration
    // measures 2, differing by exactly one mask.
    let swa_copies = if slots > 1 && unified && !ik { 1 + swa.div_ceil(tokens).min(2) } else { 1 };
    let swa_mask = if swa != 0 { swa_copies * swa_rows * tokens * width } else { 0 };

    // Two f32 hidden-state buffers on mainline, one on ik.
    let n_embd = u64::from(parsed.n_embd.unwrap_or(0));
    let mut hidden = if ik { 1 } else { 2 } * n_embd * tokens * 4;

    // ik keeps its MoE op intermediates on the CPU below a batch threshold; see
    // FINDINGS.md. mainline shows the same shape under a tensor split alone.
    let experts = u64::from(parsed.n_expert.unwrap_or(0));
    let used = u64::from(parsed.n_expert_used.unwrap_or(0));
    if charge_moe && experts != 0 && used != 0 && tokens * used < 32 * experts {
        if ik {
            hidden += tuning.ik_moe_rate(arch).max(0) as u64 * n_embd * tokens;
        } else if factors.is_hybrid() && factors.split.as_deref() == Some("tensor") {
            hidden += tuning.mainline_tensor_moe_per_nembd().max(0) as u64 * n_embd * tokens;
        }
    }

    const MIB: f64 = (1024 * 1024) as f64;
    ArenaTerms {
        mask: mask as f64 / MIB,
        swa_mask: swa_mask as f64 / MIB,
        hidden: hidden as f64 / MIB,
    }
}

/// Hold this model to the same measurements the estimator is held to.
///
/// `arena_terms` here and `pinned_graph_bytes` in `host_buffer.rs` are two
/// implementations of one model, and they have drifted once already: the analysis
/// went on modelling a single window mask after the estimator moved to three,
/// which is what made `consensus` see a 5.27 multiple among cells that are
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
    no_fa_rates: &BTreeMap<String, i64>,
    quant_rates: &BTreeMap<String, i64>,
    tolerance_mib: f64,
) -> Result<()> {
    let e_variant_rate = tuning.constant_f64("GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN", 0.0);
    let mut worst: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for record in rows {
        let (parsed, factors) = (&record.parsed, &record.factors);
        let Some(arena) = parsed.arena_mib.filter(|v| *v != 0.0) else {
            continue;
        };
        if factors.has_spec() || factors.split_or_layer() == "tensor" || factors.is_hybrid() {
            continue;
        }
        // Layers on the GPU only. With `-ngl 0` nothing is pinned and the CPU
        // backend holds op intermediates a GPU run offloads, which is a different
        // graph and not what this models.
        if factors.ngl != Some(99) {
            continue;
        }
        let terms = arena_terms(record, true, tuning);
        let cards = factors.cards_or(1);
        let copies = if cards > 1 && !factors.runtime_is_ik() { 4.0 } else { 1.0 };
        let tokens = factors.tokens() as f64;
        // Never on ik: `pinned_graph_bytes` returns before this term, so letting
        // the table's default apply here would compare against a model the
        // estimator does not use.
        let no_fa = if !factors.runtime_is_ik() && !factors.flash_attn_on() {
            table_rate(no_fa_rates, &variant_key(record, false)) as f64 * tokens / MIB
        } else {
            0.0
        };
        let quantised = if factors.kv_type.as_deref() != Some("f16") {
            table_rate(quant_rates, parsed.arch.as_deref().unwrap_or("None")) as f64 * tokens / MIB
        } else {
            0.0
        };
        let e_variant = if parsed.per_layer_token_embd.unwrap_or(false) {
            e_variant_rate * f64::from(parsed.n_layer.unwrap_or(0)) * tokens / MIB
        } else {
            0.0
        };
        let modelled =
            copies * (terms.masks() + no_fa) + terms.hidden + quantised + e_variant;
        let error = (arena - modelled).abs();
        let arch = parsed.arch.clone().unwrap_or_else(|| "None".to_string());
        let entry = worst.entry(arch).or_insert((0.0, String::new()));
        if error > entry.0 {
            *entry = (error, record.log.clone().unwrap_or_default());
        }
    }
    let bad: Vec<String> = worst
        .iter()
        .filter(|(_, (error, _))| *error > tolerance_mib)
        .map(|(arch, (error, log))| {
            let head: String = log.chars().take(40).collect();
            format!("{arch} off by {error:.1} MiB on {head}")
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

const MIB: f64 = (1024 * 1024) as f64;

/// A per-architecture rate, falling back to the table's worst where the
/// architecture has no row — the same fallback the estimator applies.
fn table_rate(table: &BTreeMap<String, i64>, key: &str) -> i64 {
    match table.get(key) {
        Some(rate) => *rate,
        None => table.values().copied().max().unwrap_or(0),
    }
}
