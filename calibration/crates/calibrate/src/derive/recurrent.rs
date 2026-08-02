//! The recurrent-state module, and the one constant read straight out of it.

use std::collections::{BTreeMap, BTreeSet};

use ananke_config::units::MIB_F64;
use ananke_gguf::keys::suffix;

use crate::{
    derive::{
        Scalar,
        error::{DeriveError, Result},
    },
    record::{Parsed, Record, RsPool},
};

/// Every measured `llama_memory_recurrent` pool, with its record's shape.
///
/// A record creates at most one recurrent module, but it appears once per context
/// the server built — including llama.cpp's parameter-fitting dry run — so the pools
/// are taken from whichever contexts hold one and deduplicated by the reader.
pub fn recurrent_pools(rows: &[Record]) -> Vec<(&RsPool, &Parsed)> {
    let mut out = Vec::new();
    for record in rows {
        for context in &record.parsed.contexts {
            if let Some(pool) = &context.rs_pool {
                out.push((pool, &record.parsed));
            }
        }
    }
    out
}

/// The R and S halves of a recurrent pool, from GGUF metadata alone.
///
/// This is `estimator/recurrent.rs` transcribed. It exists so the formula is held to
/// every measurement rather than checked once by hand: a divergence fails `emit` on
/// whichever side introduced it.
///
/// Returns `None` for a model whose metadata does not describe a recurrent block,
/// which is how a pool of zero size reads.
pub fn modelled_recurrent_mib(pool: &RsPool, parsed: &Parsed) -> Option<(f64, f64)> {
    // A convolution kernel is how an SSM block announces itself; without one the
    // model is the shortconv shape instead. Absent and zero pick the same branch.
    let (n_embd_r, n_embd_s) = match parsed.gguf(suffix::SSM_CONV_KERNEL).unwrap_or(0) {
        d_conv if d_conv != 0 => {
            // Every other SSM dimension must be there if the kernel is. A missing
            // one must not read as zero: that shrinks the modelled state silently.
            let d_inner = parsed.gguf(suffix::SSM_INNER_SIZE)?;
            let d_state = parsed.gguf(suffix::SSM_STATE_SIZE)?;
            let n_group = parsed.gguf(suffix::SSM_GROUP_COUNT)?;
            (
                (d_conv - 1) * (d_inner + 2 * n_group * d_state),
                d_state * d_inner,
            )
        }
        _ => {
            let l_cache = parsed.gguf(suffix::SHORTCONV_L_CACHE).unwrap_or(0);
            if l_cache <= 1 {
                return None;
            }
            (parsed.gguf(suffix::EMBEDDING_LENGTH)? * (l_cache - 1), 0)
        }
    };

    // The layer count the pool itself reports is the span, not the number of
    // allocating layers; the attention layers within it are subtracted the way the
    // estimator does.
    let span = pool.layers as i64;
    let interval = parsed.gguf(suffix::FULL_ATTENTION_INTERVAL).unwrap_or(0);
    if interval == 0 {
        // Without an interval the pattern is a per-layer property the log does not
        // carry (LFM2's is a fixed list inside llama.cpp), so the layer count cannot
        // be checked from the record — only the per-layer size.
        return None;
    }
    let recurrent = span - span / interval;
    let copies = pool.seqs as i64 * (pool.rs_seq + 1) as i64;
    let scale = (recurrent * copies * 4) as f64 / MIB_F64;
    Some((n_embd_r as f64 * scale, n_embd_s as f64 * scale))
}

/// Hold the recurrent-state formula to every pool the dataset recorded.
///
/// The R and S halves are checked separately, because a formula that is wrong in
/// both directions can still land on the right total: an earlier per-slot constant
/// reproduced Qwen3.6-27B's 149.62 MiB exactly while being a factor of two out on the
/// same model's two-slot reading.
///
/// The tolerance is the log's own rounding — these figures are printed to two
/// decimals — not a fitting margin. The formula reproduces every pool.
pub fn check_recurrent_model(rows: &[Record], tolerance_mib: f64) -> Result<()> {
    let mut worst: BTreeMap<String, WorstPool> = BTreeMap::new();
    for (pool, parsed) in recurrent_pools(rows) {
        let Some((modelled_r, modelled_s)) = modelled_recurrent_mib(pool, parsed) else {
            continue;
        };
        let arch = parsed.arch.clone();
        for (half, got, want) in [("R", modelled_r, pool.r_mib), ("S", modelled_s, pool.s_mib)] {
            let error = (got - want).abs();
            let key = format!("{arch} {half}");
            let entry = worst.entry(key).or_default();
            if error > entry.error {
                *entry = WorstPool {
                    error,
                    why: format!(
                        "modelled {got:.2} vs {want:.2} MiB at {} seqs, {} rs_seq",
                        pool.seqs, pool.rs_seq
                    ),
                };
            }
        }
    }
    let bad: Vec<String> = worst
        .iter()
        .filter(|(_, pool)| pool.error > tolerance_mib)
        .map(|(key, pool)| format!("{key} off by {:.2} MiB ({})", pool.error, pool.why))
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    Err(DeriveError::disagreement(format!(
        "the recurrent-state formula no longer reproduces the measurements: {}. Either \
         llama.cpp changed how it sizes the module or estimator/recurrent.rs and this \
         file have drifted apart.",
        bad.join("; ")
    )))
}

/// The log prints its figures to two decimals, so that is the whole tolerance.
pub const RECURRENT_TOLERANCE_MIB: f64 = 0.02;

/// How many extra copies of the recurrent state speculative decoding holds.
///
/// llama.cpp calls it `n_rs_seq` and prints it in the recurrent module's own summary
/// line, so it is read rather than fitted. It is a property of the runtime's
/// speculative depth — not of the model — which is why it belongs here instead of
/// being derived from GGUF metadata.
pub fn spec_rollback_depth(rows: &[Record]) -> Result<Scalar> {
    check_recurrent_model(rows, RECURRENT_TOLERANCE_MIB)?;
    let pools = recurrent_pools(rows);
    let speculative: BTreeSet<u64> = pools
        .iter()
        .map(|(pool, _)| pool.rs_seq)
        .filter(|d| *d > 0)
        .collect();
    if speculative.is_empty() {
        return Err(DeriveError::no_data(
            "no recurrent pool measured under speculative decoding",
        ));
    }
    if speculative.len() > 1 {
        let listed: Vec<String> = speculative.iter().map(u64::to_string).collect();
        return Err(DeriveError::disagreement(format!(
            "recurrent rollback depth is not one value across the dataset: [{}]. It was \
             a runtime constant; if a flag now moves it, the estimator has to read that \
             flag rather than a constant.",
            listed.join(", ")
        )));
    }
    let depth = *speculative.iter().next().expect("non-empty");
    let slot_counts: BTreeSet<u64> = pools.iter().map(|(pool, _)| pool.seqs).collect();
    Ok(Scalar {
        value: depth as i64,
        evidence: format!(
            "llama.cpp's own `rs_seq` field, {depth} on every speculative cell and 0 on \
             every other, across {} slot counts. The recurrent module is replicated \
             `parallel x (depth + 1)`: production Qwen3.6-27B measures 1197.00 MiB \
             against 149.62 at one slot without speculation, exactly 2 x 4.",
            slot_counts.len(),
        ),
    })
}

/// The pool one half of the formula reproduces worst, and what it read there.
#[derive(Debug, Default, Clone)]
struct WorstPool {
    error: f64,
    why: String,
}
