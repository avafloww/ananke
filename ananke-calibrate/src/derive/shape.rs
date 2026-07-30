//! Reading a cell's shape: how a record is keyed, how wide its attention is,
//! and what the runtime reported holding where.

use crate::record::{Parsed, Record};

/// llama.cpp's `--checkpoint-min-step` default: the prompt length past which a
/// second context checkpoint is taken, and so the line between a cell that
/// measures the probe's state and one that measures a real service's.
pub const CHECKPOINT_MIN_STEP: u32 = 8192;

/// The architecture, plus the distinctions that split one arch string.
///
/// `gemma4` covers three models whose host terms differ by more than the rolling
/// correction can travel: a mixture of experts, a dense model, and an E-variant.
/// Both discriminators are ones `host_buffer` already applies — `has_experts` and
/// `compute_buffer::is_gemma_e_variant` — so a key built from them is one the
/// estimator can construct at lookup time.
///
/// `with_environment` is asked for by the baseline offset alone. It differs by
/// runtime (ik sits 24 to 192 MiB above mainline on the same architecture) and by
/// flash attention, which shifts it by +21 to +33 MiB on most architectures and
/// +131 on lfm2 — on top of the per-token arena rate, which is a separate term.
///
/// The flash-attention *rates* must not be keyed this way: ik is excluded from
/// that derivation, so an ik-suffixed key would have no row and would inherit the
/// table's worst rate as its default.
pub fn variant_key(record: &Record, with_environment: bool) -> String {
    let parsed = &record.parsed;
    let mut key = match parsed.arch.as_deref() {
        Some(arch) => arch.to_string(),
        // Python stringifies `None`, and the resulting key is real: it appears
        // in no table only because every cell reaching a keyed deriver has an
        // architecture.
        None => "None".to_string(),
    };
    if parsed.n_expert.unwrap_or(0) != 0 {
        key.push_str("+moe");
    }
    // The same discriminator `compute_buffer::is_gemma_e_variant` uses, read
    // from the load log rather than guessed from the filename. The proxy this
    // replaces would have disagreed with the estimator the moment an E-variant
    // shipped under another name: the analysis would fit one curve while the
    // estimator selected a different one.
    if parsed.per_layer_token_embd.unwrap_or(false) {
        key.push_str("+e");
    }
    if with_environment && record.factors.runtime_is_ik() {
        key.push_str("@ik");
    }
    if with_environment && !record.factors.flash_attn_on() {
        key.push_str("@nofa");
    }
    key
}

/// Query heads, falling back to `n_embd / key_length` where the GGUF omits them.
///
/// Not every architecture writes `attention.head_count`: laguna carries only
/// `head_count_kv`, so a term built on the query head count silently evaluated to
/// zero and left 9218 MiB of unfused score matrix unreserved. Where the count has
/// to be inferred, any error in it is absorbed by the per-architecture rate — the
/// two only ever appear as a product — so the fallback costs accuracy in the
/// *attribution*, not in the reservation.
pub fn query_head_count(parsed: &Parsed) -> u64 {
    if let Some(heads) = parsed.n_head.filter(|h| *h != 0) {
        return u64::from(heads);
    }
    let arch = parsed.arch.as_deref().unwrap_or("");
    let n_embd = match parsed.gguf_int(&format!("{arch}.embedding_length")) {
        0 => i64::from(parsed.n_embd.unwrap_or(0)),
        value => value,
    };
    let head_dim = match parsed.gguf_int(&format!("{arch}.attention.key_length")) {
        0 => i64::from(parsed.n_embd_head_k.unwrap_or(0)),
        value => value,
    };
    if head_dim == 0 {
        0
    } else {
        (n_embd / head_dim).max(0) as u64
    }
}

/// Model weights this cell reported holding, host and device together.
pub fn resident_weight_mib(record: &Record) -> f64 {
    let parsed = &record.parsed;
    let mut total = parsed.cpu_model_mib.unwrap_or(0.0);
    for index in 0..8 {
        total += parsed.gpu_model_mib(index);
    }
    for context in &parsed.contexts {
        for buffers in context.buffers.values() {
            total += buffers.get("model").copied().unwrap_or(0.0);
        }
    }
    total
}

/// Whether two cells loaded the same weights.
///
/// The pairing keys pin every factor that should decide placement, but placement
/// is an *outcome* rather than a factor — an `auto` expert offload can land
/// differently between two runs — and a mismatch there lands entirely in whatever
/// delta the pair is measuring.
pub fn same_resident_weights(left: &Record, right: &Record, tolerance: f64) -> bool {
    let (a, b) = (resident_weight_mib(left), resident_weight_mib(right));
    if a == 0.0 || b == 0.0 {
        return true;
    }
    (a - b).abs() / a.max(b) <= tolerance
}

/// The default tolerance on a weight match: a percent, which is far below the
/// 8 GiB mismatch that first exposed the need for the check.
pub const WEIGHT_TOLERANCE: f64 = 0.01;

/// Per-device `compute + unaccounted` for a cell with no breakdown table.
///
/// ik_llama does not print llama.cpp's memory breakdown, so the column the curve
/// fit wants does not exist. It is still recoverable: the driver's total for the
/// process, less the weights it loaded onto each card and less the context it
/// allocated there, over the number of cards. Everything left is the graph plus
/// whatever the driver holds that the runtime never named — which is exactly what
/// the table's two columns sum to.
///
/// Reproduces the production GLM-5.2 cell: 38708 MiB from the driver against
/// 14064 of weights and 11904 of cache leaves 6370 per card, and the runtime's own
/// compute-buffer lines account for 6260 of that.
pub fn table_less_compute(record: &Record) -> Option<f64> {
    let total = record.gpu_used_mib().filter(|v| *v != 0)? as f64;
    let devices = record.factors.cards_nonempty() as f64;
    let mut weights = 0.0;
    let mut kv = 0.0;
    for context in &record.parsed.contexts {
        for (name, buffers) in &context.buffers {
            if !name.starts_with("CUDA") || name.ends_with("Host") {
                continue;
            }
            weights += buffers.get("model").copied().unwrap_or(0.0);
            kv += buffers.get("kv").copied().unwrap_or(0.0)
                + buffers.get("rs").copied().unwrap_or(0.0);
        }
    }
    if weights == 0.0 {
        return None;
    }
    let remainder = total - weights - kv;
    // A negative remainder means the buffer lines and the driver disagree about
    // what is on the cards — a partial offload the parse cannot attribute — and
    // is not a compute-buffer measurement.
    if remainder > 0.0 {
        Some(remainder / devices)
    } else {
        None
    }
}

/// Per-context device-side buffer sums for one cell, in MiB.
///
/// Host buffers are excluded — `CUDA_Host` is page-locked *host* memory, not
/// VRAM, and folding it in overstates the draft context's device cost by the
/// 40 MiB it holds. `Meta()` is one card's share under a tensor split, so the
/// caller multiplies by the card count.
///
/// Identical consecutive entries are collapsed: the load log prints the draft
/// context twice, once at creation and once at reserve.
pub fn device_context_sums(record: &Record) -> Vec<ContextSums> {
    let mut out: Vec<ContextSums> = Vec::new();
    for context in &record.parsed.contexts {
        let device = context
            .buffers
            .iter()
            .filter(|(name, _)| !name.ends_with("_Host") && !name.starts_with("CPU"));
        let entry = ContextSums {
            kv: context.kv_pools.iter().map(|p| p.total_mib).sum(),
            rs: device
                .clone()
                .map(|(_, b)| b.get("rs").copied().unwrap_or(0.0))
                .sum(),
            compute: device
                .map(|(_, b)| b.get("compute").copied().unwrap_or(0.0))
                .sum(),
        };
        // Python compares whole dicts, so a repeat only collapses when all three
        // sums match; the check is against every earlier entry, not just the last.
        if !out.contains(&entry) {
            out.push(entry);
        }
    }
    out
}

/// One context's device-side cache, recurrent state, and graph, in MiB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextSums {
    pub kv: f64,
    pub rs: f64,
    pub compute: f64,
}
