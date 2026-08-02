//! Reading a cell's shape: how wide its attention is, and what the runtime
//! reported holding where. How a record is *keyed* lives in [`crate::derive::keys`].

use ananke_dataset::BufferRole;
use ananke_gguf::keys::suffix;

use crate::record::{Parsed, Record};

/// llama.cpp's `--checkpoint-min-step` default: the prompt length past which a
/// second context checkpoint is taken, and so the line between a cell that
/// measures the probe's state and one that measures a real service's.
pub const CHECKPOINT_MIN_STEP: u32 = 8192;

/// Query heads, falling back to `n_embd / key_length` where the GGUF omits them.
///
/// Not every architecture writes `attention.head_count`: laguna carries only
/// `head_count_kv`, so a term built on the query head count silently evaluates to
/// zero and leaves 9218 MiB of unfused score matrix unreserved. Where the count has
/// to be inferred, any error in it is absorbed by the per-architecture rate — the
/// two only ever appear as a product — so the fallback costs accuracy in the
/// *attribution*, not in the reservation.
pub fn query_head_count(parsed: &Parsed) -> u64 {
    if parsed.n_head != 0 {
        return parsed.n_head;
    }
    let n_embd = match parsed.gguf(suffix::EMBEDDING_LENGTH).unwrap_or(0) {
        0 => parsed.n_embd as i64,
        value => value,
    };
    let head_dim = match parsed.gguf(suffix::ATTENTION_KEY_LENGTH).unwrap_or(0) {
        0 => parsed.n_embd_head_k as i64,
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
    let mut total = parsed.cpu_model_mib;
    // The typed breakdown rows, not the flat `gpu{n}_model_mib` mirrors of them:
    // one list, read once, rather than probing eight string keys that may or may
    // not be present.
    total += parsed
        .devices
        .iter()
        .map(|device| device.model_mib as f64)
        .sum::<f64>();
    for context in &parsed.contexts {
        for buffers in context.buffers.values() {
            total += buffers.get(&BufferRole::Model).copied().unwrap_or(0.0);
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
/// 8 GiB mismatch the check exists to catch.
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
    let total = record.rss.gpu_used_mib.filter(|v| *v != 0)? as f64;
    let devices = record.factors.cards_nonempty() as f64;
    let mut weights = 0.0;
    let mut kv = 0.0;
    for context in &record.parsed.contexts {
        for (name, buffers) in &context.buffers {
            if !name.starts_with("CUDA") || name.ends_with("Host") {
                continue;
            }
            weights += buffers.get(&BufferRole::Model).copied().unwrap_or(0.0);
            kv += buffers.get(&BufferRole::Kv).copied().unwrap_or(0.0)
                + buffers.get(&BufferRole::Rs).copied().unwrap_or(0.0);
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
pub fn device_context_sums(record: &Record) -> DeviceContexts {
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
                .map(|(_, b)| b.get(&BufferRole::Rs).copied().unwrap_or(0.0))
                .sum(),
            compute: device
                .map(|(_, b)| b.get(&BufferRole::Compute).copied().unwrap_or(0.0))
                .sum(),
        };
        // A repeat collapses only when all three sums match, and the check is
        // against every earlier entry rather than just the last.
        if !out.contains(&entry) {
            out.push(entry);
        }
    }
    DeviceContexts(out)
}

/// One cell's contexts, in the order llama.cpp created them.
///
/// The order carries the meaning: the draft context is created first and the main
/// one last, which is a fact about the load log rather than about the sums, so
/// reading it off an index leaves the argument at the call site.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceContexts(Vec<ContextSums>);

impl DeviceContexts {
    /// The draft context, present only where a second one was created.
    pub fn draft(&self) -> Option<ContextSums> {
        (self.0.len() > 1).then(|| self.0[0])
    }

    /// The main context, which llama.cpp creates last.
    pub fn main(&self) -> Option<ContextSums> {
        self.0.last().copied()
    }
}

/// One context's device-side cache, recurrent state, and graph, in MiB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextSums {
    pub kv: f64,
    pub rs: f64,
    pub compute: f64,
}
