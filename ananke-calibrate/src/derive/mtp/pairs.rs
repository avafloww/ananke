//! Matching a with-MTP cell to its without-MTP twin, and reporting the slot series.
//!
//! No single reading separates the draft context's cost from the process it lives in,
//! so every MTP term is measured as a difference between a pair. Which makes the
//! pairing key the whole correctness argument: it has to pin every factor that could
//! differ, and it has to pin the *sitting* as well, since cell identity ignores the
//! label and a cell can otherwise pair with one recorded hours earlier under different
//! machine state.

use std::collections::BTreeMap;

use crate::{
    derive::{dataset::measured_at, ordered::OrderedMap},
    record::Record,
};

/// An f16 KQ mask costs two bytes per (batch token, cache token).
pub const MASK_BYTES_PER_TOKEN_PAIR: u64 = 2;

/// How far apart two halves of a pair may have been measured.
///
/// Cell identity ignores the label, so a freshly-measured cell can pair with one
/// recorded hours earlier under different machine state — which produced a
/// *negative* delta once, a with-MTP process apparently using less VRAM than
/// without. Repeats taken back to back reproduce to the megabyte, so the machine is
/// not noisy; the pairing was.
pub const SAME_SITTING_SECONDS: f64 = 3600.0;

/// A with-MTP cell and the without-MTP twin it is measured against.
#[derive(Debug, Clone)]
pub struct MtpPair<'a> {
    pub model: String,
    pub ctx: u32,
    /// Driver VRAM the MTP flag costs, in MiB.
    pub delta: i64,
    /// Host owned memory the MTP flag costs, in MiB.
    pub host_delta: i64,
    pub on: &'a Record,
    pub without: &'a Record,
}

/// With/without MTP cells matched on every factor but the MTP flag.
///
/// `served` belongs in the identity. An idle process has not made the first-use
/// allocations a served one has — the context checkpoint above all — so pairing
/// across it reads that whole step as MTP overhead: one such pair showed 568 MiB of
/// host delta where every same-sitting served pair of the same configuration shows
/// 239 to 243.
pub fn mtp_pairs(rows: &[Record], draft: bool) -> Vec<MtpPair<'_>> {
    type Identity = (String, u32, u32, bool, String, String, u32, String, bool);
    fn identity(record: &Record) -> Identity {
        let f = &record.factors;
        (
            record.provenance.model_key.clone(),
            f.ctx,
            f.parallel.unwrap_or(0),
            f.kv_unified,
            f.split.clone().unwrap_or_else(|| "-".to_string()),
            f.gpus.clone(),
            f.ubatch.unwrap_or(0),
            f.kv_type.clone().unwrap_or_default(),
            f.served,
        )
    }

    let mut grouped: OrderedMap<Identity, BTreeMap<bool, &Record>> = OrderedMap::new();
    for record in rows {
        if record.parsed.arch.as_deref().is_some_and(|a| !a.is_empty()) {
            grouped
                .or_insert_with(identity(record), BTreeMap::new)
                .insert(record.factors.has_spec(), record);
        }
    }
    let mut out = Vec::new();
    for (key, pair) in grouped.iter() {
        if pair.len() != 2 {
            continue;
        }
        let (on, off) = (pair[&true], pair[&false]);
        if on.factors.draft.as_deref().is_some_and(|d| !d.is_empty()) != draft {
            continue;
        }
        let (Some(on_used), Some(off_used)) = (
            on.gpu_used_mib().filter(|v| *v != 0),
            off.gpu_used_mib().filter(|v| *v != 0),
        ) else {
            continue;
        };
        if (measured_at(on) - measured_at(off)).abs() > SAME_SITTING_SECONDS {
            continue;
        }
        let delta = on_used as i64 - off_used as i64;
        if delta <= 0 {
            continue;
        }
        out.push(MtpPair {
            model: key.0.clone(),
            ctx: key.1,
            delta,
            host_delta: on.owned_mib() - off.owned_mib(),
            on,
            without: off,
        });
    }
    out
}

/// Does the MTP overhead depend on the slot count, at a fixed context?
///
/// This is the question `MTP_COMPUTE_MIB` was held under review for, and the first
/// campaign could not answer it: every one-slot pair sat at ctx 32768 or 65536 and
/// the only four-slot pair at 131072, so slots and context were confounded and the
/// four-slot pair's much larger delta had two candidate causes.
///
/// Reported rather than fitted. A flat series across slots says the earlier "slot
/// dependence" was the longer context and the constant can be fitted on context
/// alone; a rising one says the model needs a slot term before any value is
/// trustworthy.
pub fn mtp_slot_scaling(rows: &[Record]) -> String {
    let mut by_key: BTreeMap<(String, u32), BTreeMap<u32, i64>> = BTreeMap::new();
    let embedded = mtp_pairs(rows, false);
    let separate = mtp_pairs(rows, true);
    for pair in embedded.iter().chain(separate.iter()) {
        let name: String = pair
            .model
            .rsplit('/')
            .next()
            .unwrap_or(&pair.model)
            .chars()
            .take(24)
            .collect();
        by_key
            .entry((name, pair.ctx))
            .or_default()
            .insert(pair.on.factors.parallel.unwrap_or(0), pair.delta);
    }
    let series: Vec<String> = by_key
        .iter()
        .filter(|(_, points)| points.len() > 1)
        .map(|((model, ctx), points)| {
            let listed = points
                .iter()
                .map(|(slots, delta)| format!("np{slots} {delta} MiB"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{model} ctx {ctx}: {listed}")
        })
        .collect();
    if series.is_empty() {
        return "no fixed-context slot series measured".to_string();
    }
    series.join("; ")
}
