//! Matching a with-MTP cell to its without-MTP twin, and reporting the slot series.
//!
//! No single reading separates the draft context's cost from the process it lives in,
//! so every MTP term is measured as a difference between a pair. Which makes the
//! pairing key the whole correctness argument: it has to pin every factor that could
//! differ, and it has to pin the *sitting* as well, since cell identity ignores the
//! label and a cell can otherwise pair with one recorded hours earlier under different
//! machine state.

use std::collections::BTreeMap;

use ananke_config::placement::SplitMode;
use ananke_dataset::KvType;
use jiff::SignedDuration;

use crate::{
    derive::{dataset::measured_at, ordered::OrderedMap, pair::Pair},
    record::Record,
};

/// How far apart two halves of a pair may have been measured.
///
/// Cell identity ignores the label, so a freshly-measured cell can pair with one
/// recorded hours earlier under different machine state — which produces a
/// *negative* delta, a with-MTP process apparently using less VRAM than without.
/// Repeats taken back to back reproduce to the megabyte, so the machine is not the
/// noisy part; the pairing is.
pub const SAME_SITTING: SignedDuration = SignedDuration::from_hours(1);

/// Where the MTP head lives, which is the thing a pair is selected on.
///
/// The two shapes cost different things — an embedded head adds a draft context, a
/// separate GGUF adds weights — so every MTP constant is derived over one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpShape {
    /// A head shipped inside the target model, run without a draft GGUF.
    Embedded,
    /// A head loaded from its own file with `-md`.
    SeparateDraft,
}

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
/// across it reads that whole step as MTP overhead: one such pair shows 568 MiB of
/// host delta where every same-sitting served pair of the same configuration shows
/// 239 to 243.
pub fn mtp_pairs(rows: &[Record], shape: MtpShape) -> Vec<MtpPair<'_>> {
    let mut grouped: OrderedMap<Identity, Pair<&Record>> = OrderedMap::new();
    for record in rows {
        if record.parsed.architecture().is_some() {
            *grouped
                .or_insert_with(Identity::of(record), Pair::default)
                .half_mut(record.factors.has_spec()) = Some(record);
        }
    }
    let mut out = Vec::new();
    for (key, pair) in grouped.iter() {
        let Some((on, off)) = pair.both() else {
            continue;
        };
        let (on, off) = (*on, *off);
        if shape_of(on) != shape {
            continue;
        }
        let (Some(on_used), Some(off_used)) = (
            on.rss.gpu_used_mib.filter(|v| *v != 0),
            off.rss.gpu_used_mib.filter(|v| *v != 0),
        ) else {
            continue;
        };
        if measured_at(on).duration_since(measured_at(off)).abs() > SAME_SITTING {
            continue;
        }
        let delta = on_used as i64 - off_used as i64;
        if delta <= 0 {
            continue;
        }
        out.push(MtpPair {
            model: key.model.clone(),
            ctx: key.ctx,
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
/// This is the question `MTP_COMPUTE_MIB` is held under review for. Slots and
/// context are easily confounded: one-slot pairs at ctx 32768 or 65536 against a
/// four-slot pair at 131072 leave that pair's much larger delta with two candidate
/// causes.
///
/// Reported rather than fitted. A flat series across slots says an apparent slot
/// dependence is the longer context and the constant can be fitted on context
/// alone; a rising one says the model needs a slot term before any value is
/// trustworthy.
pub fn mtp_slot_scaling(rows: &[Record]) -> String {
    let mut by_key: BTreeMap<(String, u32), BTreeMap<u32, i64>> = BTreeMap::new();
    let embedded = mtp_pairs(rows, MtpShape::Embedded);
    let separate = mtp_pairs(rows, MtpShape::SeparateDraft);
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
            .insert(pair.on.factors.parallel, pair.delta);
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

/// Which shape a cell actually ran, read from whether it names a draft file.
fn shape_of(record: &Record) -> MtpShape {
    match record.factors.draft.as_deref() {
        Some(path) if !path.is_empty() => MtpShape::SeparateDraft,
        _ => MtpShape::Embedded,
    }
}

/// Every factor a pair must agree on, so the MTP flag is the only difference left.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Identity {
    model: String,
    ctx: u32,
    parallel: u32,
    kv_unified: bool,
    split: Option<SplitMode>,
    gpus: String,
    ubatch: u32,
    kv_type: KvType,
    served: bool,
}

impl Identity {
    fn of(record: &Record) -> Self {
        let f = &record.factors;
        Self {
            model: record.provenance.model_key.clone(),
            ctx: f.ctx,
            parallel: f.parallel,
            kv_unified: f.kv_unified,
            split: f.split,
            gpus: f.gpus.clone(),
            ubatch: f.ubatch,
            kv_type: f.kv_type,
            served: f.served,
        }
    }
}
