//! Writing the derived tables into the document, and the one invariant
//! ([`check_table_signs`]) that must hold before it is written.

use std::{borrow::Cow, collections::BTreeMap};

use serde::Serialize;
use serde_json::Value;

use crate::derive::{
    NestedTable, Table,
    error::Result,
    keys::{ArchCardsKey, ArchKey, VariantEnvironmentKey, VariantKey},
    mtp,
};

/// Every table `emit` writes, gathered so the writer takes one argument.
pub(super) struct Tables<'a> {
    pub(super) baseline: Option<&'a Table<VariantEnvironmentKey>>,
    pub(super) tensor_base: Option<&'a Table<ArchKey>>,
    pub(super) draft_compute: Option<&'a mtp::DraftComputeFit>,
    pub(super) slot_scaling: &'a str,
    pub(super) checkpoint: Option<&'a Table<VariantKey>>,
    pub(super) per_slot: Option<&'a Table<ArchKey>>,
    pub(super) table_less: Option<&'a NestedTable>,
    pub(super) score: Option<&'a Table<ArchKey>>,
    pub(super) no_fa: Option<&'a Table<VariantKey>>,
    pub(super) quantised: Option<&'a Table<ArchKey>>,
    pub(super) ik_moe: Option<&'a Table<ArchCardsKey>>,
}

pub(super) fn write_tables(document: &mut Value, tables: Tables<'_>) {
    if let Some(table) = tables.baseline {
        document["baseline_offset"] = value(RateTable {
            comment: Cow::Borrowed(
                "Per-architecture correction to the process baseline, in bytes. \
                 The layer-count model leaves a residual that is \
                 architecture-shaped and reproducible; this charges it where it \
                 is positive. `default` is zero, since an unmeasured \
                 architecture has no evidence either way.",
            ),
            default: 0,
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.tensor_base {
        document["tensor_split_baseline"] = value(RateTable {
            comment: Cow::Borrowed(
                "Extra host baseline bytes a tensor split costs beyond a layer \
                 split, by architecture. `default` applies to an architecture \
                 not listed.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(fit) = tables.draft_compute {
        document["mtp_draft_compute_base_mib"] = value(RateTable {
            comment: Cow::Owned(format!(
                "The MTP draft context's own per-device compute buffer at zero context, \
                 by architecture. {}",
                fit.evidence,
            )),
            default: fit.bases.values().copied().max().unwrap_or(0),
            by_arch: &fit.bases,
        });
        document["mtp_draft_compute_mib_per_1k"] = value(RateTable {
            comment: Cow::Borrowed(
                "Slope of the same buffer, in thousandths of a MiB per 1024 \
                 context tokens. Far below the 1000 a full `ubatch x n_kv x 2` \
                 f16 mask would cost at ubatch 512, which is why the mask shape \
                 the old model assumed over-reserved at long contexts.",
            ),
            default: fit.slopes.values().copied().max().unwrap_or(0),
            by_arch: &fit.slopes,
        });
    }
    if !tables.slot_scaling.is_empty() {
        document["mtp_slot_scaling"] = value(SlotScalingTable {
            comment: "Reported, not fitted. The MTP overhead measured at a fixed \
                      context across slot counts, which is what separates a genuine \
                      slot term from the longer context the first campaign confounded \
                      it with.",
            observed: tables.slot_scaling,
        });
    }
    if let Some(table) = tables.checkpoint {
        document["checkpoint_headroom_bytes"] = value(RateTable {
            comment: Cow::Borrowed(
                "What a real prompt adds over the campaign's probe, by \
                 architecture, from llama.cpp's context checkpoints. Reserved as \
                 slop rather than charged to the correction: a service that never \
                 sees a long prompt would otherwise read as a large \
                 over-reservation and clamp unreachably.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.per_slot {
        document["per_slot_host_bytes"] = value(RateTable {
            comment: Cow::Borrowed(
                "Host memory each concurrently active slot costs, by \
                 architecture. Recorded, and deliberately not charged in \
                 `host_overhead_bytes`: that figure is what the rolling \
                 correction divides an observation by, so it must model what a \
                 process holds rather than the worst case, and charging all four \
                 slots took the cells outside the band from 2 to 44. A worst-case \
                 allowance belongs in the packer's slop, beside the prompt cache.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.table_less {
        document["table_less_compute_observations"] = value(ObservationTable {
            comment: "Per-device `compute + unaccounted` for architectures whose \
                      runtime prints no memory-breakdown table, recovered from the \
                      driver total less the weights and the context. Recorded, not \
                      fitted: these are the measurements behind a \
                      `compute_buffer_curves` entry that has to be held by hand, so \
                      the held value's justification cannot go stale. Deliberately not \
                      pooled into the curves — the two runtimes build different graphs \
                      for one architecture, and pooling them moved four curves at \
                      once. MiB per device, keyed by context and batch.",
            by_arch: &table.by_arch,
        });
    }
    if let Some(table) = tables.score {
        document["no_flash_attn_score_centibytes"] = value(RateTable {
            comment: Cow::Borrowed(
                "Bytes of unfused attention score matrix per (head x cache token \
                 x batch token), by architecture. One f32 for every dense and MoE \
                 architecture measured; deepseek4's MLA shares one latent across \
                 heads and needs essentially none. `default` is the largest, so \
                 an unmeasured architecture over-reserves rather than OOMs. \
                 Distinct from `no_flash_attn_rates`, which is the *host* pinned \
                 buffer's rate.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.no_fa {
        document["no_flash_attn_rates"] = value(RateTable {
            comment: Cow::Borrowed(
                "Extra pinned bytes per batch token when flash attention is off, \
                 by architecture. The residual is flat in context and \
                 proportional to batch, and the rates differ fourfold between \
                 sliding-window models and the rest. `default` applies to an \
                 architecture not listed.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.quantised {
        document["quantised_cache_rates"] = value(RateTable {
            comment: Cow::Borrowed(
                "Extra pinned bytes per batch token when the KV cache is \
                 quantised, by architecture. They span a factor of forty, so one \
                 value would either under-reserve deepseek4 or over-reserve \
                 everything else by ~3 MiB. `default` applies to an architecture \
                 not listed.",
            ),
            default: table.worst(),
            by_arch: &table.by_key,
        });
    }
    if let Some(table) = tables.ik_moe {
        document["ik_moe_rates"] = ik_moe_value(table);
    }
}

/// The `ik_moe_rates` section, as both the document and the live view need it.
/// Per architecture because the rates differ enough that one number would either
/// under-reserve the worst or over-reserve the rest.
pub(super) fn ik_moe_value(table: &Table<ArchCardsKey>) -> Value {
    value(RateTable {
        comment: Cow::Borrowed(
            "Bytes per batch token per unit of hidden size for ik's \
             CPU-resident MoE intermediates, by architecture. `default` \
             applies to an architecture not listed.",
        ),
        default: table.worst(),
        by_arch: &table.by_key,
    })
}

/// Tables whose values may be negative, and which `build.rs` must therefore emit
/// through `generate_signed_rate_table`.
pub const SIGNED_TABLES: &[&str] = &["baseline_offset"];

/// Refuse to write a negative into a table read as unsigned.
///
/// `build.rs` reads every table but the signed ones through `as_u64`, which turns
/// a negative into the table's default rather than raising — the value vanishes
/// with no error anywhere, which is how the negative baseline offsets would have
/// failed had they gone out through the unsigned path.
pub fn check_table_signs(document: &Value) -> Result<()> {
    let Some(tables) = document.as_object() else {
        return Ok(());
    };
    for (name, table) in tables {
        if !table.is_object() || SIGNED_TABLES.contains(&name.as_str()) {
            continue;
        }
        // A nested `by_arch` holds observations rather than one rate per architecture,
        // and `build.rs` does not read those at all.
        let negative: Vec<String> = table
            .get("by_arch")
            .and_then(Value::as_object)
            .map(|by_arch| {
                by_arch
                    .iter()
                    .filter(|(_, value)| value.as_f64().is_some_and(|v| v < 0.0))
                    .map(|(arch, value)| format!("{arch}: {value}"))
                    .collect()
            })
            .unwrap_or_default();
        if !negative.is_empty() {
            return Err(crate::derive::error::DeriveError::disagreement(format!(
                "{name} holds negative values {{{}}} but is read through the unsigned \
                 path in build.rs, which would silently replace each with the default. \
                 Emit it via `generate_signed_rate_table` and add it to `SIGNED_TABLES`.",
                negative.join(", ")
            )));
        }
    }
    Ok(())
}

/// The shape nearly every table takes. Field order is the document's key order,
/// since `serde_json` serializes a struct in declaration order.
#[derive(Serialize)]
struct RateTable<'a, K> {
    #[serde(rename = "$comment")]
    comment: Cow<'a, str>,
    default: i64,
    by_arch: &'a BTreeMap<K, i64>,
}

/// `mtp_slot_scaling`, which reports what was observed rather than a fitted rate and
/// so has no per-architecture breakdown to fall back from.
#[derive(Serialize)]
struct SlotScalingTable<'a> {
    #[serde(rename = "$comment")]
    comment: &'a str,
    observed: &'a str,
}

/// Observations keyed by architecture and then by the configuration each was
/// taken at. No `default`: nothing reads these, they are written down so a
/// hand-held value's justification cannot go stale.
#[derive(Serialize)]
struct ObservationTable<'a> {
    #[serde(rename = "$comment")]
    comment: &'a str,
    by_arch: &'a BTreeMap<String, BTreeMap<String, i64>>,
}

/// # Panics
///
/// Never: every field is a string, an integer, or a map keyed by one.
fn value<T: Serialize>(table: T) -> Value {
    serde_json::to_value(table).expect("a table serializes to JSON")
}
