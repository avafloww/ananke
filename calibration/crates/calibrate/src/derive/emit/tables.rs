//! Writing the derived tables into the document, and the one invariant
//! ([`check_table_signs`]) that must hold before it is written.

use std::{collections::BTreeMap, fmt};

use ananke_tuning_schema::{Document, ObservationTable, RateTable, SlotScaling};

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

pub(super) fn write_tables(document: &mut Document, tables: Tables<'_>) {
    if let Some(table) = tables.baseline {
        document.baseline_offset = rates(
            "Per-architecture correction to the process baseline, in bytes. \
             The layer-count model leaves a residual that is \
             architecture-shaped and reproducible; this charges it where it \
             is positive. `default` is zero, since an unmeasured \
             architecture has no evidence either way.",
            0,
            &table.by_key,
        );
    }
    if let Some(table) = tables.tensor_base {
        document.tensor_split_baseline = rates(
            "Extra host baseline bytes a tensor split costs beyond a layer \
             split, by architecture. `default` applies to an architecture \
             not listed.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(fit) = tables.draft_compute {
        document.mtp_draft_compute_base_mib = rates(
            format!(
                "The MTP draft context's own per-device compute buffer at zero context, \
                 by architecture. {}",
                fit.evidence,
            ),
            fit.bases.values().copied().max().unwrap_or(0),
            &fit.bases,
        );
        document.mtp_draft_compute_mib_per_1k = rates(
            "Slope of the same buffer, in thousandths of a MiB per 1024 \
             context tokens. Far below the 1000 a full `ubatch x n_kv x 2` \
             f16 mask would cost at ubatch 512, which is why the mask shape \
             the old model assumed over-reserved at long contexts.",
            fit.slopes.values().copied().max().unwrap_or(0),
            &fit.slopes,
        );
    }
    if !tables.slot_scaling.is_empty() {
        document.mtp_slot_scaling = SlotScaling {
            comment: "Reported, not fitted. The MTP overhead measured at a fixed \
                      context across slot counts, which is what separates a genuine \
                      slot term from the longer context the first campaign confounded \
                      it with."
                .to_string(),
            observed: tables.slot_scaling.to_string(),
        };
    }
    if let Some(table) = tables.checkpoint {
        document.checkpoint_headroom_bytes = rates(
            "What a real prompt adds over the campaign's probe, by \
             architecture, from llama.cpp's context checkpoints. Reserved as \
             slop rather than charged to the correction: a service that never \
             sees a long prompt would otherwise read as a large \
             over-reservation and clamp unreachably.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(table) = tables.per_slot {
        document.per_slot_host_bytes = rates(
            "Host memory each concurrently active slot costs, by \
             architecture. Recorded, and deliberately not charged in \
             `host_overhead_bytes`: that figure is what the rolling \
             correction divides an observation by, so it must model what a \
             process holds rather than the worst case, and charging all four \
             slots took the cells outside the band from 2 to 44. A worst-case \
             allowance belongs in the packer's slop, beside the prompt cache.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(table) = tables.table_less {
        document.table_less_compute_observations = ObservationTable {
            comment: "Per-device `compute + unaccounted` for architectures whose \
                      runtime prints no memory-breakdown table, recovered from the \
                      driver total less the weights and the context. Recorded, not \
                      fitted: these are the measurements behind a \
                      `compute_buffer_curves` entry that has to be held by hand, so \
                      the held value's justification cannot go stale. Deliberately not \
                      pooled into the curves — the two runtimes build different graphs \
                      for one architecture, and pooling them moved four curves at \
                      once. MiB per device, keyed by context and batch."
                .to_string(),
            by_arch: table.by_arch.clone(),
        };
    }
    if let Some(table) = tables.score {
        document.no_flash_attn_score_centibytes = rates(
            "Bytes of unfused attention score matrix per (head x cache token \
             x batch token), by architecture. One f32 for every dense and MoE \
             architecture measured; deepseek4's MLA shares one latent across \
             heads and needs essentially none. `default` is the largest, so \
             an unmeasured architecture over-reserves rather than OOMs. \
             Distinct from `no_flash_attn_rates`, which is the *host* pinned \
             buffer's rate.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(table) = tables.no_fa {
        document.no_flash_attn_rates = rates(
            "Extra pinned bytes per batch token when flash attention is off, \
             by architecture. The residual is flat in context and \
             proportional to batch, and the rates differ fourfold between \
             sliding-window models and the rest. `default` applies to an \
             architecture not listed.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(table) = tables.quantised {
        document.quantised_cache_rates = rates(
            "Extra pinned bytes per batch token when the KV cache is \
             quantised, by architecture. They span a factor of forty, so one \
             value would either under-reserve deepseek4 or over-reserve \
             everything else by ~3 MiB. `default` applies to an architecture \
             not listed.",
            table.worst(),
            &table.by_key,
        );
    }
    if let Some(table) = tables.ik_moe {
        document.ik_moe_rates = ik_moe_rates(table);
    }
}

/// The `ik_moe_rates` section, as both the document and the live view need it.
/// Per architecture because the rates differ enough that one number would either
/// under-reserve the worst or over-reserve the rest.
pub(super) fn ik_moe_rates(table: &Table<ArchCardsKey>) -> RateTable {
    rates(
        "Bytes per batch token per unit of hidden size for ik's \
         CPU-resident MoE intermediates, by architecture. `default` \
         applies to an architecture not listed.",
        table.worst(),
        &table.by_key,
    )
}

/// Refuse to write a negative into a table read as unsigned.
///
/// `build.rs` reads every table but the signed ones through `as u64`, which turns
/// a negative into the table's default rather than raising — the value vanishes
/// with no error anywhere, which is how the negative baseline offsets would have
/// failed had they gone out through the unsigned path. Which tables are signed is
/// [`ananke_tuning_schema::RateTableName::signed`], so the generator and this
/// check cannot disagree about one.
pub fn check_table_signs(document: &Document) -> Result<()> {
    for (name, table) in document.rate_tables() {
        if name.signed() {
            continue;
        }
        let negative: Vec<String> = table
            .by_arch
            .iter()
            .filter(|(_, value)| **value < 0)
            .map(|(arch, value)| format!("{arch}: {value}"))
            .collect();
        if !negative.is_empty() {
            return Err(crate::derive::error::DeriveError::disagreement(format!(
                "{} holds negative values {{{}}} but is read through the unsigned \
                 path in build.rs, which would silently replace each with the default. \
                 Mark it signed in `RateTableName::signed`.",
                name.as_str(),
                negative.join(", ")
            )));
        }
    }
    Ok(())
}

/// One rate table, at the keys the document spells rather than the vocabulary
/// the deriver reduced over.
fn rates<K: fmt::Display>(
    comment: impl Into<String>,
    default: i64,
    by_key: &BTreeMap<K, i64>,
) -> RateTable {
    RateTable {
        comment: comment.into(),
        by_arch: by_key
            .iter()
            .map(|(key, value)| (key.to_string(), *value))
            .collect(),
        default,
    }
}
