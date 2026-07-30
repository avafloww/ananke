//! Writing the derived tables into the document, and the one invariant that must hold
//! before it is written.

use serde_json::{Value, json};

use crate::derive::{NestedTable, Table, error::Result, mtp};

/// Every table `emit` writes, gathered so the writer takes one argument.
pub(super) struct Tables<'a> {
    pub(super) baseline: Option<&'a Table>,
    pub(super) tensor_base: Option<&'a Table>,
    pub(super) draft_compute: Option<&'a mtp::DraftComputeFit>,
    pub(super) slot_scaling: &'a str,
    pub(super) checkpoint: Option<&'a Table>,
    pub(super) per_slot: Option<&'a Table>,
    pub(super) table_less: Option<&'a NestedTable>,
    pub(super) score: Option<&'a Table>,
    pub(super) no_fa: Option<&'a Table>,
    pub(super) quantised: Option<&'a Table>,
    pub(super) ik_moe: Option<&'a Table>,
}

pub(super) fn write_tables(document: &mut Value, tables: Tables<'_>) {
    if let Some(table) = tables.baseline {
        document["baseline_offset"] = json!({
            "$comment": "Per-architecture correction to the process baseline, in bytes. \
                         The layer-count model leaves a residual that is \
                         architecture-shaped and reproducible; this charges it where it \
                         is positive. `default` is zero, since an unmeasured \
                         architecture has no evidence either way.",
            "default": 0,
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.tensor_base {
        document["tensor_split_baseline"] = json!({
            "$comment": "Extra host baseline bytes a tensor split costs beyond a layer \
                         split, by architecture. `default` applies to an architecture \
                         not listed.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(fit) = tables.draft_compute {
        document["mtp_draft_compute_base_mib"] = json!({
            "$comment": format!(
                "The MTP draft context's own per-device compute buffer at zero context, \
                 by architecture. {}",
                fit.evidence,
            ),
            "default": fit.bases.values().copied().max().unwrap_or(0),
            "by_arch": fit.bases,
        });
        document["mtp_draft_compute_mib_per_1k"] = json!({
            "$comment": "Slope of the same buffer, in thousandths of a MiB per 1024 \
                         context tokens. Far below the 1000 a full `ubatch x n_kv x 2` \
                         f16 mask would cost at ubatch 512, which is why the mask shape \
                         the old model assumed over-reserved at long contexts.",
            "default": fit.slopes.values().copied().max().unwrap_or(0),
            "by_arch": fit.slopes,
        });
    }
    if !tables.slot_scaling.is_empty() {
        document["mtp_slot_scaling"] = json!({
            "$comment": "Reported, not fitted. The MTP overhead measured at a fixed \
                         context across slot counts, which is what separates a genuine \
                         slot term from the longer context the first campaign confounded \
                         it with.",
            "observed": tables.slot_scaling,
        });
    }
    if let Some(table) = tables.checkpoint {
        document["checkpoint_headroom_bytes"] = json!({
            "$comment": "What a real prompt adds over the campaign's probe, by \
                         architecture, from llama.cpp's context checkpoints. Reserved as \
                         slop rather than charged to the correction: a service that never \
                         sees a long prompt would otherwise read as a large \
                         over-reservation and clamp unreachably.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.per_slot {
        document["per_slot_host_bytes"] = json!({
            "$comment": "Host memory each concurrently active slot costs, by \
                         architecture. Recorded, and deliberately not charged in \
                         `host_overhead_bytes`: that figure is what the rolling \
                         correction divides an observation by, so it must model what a \
                         process holds rather than the worst case, and charging all four \
                         slots took the cells outside the band from 2 to 44. A worst-case \
                         allowance belongs in the packer's slop, beside the prompt cache.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.table_less {
        document["table_less_compute_observations"] = json!({
            "$comment": "Per-device `compute + unaccounted` for architectures whose \
                         runtime prints no memory-breakdown table, recovered from the \
                         driver total less the weights and the context. Recorded, not \
                         fitted: these are the measurements behind a \
                         `compute_buffer_curves` entry that has to be held by hand, so \
                         the held value's justification cannot go stale. Deliberately not \
                         pooled into the curves — the two runtimes build different graphs \
                         for one architecture, and pooling them moved four curves at \
                         once. MiB per device, keyed by context and batch.",
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.score {
        document["no_flash_attn_score_centibytes"] = json!({
            "$comment": "Bytes of unfused attention score matrix per (head x cache token \
                         x batch token), by architecture. One f32 for every dense and MoE \
                         architecture measured; deepseek4's MLA shares one latent across \
                         heads and needs essentially none. `default` is the largest, so \
                         an unmeasured architecture over-reserves rather than OOMs. \
                         Distinct from `no_flash_attn_rates`, which is the *host* pinned \
                         buffer's rate.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.no_fa {
        document["no_flash_attn_rates"] = json!({
            "$comment": "Extra pinned bytes per batch token when flash attention is off, \
                         by architecture. The residual is flat in context and \
                         proportional to batch, and the rates differ fourfold between \
                         sliding-window models and the rest. `default` applies to an \
                         architecture not listed.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.quantised {
        document["quantised_cache_rates"] = json!({
            "$comment": "Extra pinned bytes per batch token when the KV cache is \
                         quantised, by architecture. They span a factor of forty, so one \
                         value would either under-reserve deepseek4 or over-reserve \
                         everything else by ~3 MiB. `default` applies to an architecture \
                         not listed.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
    if let Some(table) = tables.ik_moe {
        // Per architecture, because they differ and one number cannot serve all three
        // without either under-reserving the worst or over-reserving the rest. The
        // fallback is the worst seen, for an ik mixture of experts this dataset has
        // never measured.
        document["ik_moe_rates"] = json!({
            "$comment": "Bytes per batch token per unit of hidden size for ik's \
                         CPU-resident MoE intermediates, by architecture. `default` \
                         applies to an architecture not listed.",
            "default": table.worst(),
            "by_arch": table.by_arch,
        });
    }
}

/// Tables whose values may be negative, and which `build.rs` must therefore emit
/// through `generate_signed_rate_table`.
pub const SIGNED_TABLES: &[&str] = &["baseline_offset"];

/// Refuse to write a negative into a table read as unsigned.
///
/// `build.rs` reads every table but the signed ones through `as_u64`, which turns a
/// negative into the table's default rather than raising — the value vanishes with no
/// error anywhere. That is how the negative baseline offsets would have failed had
/// they gone out through the unsigned path, so the invariant is asserted here, where
/// it can still be seen.
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
