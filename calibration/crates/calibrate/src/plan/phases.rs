//! The campaign's original phases: a noise floor, a factor screen, the per-model
//! baseline, the context and batch curves, the fork comparison, the switched
//! terms, and the held-out production configurations.
//!
//! The ordering is the argument. Screening on the cheapest model first is what
//! makes sweeping the expensive ones affordable, and fixing the factor set before
//! varying the model is what stops a constant fitted to one shape from being
//! generalised — the mistake this campaign exists to stop repeating.

use ananke_config::placement::SplitMode;
use ananke_measure::record::{Factors, FlashAttn, KvType, Runtime};

use crate::plan::library::{Library, MODELS, Model, model, runtime_name};

/// Repeat one cell, to establish what a difference has to exceed to matter.
pub fn noise(lib: &Library) -> Vec<Factors> {
    let m = model("qwen3-4b");
    (0..5)
        .map(|repeat| Factors {
            label: "noise".to_owned(),
            model: lib.path_of(m.path),
            parallel: 4,
            repeat,
            ..Factors::default()
        })
        .collect()
}

/// Full factorial on the fastest model.
///
/// Establishes which factors move the host baseline at all, and which interact,
/// before any expensive model is loaded. `split` is included at one GPU as well,
/// where it should be inert — a factor measured to be irrelevant is worth as much
/// as one that matters.
pub fn factor_screen(lib: &Library) -> Vec<Factors> {
    let m = model("qwen3-4b");
    let mut cells = Vec::new();
    for gpus in ["0", "0,1"] {
        for split in [SplitMode::Layer, SplitMode::Tensor] {
            for kv in [KvType::F16, KvType::Q80] {
                for served in [true, false] {
                    for parallel in [1, 4] {
                        for ctx in [8192, 32768] {
                            let serving = if served { "srv" } else { "idle" };
                            cells.push(Factors {
                                label: format!(
                                    "p1-{}-{split}-{kv}-{serving}-np{parallel}-c{ctx}",
                                    gpus.replace(',', "")
                                ),
                                model: lib.path_of(m.path),
                                gpus: gpus.to_owned(),
                                split: Some(split),
                                kv_type: kv,
                                served,
                                parallel,
                                ctx,
                                ..Factors::default()
                            });
                        }
                    }
                }
            }
        }
    }
    cells
}

/// Does the per-model term follow layers, hidden size, vocabulary, or none?
///
/// The factor screen fixed the factor set on one model; this varies the model
/// with that set held constant. Without it the baseline is a constant fitted to
/// one shape and generalised.
pub fn model_baseline(lib: &Library) -> Vec<Factors> {
    MODELS
        .iter()
        .filter(|m| m.runtimes.contains(&Runtime::Mainline))
        .flat_map(|m| significant(lib, m, Runtime::Mainline))
        .collect()
}

/// The same cells on ik_llama, which sizes its graph arena by different rules.
pub fn fork(lib: &Library) -> Vec<Factors> {
    MODELS
        .iter()
        .filter(|m| m.runtimes.contains(&Runtime::Ik))
        .flat_map(|m| significant(lib, m, Runtime::Ik))
        .collect()
}

/// Two contexts and two batches per model, so the slopes are fittable.
///
/// Everything else in this campaign holds `ctx` at 32768 and `ubatch` at 512,
/// which is enough for the host baseline — the factor screen measured it flat in
/// both — but leaves several constants underdetermined. A slope needs two points:
/// every per-architecture curve in `estimator/compute_buffer.rs` is
/// `base + slope * (ctx / 1024)`, `deepseek4`'s slope is additionally linear in
/// `ubatch`, and ik's CPU-MoE term is per batch token with a batch-size
/// threshold. One point per model can fit none of them.
///
/// Flash attention is varied here for the same reason: it changes the KQ mask
/// element width and is the sole justification for
/// `NO_FLASH_ATTN_BYTES_PER_TOKEN`, which no cell had ever exercised.
pub fn curves(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for m in MODELS {
        let gpus = m.preferred_gpus();
        for &runtime in m.runtimes {
            for (ctx, ubatch, fa) in curve_points(m) {
                // Flash attention is a KV-cache property; `-dsa` rejects a
                // quantised cache and the fa-off point is not meaningful for a
                // model that must run with it on.
                if fa == FlashAttn::Off && m.kv_types == [KvType::F16] && !m.extra.is_empty() {
                    continue;
                }
                cells.push(Factors {
                    label: format!(
                        "curve-{}-{}-c{ctx}-ub{ubatch}-fa{fa}",
                        m.key,
                        runtime_name(runtime)
                    ),
                    model: lib.path_of(m.path),
                    runtime,
                    gpus: gpus.to_owned(),
                    split: Some(SplitMode::Layer),
                    ctx,
                    ubatch,
                    flash_attn: fa,
                    ..m.flags(gpus)
                });
            }
        }
    }
    cells
}

/// Terms with their own switch rather than a continuous factor.
///
/// Each of these is a constant in the estimator justified by one measurement or
/// none: the two MTP shapes, the vision projector, the offload regimes, ik's
/// repacking and mapping paths, the embedding modality, and what actually
/// accumulates once a server is doing an agent's work.
pub fn switches(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    let g4 = model("gemma4-31b-qat");
    let q27 = model("qwen36-27b");
    // MTP: none, embedded head, separate draft GGUF. The estimator charges a
    // different constant for each and has one measurement apiece. Two contexts
    // and a production-shaped slot configuration, not one point: the constants
    // these cells justify were calibrated at ctx 240000 with `-np 4 --kv-unified`
    // (the separate-draft shape) and at ctx 32768 with `-np 2` (the embedded
    // shape), and a single small-context point can neither reproduce those nor
    // test the claim that the separate draft's cost does not scale with context.
    let shapes: [(&str, &Model, Option<&str>, Option<&str>); 4] = [
        ("mtp-none-g4", g4, None, None),
        ("mtp-draft-g4", g4, Some("draft-mtp"), g4.draft),
        ("mtp-none-q27", q27, None, None),
        ("mtp-embedded-q27", q27, Some("draft-mtp"), None),
    ];
    for (label, m, spec, draft) in shapes {
        for (ctx, parallel, unified) in [(32768, 1, false), (131072, 4, true)] {
            cells.push(Factors {
                label: format!("{label}-c{ctx}-np{parallel}"),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Tensor),
                ctx,
                parallel,
                kv_unified: unified,
                spec_type: spec.map(str::to_owned),
                draft: lib.path_opt(draft),
                ..Factors::default()
            });
        }
    }
    // The vision projector, which was worth ~3 MiB in one observation.
    for (label, mmproj) in [("mmproj-off", None), ("mmproj-on", g4.mmproj)] {
        cells.push(Factors {
            label: format!("{label}-g4"),
            model: lib.path_of(g4.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
            mmproj: lib.path_opt(mmproj),
            ..Factors::default()
        });
    }
    // Offload regimes: the arena was measured invariant across them, and a host
    // baseline with no GPU visible is a different shape entirely.
    let q4 = model("qwen3-4b");
    for (ngl, gpus, label) in [
        (99, "0", "ngl99"),
        (18, "0", "ngl18"),
        (0, "0", "ngl0"),
        (0, "", "no-cuda"),
    ] {
        cells.push(Factors {
            label: format!("offload-{label}"),
            model: lib.path_of(q4.path),
            gpus: gpus.to_owned(),
            ngl,
            ..Factors::default()
        });
    }
    // ik-only paths that change where weights land, and so which counter the
    // daemon's weights detection reads.
    //
    // `--use-thp` is absent from this ik build — passing it is an argument error,
    // not a no-op. Restore the cell on a build that has the flag; there is no way
    // to ask for it conditionally from here, since the failure looks like any
    // other load failure.
    let lag = model("laguna");
    for (label, rtr, no_mmap) in [
        ("plain", false, false),
        ("rtr", true, false),
        ("nommap", false, true),
    ] {
        cells.push(Factors {
            label: format!("ik-laguna-{label}"),
            model: lib.path_of(lag.path),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            n_cpu_moe: lag.n_cpu_moe,
            rtr,
            no_mmap,
            ..Factors::default()
        });
    }
    // The embedding modality, which has its own graph and no generation.
    let emb = model("lfm2-embed");
    cells.push(Factors {
        label: "embeddings".to_owned(),
        model: lib.path_of(emb.path),
        ctx: 2048,
        embeddings: true,
        ..Factors::default()
    });
    // Growth: what accumulates once the server is doing an agent's work, with and
    // without the prompt cache the estimator reserves 8 GiB for.
    for cram in [0, 8192] {
        cells.push(Factors {
            label: format!("growth-cram{cram}"),
            model: lib.path_of(q4.path),
            cram,
            bench: true,
            bench_turns: 40,
            verbose_log: false,
            ..Factors::default()
        });
    }
    // The same question for the models where host memory actually matters. A 4B
    // dense model is the one whose growth is least interesting and was the only
    // one measured; a hybrid MoE holds tens of GiB on the host, and if it
    // accumulates over an agent session nothing else here would see it.
    for m in MODELS {
        if m.key == q4.key {
            continue; // Already covered by the cram pair above.
        }
        let gpus = m.preferred_gpus();
        cells.push(Factors {
            label: format!("growth-{}", m.key),
            model: lib.path_of(m.path),
            runtime: m.runtimes[0],
            gpus: gpus.to_owned(),
            split: Some(SplitMode::Layer),
            bench: true,
            bench_turns: GROWTH_TURNS,
            verbose_log: false,
            ..m.flags(gpus)
        });
    }
    cells
}

/// The operator's real service configurations, predicted before they were
/// measured.
///
/// The prediction-before-measurement is the honest part and it happened once, at
/// the moment these were first run. It does not survive into the fit: `emit` takes
/// every `ok` row, so these cells are in the fitting set like any other, and the
/// scoreboard's drift is an in-sample figure.
///
/// Deliberately, because they carry evidence nothing else has — 3 of the 4 mmproj
/// cells, including the larger vision configuration. Holding them back drops
/// `MMPROJ_GRAPH_BYTES` from 260 MiB to 147 MiB by leaving one configuration to fit
/// two degrees of freedom, which is a worse estimator bought with a cleaner story.
///
/// The out-of-sample figure is supposed to come from leave-one-model-out
/// cross-validation instead — see the analysis protocol in `PLAN.md` — which costs
/// no extra measurement and is not yet implemented. Until it is, there is no
/// generalisation number here, and the scoreboard should not be read as one.
pub fn holdout(lib: &Library) -> Vec<Factors> {
    let g4 = model("gemma4-31b-qat");
    let q27 = model("qwen36-27b");
    let q35 = model("qwen36-35b-a3b");
    let lag = model("laguna");
    let glm = model("glm52");
    let ds = model("dsv4f");
    vec![
        Factors {
            label: "prod-gemma4-31b-qat".to_owned(),
            model: lib.path_of(g4.path),
            mmproj: lib.path_opt(g4.mmproj),
            draft: lib.path_opt(g4.draft),
            spec_type: Some("draft-mtp".to_owned()),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
            ctx: 240000,
            parallel: 4,
            kv_unified: true,
            extra: vec!["-n".to_owned(), "16384".to_owned()],
            ..Factors::default()
        },
        Factors {
            label: "prod-qwen36-27b".to_owned(),
            model: lib.path_of(q27.path),
            mmproj: lib.path_opt(q27.mmproj),
            spec_type: Some("draft-mtp".to_owned()),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
            ctx: 360000,
            parallel: 2,
            kv_type: KvType::Q80,
            ..Factors::default()
        },
        Factors {
            label: "prod-qwen36-35b-a3b".to_owned(),
            model: lib.path_of(q35.path),
            mmproj: lib.path_opt(q35.mmproj),
            spec_type: Some("draft-mtp".to_owned()),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
            ctx: 524288,
            parallel: 2,
            kv_type: KvType::Q80,
            n_cpu_moe: q35.n_cpu_moe,
            ..Factors::default()
        },
        Factors {
            label: "prod-laguna".to_owned(),
            model: lib.path_of(lag.path),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            ctx: 131072,
            batch: Some(2048),
            ubatch: 2048,
            kv_type: KvType::Q80,
            no_mmap: true,
            threads: Some(24),
            numa: Some("distribute".to_owned()),
            n_cpu_moe: lag.n_cpu_moe,
            ..Factors::default()
        },
        Factors {
            label: "prod-glm52".to_owned(),
            model: lib.path_of(glm.path),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            ctx: 131072,
            batch: Some(2048),
            ubatch: 2048,
            no_mmap: true,
            threads: Some(24),
            n_cpu_moe: glm.n_cpu_moe,
            extra: glm.extra.iter().map(|s| (*s).to_owned()).collect(),
            ..Factors::default()
        },
        Factors {
            label: "prod-dsv4f".to_owned(),
            model: lib.path_of(ds.path),
            gpus: "0,1".to_owned(),
            ctx: 131072,
            n_cpu_moe: ds.n_cpu_moe,
            ..Factors::default()
        },
        Factors {
            label: "prod-talkie".to_owned(),
            model: lib.path_of(model("talkie-13b").path),
            ctx: 2048,
            ..Factors::default()
        },
    ]
}

/// Turns every growth cell runs, held constant across models.
///
/// Growth as a function of turn count only means anything if the turn count is
/// the same everywhere, so this does not scale with model speed even though the
/// slowest model costs twenty times the fastest per turn.
///
/// Ten turns reaches roughly five thousand generated tokens. That is enough to
/// separate "grows per token" from "allocates once on first use", which is the
/// question; it is not enough to characterise a slow leak, and nothing here
/// claims to.
pub const GROWTH_TURNS: u32 = 10;

/// The cells the factor screen showed matter, for one model.
///
/// The screen measured `ctx` and `parallel` to be irrelevant to the baseline and
/// `gpus`/`split`/`served`/`kv_type` to matter, so only those are varied here.
fn significant(lib: &Library, m: &Model, runtime: Runtime) -> Vec<Factors> {
    let mut cells = Vec::new();
    for gpus in m.gpus {
        for split in m.splits_for(runtime) {
            for kv in m.kv_types {
                for served in [true, false] {
                    let serving = if served { "srv" } else { "idle" };
                    let mut cell = Factors {
                        label: format!(
                            "{}-{}-{}-{split}-{kv}-{serving}",
                            m.key,
                            runtime_name(runtime),
                            gpus.replace(',', "")
                        ),
                        model: lib.path_of(m.path),
                        runtime,
                        gpus: (*gpus).to_owned(),
                        split: Some(split),
                        kv_type: *kv,
                        served,
                        ..m.flags(gpus)
                    };
                    // A model whose native context is below the sweep's stays
                    // inside its own range rather than being extrapolated past
                    // `n_ctx_train`.
                    if let Some(top) = m.max_ctx {
                        cell.ctx = top;
                    }
                    cells.push(cell);
                }
            }
        }
    }
    cells
}

/// The (context, ubatch, flash-attn) points a model's curve is fitted on.
///
/// Three contexts to check the fit is linear rather than merely fitted, one
/// larger batch for the terms that scale with it, and one flash-attention-off
/// point. A model whose native context is below the standard sweep gets the same
/// shape scaled into its own range instead of being pushed past it.
fn curve_points(m: &Model) -> Vec<(u32, u32, FlashAttn)> {
    let contexts = match m.max_ctx {
        Some(top) if top < 65536 => [(top / 4).max(512), (top / 2).max(1024), top],
        _ => [8192, 32768, 65536],
    };
    let mid = contexts[1];
    let mut points: Vec<_> = contexts.iter().map(|&c| (c, 512, FlashAttn::On)).collect();
    points.push((mid, 2048, FlashAttn::On));
    points.push((mid, 512, FlashAttn::Off));
    points
}
