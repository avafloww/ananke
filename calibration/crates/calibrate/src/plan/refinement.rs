//! Sweeps that turn a bracketed relationship into a measured one.
//!
//! The phases in [`crate::plan::phases`] establish which factors matter; these
//! establish *how*. Several terms were claimed linear on the strength of two
//! samples, which cannot distinguish a line through the origin from an affine
//! one — and the difference is charged to the slope, so it grows with every
//! extrapolation.

use ananke_measure::record::{Factors, Runtime};

use crate::plan::library::{Library, model};

/// The points that turn a two-point line into a measured relationship.
///
/// Each cell here adds an interior or a wider point to a relationship the rest of
/// the campaign only brackets.
pub fn interior(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    // ik's CPU-MoE term is per batch token with a threshold between them; one
    // point below and one above forces the slope through the origin and leaves the
    // threshold bracketed to a factor of four.
    for key in ["qwen36-35b-a3b", "laguna", "glm52"] {
        let m = model(key);
        if !m.runtimes.contains(&Runtime::Ik) {
            continue;
        }
        for ubatch in [256, 1024] {
            cells.push(Factors {
                label: format!("ikmoe-{}-ub{ubatch}", m.key),
                model: lib.path_of(m.path),
                runtime: Runtime::Ik,
                gpus: "0,1".to_owned(),
                split: Some("layer".to_owned()),
                ubatch,
                ..m.flags("0,1")
            });
        }
    }
    // The no-flash-attention term is claimed per-token but only ever sampled at
    // one batch size, where a flat 4 MiB fits exactly as well.
    for key in ["qwen3-4b", "gemma3-27b"] {
        let m = model(key);
        cells.push(Factors {
            label: format!("nofa-{}-ub2048", m.key),
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some("layer".to_owned()),
            ubatch: 2048,
            flash_attn: "off".to_owned(),
            ..m.flags("0,1")
        });
    }
    // deepseek4's slope is claimed linear in ubatch from two points.
    let ds = model("dsv4f");
    cells.push(Factors {
        label: "dsv4f-ub1024".to_owned(),
        model: lib.path_of(ds.path),
        gpus: "0,1".to_owned(),
        split: Some("layer".to_owned()),
        ubatch: 1024,
        ..ds.flags("0,1")
    });
    // The curves are fitted to 65536 and used to 524288. One long point per steep
    // architecture bounds how far the extrapolation can drift.
    //
    // The runtime is the one the operator actually serves each of these with, not
    // the first in the model's tuple: laguna's production runtime is ik, and
    // anchoring its curve on mainline measures a combination nobody runs — and, at
    // this context, one that does not fit.
    for (key, ctx, runtime) in [
        ("dsv4f", 131072, Runtime::Mainline),
        ("glm52", 131072, Runtime::Ik),
        ("laguna", 131072, Runtime::Ik),
    ] {
        let m = model(key);
        cells.push(Factors {
            label: format!("longctx-{}-c{ctx}", m.key),
            model: lib.path_of(m.path),
            runtime,
            gpus: "0,1".to_owned(),
            split: Some("layer".to_owned()),
            ctx,
            ..m.flags("0,1")
        });
    }
    // The embedded-MTP constant is a one-model number, and the second model that
    // runs it in production differs in kv-head count — the factor the KV formula
    // multiplies by and which one model cannot verify.
    let q35 = model("qwen36-35b-a3b");
    for (spec, name) in [(None, "none"), (Some("draft-mtp"), "embedded")] {
        cells.push(Factors {
            label: format!("mtp-{name}-35b"),
            model: lib.path_of(q35.path),
            gpus: "0,1".to_owned(),
            split: Some("tensor".to_owned()),
            spec_type: spec.map(str::to_owned),
            ..q35.flags("0,1")
        });
    }
    cells
}

/// Whether the curves hold in the regime production actually runs.
///
/// Every curve cell is f16, one slot, layer split. Production is q8_0, two to
/// four slots, tensor split. Fitting in one regime and serving in another is only
/// safe if the curve's *slope* does not depend on those settings — which is an
/// assumption nobody has tested, and one whose failure would appear as
/// unexplained holdout error with no cell to attribute it to.
///
/// Two contexts per variant is the minimum that can distinguish "shifts the base"
/// from "changes the slope"; one point could only see the former.
pub fn interactions(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["qwen3-4b", "gemma3-27b", "qwen36-27b", "qwen36-35b-a3b"] {
        let m = model(key);
        if m.kv_types == ["f16"] {
            continue;
        }
        for ctx in [8192, 65536] {
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-q8", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some("layer".to_owned()),
                ctx,
                kv_type: "q8_0".to_owned(),
                ..m.flags("0,1")
            });
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-np4", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some("layer".to_owned()),
                ctx,
                parallel: 4,
                kv_unified: true,
                ..m.flags("0,1")
            });
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-tensor", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some("tensor".to_owned()),
                ctx,
                ..m.flags("0,1")
            });
        }
    }
    cells
}

/// The cells the two constants held under review actually need.
///
/// Both were left unchanged after the first campaign because the evidence said
/// the current value is wrong without saying what is right.
///
/// deepseek4's curve carries a slope of 66 MiB per 1024 tokens that the
/// production hybrid does not show — flat across a sixteenfold range of context.
/// But every cell measuring that was the hybrid, with 40 of 43 layers on the CPU.
/// If the original calibration measured a GPU-resident configuration then both
/// figures are correct and the curve is being applied outside the regime it was
/// fitted in, which is a different fix from a wrong number. Sweeping the offload
/// axis at two contexts separates them: if VRAM climbs with context once layers
/// are resident, the curve is right and misapplied; if it stays flat, the slope
/// is wrong.
///
/// MTP's compute constant was fitted against llama.cpp's own `[spec]` log line,
/// which reports a quantity four times smaller than the driver delta between
/// paired with- and without-MTP cells. Correcting it needs those pairs at more
/// shapes than the first campaign ran, and on both models that carry an embedded
/// head, since they differ in the kv-head count the KV formula multiplies by.
pub fn review_followup(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    let ds = model("dsv4f");
    for n_cpu_moe in [0, 20, 40] {
        for ctx in [8192, 65536] {
            cells.push(Factors {
                label: format!("ds4-offload{n_cpu_moe}-c{ctx}"),
                model: lib.path_of(ds.path),
                gpus: "0,1".to_owned(),
                split: Some("layer".to_owned()),
                ctx,
                // Zero means "nothing on the CPU", which is the absence of the
                // flag rather than the flag with a zero.
                n_cpu_moe: (n_cpu_moe != 0).then_some(n_cpu_moe),
                ..ds.flags("0,1")
            });
        }
    }
    // Paired with/without MTP on both embedded-head models and the separate draft,
    // at a second context each, so the constant is fitted against the driver delta
    // rather than against a log line.
    for (key, ctx) in [
        ("qwen36-27b", 65536),
        ("qwen36-35b-a3b", 65536),
        ("gemma4-31b-qat", 65536),
    ] {
        let m = model(key);
        for (spec, name) in [(None, "none"), (Some("draft-mtp"), "mtp")] {
            cells.push(Factors {
                label: format!("mtprev-{}-{name}-c{ctx}", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some("tensor".to_owned()),
                ctx,
                spec_type: spec.map(str::to_owned),
                draft: spec.and_then(|_| lib.path_opt(m.draft)),
                ..m.flags("0,1")
            });
        }
    }
    cells
}

/// Separate MTP's slot count from its context.
///
/// The first campaign left `MTP_COMPUTE_MIB` under review because the paired
/// with- and without-MTP cells disagreed with the constant by a factor of four,
/// and the disagreement grew at `parallel = 4`. But the design cannot say that:
/// every one-slot pair sits at ctx 32768 or 65536 and the only four-slot pair sits
/// at 131072, so slots and context are confounded and the "slot dependence" may be
/// nothing but the longer context.
///
/// Context is therefore held fixed and only `parallel` moves. Both models with an
/// embedded head are swept, since they differ in the kv-head count the KV formula
/// multiplies by, and the separate-draft model is swept too because its overhead
/// has no context-scaling term at all — its draft shares the target's KV cache —
/// so it is the control: if its delta moves with slots, the cause is not the MTP
/// KV.
pub fn mtp_slots(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["qwen36-27b", "qwen36-35b-a3b", "gemma4-31b-qat"] {
        let m = model(key);
        for parallel in [1, 2, 4] {
            for (spec, name) in [(None, "none"), (Some("draft-mtp"), "mtp")] {
                cells.push(Factors {
                    label: format!("mtpslot-{}-{name}-np{parallel}", m.key),
                    purpose: vec!["mtp-slots".to_owned()],
                    model: lib.path_of(m.path),
                    gpus: "0,1".to_owned(),
                    split: Some("tensor".to_owned()),
                    parallel,
                    kv_unified: true,
                    spec_type: spec.map(str::to_owned),
                    draft: spec.and_then(|_| lib.path_opt(m.draft)),
                    ..m.flags("0,1")
                });
            }
        }
    }
    cells
}

/// Repeats in the regimes the noise floor never visited.
///
/// The floor is five repeats of one small dense model on one card with a hot page
/// cache. It licenses a significance claim for approximately that cell. A hybrid
/// under page-cache pressure, an ik no-mmap load, and a two-card tensor split are
/// all obviously noisier, and every per-model constant is otherwise a single
/// sample.
///
/// The repeats are spread across the run rather than run back to back, so a
/// monotone drift — thermal, fragmentation, page-cache composition — shows up as a
/// difference between them instead of loading onto model size, which the
/// smallest-first ordering would otherwise confound it with.
pub fn replication(lib: &Library) -> Vec<Factors> {
    let lag = model("laguna");
    let gemma3 = model("gemma3-27b");
    let mut cells = Vec::new();
    for repeat in 0..3 {
        cells.push(Factors {
            label: format!("repeat-laguna-hybrid-{repeat}"),
            model: lib.path_of(lag.path),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            split: Some("layer".to_owned()),
            repeat,
            ..lag.flags("0,1")
        });
        cells.push(Factors {
            label: format!("repeat-gemma3-tensor-{repeat}"),
            model: lib.path_of(gemma3.path),
            gpus: "0,1".to_owned(),
            split: Some("tensor".to_owned()),
            repeat,
            ..Factors::default()
        });
    }
    cells
}

/// Per-slot state, which no other cell allocates.
///
/// Production runs `parallel` 2-4. Every other cell here sends strictly
/// sequential requests, so only the first slot is ever touched and the rest stay
/// unallocated — `soak` with `concurrency` is what reaches them.
pub fn concurrency(lib: &Library) -> Vec<Factors> {
    let m = model("qwen36-27b");
    [(4, 4, false), (4, 4, true), (2, 2, false)]
        .into_iter()
        .map(|(parallel, conc, unified)| Factors {
            label: format!("slots-np{parallel}-c{conc}"),
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some("layer".to_owned()),
            parallel,
            kv_unified: unified,
            soak: 6,
            concurrency: conc,
            ..Factors::default()
        })
        .collect()
}

/// Separate the per-device CUDA cost from everything that scales with model.
///
/// The `gpus` axis elsewhere varies placement *and* device count together, so it
/// cannot say which of the two moved a number. These cells pin placement to the
/// CPU — `-ngl 0`, no weights on any card — and vary only how many CUDA contexts
/// get initialised. The difference between them is the host cost of a visible
/// device and nothing else.
///
/// That difference is the term the estimator does not have. `PROCESS_BASE_BYTES`
/// is a compiled scalar fitted on a two-card box; an operator with four or eight
/// cards inherits it wrong by an increment nobody has measured. Three cells per
/// model, on three models of different shape, establish whether the increment is
/// constant and whether it is model-independent.
pub fn device_scaling(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["qwen3-4b", "gemma3-27b", "qwen36-35b-a3b"] {
        let m = model(key);
        for (gpus, name) in [("", "none"), ("0", "one"), ("0,1", "two")] {
            cells.push(Factors {
                label: format!("devices-{}-{name}", m.key),
                model: lib.path_of(m.path),
                gpus: gpus.to_owned(),
                ngl: 0,
                split: None,
                extra: m.extra.iter().map(|s| (*s).to_owned()).collect(),
                threads: m.threads,
                ..Factors::default()
            });
        }
    }
    // Whether the CPU-side terms depend on core count, which every contributor
    // with a different CPU inherits blind.
    let lag = model("laguna");
    for threads in [8, 16, 32] {
        cells.push(Factors {
            label: format!("threads-{threads}-laguna"),
            model: lib.path_of(lag.path),
            runtime: Runtime::Ik,
            gpus: "0,1".to_owned(),
            split: Some("layer".to_owned()),
            threads: Some(threads),
            n_cpu_moe: lag.n_cpu_moe,
            ..Factors::default()
        });
    }
    cells
}
