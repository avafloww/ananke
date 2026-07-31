//! Sweeps that turn a bracketed relationship into a measured one.
//!
//! The phases in [`crate::plan::phases`] establish which factors matter; these
//! establish *how*. Two samples cannot distinguish a line through the origin
//! from an affine one, and the difference is charged to the slope, so it grows
//! with every extrapolation.

use ananke_config::placement::SplitMode;
use ananke_measure::record::{Factors, FlashAttn, KvType, Runtime};

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
                split: Some(SplitMode::Layer),
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
            split: Some(SplitMode::Layer),
            ubatch: 2048,
            flash_attn: FlashAttn::Off,
            ..m.flags("0,1")
        });
    }
    // deepseek4's slope is claimed linear in ubatch from two points.
    let ds = model("dsv4f");
    cells.push(Factors {
        label: "dsv4f-ub1024".to_owned(),
        model: lib.path_of(ds.path),
        gpus: "0,1".to_owned(),
        split: Some(SplitMode::Layer),
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
    for anchor in [
        LongContextAnchor {
            key: "dsv4f",
            ctx: 131072,
            runtime: Runtime::Mainline,
        },
        LongContextAnchor {
            key: "glm52",
            ctx: 131072,
            runtime: Runtime::Ik,
        },
        LongContextAnchor {
            key: "laguna",
            ctx: 131072,
            runtime: Runtime::Ik,
        },
    ] {
        let m = model(anchor.key);
        let ctx = anchor.ctx;
        cells.push(Factors {
            label: format!("longctx-{}-c{ctx}", m.key),
            model: lib.path_of(m.path),
            runtime: anchor.runtime,
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Layer),
            ctx,
            ..m.flags("0,1")
        });
    }
    // The embedded-MTP constant is a one-model number, and the second model that
    // runs it in production differs in kv-head count — the factor the KV formula
    // multiplies by and which one model cannot verify.
    let q35 = model("qwen36-35b-a3b");
    for arm in MTP_ARMS_EMBEDDED {
        cells.push(Factors {
            label: format!("mtp-{}-35b", arm.name),
            model: lib.path_of(q35.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
            spec_type: arm.spec_type.map(str::to_owned),
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
        if m.kv_types == [KvType::F16] {
            continue;
        }
        for ctx in [8192, 65536] {
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-q8", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Layer),
                ctx,
                kv_type: KvType::Q80,
                ..m.flags("0,1")
            });
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-np4", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Layer),
                ctx,
                parallel: 4,
                kv_unified: true,
                ..m.flags("0,1")
            });
            cells.push(Factors {
                label: format!("interact-{}-c{ctx}-tensor", m.key),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Tensor),
                ctx,
                ..m.flags("0,1")
            });
        }
    }
    cells
}

/// The cells the two constants held under review actually need.
///
/// Both are left unchanged because the evidence says the shipped value is wrong
/// without saying what is right.
///
/// deepseek4's curve carries a slope of 66 MiB per 1024 tokens that the
/// production hybrid does not show — flat across a sixteenfold range of context.
/// But every cell measuring that is the hybrid, with 40 of 43 layers on the CPU.
/// If the calibration behind the curve measured a GPU-resident configuration then both
/// figures are correct and the curve is being applied outside the regime it was
/// fitted in, which is a different fix from a wrong number. Sweeping the offload
/// axis at two contexts separates them: if VRAM climbs with context once layers
/// are resident, the curve is right and misapplied; if it stays flat, the slope
/// is wrong.
///
/// MTP's compute constant was fitted against llama.cpp's own `[spec]` log line,
/// which reports a quantity four times smaller than the driver delta between
/// paired with- and without-MTP cells. Correcting it needs those pairs at more
/// shapes than the dataset holds, and on both models that carry an embedded
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
                split: Some(SplitMode::Layer),
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
        for arm in MTP_ARMS {
            cells.push(Factors {
                label: format!("mtprev-{}-{}-c{ctx}", m.key, arm.name),
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Tensor),
                ctx,
                spec_type: arm.spec_type.map(str::to_owned),
                draft: arm.spec_type.and_then(|_| lib.path_opt(m.draft)),
                ..m.flags("0,1")
            });
        }
    }
    cells
}

/// Separate MTP's slot count from its context.
///
/// `MTP_COMPUTE_MIB` is under review because the paired with- and without-MTP
/// cells disagree with the constant by a factor of four, and the disagreement
/// grows at `parallel = 4`. But the existing design cannot say that:
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
            for arm in MTP_ARMS {
                cells.push(Factors {
                    label: format!("mtpslot-{}-{}-np{parallel}", m.key, arm.name),
                    purpose: vec!["mtp-slots".to_owned()],
                    model: lib.path_of(m.path),
                    gpus: "0,1".to_owned(),
                    split: Some(SplitMode::Tensor),
                    parallel,
                    kv_unified: true,
                    spec_type: arm.spec_type.map(str::to_owned),
                    draft: arm.spec_type.and_then(|_| lib.path_opt(m.draft)),
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
            split: Some(SplitMode::Layer),
            repeat,
            ..lag.flags("0,1")
        });
        cells.push(Factors {
            label: format!("repeat-gemma3-tensor-{repeat}"),
            model: lib.path_of(gemma3.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Tensor),
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
    [
        SlotShape {
            parallel: 4,
            concurrency: 4,
            kv_unified: false,
        },
        SlotShape {
            parallel: 4,
            concurrency: 4,
            kv_unified: true,
        },
        SlotShape {
            parallel: 2,
            concurrency: 2,
            kv_unified: false,
        },
    ]
    .into_iter()
    .map(|shape| Factors {
        label: format!("slots-np{}-c{}", shape.parallel, shape.concurrency),
        model: lib.path_of(m.path),
        gpus: "0,1".to_owned(),
        split: Some(SplitMode::Layer),
        parallel: shape.parallel,
        kv_unified: shape.kv_unified,
        soak: 6,
        concurrency: shape.concurrency,
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
        for visible in DEVICE_COUNTS {
            cells.push(Factors {
                label: format!("devices-{}-{}", m.key, visible.name),
                model: lib.path_of(m.path),
                gpus: visible.gpus.to_owned(),
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
            split: Some(SplitMode::Layer),
            threads: Some(threads),
            n_cpu_moe: lag.n_cpu_moe,
            ..Factors::default()
        });
    }
    cells
}

/// A model whose curve is anchored at one long context, on the runtime the
/// operator actually serves it with.
#[derive(Debug, Clone, Copy)]
struct LongContextAnchor {
    key: &'static str,
    ctx: u32,
    runtime: Runtime,
}

/// The two arms of an MTP comparison: the flag off, and the flag on.
#[derive(Debug, Clone, Copy)]
struct MtpArm {
    spec_type: Option<&'static str>,
    name: &'static str,
}

const MTP_ARMS: [MtpArm; 2] = [
    MtpArm {
        spec_type: None,
        name: "none",
    },
    MtpArm {
        spec_type: Some("draft-mtp"),
        name: "mtp",
    },
];

/// The same two arms, named for the shape the second one exercises.
const MTP_ARMS_EMBEDDED: [MtpArm; 2] = [
    MtpArm {
        spec_type: None,
        name: "none",
    },
    MtpArm {
        spec_type: Some("draft-mtp"),
        name: "embedded",
    },
];

/// How many CUDA devices a cell can see, with placement pinned to the CPU.
#[derive(Debug, Clone, Copy)]
struct VisibleDevices {
    gpus: &'static str,
    name: &'static str,
}

const DEVICE_COUNTS: [VisibleDevices; 3] = [
    VisibleDevices {
        gpus: "",
        name: "none",
    },
    VisibleDevices {
        gpus: "0",
        name: "one",
    },
    VisibleDevices {
        gpus: "0,1",
        name: "two",
    },
];

/// How many slots a cell serves, and how many requests reach them at once.
#[derive(Debug, Clone, Copy)]
struct SlotShape {
    parallel: u32,
    concurrency: u32,
    kv_unified: bool,
}
