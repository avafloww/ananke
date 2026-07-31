//! Sweeps that add a second point to an axis a rule depends on.
//!
//! These exist because `coverage.rs` audits the dataset for exactly one failure
//! mode — a regime whose rule varies in an axis the measurements hold fixed — and
//! that failure mode has produced four wrong constants here: the
//! flash-attention rate, the shared-cache window mask, the separate-draft compute,
//! and the checkpoint headroom all looked flat until a second point in the axis
//! that mattered. Every sweep below closes one of the gaps that audit reports.

use ananke_config::placement::SplitMode;
use ananke_measure::record::{Factors, FlashAttn, Runtime};

use crate::plan::library::{Library, model};

/// The no-flash-attention regime, over enough shapes to fit a term.
///
/// Nineteen cells ran with it off and seventeen of them sit at exactly ctx 32768,
/// ubatch 512, one slot. That measures the *offset* at one point and says nothing
/// about how it scales, which is why the regime is excluded from every derivation
/// rather than modelled: it costs 30 to 254 MiB of host residual depending on the
/// architecture, and it is the single largest remaining compute over-reservation.
///
/// Both axes the mask depends on are swept — the KQ mask is `n_kv x n_tokens` and
/// loses its f16 packing when flash attention is off, so the term should be linear
/// in both context and batch. Four architectures, chosen for the mask shapes that
/// differ: interleaved SWA, plain causal, full MHA, and the embedding model whose
/// no-flash-attention residual is the largest measured.
///
/// Each cell has a flash-attention-on twin already in the dataset from the curve
/// sweep, so the pairs are formed without measuring the twins again.
pub fn flash_attention(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "qwen36-27b", "magidonia-24b", "lfm2-embed"] {
        let m = model(key);
        // The widest card count the model is worth measuring on, which for the
        // embedding model is still one.
        let gpus = *m.gpus.last().expect("every model names a card count");
        for (ctx, ubatch) in [
            (8192, 512),
            (32768, 512),
            (131072, 512),
            (32768, 2048),
            (8192, 2048),
        ] {
            if m.max_ctx.is_some_and(|top| ctx > top) {
                continue;
            }
            cells.push(Factors {
                label: format!("faoff-{}-c{ctx}-ub{ubatch}", m.key),
                purpose: vec!["flash-attention".to_owned()],
                model: lib.path_of(m.path),
                gpus: gpus.to_owned(),
                split: Some(m.splits_for(Runtime::Mainline)[0].to_owned()),
                ctx,
                ubatch,
                flash_attn: FlashAttn::Off,
                ..m.flags(gpus)
            });
        }
    }
    cells
}

/// The slot rules at a second batch size.
///
/// Every cell with `parallel > 1` or `--kv-unified` was measured at ubatch 512,
/// and both feed rules that multiply terms which scale with the batch: the stream
/// division that sizes the KQ mask, and the three window masks an interleaved-SWA
/// model builds when slots share one cache. A rule that is wrong in its batch
/// dependence is invisible at one batch size.
///
/// That is exactly how flash-attention-off spent the first campaign recorded as an
/// inconsistent baseline shift when it is a clean per-token rate — the cells that
/// would have shown it all sat at one ubatch. This is the same hole in the two
/// remaining places it exists.
///
/// An SWA model and a plain causal one, since only the former exercises the
/// window-mask rule.
pub fn slot_batch(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "qwen3-4b"] {
        let m = model(key);
        for (parallel, unified) in [(4, true), (4, false), (1, false)] {
            for ubatch in [512, 2048] {
                cells.push(Factors {
                    label: format!(
                        "slotbatch-{}-np{parallel}{}-ub{ubatch}",
                        m.key,
                        if unified { "-unified" } else { "" }
                    ),
                    purpose: vec!["slot-batch".to_owned()],
                    model: lib.path_of(m.path),
                    gpus: "0,1".to_owned(),
                    split: Some(SplitMode::Layer),
                    ubatch,
                    parallel,
                    kv_unified: unified,
                    ..m.flags("0,1")
                });
            }
        }
    }
    cells
}

/// The per-slot cost, across architectures rather than one.
///
/// Qwen3.6-27B holds 602, 767, and 1083 MiB of anonymous memory at one, two, and
/// four *concurrent* requests, all else equal — about 162 MiB per additional
/// active slot, linear. Slots that stay idle cost nothing: the same model at
/// `parallel` 1, 2, and 4 with a single sequential probe reads 716 MiB at every
/// one. A reservation has to assume every slot can become active, so this belongs
/// in the model.
///
/// It is measured on exactly one architecture, at one context and one split. That
/// is the coverage that has produced a wrong constant three times in this
/// campaign, so the term is measured across architectures before it is modelled,
/// not fitted from the one series and generalised.
///
/// An interleaved-SWA model, a plain causal one, and the one with the existing
/// series as a control.
pub fn concurrency_models(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "magidonia-24b", "qwen36-27b"] {
        let m = model(key);
        for conc in [1, 2, 4] {
            cells.push(Factors {
                label: format!("conc-{}-c{conc}", m.key),
                purpose: vec!["concurrency".to_owned()],
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Layer),
                parallel: 4,
                kv_unified: true,
                soak: 6,
                concurrency: conc,
                ..m.flags("0,1")
            });
        }
    }
    cells
}

/// Two claims this campaign asserted on data outside the dataset.
///
/// The flash-attention term is divided by the stream count, on the strength of
/// hardcoded Qwen3-4B points in a unit test from an earlier sweep: every such cell
/// in `measurements.ndjson` runs one slot, so nothing there can falsify it. If the
/// division is right these cells show a quarter of the one-slot rate; if it is
/// wrong they show the whole of it.
///
/// And the per-slot host cost was measured at one context and one batch. It is
/// reserved as slop rather than charged to the correction, so an error is less
/// costly — but "measured at one point in the axis" is what made three other
/// constants wrong here, so it gets a second batch size.
pub fn loose_ends(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "qwen36-27b"] {
        let m = model(key);
        cells.push(Factors {
            label: format!("faoff-slots-{}-np4", m.key),
            purpose: vec!["flash-attention".to_owned()],
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Layer),
            parallel: 4,
            flash_attn: FlashAttn::Off,
            ..m.flags("0,1")
        });
    }
    let m = model("gemma3-27b");
    for conc in [1, 4] {
        cells.push(Factors {
            label: format!("conc-ub2048-{}-c{conc}", m.key),
            purpose: vec!["concurrency".to_owned()],
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Layer),
            ubatch: 2048,
            parallel: 4,
            kv_unified: true,
            soak: 6,
            concurrency: conc,
            ..m.flags("0,1")
        });
    }
    cells
}

/// Context and batch on one card, where the mask is not replicated.
///
/// Seven of eleven architectures have exactly one single-card point, at ctx 32768
/// and ubatch 512. Everything else about the arena — the mask copies above all —
/// is fitted from two-card cells, so a rule that is right at four copies and wrong
/// at one would show up nowhere: the copy factor multiplies terms that scale with
/// both context and batch, and at a single point any factor can be made to fit.
pub fn single_card(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "magidonia-24b"] {
        let m = model(key);
        for (ctx, ubatch) in [(8192, 512), (65536, 512), (32768, 2048)] {
            cells.push(Factors {
                label: format!("onecard-{}-c{ctx}-ub{ubatch}", m.key),
                purpose: vec!["curves".to_owned()],
                model: lib.path_of(m.path),
                gpus: "0".to_owned(),
                split: Some(SplitMode::Layer),
                ctx,
                ubatch,
                ..m.flags("0")
            });
        }
    }
    cells
}

/// The three switches with one cell each, given a matched pair.
///
/// `--use-thp`, `-rtr`, and `--numa distribute` are recorded on every row and
/// modelled by nothing. That is defensible — `RollingBase::host_peak` reads the
/// measured `RssFile` precisely because `-rtr`'s effect cannot be predicted from
/// the flag — but "modelled by nothing" is a claim about the data, and one cell
/// cannot support it. A switch measured once has no counterfactual: it might move
/// host memory by 500 MiB and nothing here would show it.
///
/// Each gets its off/on pair at otherwise identical settings, so the claim becomes
/// a measurement instead of an assumption. `-rtr` is ik-only and forces
/// `--no-mmap`, so its pair carries that on both sides to keep the mapping mode
/// from confounding it.
///
/// `--use-thp` is absent because this llama.cpp build rejects the flag outright —
/// `error: invalid argument: --use-thp` — so such a cell can only fail to load,
/// and it sat out the full load timeout doing it. The `thp` field is gone from the
/// factor set entirely, which identity-by-non-default makes free.
pub fn sparse_switches(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    let dense = model("qwen3-4b");
    for value in [false, true] {
        let state = if value { "on" } else { "off" };
        cells.push(Factors {
            label: format!("switch-numa-{state}"),
            purpose: vec!["switches".to_owned()],
            model: lib.path_of(dense.path),
            gpus: "0".to_owned(),
            split: Some(SplitMode::Layer),
            numa: value.then(|| "distribute".to_owned()),
            ..dense.flags("0")
        });
        cells.push(Factors {
            label: format!("switch-rtr-{state}"),
            purpose: vec!["switches".to_owned()],
            model: lib.path_of(dense.path),
            gpus: "0".to_owned(),
            split: Some(SplitMode::Layer),
            runtime: Runtime::Ik,
            rtr: value,
            no_mmap: true,
            ..dense.flags("0")
        });
    }
    cells
}

/// The host cost a real prompt reaches, against the probe's.
///
/// llama.cpp's server takes a context checkpoint while decoding a prompt, spaced
/// by `--checkpoint-min-step` (8192 tokens) and capped by `--ctx-checkpoints`
/// (32). The campaign's probe is four tokens, so every baseline cell captures at
/// most one checkpoint — and, since the step measures 11 MiB at a one-token prompt
/// against 274 at sixty-four, possibly only part of one. A prompt past the spacing
/// reaches the steady state, which is two checkpoints: measured 274, 426, 428, and
/// 431 MiB at 64, 8192, 16384, and 24576 tokens.
///
/// Each cell here mirrors an existing baseline cell exactly but for
/// `probe_prompt_tokens`, so the difference between them is the whole correction —
/// the unmeasured part of the first checkpoint plus the second — without
/// re-measuring the 156 cells the offsets are fitted from.
pub fn checkpoint_steady(lib: &Library) -> Vec<Factors> {
    [
        "qwen3-4b",
        "qwen36-27b",
        "gemma3-27b",
        "magidonia-24b",
        "talkie-13b",
        "gemma4-31b-qat",
        "gemma4-26b-a4b",
        "gemma4-e4b",
        "qwen36-35b-a3b",
    ]
    .into_iter()
    .map(|key| {
        let m = model(key);
        Factors {
            label: format!("ckpt-steady-{}", m.key),
            purpose: vec!["switches".to_owned()],
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Layer),
            probe_prompt_tokens: 16384,
            ..m.flags("0,1")
        }
    })
    .collect()
}

/// A second context for the two regimes the coverage audit reports as thin.
///
/// `checkpoint_headroom_bytes` and `per_slot_host_bytes` were both fitted from
/// cells at ctx 32768 alone, which is the shape that has produced a wrong constant
/// four times here. Neither is charged to the rolling correction — both are
/// reserved as slop — so an error costs capacity rather than a failed load, but a
/// context dependence would still be invisible.
///
/// Checkpoint headroom needs a steady/short pair at the new context; the short
/// halves already exist from the curve sweep, so only the steady halves are
/// measured. The per-slot cost needs its concurrency series repeated.
pub fn second_context(lib: &Library) -> Vec<Factors> {
    let mut cells = Vec::new();
    for key in ["gemma3-27b", "gemma4-31b-qat", "qwen36-27b"] {
        let m = model(key);
        cells.push(Factors {
            label: format!("ckpt-steady-c65536-{}", m.key),
            purpose: vec!["switches".to_owned()],
            model: lib.path_of(m.path),
            gpus: "0,1".to_owned(),
            split: Some(SplitMode::Layer),
            ctx: 65536,
            probe_prompt_tokens: 16384,
            ..m.flags("0,1")
        });
    }
    for key in ["gemma3-27b", "qwen36-27b"] {
        let m = model(key);
        for conc in [1, 4] {
            cells.push(Factors {
                label: format!("conc-c65536-{}-c{conc}", m.key),
                purpose: vec!["concurrency".to_owned()],
                model: lib.path_of(m.path),
                gpus: "0,1".to_owned(),
                split: Some(SplitMode::Layer),
                ctx: 65536,
                parallel: 4,
                kv_unified: true,
                soak: 6,
                concurrency: conc,
                ..m.flags("0,1")
            });
        }
    }
    // And on one card. Every steady-state cell above runs two, so the checkpoint
    // headroom is fitted without ever varying the card count — which matters,
    // since the term is charged per slot and the masks beside it replicate per
    // device.
    for key in ["gemma3-27b", "qwen3-4b"] {
        let m = model(key);
        cells.push(Factors {
            label: format!("ckpt-steady-1card-{}", m.key),
            purpose: vec!["switches".to_owned()],
            model: lib.path_of(m.path),
            gpus: "0".to_owned(),
            split: Some(SplitMode::Layer),
            probe_prompt_tokens: 16384,
            ..m.flags("0")
        });
    }
    cells
}
