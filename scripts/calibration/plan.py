"""Generate campaign plans for `measure.py`.

Plans are generated rather than hand-written so that the factor coverage is
stated once, as code, and can be re-derived or extended later. Model paths come
from ``$LLM_DIR`` so a plan is portable to another machine with the same
library.

    python plan.py phase1 > phase1.json
    python measure.py --out data/calibration/phase1.csv --plan phase1.json
"""

from __future__ import annotations

import argparse
import re
import dataclasses
import json
import os
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure import Cell  # noqa: E402

LLM_DIR = Path(os.environ.get("LLM_DIR", "/mnt/ssd0/ai/llm"))


@dataclasses.dataclass(frozen=True)
class Model:
    """A model in the local library, with what the estimator would need to know."""

    key: str
    path: str
    runtimes: tuple[str, ...] = ("mainline",)
    mmproj: str | None = None
    draft: str | None = None
    splits: tuple[str, ...] = ("layer", "tensor")
    """Split modes the runtime can actually serve this architecture with.

    Not every architecture supports every mode. mainline's
    `llm_arch_supports_sm_tensor` (llama-arch.cpp) blocklists the mode, and
    of the models here that catches `deepseek4` and `lfm2`; `glm-dsa` is on
    the same list, and although ik does not gate on architecture at all, the
    operator serves that model hybrid rather than tensor-split, so measuring
    it would characterise a configuration nobody runs.

    Planning cells that cannot load wastes the slot and buries the failures
    that are real.
    """
    extra: tuple[str, ...] = ()
    """Flags the architecture needs to run the way production runs it.

    glm52 is served with `-mla 1 -dsa -amb 512`; without them ik takes a
    different attention path entirely, so a cell that omits them measures
    something the operator never runs — and `glm-dsa`'s curve is calibrated
    against the DSA path specifically.
    """
    kv_types: tuple[str, ...] = ("f16", "q8_0")
    """KV cache types the model can actually be served with.

    `-dsa` rejects a quantised cache, so sweeping q8_0 on glm52 plans eight
    cells that cannot load.
    """
    gpus: tuple[str, ...] = ("0", "0,1")
    """Card counts worth measuring. A 350M embedding model does not need two,
    and spreading it changes the very baseline the cell exists to isolate."""
    max_ctx: int | None = None
    """Native context, where it is below the sweep's range.

    talkie tops out at 2048. Requesting more does not fail — llama.cpp runs
    past `n_ctx_train` with a warning — it just makes every point on the curve
    an extrapolation from a regime the model was never trained for, which is
    not a calibration.
    """
    embeddings: bool = False
    """Whether the model is served with `--embeddings` and has no generation."""
    threads: int | None = None
    no_mmap: bool = False
    n_cpu_moe: int | None = None
    n_cpu_moe_1gpu: int | None = None
    """Expert layers to keep on the CPU when only one card is visible.

    `n_cpu_moe` is sized for both cards. A hybrid tuned that way does not fit
    on one — laguna keeps 18 layers on the GPU, which aborts a single-card
    load — so the single-GPU cells need their own figure or the `gpus` axis is
    simply missing for every large model.
    """
    note: str = ""


MODELS = {
    m.key: m
    for m in [
        Model("qwen3-4b", "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf",
              ("mainline", "ik"), note="dense, 36L, n_embd 2560 — the fast factorial subject"),
        Model("qwen36-27b", "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf",
              ("mainline", "ik"), mmproj="unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf",
              note="dense, 65L, n_embd 5120, embedded MTP head"),
        Model("gemma4-31b-qat", "unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
              ("mainline",), mmproj="unsloth/gemma-4-31B-it-qat-GGUF/mmproj-F16.gguf",
              draft="unsloth/gemma-4-31B-it-qat-GGUF/mtp-gemma-4-31B-it.gguf",
              note="dense + SWA 1024, separate draft GGUF"),
        Model("gemma3-27b", "mlabonne/gemma-3-27b-it-abliterated-GGUF/gemma-3-27b-it-abliterated.q4_k_m.gguf",
              ("mainline", "ik"), note="dense + SWA, no MoE — isolates SWA from experts"),
        Model("qwen36-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
              ("mainline", "ik"), mmproj="unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf",
              n_cpu_moe=40, note="MoE 256/8, 41L"),
        Model("laguna", "unsloth/Laguna-S-2.1-GGUF/UD-IQ4_NL/Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf",
              ("mainline", "ik"), n_cpu_moe=30, n_cpu_moe_1gpu=39,
              note="MoE 256/10 + SWA 512, 48L"),
        Model("dsv4f", "unsloth/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
              ("mainline",), n_cpu_moe=40, splits=("layer",),
              note="MoE + MLA + NSA indexer, 43L; mainline rejects tensor split"),
        Model("glm52", "muzzy/GLM-5.2-GGUF/IQ2_KS/GLM-5.2-smol-IQ2_KS-00001-of-00033.gguf",
              ("ik",), n_cpu_moe=92, n_cpu_moe_1gpu=96,
              extra=("-mla", "1", "-dsa", "-amb", "512"), kv_types=("f16",),
              threads=24, no_mmap=True, splits=("layer",),
              note="MoE + MLA + DSA, 79L — the production quant"),
        # Every remaining llama.cpp service in the operator's config. These
        # were absent from the registry while being served in production
        # daily, which meant the campaign's "holdout" covered under half of
        # what the daemon actually runs.
        Model("gemma4-26b-a4b", "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
              ("mainline",), mmproj="unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-F16.gguf",
              note="gemma4 MoE, 30L"),
        Model("gemma4-e4b", "unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q5_K_XL.gguf",
              ("mainline",), mmproj="unsloth/gemma-4-E4B-it-GGUF/mmproj-F16.gguf",
              note="gemma4 E-variant, 42L — the 1100/7 curve"),
        Model("magidonia-24b", "bartowski/TheDrummer_Magidonia-24B-v4.3-GGUF/TheDrummer_Magidonia-24B-v4.3-Q5_K_M.gguf",
              ("mainline", "ik"), note="llama arch, 40L — the llama-family default curve"),
        Model("talkie-13b", "mradermacher/talkie-1930-13b-it-hf-GGUF/talkie-1930-13b-it-hf.Q6_K.gguf",
              ("mainline",), max_ctx=2048,
              note="talkie arch, 40L, full MHA; native context 2048"),
        Model("lfm2-embed", "LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf",
              ("mainline",), embeddings=True, gpus=("0",), splits=("layer",),
              note="embedding modality; 128k native context. mainline's "
                   "llm_arch_supports_sm_tensor blocklists lfm2"),
    ]
}


def path_of(rel: str | None) -> str | None:
    return str(LLM_DIR / rel) if rel else None


def phase0_noise() -> list[Cell]:
    """Repeat one cell, to establish what a difference has to exceed to matter."""
    m = MODELS["qwen3-4b"]
    return [Cell(label="noise", model=path_of(m.path), ctx=32768, ubatch=512,
                 parallel=4, repeat=i) for i in range(5)]


def phase1_factorial() -> list[Cell]:
    """Full factorial on the fastest model.

    Establishes which factors move the host baseline at all, and which
    interact, before any expensive model is loaded. `split` is included at one
    GPU as well, where it should be inert — a factor measured to be irrelevant
    is worth as much as one that matters.
    """
    m = MODELS["qwen3-4b"]
    cells = []
    for gpus, split, kv, served, parallel, ctx in product(
        ["0", "0,1"], ["layer", "tensor"], ["f16", "q8_0"],
        [True, False], [1, 4], [8192, 32768],
    ):
        cells.append(Cell(
            label=f"p1-{gpus.replace(',', '')}-{split}-{kv}-"
                  f"{'srv' if served else 'idle'}-np{parallel}-c{ctx}",
            model=path_of(m.path), gpus=gpus, split=split, kv_type=kv,
            served=served, parallel=parallel, ctx=ctx,
        ))
    return cells


def _significant(model: Model, runtime: str = "mainline", **over) -> list[Cell]:
    """The cells Phase 1 showed matter, for one model.

    Phase 1 measured `ctx` and `parallel` to be irrelevant to the baseline and
    `gpus`/`split`/`served`/`kv_type` to matter, so only those are varied here.
    Screening on the cheapest model first is what makes sweeping the expensive
    ones affordable.
    """
    cells = []
    for gpus, split, kv, served in product(model.gpus,
                                           _splits_for(model, runtime),
                                           model.kv_types, [True, False]):
        cells.append(Cell(
            label=f"{model.key}-{runtime}-{gpus.replace(',', '')}-{split}-{kv}-"
                  f"{'srv' if served else 'idle'}",
            model=path_of(model.path), runtime=runtime, gpus=gpus, split=split,
            kv_type=kv, served=served, **_model_flags(model, gpus),
            **({"ctx": model.max_ctx} if model.max_ctx else {}), **over,
        ))
    return cells


# ik_llama's `--split-mode` takes none/graph/layer — there is no `tensor`, and
# passing one is a hard argument error rather than a fallback. Its analogue is
# `graph`, which the operator does not run, so restricting to layer keeps the
# fork's cells comparable to mainline's rather than measuring a mode nobody uses.
_RUNTIME_SPLITS = {"ik": ("layer",)}


def _splits_for(model: Model, runtime: str) -> tuple[str, ...]:
    """Split modes both the architecture and the runtime will accept."""
    allowed = _RUNTIME_SPLITS.get(runtime)
    if allowed is None:
        return model.splits
    return tuple(s for s in model.splits if s in allowed) or allowed


def _model_flags(model: Model, gpus: str) -> dict:
    """The per-model settings every phase has to carry, in one place.

    Scattering these across the phase functions is how glm52 came to be
    planned without the DSA flags it is always served with.
    """
    return {
        "extra": model.extra,
        "embeddings": model.embeddings,
        "threads": model.threads,
        "no_mmap": model.no_mmap,
        "n_cpu_moe": (model.n_cpu_moe_1gpu or model.n_cpu_moe)
        if gpus == "0" else model.n_cpu_moe,
    }


def phase2_models() -> list[Cell]:
    """Does the per-model term follow layers, hidden size, vocabulary, or none?

    Phase 1 fixed the factor set on one model; this varies the model with that
    set held constant. Without it the baseline is a constant fitted to one
    shape and generalised — which is the mistake this campaign exists to stop
    repeating.
    """
    return [c for m in MODELS.values() if "mainline" in m.runtimes
            for c in _significant(m)]


def phase3_fork() -> list[Cell]:
    """The same cells on ik_llama, which sizes its graph by different rules."""
    keys = [k for k, m in MODELS.items() if "ik" in m.runtimes]
    return [c for k in keys for c in _significant(MODELS[k], runtime="ik")]


def phase2b_curves() -> list[Cell]:
    """Two contexts and two batches per model, so the slopes are fittable.

    Everything else in this campaign holds `ctx` at 32768 and `ubatch` at 512,
    which is enough for the host baseline — Phase 1 measured it flat in both —
    but leaves several constants underdetermined. A slope needs two points:
    every per-architecture curve in `estimator/compute_buffer.rs` is
    `base + slope * (ctx / 1024)`, `deepseek4`'s slope is additionally linear
    in `ubatch`, and ik's CPU-MoE term is per batch token with a batch-size
    threshold. One point per model can fit none of them.

    Flash attention is varied here for the same reason: it changes the KQ mask
    element width and is the sole justification for
    `NO_FLASH_ATTN_BYTES_PER_TOKEN`, which no cell has ever exercised.
    """
    cells: list[Cell] = []
    for model in MODELS.values():
        # A model that needs one card keeps one: spreading a 350M embedding
        # model across two changes the baseline the curve is fitted against.
        gpus = "0,1" if "0,1" in model.gpus else "0"
        for runtime in model.runtimes:
            for ctx, ubatch, fa in _curve_points(model):
                # Flash attention is a KV-cache property; `-dsa` rejects a
                # quantised cache and the fa-off point is not meaningful for a
                # model that must run with it on.
                if fa == "off" and model.kv_types == ("f16",) and model.extra:
                    continue
                cells.append(Cell(
                    label=f"curve-{model.key}-{runtime}-c{ctx}-ub{ubatch}-fa{fa}",
                    model=path_of(model.path), runtime=runtime, gpus=gpus,
                    split="layer", ctx=ctx, ubatch=ubatch, flash_attn=fa,
                    **_model_flags(model, gpus)))
    return cells


def phase4_special() -> list[Cell]:
    """Terms with their own switch rather than a continuous factor.

    Each of these is currently a constant in the estimator justified by one
    measurement or none: the two MTP shapes, the vision projector, the offload
    regimes, ik's repacking and huge-page paths, the embedding modality, and
    what actually accumulates once a server is doing an agent's work.
    """
    cells: list[Cell] = []
    g4 = MODELS["gemma4-31b-qat"]
    q27 = MODELS["qwen36-27b"]
    # MTP: none, embedded head, separate draft GGUF. The estimator charges a
    # different constant for each and has one measurement apiece.
    # Two contexts and a production-shaped slot configuration, not one point.
    # The constants these cells justify were calibrated at ctx 240000 with
    # `-np 4 --kv-unified` (the separate-draft shape) and at ctx 32768 with
    # `-np 2` (the embedded shape); a single small-context point can neither
    # reproduce those nor test the claim that the separate draft's cost does
    # not scale with context.
    for label, model, spec, draft in [
        ("mtp-none-g4", g4, None, None),
        ("mtp-draft-g4", g4, "draft-mtp", path_of(g4.draft)),
        ("mtp-none-q27", q27, None, None),
        ("mtp-embedded-q27", q27, "draft-mtp", None),
    ]:
        for ctx, parallel, unified in [(32768, 1, False), (131072, 4, True)]:
            cells.append(Cell(label=f"{label}-c{ctx}-np{parallel}",
                              model=path_of(model.path), gpus="0,1",
                              split="tensor", ctx=ctx, parallel=parallel,
                              kv_unified=unified, spec_type=spec, draft=draft))
    # The vision projector, which was worth ~3 MiB in one observation.
    for label, mmproj in [("mmproj-off", None), ("mmproj-on", path_of(g4.mmproj))]:
        cells.append(Cell(label=f"{label}-g4", model=path_of(g4.path), gpus="0,1",
                          split="tensor", ctx=32768, mmproj=mmproj))
    # Offload regimes: the arena was measured invariant across them, and a
    # host baseline with no GPU visible is a different shape entirely.
    q4 = MODELS["qwen3-4b"]
    for ngl, gpus, label in [(99, "0", "ngl99"), (18, "0", "ngl18"),
                             (0, "0", "ngl0"), (0, "", "no-cuda")]:
        cells.append(Cell(label=f"offload-{label}", model=path_of(q4.path),
                          gpus=gpus, ngl=ngl, ctx=32768))
    # ik-only paths that change where weights land, and so which counter the
    # daemon's weights detection reads.
    lag = MODELS["laguna"]
    # `--use-thp` is absent from this ik build — passing it is an argument
    # error, not a no-op. Restore the cell on a build that has the flag; there
    # is no way to ask for it conditionally from here, since the failure looks
    # like any other load failure.
    for label, over in [("plain", {}), ("rtr", {"rtr": True}),
                        ("nommap", {"no_mmap": True})]:
        cells.append(Cell(label=f"ik-laguna-{label}", model=path_of(lag.path),
                          runtime="ik", gpus="0,1", ctx=32768, ubatch=512,
                          n_cpu_moe=lag.n_cpu_moe, **over))
    # The embedding modality, which has its own graph and no generation.
    emb = MODELS["lfm2-embed"]
    cells.append(Cell(label="embeddings", model=path_of(emb.path), ctx=2048,
                      embeddings=True))
    # Growth: what accumulates once the server is doing an agent's work, with
    # and without the prompt cache the estimator reserves 8 GiB for.
    for cram in [0, 8192]:
        cells.append(Cell(label=f"growth-cram{cram}", model=path_of(q4.path),
                          gpus="0", ctx=32768, cram=cram, bench=True,
                          bench_turns=40, verbose_log=False))
    # The same question for the models where host memory actually matters. A
    # 4B dense model is the one whose growth is least interesting and was the
    # only one measured; a hybrid MoE holds tens of GiB on the host, and if it
    # accumulates over an agent session nothing else here would see it.
    #
    # Turns are scaled to generation speed, not held constant: a hybrid at
    # ~3 tok/s would spend an hour on 40 turns, and growth that only appears
    # after 40 turns but not 12 is not a shape this campaign can resolve
    # anyway.
    for model in MODELS.values():
        if model.key == "qwen3-4b":
            continue  # already covered by the cram pair above
        gpus = "0,1" if "0,1" in model.gpus else "0"
        cells.append(Cell(label=f"growth-{model.key}", model=path_of(model.path),
                          runtime=model.runtimes[0], gpus=gpus, split="layer",
                          ctx=32768, cram=0, bench=True,
                          bench_turns=GROWTH_TURNS, verbose_log=False,
                          **_model_flags(model, gpus)))
    return cells


def _curve_points(model: Model) -> list[tuple[int, int, str]]:
    """The (context, ubatch, flash-attn) points a model's curve is fitted on.

    Three contexts to check the fit is linear rather than merely fitted, one
    larger batch for the terms that scale with it, and one flash-attention-off
    point. A model whose native context is below the standard sweep gets the
    same shape scaled into its own range instead of being pushed past it.
    """
    if model.max_ctx and model.max_ctx < 65536:
        top = model.max_ctx
        contexts = [max(512, top // 4), max(1024, top // 2), top]
    else:
        contexts = [8192, 32768, 65536]
    mid = contexts[1]
    points = [(c, 512, "on") for c in contexts]
    points.append((mid, 2048, "on"))
    points.append((mid, 512, "off"))
    return points


GROWTH_TURNS = 10
"""Turns every growth cell runs, held constant across models.

Growth as a function of turn count only means anything if the turn count is
the same everywhere, so this does not scale with model speed even though the
slowest model costs twenty times the fastest per turn.

Ten turns reaches roughly five thousand generated tokens. That is enough to
separate "grows per token" from "allocates once on first use", which is the
question; it is not enough to characterise a slow leak, and nothing here
claims to.
"""


def interior_points() -> list[Cell]:
    """The points that turn a two-point line into a measured relationship.

    Several terms are claimed linear on the strength of two samples, which
    cannot distinguish a line through the origin from an affine one — and the
    difference is charged to the slope, so it grows with every extrapolation.
    Each cell here adds an interior or a wider point to a relationship the
    rest of the campaign only brackets.
    """
    cells: list[Cell] = []
    # ik's CPU-MoE term is per batch token with a threshold between them; one
    # point below and one above forces the slope through the origin and leaves
    # the threshold bracketed to a factor of four.
    for key in ("qwen36-35b-a3b", "laguna", "glm52"):
        model = MODELS[key]
        if "ik" not in model.runtimes:
            continue
        for ubatch in (256, 1024):
            cells.append(Cell(
                label=f"ikmoe-{model.key}-ub{ubatch}", model=path_of(model.path),
                runtime="ik", gpus="0,1", split="layer", ctx=32768,
                ubatch=ubatch, **_model_flags(model, "0,1")))
    # The no-flash-attention term is claimed per-token but only ever sampled at
    # one batch size, where a flat 4 MiB fits exactly as well.
    for key in ("qwen3-4b", "gemma3-27b"):
        model = MODELS[key]
        cells.append(Cell(label=f"nofa-{model.key}-ub2048",
                          model=path_of(model.path), gpus="0,1", split="layer",
                          ctx=32768, ubatch=2048, flash_attn="off",
                          **_model_flags(model, "0,1")))
    # deepseek4's slope is claimed linear in ubatch from two points.
    ds = MODELS["dsv4f"]
    cells.append(Cell(label="dsv4f-ub1024", model=path_of(ds.path), gpus="0,1",
                      split="layer", ctx=32768, ubatch=1024,
                      **_model_flags(ds, "0,1")))
    # The curves are fitted to 65536 and used to 524288. One long point per
    # steep architecture bounds how far the extrapolation can drift.
    # The runtime the operator actually serves each of these with, not the
    # first in the tuple: laguna's production runtime is ik, and anchoring its
    # curve on mainline measures a combination nobody runs — and, at this
    # context, one that does not fit.
    for key, ctx, runtime in (("dsv4f", 131072, "mainline"),
                              ("glm52", 131072, "ik"),
                              ("laguna", 131072, "ik")):
        model = MODELS[key]
        cells.append(Cell(
            label=f"longctx-{model.key}-c{ctx}", model=path_of(model.path),
            runtime=runtime, gpus="0,1", split="layer", ctx=ctx,
            **_model_flags(model, "0,1")))
    # The embedded-MTP constant is a one-model number, and the second model
    # that runs it in production differs in kv-head count — the factor the KV
    # formula multiplies by and which one model cannot verify.
    q35 = MODELS["qwen36-35b-a3b"]
    for spec, name in ((None, "none"), ("draft-mtp", "embedded")):
        cells.append(Cell(label=f"mtp-{name}-35b", model=path_of(q35.path),
                          gpus="0,1", split="tensor", ctx=32768,
                          spec_type=spec, **_model_flags(q35, "0,1")))
    return cells


def interactions() -> list[Cell]:
    """Whether the curves hold in the regime production actually runs.

    Every curve cell is f16, one slot, layer split. Production is q8_0, two to
    four slots, tensor split. Fitting in one regime and serving in another is
    only safe if the curve's *slope* does not depend on those settings — which
    is an assumption nobody has tested, and one whose failure would appear as
    unexplained holdout error with no cell to attribute it to.

    Two contexts per variant is the minimum that can distinguish "shifts the
    base" from "changes the slope"; one point could only see the former.
    """
    cells: list[Cell] = []
    for key in ("qwen3-4b", "gemma3-27b", "qwen36-27b", "qwen36-35b-a3b"):
        model = MODELS[key]
        if model.kv_types == ("f16",):
            continue
        for ctx in (8192, 65536):
            cells.append(Cell(
                label=f"interact-{model.key}-c{ctx}-q8", model=path_of(model.path),
                gpus="0,1", split="layer", ctx=ctx, kv_type="q8_0",
                **_model_flags(model, "0,1")))
            cells.append(Cell(
                label=f"interact-{model.key}-c{ctx}-np4", model=path_of(model.path),
                gpus="0,1", split="layer", ctx=ctx, parallel=4, kv_unified=True,
                **_model_flags(model, "0,1")))
            cells.append(Cell(
                label=f"interact-{model.key}-c{ctx}-tensor", model=path_of(model.path),
                gpus="0,1", split="tensor", ctx=ctx,
                **_model_flags(model, "0,1")))
    return cells


def review_followup() -> list[Cell]:
    """The cells the two constants held under review actually need.

    Both were left unchanged after the first campaign because the evidence
    said the current value is wrong without saying what is right.

    deepseek4's curve carries a slope of 66 MiB per 1024 tokens that the
    production hybrid does not show — flat across a sixteenfold range of
    context. But every cell measuring that was the hybrid, with 40 of 43
    layers on the CPU. If the original calibration measured a GPU-resident
    configuration then both figures are correct and the curve is being applied
    outside the regime it was fitted in, which is a different fix from a wrong
    number. Sweeping the offload axis at two contexts separates them: if VRAM
    climbs with context once layers are resident, the curve is right and
    misapplied; if it stays flat, the slope is wrong.

    MTP's compute constant was fitted against llama.cpp's own `[spec]` log
    line, which reports a quantity four times smaller than the driver delta
    between paired with- and without-MTP cells. Correcting it needs those
    pairs at more shapes than the first campaign ran, and on both models that
    carry an embedded head, since they differ in the kv-head count the KV
    formula multiplies by.
    """
    cells: list[Cell] = []
    ds = MODELS["dsv4f"]
    for n_cpu_moe in (0, 20, 40):
        for ctx in (8192, 65536):
            cells.append(Cell(
                label=f"ds4-offload{n_cpu_moe}-c{ctx}", model=path_of(ds.path),
                gpus="0,1", split="layer", ctx=ctx,
                **{**_model_flags(ds, "0,1"), "n_cpu_moe": n_cpu_moe or None}))
    # Paired with/without MTP on both embedded-head models and the separate
    # draft, at a second context each, so the constant is fitted against the
    # driver delta rather than against a log line.
    for key, ctx in (("qwen36-27b", 65536), ("qwen36-35b-a3b", 65536),
                     ("gemma4-31b-qat", 65536)):
        model = MODELS[key]
        draft = path_of(model.draft) if model.draft else None
        for spec, name in ((None, "none"), ("draft-mtp", "mtp")):
            cells.append(Cell(
                label=f"mtprev-{model.key}-{name}-c{ctx}", model=path_of(model.path),
                gpus="0,1", split="tensor", ctx=ctx, spec_type=spec,
                draft=draft if spec else None, **_model_flags(model, "0,1")))
    return cells


def mtp_slots() -> list[Cell]:
    """Separate MTP's slot count from its context.

    The first campaign left `MTP_COMPUTE_MIB` under review because the paired
    with- and without-MTP cells disagreed with the constant by a factor of
    four, and the disagreement grew at `parallel = 4`. But the design cannot
    say that: every one-slot pair sits at ctx 32768 or 65536 and the only
    four-slot pair sits at 131072, so slots and context are confounded and the
    "slot dependence" may be nothing but the longer context.

    Context is therefore held fixed and only `parallel` moves. Both models
    with an embedded head are swept, since they differ in the kv-head count
    the KV formula multiplies by, and the separate-draft model is swept too
    because its overhead has no context-scaling term at all — its draft shares
    the target's KV cache — so it is the control: if its delta moves with
    slots, the cause is not the MTP KV.
    """
    cells: list[Cell] = []
    for key in ("qwen36-27b", "qwen36-35b-a3b", "gemma4-31b-qat"):
        model = MODELS[key]
        draft = path_of(model.draft) if model.draft else None
        for parallel in (1, 2, 4):
            for spec, name in ((None, "none"), ("draft-mtp", "mtp")):
                cells.append(Cell(
                    label=f"mtpslot-{model.key}-{name}-np{parallel}",
                    purpose=("mtp-slots",), model=path_of(model.path),
                    gpus="0,1", split="tensor", ctx=32768, parallel=parallel,
                    kv_unified=True, spec_type=spec,
                    draft=draft if spec else None,
                    **_model_flags(model, "0,1")))
    return cells


def flash_attention_off() -> list[Cell]:
    """The no-flash-attention regime, over enough shapes to fit a term.

    Nineteen cells ran with it off and seventeen of them sit at exactly ctx
    32768, ubatch 512, one slot. That measures the *offset* at one point and
    says nothing about how it scales, which is why the regime is excluded from
    every derivation rather than modelled: it costs 30 to 254 MiB of host
    residual depending on the architecture, and it is the single largest
    remaining compute over-reservation.

    Both axes the mask depends on are swept — the KQ mask is `n_kv x n_tokens`
    and loses its f16 packing when flash attention is off, so the term should
    be linear in both context and batch. Four architectures, chosen for the
    mask shapes that differ: interleaved SWA, plain causal, full MHA, and the
    embedding model whose no-flash-attention residual is the largest measured.

    Each cell has a flash-attention-on twin already in the dataset from the
    curve sweep, so the pairs are formed without measuring the twins again.
    """
    cells: list[Cell] = []
    for key in ("gemma3-27b", "qwen36-27b", "magidonia-24b", "lfm2-embed"):
        model = MODELS[key]
        gpus = model.gpus[-1]
        for ctx, ubatch in ((8192, 512), (32768, 512), (131072, 512),
                            (32768, 2048), (8192, 2048)):
            if model.max_ctx and ctx > model.max_ctx:
                continue
            cells.append(Cell(
                label=f"faoff-{model.key}-c{ctx}-ub{ubatch}",
                purpose=("flash-attention",), model=path_of(model.path),
                gpus=gpus, split=_splits_for(model, "mainline")[0], ctx=ctx,
                ubatch=ubatch, flash_attn="off", **_model_flags(model, gpus)))
    return cells


def slot_batch() -> list[Cell]:
    """The slot rules at a second batch size.

    Every cell with `parallel > 1` or `--kv-unified` was measured at ubatch
    512, and both feed rules that multiply terms which scale with the batch:
    the stream division that sizes the KQ mask, and the three window masks an
    interleaved-SWA model builds when slots share one cache. A rule that is
    wrong in its batch dependence is invisible at one batch size.

    That is exactly how flash-attention-off spent the first campaign recorded
    as an inconsistent baseline shift when it is a clean per-token rate — the
    cells that would have shown it all sat at one ubatch. This is the same
    hole in the two remaining places it exists.

    An SWA model and a plain causal one, since only the former exercises the
    window-mask rule.
    """
    cells: list[Cell] = []
    for key in ("gemma3-27b", "qwen3-4b"):
        model = MODELS[key]
        for parallel, unified in ((4, True), (4, False), (1, False)):
            for ubatch in (512, 2048):
                cells.append(Cell(
                    label=f"slotbatch-{model.key}-np{parallel}"
                          f"{'-unified' if unified else ''}-ub{ubatch}",
                    purpose=("slot-batch",), model=path_of(model.path),
                    gpus="0,1", split="layer", ctx=32768, ubatch=ubatch,
                    parallel=parallel, kv_unified=unified,
                    **_model_flags(model, "0,1")))
    return cells


def replication() -> list[Cell]:
    """Repeats in the regimes the noise floor never visited.

    The floor is five repeats of one small dense model on one card with a hot
    page cache. It licenses a significance claim for approximately that cell.
    A hybrid under page-cache pressure, an ik no-mmap load, and a two-card
    tensor split are all obviously noisier, and every per-model constant is
    otherwise a single sample.

    The repeats are spread across the run rather than run back to back, so a
    monotone drift — thermal, fragmentation, page-cache composition — shows up
    as a difference between them instead of loading onto model size, which the
    smallest-first ordering would otherwise confound it with.
    """
    cells: list[Cell] = []
    for repeat in range(3):
        cells.append(Cell(label=f"repeat-laguna-hybrid-{repeat}",
                          model=path_of(MODELS["laguna"].path), runtime="ik",
                          gpus="0,1", split="layer", ctx=32768, repeat=repeat,
                          **_model_flags(MODELS["laguna"], "0,1")))
        cells.append(Cell(label=f"repeat-gemma3-tensor-{repeat}",
                          model=path_of(MODELS["gemma3-27b"].path), gpus="0,1",
                          split="tensor", ctx=32768, repeat=repeat))
    return cells


def concurrency() -> list[Cell]:
    """Per-slot state, which no other cell allocates.

    Production runs `parallel` 2-4. Every other cell here sends strictly
    sequential requests, so only the first slot is ever touched and the rest
    stay unallocated — `soak` with `concurrency` is what reaches them, and
    until now nothing set either.
    """
    return [
        Cell(label=f"slots-np{np}-c{conc}", model=path_of(MODELS["qwen36-27b"].path),
             gpus="0,1", split="layer", ctx=32768, parallel=np,
             kv_unified=unified, soak=6, concurrency=conc)
        for np, conc, unified in ((4, 4, False), (4, 4, True), (2, 2, False))
    ]


def concurrency_models() -> list[Cell]:
    """The per-slot cost, across architectures rather than one.

    Qwen3.6-27B holds 602, 767, and 1083 MiB of anonymous memory at one, two,
    and four *concurrent* requests, all else equal — about 162 MiB per
    additional active slot, linear. Slots that stay idle cost nothing: the
    same model at `parallel` 1, 2, and 4 with a single sequential probe reads
    716 MiB at every one. A reservation has to assume every slot can become
    active, so this belongs in the model.

    It is measured on exactly one architecture, at one context and one split.
    That is the coverage that has produced a wrong constant three times in
    this campaign — the flash-attention rate, the shared-cache window mask,
    and the separate-draft compute all looked flat until a second point in the
    axis that mattered. So the term is measured across architectures before it
    is modelled, not fitted from the one series and generalised.

    An interleaved-SWA model, a plain causal one, and the one with the
    existing series as a control.
    """
    return [
        Cell(label=f"conc-{MODELS[key].key}-c{conc}", purpose=("concurrency",),
             model=path_of(MODELS[key].path), gpus="0,1", split="layer",
             ctx=32768, parallel=4, kv_unified=True, soak=6, concurrency=conc,
             **_model_flags(MODELS[key], "0,1"))
        for key in ("gemma3-27b", "magidonia-24b", "qwen36-27b")
        for conc in (1, 2, 4)
    ]


def loose_ends() -> list[Cell]:
    """Two claims this campaign asserted on data outside the dataset.

    The flash-attention term is divided by the stream count, on the strength
    of hardcoded Qwen3-4B points in a unit test from an earlier sweep: every
    such cell in `measurements.ndjson` runs one slot, so nothing here can
    falsify it. If the division is right these cells show a quarter of the
    one-slot rate; if it is wrong they show the whole of it.

    And the per-slot host cost was measured at one context and one batch. It
    is reserved as slop rather than charged to the correction, so an error is
    less costly — but "measured at one point in the axis" is what made three
    other constants wrong here, so it gets a second batch size.
    """
    cells: list[Cell] = []
    for key in ("gemma3-27b", "qwen36-27b"):
        model = MODELS[key]
        cells.append(Cell(
            label=f"faoff-slots-{model.key}-np4", purpose=("flash-attention",),
            model=path_of(model.path), gpus="0,1", split="layer", ctx=32768,
            ubatch=512, parallel=4, flash_attn="off",
            **_model_flags(model, "0,1")))
    model = MODELS["gemma3-27b"]
    for conc in (1, 4):
        cells.append(Cell(
            label=f"conc-ub2048-{model.key}-c{conc}", purpose=("concurrency",),
            model=path_of(model.path), gpus="0,1", split="layer", ctx=32768,
            ubatch=2048, parallel=4, kv_unified=True, soak=6, concurrency=conc,
            **_model_flags(model, "0,1")))
    return cells


def single_card_curves() -> list[Cell]:
    """Context and batch on one card, where the mask is not replicated.

    Seven of eleven architectures have exactly one single-card point, at ctx
    32768 and ubatch 512. Everything else about the arena — the mask copies
    above all — is fitted from two-card cells, so a rule that is right at four
    copies and wrong at one would show up nowhere: the copy factor multiplies
    terms that scale with both context and batch, and at a single point any
    factor can be made to fit.

    That is the same hole that made the flash-attention rate, the shared-cache
    window mask, and the separate-draft compute wrong here, and once more the
    axis is the one the rule is about.
    """
    return [
        Cell(label=f"onecard-{MODELS[key].key}-c{ctx}-ub{ubatch}",
             purpose=("curves",), model=path_of(MODELS[key].path), gpus="0",
             split="layer", ctx=ctx, ubatch=ubatch,
             **_model_flags(MODELS[key], "0"))
        for key in ("gemma3-27b", "magidonia-24b")
        for ctx, ubatch in ((8192, 512), (65536, 512), (32768, 2048))
    ]


def device_scaling() -> list[Cell]:
    """Separate the per-device CUDA cost from everything that scales with model.

    The `gpus` axis elsewhere varies placement *and* device count together, so
    it cannot say which of the two moved a number. These cells pin placement
    to the CPU — `-ngl 0`, no weights on any card — and vary only how many
    CUDA contexts get initialised. The difference between them is the host
    cost of a visible device and nothing else.

    That difference is the term the estimator does not have. `PROCESS_BASE_BYTES`
    is a compiled scalar fitted on a two-card box; an operator with four or
    eight cards inherits it wrong by an increment nobody has measured. Three
    cells per model, on three models of different shape, establish whether the
    increment is constant and whether it is model-independent.
    """
    cells: list[Cell] = []
    for key in ("qwen3-4b", "gemma3-27b", "qwen36-35b-a3b"):
        model = MODELS[key]
        for gpus, name in [("", "none"), ("0", "one"), ("0,1", "two")]:
            cells.append(Cell(
                label=f"devices-{model.key}-{name}",
                model=path_of(model.path), gpus=gpus, ngl=0, ctx=32768,
                split=None, extra=model.extra, threads=model.threads))
    # Whether the CPU-side terms depend on core count, which every contributor
    # with a different CPU inherits blind.
    for threads in (8, 16, 32):
        cells.append(Cell(label=f"threads-{threads}-laguna",
                          model=path_of(MODELS["laguna"].path), runtime="ik",
                          gpus="0,1", split="layer", ctx=32768, threads=threads,
                          n_cpu_moe=MODELS["laguna"].n_cpu_moe))
    return cells


def phase5_holdout() -> list[Cell]:
    """The operator's real service configurations, held out of every fit.

    Every accuracy figure quoted so far has been in-sample. These are the
    configurations the daemon actually runs; predicting them before measuring
    is the only test that says whether the model generalises.
    """
    return [
        Cell(label="prod-gemma4-31b-qat", model=path_of(MODELS["gemma4-31b-qat"].path),
             mmproj=path_of(MODELS["gemma4-31b-qat"].mmproj),
             draft=path_of(MODELS["gemma4-31b-qat"].draft), spec_type="draft-mtp",
             gpus="0,1", split="tensor", ctx=240000, parallel=4, kv_unified=True,
             kv_type="f16", cram=0, extra=("-n", "16384")),
        Cell(label="prod-qwen36-27b", model=path_of(MODELS["qwen36-27b"].path),
             mmproj=path_of(MODELS["qwen36-27b"].mmproj), spec_type="draft-mtp",
             gpus="0,1", split="tensor", ctx=360000, parallel=2, kv_type="q8_0"),
        Cell(label="prod-qwen36-35b-a3b", model=path_of(MODELS["qwen36-35b-a3b"].path),
             mmproj=path_of(MODELS["qwen36-35b-a3b"].mmproj), spec_type="draft-mtp",
             gpus="0,1", split="tensor", ctx=524288, parallel=2, kv_type="q8_0",
             n_cpu_moe=MODELS["qwen36-35b-a3b"].n_cpu_moe),
        Cell(label="prod-laguna", model=path_of(MODELS["laguna"].path), runtime="ik",
             gpus="0,1", ctx=131072, batch=2048, ubatch=2048, parallel=1,
             kv_type="q8_0", no_mmap=True, threads=24, numa="distribute",
             n_cpu_moe=MODELS["laguna"].n_cpu_moe),
        Cell(label="prod-glm52", model=path_of(MODELS["glm52"].path), runtime="ik",
             gpus="0,1", ctx=131072, batch=2048, ubatch=2048, parallel=1,
             no_mmap=True, threads=24, n_cpu_moe=MODELS["glm52"].n_cpu_moe,
             extra=("-mla", "1", "-dsa", "-amb", "512")),
        Cell(label="prod-dsv4f", model=path_of(MODELS["dsv4f"].path),
             gpus="0,1", ctx=131072, parallel=1, ubatch=512,
             n_cpu_moe=MODELS["dsv4f"].n_cpu_moe),
        Cell(label="prod-talkie", model=path_of(MODELS["talkie-13b"].path), ctx=2048),
    ]


def all_cells() -> list[Cell]:
    """Every configuration worth measuring, once each, in the cheapest order.

    The questions are separate — a noise floor, a per-model baseline, a
    context curve, a fork comparison, growth — but they are not separate
    *schedules*. Running them as separate passes reloads each model once per
    question, and a reload is the single most expensive thing here: the 205
    GiB production quant cannot even stay in the page cache alongside
    anything else, so every revisit pays full disk cost again.

    So the questions become tags and the schedule becomes one list, ordered so
    consecutive cells disturb as little as possible: all of a model's work
    happens while its weights are hot, and models run smallest first, because
    the largest evicts everything behind it on the way past.
    """
    seen: dict[str, Cell] = {}
    for name, build in QUESTIONS.items():
        for cell in build():
            existing = seen.get(cell.cell_id)
            if existing is None:
                seen[cell.cell_id] = dataclasses.replace(cell, purpose=(name,))
            elif name not in existing.purpose:
                seen[cell.cell_id] = dataclasses.replace(
                    existing, purpose=existing.purpose + (name,))
    return sorted(seen.values(), key=_disturbance)


def _disturbance(cell: Cell) -> tuple:
    """Sort key: what it costs to move from one cell to the next.

    Changing the model is the expensive move, so it is the outermost key and
    models are ordered by size. Everything below it — runtime, then the load
    path, then placement, then the cache and batch knobs — is progressively
    cheaper to vary while the same weights stay resident.
    """
    return (
        _model_size(cell.model),
        cell.model,
        cell.runtime,
        cell.no_mmap, cell.rtr, cell.thp,
        cell.gpus, cell.split or "", cell.ngl,
        cell.spec_type or "", cell.draft or "", cell.mmproj or "",
        cell.ctx, cell.ubatch, cell.parallel, cell.kv_type,
        cell.flash_attn, not cell.served, cell.bench,
    )


_SIZE_CACHE: dict[str, int] = {}


def _model_size(path: str | None) -> int:
    """Total bytes across a model's shards, cached; unreadable sorts last."""
    if not path:
        return 0
    if path not in _SIZE_CACHE:
        first = Path(path)
        if not first.exists():
            _SIZE_CACHE[path] = 1 << 62
        else:
            stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first.name)
            shards = ([first] if stem == first.name
                      else sorted(first.parent.glob(f"{stem}-*-of-*.gguf")))
            _SIZE_CACHE[path] = sum(s.stat().st_size for s in shards)
    return _SIZE_CACHE[path]


# What each cell is for. These are questions, not a schedule — `all_cells`
# merges them into one ordered run, and a cell wanted by two questions is
# measured once and tagged with both.
QUESTIONS = {
    "noise": phase0_noise,
    "factor-screen": phase1_factorial,
    "model-baseline": phase2_models,
    "curves": phase2b_curves,
    "fork": phase3_fork,
    "switches": phase4_special,
    "device-scaling": device_scaling,
    "interior": interior_points,
    "interactions": interactions,
    "replication": replication,
    "concurrency": concurrency,
    "review-followup": review_followup,
    "mtp-slots": mtp_slots,
    "slot-batch": slot_batch,
    "concurrency-models": concurrency_models,
    "loose-ends": loose_ends,
    "single-card": single_card_curves,
    "flash-attention": flash_attention_off,
    "holdout": phase5_holdout,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(QUESTIONS) + ["all"])
    args = parser.parse_args()
    cells = (all_cells() if args.phase == "all" else QUESTIONS[args.phase]())
    json.dump([{k: v for k, v in dataclasses.asdict(c).items()} for c in cells],
              sys.stdout, indent=1, default=list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
