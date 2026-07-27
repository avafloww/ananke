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

    Not every architecture supports every mode: mainline rejects
    `LLAMA_SPLIT_MODE_TENSOR` for `deepseek4` outright, at load. Planning
    cells that cannot load wastes the slot and buries the real failures.
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
              threads=24, no_mmap=True,
              note="MoE + MLA + DSA, 79L — the production quant"),
        # Every remaining llama.cpp service in the operator's config. These
        # were absent from the registry while being served in production
        # daily, which meant the campaign's "holdout" covered under half of
        # what the daemon actually runs.
        Model("gemma3-glitter-27b", "mradermacher/Gemma-3-Glitter-27B-i1-GGUF/Gemma-3-Glitter-27B.i1-Q5_K_M.gguf",
              ("mainline",), note="gemma3, 62L — a second layer count for the baseline fit"),
        Model("gemma4-31b", "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf",
              ("mainline",), mmproj="unsloth/gemma-4-31B-it-GGUF/mmproj-F16.gguf",
              note="gemma4 dense, the non-QAT sibling"),
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
              ("mainline",), embeddings=True, gpus=("0",),
              note="embedding modality; 128k native context"),
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
    for gpus, split, kv, served in product(model.gpus, model.splits,
                                           model.kv_types, [True, False]):
        cells.append(Cell(
            label=f"{model.key}-{runtime}-{gpus.replace(',', '')}-{split}-{kv}-"
                  f"{'srv' if served else 'idle'}",
            model=path_of(model.path), runtime=runtime, gpus=gpus, split=split,
            kv_type=kv, served=served, **_model_flags(model, gpus),
            **({"ctx": model.max_ctx} if model.max_ctx else {}), **over,
        ))
    return cells


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
    for label, over in [("plain", {}), ("rtr", {"rtr": True}),
                        ("thp", {"thp": True}), ("nommap", {"no_mmap": True})]:
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


GROWTH_TURNS = 40
"""Turns every growth cell runs, held constant across models.

Growth as a function of turn count only means anything if the turn count is
the same everywhere. Scaling it down for slow models — which is tempting,
since a hybrid at ~3 tok/s spends over an hour here — makes the resulting
curves incomparable, which defeats the measurement.
"""


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
        Cell(label="prod-talkie", model=path_of(MODELS["qwen3-4b"].path), ctx=2048),
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
