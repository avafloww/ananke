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
              note="MoE + MLA + DSA, 79L — the production quant"),
        Model("lfm2-embed", "LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf",
              ("mainline",), note="embedding modality"),
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
    for gpus, split, kv, served in product(["0", "0,1"], model.splits,
                                           ["f16", "q8_0"], [True, False]):
        cells.append(Cell(
            label=f"{model.key}-{runtime}-{gpus.replace(',', '')}-{split}-{kv}-"
                  f"{'srv' if served else 'idle'}",
            model=path_of(model.path), runtime=runtime, gpus=gpus, split=split,
            kv_type=kv, served=served,
            n_cpu_moe=(model.n_cpu_moe_1gpu or model.n_cpu_moe)
            if gpus == "0" else model.n_cpu_moe, **over,
        ))
    return cells


def phase2_models() -> list[Cell]:
    """Does the per-model term follow layers, hidden size, vocabulary, or none?

    Phase 1 fixed the factor set on one model; this varies the model with that
    set held constant. Without it the baseline is a constant fitted to one
    shape and generalised — which is the mistake this campaign exists to stop
    repeating.
    """
    keys = ["qwen3-4b", "qwen36-27b", "gemma3-27b", "gemma4-31b-qat",
            "qwen36-35b-a3b", "laguna", "dsv4f"]
    return [c for k in keys for c in _significant(MODELS[k])]


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
    keys = ["qwen3-4b", "qwen36-27b", "gemma3-27b", "gemma4-31b-qat",
            "qwen36-35b-a3b", "laguna", "dsv4f"]
    for key in keys:
        model = MODELS[key]
        for runtime in model.runtimes:
            for ctx, ubatch, fa in [(8192, 512, "on"), (32768, 512, "on"),
                                    (65536, 512, "on"), (32768, 2048, "on"),
                                    (32768, 512, "off")]:
                cells.append(Cell(
                    label=f"curve-{model.key}-{runtime}-c{ctx}-ub{ubatch}-fa{fa}",
                    model=path_of(model.path), runtime=runtime, gpus="0,1",
                    split="layer", ctx=ctx, ubatch=ubatch, flash_attn=fa,
                    n_cpu_moe=model.n_cpu_moe))
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
    for label, model, spec, draft in [
        ("mtp-none-g4", g4, None, None),
        ("mtp-draft-g4", g4, "draft-mtp", path_of(g4.draft)),
        ("mtp-none-q27", q27, None, None),
        ("mtp-embedded-q27", q27, "draft-mtp", None),
    ]:
        cells.append(Cell(label=label, model=path_of(model.path), gpus="0,1",
                          split="tensor", ctx=32768, spec_type=spec, draft=draft))
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
        Cell(label="prod-talkie", model=path_of(MODELS["qwen3-4b"].path), ctx=2048),
    ]


PHASES = {
    "phase0": phase0_noise,
    "phase1": phase1_factorial,
    "phase2": phase2_models,
    "phase3": phase3_fork,
    "phase2b": phase2b_curves,
    "phase4": phase4_special,
    "phase5": phase5_holdout,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    args = parser.parse_args()
    cells = PHASES[args.phase]()
    json.dump([{k: v for k, v in dataclasses.asdict(c).items()} for c in cells],
              sys.stdout, indent=1, default=list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
