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
    n_cpu_moe: int | None = None
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
              ("mainline", "ik"), n_cpu_moe=30, note="MoE 256/10 + SWA 512, 48L"),
        Model("dsv4f", "unsloth/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
              ("mainline",), n_cpu_moe=40, note="MoE + MLA + NSA indexer, 43L"),
        Model("glm52", "muzzy/GLM-5.2-GGUF/IQ2_KS/GLM-5.2-smol-IQ2_KS-00001-of-00033.gguf",
              ("ik",), n_cpu_moe=92, note="MoE + MLA + DSA, 79L — the production quant"),
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


PHASES = {"phase0": phase0_noise, "phase1": phase1_factorial}


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
