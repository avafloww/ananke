"""One per-device compute model, fitted per (runtime, split, architecture).

This replaces three separate mechanisms — `compute_buffer_curves` for a
mainline layer split, `tensor_compute_curves` plus its intermediates/quantised/
shadow tables for a tensor split, and the four `ik_compute_*` rate tables —
with a single design matrix. They were three because each was derived against
a different target: llama.cpp's `compute` column plus `unaccounted` for one,
the fused `Meta()` row's `compute` column alone for another, and the driver
total minus a modelled remainder for ik, which prints no breakdown table at
all. Three targets meant three shapes, and a cell that fell between them was
covered by whichever mechanism claimed it rather than by the one that described
it.

The target here is uniform and per *device*:

    target = gpu{card}_used_mib - model_share - kv_share

It is available for every cell, ik's included. ik prints no breakdown table at
all, but the same quantity is recoverable from the driver total less the weights
and context its own buffer lines name, over the number of cards — which is what
`table_less_compute` does, and what the caller injects as `table_less_target`.
Those cells enter as one averaged observation each, exactly as a fused tensor
cell does.

A layer split reports one breakdown row per card, so `model_share` and
`kv_share` are that card's own and each card contributes an observation.

A tensor split reports a single fused row whose `total`, `free`, and `self` are
summed across cards but whose `model`, `kv`, and `compute` columns are one
card's share. Both cards therefore get charged the *same* share, and where the
real split is uneven — a mixture of experts under a tensor split need not divide
evenly — the whole difference lands in the residual with opposite signs on the
two cards. gemma-4-26B-A4B reads 184 MiB on device 0 against 490 on device 1 at
identical settings, which is not 306 MiB of extra compute on one card but the
fused share being wrong for both. So a fused cell contributes one observation at
the per-card *average*, which is the quantity the fused row's own semantics
support and the quantity the packer charges per spanned GPU.

## Columns

Every column is dimensionally normalised, so models of different width share
coefficients instead of each needing its own absolute flat term. `hidden` is
the one that matters most: graph intermediates are some count of hidden-width
f32 buffers per batch token, so the column is `n_embd * ubatch` and the
coefficient is that count times four bytes. Fitting three gemma4 widths against
a shared *absolute* term is what previously left that group at 149% error.

    flat          1                            CUDA context, workspaces, graph metadata
    head_flat     head_share                   the head card's output graph and scratch
    hidden        n_embd * ub                  per-token hidden-width intermediates
    doubling      log2(max(ub, CHUNK)/CHUNK)   ik's step per batch doubling above -amb
    mask          copies * ub * n_kv           the attention mask and its copies
    quant         ub * ctx                     dequantisation of a quantised cache
    logits        head_share * n_vocab * ub    the head card also materialises logits
    offload_head  head_share when offloading   expert staging, which the head card holds

`head_share` is 1 on the primary card and 0 elsewhere for a layer split, and
`1 / cards` for a fused tensor row, whose target is already an average over the
cards it spans.

`flat`, `head_flat`, `doubling`, and `offload_flat` carry MiB; the rest carry
bytes per element, the columns being pre-divided by 2^20.

`head_flat` is not bookkeeping for `logits`, which is already per-head and
scales with the batch. It is there because the head card's cost is measurably
*flat*: deepseek4 on two cards holds 2359 MiB on device 0 at ctx 8192 ub 512 and
2381 at ctx 32768 ub 2048, while device 1 moves from 585 to 1287 across the same
pair. Without the column that asymmetry has nowhere to go and is paid for by
inflating terms that do scale, which is what left the group at 65%.

The offload axis enters as that one boolean interaction rather than through the
`n_cpu_moe` count, because the count carries no additional signal. Columns
proportional to it — a flat per-offloaded-layer cost and a per-layer per-token
one — were fitted and changed nothing: 151 of 633 observations outside +/-5% with
them and 151 without, at a marginally better median. Dropping `offload_head`
instead takes qwen35moe from 11% to 71%. So the effect is real and the head card
is where it lands, but it does not scale with how many layers are offloaded, and
carrying the count would mean threading a placement outcome back into the
estimate that feeds placement for no measured gain.

`offload_head` exists because expert offload, not the output head, is what makes
a layer split asymmetric. Qwen3.6-35B-A3B fully resident on two cards holds 428
MiB on each, while the same model under `--n-cpu-moe 40` holds visibly more on
the primary — so a plain `head_flat` had to average the two and missed both by
41%. Gathering and scattering CPU-resident expert activations stages through the
primary device, which is where the buffers land.

`mask` carries its replication count rather than leaving the fit to discover it.
Mainline replicates the graph's masks a fixed number of times when layers are
split across more than one device — separately derived as
`MAINLINE_LAYER_SPLIT_MASK_COPIES`, 99 layer-split cells at 4.00 against 147
single-card and tensor-split cells at 1.00, flat across context, batch, slot
count, and cache mode — and the caller passes that count in. Pooling the two
card counts into one unreplicated column instead had the fit report 7.65 bytes
per token-pair for a buffer whose element is an f16. With the count supplied,
the coefficient is free to land on 2, which is then a check on the model rather
than a parameter of it.

## What is deliberately not in it

Flash-attention-off cells are excluded and keep their own paired deriver. The
unfused score matrix is worth thousands of MiB against cells whose other terms
are hundreds, so fitting it here lets one term dominate a model it is not part
of. The estimator adds it on top.

Speculative-decoding cells are excluded because the MTP overhead has its own
derived model and would otherwise be counted twice. Vision cells are excluded
for the same reason: the mmproj weights and CLIP graph buffer are charged
separately as `MMPROJ_GRAPH_BYTES`, and leaving them in put gemma-4-31B-it-qat
at 1870 MiB on its primary card against the 26B's 184 under otherwise identical
settings, which the fit could only split the difference on.

Rows whose card holds no layers are excluded. A card holding no layers is not
doing compute — its whole cost is the bare CUDA context, which several groups
show as exactly 256 MiB — and those rows inform the per-device shadow instead.
Leaving them in made the fit predict 484 MiB where the card held 256.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple

# ik_llama's attention chunk (`-amb`, default 512). Above it the compute buffer
# grows by a constant per doubling of the batch rather than proportionally.
CHUNK = 512

COLUMNS = ("flat", "head_flat", "hidden", "doubling", "mask",
           "quant", "logits", "offload_head")

# Greedy selection order. A column joins the design only if the system stays
# solvable with it, so the order decides which of two collinear columns wins.
# Terms that generalise across architectures come first; `doubling` comes last
# because it is a pure function of the batch and so is collinear with `hidden`
# in any group holding a single model width and only two batch sizes.
PRIORITY = ("flat", "mask", "quant", "hidden", "head_flat",
            "logits", "offload_head", "doubling")

# How many rows a group must hold beyond the columns it is fitting. Checked
# against the *selected* columns, not all of them: a group with few cells fits a
# short design honestly, and testing against the full column list benched four
# groups that had plenty of rows for the three or four columns they varied.
MIN_SURPLUS_ROWS = 3


def columns(*, ubatch: int, n_kv: int, ctx: int, quantised: bool,
            head_share: float, n_vocab: int, n_embd: int, offloaded: int,
            mask_copies: int) -> dict[str, float]:
    """The design row for one device, keyed by column name.

    This is the single source of truth for the model's shape: the fitter builds
    its design matrix from it and the estimator evaluates the dot product of a
    group's coefficients against it, so a new axis is one edit rather than two
    that can drift apart.
    """
    return {
        "flat": 1.0,
        "head_flat": head_share,
        "hidden": n_embd * ubatch / 2**20,
        "doubling": math.log2(max(ubatch, CHUNK) / CHUNK),
        "mask": mask_copies * ubatch * n_kv / 2**20,
        "quant": (ubatch * ctx / 2**20) if quantised else 0.0,
        "logits": head_share * n_vocab * ubatch / 2**20,
        "offload_head": head_share if offloaded else 0.0,
    }


class Group(NamedTuple):
    """What a single set of coefficients is fitted against.

    The variant is carried separately rather than folded into the architecture
    string. Concatenating them made `gemma4` + `e` indistinguishable from an
    architecture literally named `gemma4e`, and splitting it back off cost the
    arch its trailing letter.
    """

    runtime: str
    split: str
    arch: str
    variant: str | None


class Row:
    """One measured device, with its design row and what it actually held."""

    __slots__ = ("cols", "target", "tag")

    def __init__(self, cols: dict[str, float], target: float, tag: dict):
        self.cols = cols
        self.target = target
        self.tag = tag


def collect(rows: list[dict], *, split_mask_copies: int,
            table_less_target=None,
            include_spec: bool = False) -> dict[Group, list[Row]]:
    """Group every usable per-device observation by runtime, split, arch, and variant.

    `split_mask_copies` is how many copies of the attention mask mainline holds
    when layers span more than one device, derived separately as
    `MAINLINE_LAYER_SPLIT_MASK_COPIES`.

    `table_less_target` recovers the per-device target for a cell whose runtime
    prints no memory breakdown. It is injected rather than reimplemented here so
    that ik's cells and the reporting that predates this model agree by
    construction.
    """
    groups: dict[Group, list[Row]] = defaultdict(list)
    for record in rows:
        factors, parsed, rss = record["factors"], record["parsed"], record["rss"]
        arch, n_embd = parsed.get("arch"), parsed.get("n_embd")
        if not arch or not n_embd:
            continue
        devices = parsed.get("devices") or []
        if factors["flash_attn"] == "off":
            continue
        if factors["spec_type"] and not include_spec:
            continue
        if factors.get("mmproj"):
            continue
        cards = [c for c in factors["gpus"].split(",") if c]
        fused = bool(devices) and devices[0]["device"].startswith("Meta")
        ubatch = factors["ubatch"] or 512
        streams = 1 if factors["kv_unified"] else max(1, factors["parallel"] or 1)
        # The E-variants keep their embeddings per layer on the host, which is a
        # different graph under the same architecture string.
        variant = "gemma_e" if parsed.get("per_layer_token_embd") else None
        split = factors["split"] or "layer"
        # A tensor split reports one card's share, so its mask is unreplicated
        # in the figure regardless of card count; only a layer split spanning
        # more than one device pays for copies. A hybrid does not replicate them
        # either — measured at 1.00 against 4.00 — so the replication follows the
        # placement, not the model.
        hybrid = bool(factors["n_cpu_moe"])
        copies = (split_mask_copies
                  if split == "layer" and len(cards) > 1 and not hybrid else 1)
        key = Group(factors["runtime"], split, arch, variant)
        readings = [(index, rss.get(f"gpu{card}_used_mib"))
                    for index, card in enumerate(cards)]
        # Keyed by the *physical* id: the sampler records `gpu<id>_used_mib`
        # while the loader's device rows are in visible order, so a cell pinned
        # to GPU 1 has its usage under `gpu1_used_mib` and its breakdown row
        # under `CUDA0`.
        readings = [(index, used) for index, used in readings if used]
        if not readings:
            continue
        shared = dict(ubatch=ubatch, n_kv=factors["ctx"] // streams,
                      ctx=factors["ctx"], quantised=factors["kv_type"] != "f16",
                      n_vocab=parsed.get("n_vocab") or 0, n_embd=n_embd,
                      offloaded=factors["n_cpu_moe"] or 0, mask_copies=copies)
        tag = {"ctx": factors["ctx"], "ubatch": ubatch,
               "kv_type": factors["kv_type"], "cards": len(cards),
               "model": record["provenance"]["model_key"],
               "parallel": factors["parallel"], "n_cpu_moe": factors["n_cpu_moe"],
               "mmproj": bool(factors.get("mmproj")),
               "label": factors.get("label") or ""}
        if not devices:
            # No breakdown table — ik. One averaged observation, recovered from
            # the driver total and the runtime's own buffer lines.
            target = table_less_target(record) if table_less_target else None
            if not target or target <= 0:
                continue
            groups[key].append(Row(
                columns(head_share=1.0 / len(readings), **shared),
                target, tag | {"device": -1}))
            continue
        if fused:
            device = devices[0]
            if not device["model_mib"]:
                continue
            targets = [used - device["model_mib"] - device["kv_mib"]
                       for _, used in readings]
            if any(t <= 0 for t in targets):
                continue
            groups[key].append(Row(
                columns(head_share=1.0 / len(targets), **shared),
                sum(targets) / len(targets), tag | {"device": -1}))
            continue
        for index, used in readings:
            device = devices[index] if index < len(devices) else None
            # A card holding no layers is not doing compute: its whole cost is
            # the bare CUDA context, and those rows inform the per-device shadow
            # rather than this model.
            if device is None or not device["model_mib"]:
                continue
            target = used - device["model_mib"] - device["kv_mib"]
            if target <= 0:
                continue
            groups[key].append(Row(
                columns(head_share=1.0 if index == 0 else 0.0, **shared),
                target, tag | {"device": index}))
    return groups


def fit(points: list[Row]) -> tuple[dict[str, float], float] | None:
    """Coefficients for one group and its worst residual as a fraction.

    Weighted by 1/y, because the criterion is relative: a 200 MiB miss on a
    5 GiB card is the failure and the same miss on a 40 GiB card is not.
    Unweighted, the largest cells dominate the sum of squares and the small ones
    fit terribly.

    Constrained to non-negative coefficients, which is not a regularisation
    convenience but the physics: every column counts bytes of a buffer that
    either exists or does not, so a negative coefficient is always the fit
    paying for one column's error with another's. Unconstrained, `logits` came
    out negative for eleven of nineteen groups — it is near-collinear with
    `hidden`, both being proportional to the batch, and separated only by which
    card is the head — and `quant` went negative wherever a quantised cache
    happened to correlate with something else. The active set is found by
    dropping the most negative column and re-solving until none remain, which is
    Lawson-Hanson's elimination step without its exchange step: enough here,
    because the columns are few and a dropped one is genuinely unidentifiable
    rather than merely awkward.
    """
    live: list[str] = []
    for name in PRIORITY:
        # A column that does not vary within the group is perfectly collinear
        # with `flat`, which makes the system singular and its coefficient
        # meaningless. `flat` itself is kept, being the intercept.
        if name != "flat" and len({round(p.cols[name], 9) for p in points}) == 1:
            continue
        if len(points) < len(live) + 1 + MIN_SURPLUS_ROWS:
            break
        if _solve_weighted(points, live + [name]) is not None:
            live.append(name)
    if not live:
        return None
    coefficients = _solve_non_negative(points, live)
    if coefficients is None:
        return None
    worst = max(abs(_evaluate(coefficients, p.cols) - p.target) / p.target
                for p in points)
    return coefficients, worst


def _solve_non_negative(points: list[Row], live: list[str]) -> dict[str, float] | None:
    active = list(live)
    while active:
        solution = _solve_weighted(points, active)
        if solution is None:
            return None
        worst = min(range(len(active)), key=lambda i: solution[i])
        if solution[worst] >= 0:
            return dict(zip(active, solution))
        active.pop(worst)
    return None


def evaluate(coefficients: dict[str, float], cols: dict[str, float]) -> float:
    """The modelled per-device compute, in MiB."""
    return _evaluate(coefficients, cols)


def _evaluate(coefficients: dict[str, float], cols: dict[str, float]) -> float:
    return sum(value * cols[name] for name, value in coefficients.items())


def _solve_weighted(points: list[Row], live: list[str]) -> list[float] | None:
    weights = [1.0 / p.target for p in points]
    normal = [[sum(w * p.cols[a] * p.cols[b] for p, w in zip(points, weights))
               for b in live] for a in live]
    right = [sum(w * p.cols[a] * p.target for p, w in zip(points, weights))
             for a in live]
    return _solve(normal, right)


def _solve(matrix: list[list[float]], right: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting, or `None` if singular."""
    n = len(matrix)
    if not n:
        return None
    augmented = [matrix[i][:] + [right[i]] for i in range(n)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(augmented[r][i]))
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        if abs(augmented[i][i]) < 1e-9:
            return None
        for r in range(i + 1, n):
            factor = augmented[r][i] / augmented[i][i]
            for c in range(i, n + 1):
                augmented[r][c] -= factor * augmented[i][c]
    solution = [0.0] * n
    for i in reversed(range(n)):
        solution[i] = (augmented[i][n] - sum(augmented[i][j] * solution[j]
                                             for j in range(i + 1, n))) / augmented[i][i]
    return solution


def document_section(groups: dict[Group, list[Row]]) -> tuple[dict, list[str]]:
    """The `compute_model` section for `tuning.json`, and per-group coverage notes.

    Entries are ordered variant-guarded first, so a lookup that scans for the
    first matching architecture finds the specific graph before the general one,
    matching the convention the curve tables already use. The `default` entry
    pools every mainline layer-split observation: with the columns dimensionally
    normalised, pooling across architectures is what the design is for, and it
    gives an architecture nobody has measured a fallback derived from data rather
    than borrowed from whichever entry happened to be listed first.
    """
    entries, notes = [], []
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        points = groups[key]
        label = f"{key.runtime}/{key.split}/{key.arch}" + \
            (f"@{key.variant}" if key.variant else "")
        fitted = fit(points)
        if fitted is None:
            notes.append(f"{label}: {len(points)} row(s), not enough to fit")
            continue
        coefficients, worst = fitted
        outside = sum(1 for p in points
                      if abs(evaluate(coefficients, p.cols) - p.target) / p.target > 0.05)
        entries.append({
            "archs": [key.arch],
            "variant": key.variant,
            "runtime": key.runtime if key.runtime != "mainline" else None,
            "split": key.split,
            "coefficients": {name: round(value, 6)
                             for name, value in coefficients.items()},
            "evidence": (
                f"non-negative weighted least squares over {len(points)} "
                f"per-device observation(s); worst residual {worst * 100:.1f}%, "
                f"{outside} outside +/-5%"),
        })
        notes.append(f"{label}: {len(points)} rows, {outside} outside +/-5%, "
                     f"worst {worst * 100:.1f}%")
    entries.sort(key=lambda e: (e["variant"] is None, e["archs"], e["split"]))
    pooled = [p for key, points in groups.items()
              if key.runtime == "mainline" and key.split == "layer" for p in points]
    fitted = fit(pooled)
    if fitted is None:
        raise ValueError("no pooled mainline layer-split rows to fit a default from")
    coefficients, worst = fitted
    return {
        "$comment": (
            "One per-device compute model per (runtime, split, architecture), "
            "replacing `compute_buffer_curves`, `tensor_compute_curves` and its "
            "companion tables, and the `ik_compute_*` rates. Columns are "
            "dimensionally normalised so architectures of different width share "
            "coefficients; see scripts/calibration/compute_model.py for what each "
            "one counts and why. Coefficients are constrained non-negative "
            "because every column counts bytes of a buffer that either exists or "
            "does not. Ordered variant-guarded first."),
        "columns": list(COLUMNS),
        "entries": entries,
        "default": {
            "coefficients": {name: round(value, 6)
                             for name, value in coefficients.items()},
            "evidence": (
                f"pooled over {len(pooled)} mainline layer-split observations "
                f"across every measured architecture; worst residual "
                f"{worst * 100:.1f}%"),
        },
    }, notes
