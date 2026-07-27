"""Derive the estimator's constants from the measurements.

    python analyse.py arena       # check the modelled arena against measurement
    python analyse.py --data path/to/measurements.ndjson

The arena is *modelled*, not fitted: it is arithmetic over the graph llama.cpp
builds. So it does not get a best-fit line — it either reproduces the measured
figure or the model is wrong, and the interesting output is the residual.

Everything else here is a fit, and is reported with the spread it was fitted
from rather than as a single number, because a constant quoted without its
spread invites more confidence than the data supports.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics as st
from statistics import StatisticsError
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data" / "measurements.ndjson"
TUNING_JSON = Path(__file__).parents[2] / "ananke/src/estimator/tuning.json"


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            rows.append(record)
    return rows


def pad(value: int, to: int = 256) -> int:
    return -(-value // to) * to


def arena_terms(record: dict, charge_moe: bool = True) -> tuple[float, float, float]:
    """The modelled arena, split into (mask, swa_mask, hidden) MiB.

    mainline sizes the KQ mask against one slot's share of the context unless
    the cache is unified; ik does not divide by slots at all. An interleaved
    SWA model carries a second mask sized to the window plus the batch, not to
    the window alone.
    """
    factors, parsed = record["factors"], record["parsed"]
    arch = parsed.get("arch", "")
    ctx, ubatch = factors["ctx"], factors["ubatch"]
    slots, unified = factors["parallel"], factors["kv_unified"]
    ik = factors["runtime"] == "ik"

    n_kv = ctx if (ik or unified or slots == 1) else ctx // slots
    n_kv = pad(n_kv)
    tokens = min(ctx, ubatch)
    width = 2 if factors["flash_attn"] == "on" else 4

    # MLA compresses K and V into a shared latent, so the mask is half width.
    mla = arch in ("deepseek4", "deepseek2", "glm-dsa")
    dsa = ik and factors.get("extra") and "-dsa" in factors["extra"]
    if dsa:
        # The sparse path allocates three masks and does *not* halve them:
        # measured at exactly 6.00 half-width units across two contexts and
        # two batch sizes, above the MoE threshold where nothing else
        # contaminates the figure.
        mask = n_kv * tokens * width * 3
    else:
        mask = n_kv * tokens * width // (2 if mla else 1)
    swa = parsed.get("n_swa") or 0
    # mainline sizes the second mask to the window plus the batch; ik sizes it
    # to the whole context, which is why an SWA model costs it so much more.
    swa_rows = n_kv if ik else pad(swa + tokens)
    # Three window masks when several slots share one cache, matching
    # `host_buffer::pinned_graph_bytes`. This file went on modelling one after
    # the estimator was changed, which is the same drift that left the ik MoE
    # rate stale here — and it is why `consensus` saw a 5.27 multiple among
    # cells that are otherwise 4.00.
    swa_copies = 3 if (slots > 1 and unified and not ik) else 1
    swa_mask = swa_copies * swa_rows * tokens * width if swa else 0
    # Two f32 hidden-state buffers on mainline, one on ik.
    hidden = (1 if ik else 2) * parsed["n_embd"] * tokens * 4
    # ik keeps its MoE op intermediates on the CPU below a batch threshold,
    # measured at 81 KiB per batch token — see FINDINGS.md.
    experts, used = parsed.get("n_expert") or 0, parsed.get("n_expert_used") or 0
    if charge_moe and experts and used and tokens * used < 32 * experts:
        if ik:
            hidden += _ik_rate(arch) * parsed["n_embd"] * tokens
        elif factors.get("n_cpu_moe") and factors.get("split") == "tensor":
            hidden += MAINLINE_TENSOR_MOE_PER_NEMBD * parsed["n_embd"] * tokens
    return mask / 1024**2, swa_mask / 1024**2, hidden / 1024**2


def _constant(name: str, default: int) -> int:
    """Read a tuning constant from the file the estimator compiles in.

    Copied by hand once, and the copy went stale the moment the constant was
    re-derived — inflating every residual this file computed for an ik mixture
    of experts until it was noticed. Read it instead: the JSON is the source of
    truth for the Rust side already, and there is no reason for the analysis to
    hold its own opinion. The default only applies before the constant exists,
    which is the bootstrap case when a new term is being added.
    """
    try:
        document = json.loads(TUNING_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return default
    entry = document.get("constants", {}).get(name)
    return int(entry["value"]) if entry else default


def _ik_rate(arch: str | None) -> int:
    """The per-architecture ik MoE rate, as the estimator resolves it."""
    try:
        rates = json.loads(TUNING_JSON.read_text()).get("ik_moe_rates", {})
    except (OSError, json.JSONDecodeError):
        return 54
    return rates.get("by_arch", {}).get(arch or "", rates.get("default", 54))
MAINLINE_TENSOR_MOE_PER_NEMBD = _constant("MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD", 57)


def check_arena(rows: list[dict]) -> None:
    """Compare measured arena against the model, grouped by model and cards.

    Grouping by card count matters: an early pass that pooled them concluded
    the law was wrong, when in fact it is exact on one card and the second card
    multiplies one of the terms.
    """
    groups: dict[tuple, list[tuple[float, float, float, float, dict]]] = defaultdict(list)
    for record in rows:
        parsed = record["parsed"]
        if not parsed.get("arena_mib") or not parsed.get("n_embd"):
            continue
        factors = record["factors"]
        if factors["ngl"] != 99 or not factors["gpus"]:
            continue  # A partly- or un-offloaded run is a different shape.
        mask, swa_mask, hidden = arena_terms(record)
        cards = len(factors["gpus"].split(","))
        # Tensor split fuses the cards into one device, so it is the split
        # mode and not the card count that decides whether the mask is
        # replicated. Keying on cards alone attributes a layer-split effect to
        # every two-card run.
        placement = (factors["split"] or "layer") if cards > 1 else "single"
        key = (parsed.get("arch", "?"), factors["runtime"], placement,
               factors["flash_attn"])
        groups[key].append((parsed["arena_mib"], mask, swa_mask, hidden, record))

    print(f"{'arch':12}{'runtime':9}{'placement':>10}{'fa':>4}{'n':>4}"
          f"{'K (mask multiple)':>20}{'residual MiB':>14}")
    for key in sorted(groups):
        arch, runtime, placement, fa = key
        rows_here = groups[key]
        multiples, residuals = [], []
        for measured, mask, swa_mask, hidden, _ in rows_here:
            total_mask = mask + swa_mask
            if total_mask > 0:
                multiples.append((measured - hidden) / total_mask)
            residuals.append(measured - (total_mask + hidden))
        k = f"{st.median(multiples):.2f}" if multiples else "-"
        spread = (f" ±{(max(multiples) - min(multiples)) / 2:.2f}"
                  if len(multiples) > 1 else "")
        print(f"{arch:12}{runtime:9}{placement:>10}{fa:>4}{len(rows_here):>4}"
              f"{k + spread:>20}{st.median(residuals):>14.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["arena", "emit"], nargs="?", default="arena")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--tuning", type=Path, default=TUNING_JSON)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed tuning.json against the data "
                             "instead of rewriting it")
    args = parser.parse_args()
    rows = load(args.data)
    if args.what == "emit":
        return emit(rows, args.tuning, args.check)
    print(f"{len(rows)} completed measurements\n")
    check_arena(rows)
    return 0



# --- Emitting tuning.json ------------------------------------------------
#
# Not every constant can be derived from measurement, and pretending otherwise
# would be the dishonesty this file exists to prevent. Each carries a `kind`:
#
#   derived   — computed here from the dataset; the deriver runs and its
#               result is what ships. If it cannot run, emitting fails.
#   policy    — a choice, not a measurement (a runtime's documented default).
#   reachable — measured, but the spread is wide enough that the value is
#               chosen so every model lands inside the rolling correction's
#               [0.8, 1.5] clamp rather than to minimise error.
#   review    — has data, but the fit is contested and the value is held
#               pending another run.

class Disagreement(ValueError):
    """The cells behind a constant do not agree, so no single value fits them."""


def consensus(
    values: list[float],
    name: str,
    tolerance: float = 0.15,
    absolute_floor: float = 0.0,
) -> float:
    """Reduce measurements to one number, refusing when they disagree.

    Every deriver here used to take a median and say nothing about the spread
    behind it. That is how a constant absorbs a factor nobody thought to group
    by: the median of a bimodal set is a number that describes none of its
    members, and it looks exactly like the median of a tight one.

    Ten conclusions in this campaign were drawn that way and were wrong — card
    count, split mode, cell label, runtime, flash-attention state, placement,
    architecture, measurement time, serving state, slot count. Each produced a
    plausible law. So a spread wider than the tolerance is treated as a
    *failure to have grouped properly*, not as noise to average over, and it
    stops the derivation rather than quietly widening the constant.
    """
    if not values:
        raise Disagreement(f"{name}: no measurements")
    middle = st.median(values)
    spread_abs = max(values) - min(values)
    # A relative tolerance is meaningless around zero: a term whose median is
    # 0.1 and whose values span 21 reads as 218% disagreement while being 21
    # units wide. Where the caller knows what "small" means in its own units,
    # it says so, and a spread below that is not worth blocking on.
    if spread_abs <= absolute_floor:
        return middle
    if middle == 0:
        return middle
    spread = spread_abs / abs(middle)
    if spread > tolerance:
        raise Disagreement(
            f"{name}: {len(values)} measurements span {min(values):.4g} to "
            f"{max(values):.4g}, which is {spread:.0%} of the median — they do not "
            f"agree. A single value would collapse a real difference; find the "
            f"factor that separates them and group by it."
        )
    return middle


def derive_ik_moe_per_nembd(rows: list[dict]) -> tuple[int, str]:
    """Bytes per batch token per unit of hidden size for ik's CPU-MoE buffers.

    The *worst* rate across architectures, not the median. They differ —
    qwen35moe 40.5, glm-dsa 42.8, laguna 53.7 — and a median under-reserves
    every architecture above it, which is the direction that OOMs. Taking the
    maximum over-reserves the others by at most a third.

    Within an architecture the rate is exact: glm-dsa measures 42.8 on nine
    cells spanning ctx 8192-131072 and ub 256-512 without deviating. It does
    vary with card count on two of the three — glm-dsa 28.0 on one card
    against 42.8 on two, laguna 36.0 against 53.7 — and qwen35moe not at all.
    That is unexplained, and it is bounded by taking the maximum.
    """
    points = []
    for record in rows:
        factors, parsed = record["factors"], record["parsed"]
        if factors["runtime"] != "ik" or not parsed.get("arena_mib"):
            continue
        experts, used = parsed.get("n_expert") or 0, parsed.get("n_expert_used") or 0
        tokens = min(factors["ctx"], factors["ubatch"])
        if not experts or not used or tokens * used >= 32 * experts:
            continue
        if factors["flash_attn"] != "on" or factors["ngl"] != 99:
            continue
        mask, swa_mask, hidden = arena_terms(record, charge_moe=False)
        excess = (parsed["arena_mib"] - (mask + swa_mask + hidden)) * 1024**2 / tokens
        cards = len(factors["gpus"].split(",")) if factors["gpus"] else 1
        points.append((parsed["n_embd"], excess / parsed["n_embd"],
                       parsed.get("arch"), cards))
    if not points:
        raise ValueError("no ik MoE cells below the offload threshold")
    # Grouped by architecture first: they genuinely differ, and the table
    # below carries each separately. Within an architecture they must agree.
    # Keyed by card count as well as architecture. glm-dsa measures 28.0 per
    # unit on one card and 42.8 on two at *identical* placement — both its
    # offload values exceed its layer count, and both logs show 80 of 80 layers
    # on the GPU — and the difference scales with tokens, so it is a rate that
    # depends on the device count rather than a separate term. Above the MoE
    # threshold the two card counts agree exactly, which is what localises it
    # here rather than in the arena.
    for arch, cards in {(a, c) for _, _, a, c in points}:
        group = [r for _, r, a, c in points if a == arch and c == cards]
        consensus(group, f"ik MoE rate for {arch} on {cards} card(s)")
    for arch in {a for _, _, a, _ in points}:
        rates_here = [r for _, r, a, _ in points if a == arch]
        try:
            consensus(rates_here, f"ik MoE rate for {arch}")
        except Disagreement:
            # glm-dsa measures 28.0 per unit on one card and 42.8 on two, at
            # identical expert placement, each figure exact within its group.
            # The cause is not in the dataset. Taking the larger over-reserves
            # the single-card case rather than under-reserving the other, which
            # is the direction that does not OOM — so the disagreement is
            # carried deliberately rather than papered over.
            # laguna shows the same shape — 36.0 on one card against 53.7 on
            # two — though its expert offload differs between them too, so its
            # cause is confounded where glm's is not.
            if arch not in ("glm-dsa", "laguna"):
                raise
    per_unit = max(p[1] for p in points)
    by_key: dict[str, float] = {}
    for _, rate, arch, cards in points:
        key = f"{arch}@{cards}"
        by_key[key] = max(by_key.get(key, 0.0), rate)
    detail = ", ".join(
        f"{arch} {rate * embd:.0f} B/token at n_embd {embd} ({rate:.1f}/unit)"
        for embd, rate, arch in sorted({(p[0], round(p[1], 1), p[2]) for p in points})
    )
    _record_ik_rates(by_key)
    return round(per_unit), (
        f"{len(points)} cells below the offload threshold: {detail}. The worst "
        "rate is taken rather than the median: the architectures differ, and a "
        "median under-reserves every one above it. Replaces a flat 81 KiB, "
        "which was this term evaluated at qwen35moe's hidden size and frozen."
    )


def derive_mainline_tensor_moe(rows: list[dict]) -> tuple[int, str]:
    """mainline's host-resident MoE buffers under tensor split.

    A hybrid served with `--split-mode tensor` keeps per-token MoE
    intermediates on the host, the same shape as ik's term and at a higher
    rate. Under layer split the same models show none of it.
    """
    points = []
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["runtime"] != "mainline" or not factors["n_cpu_moe"]:
            continue
        if factors["split"] != "tensor" or factors["flash_attn"] != "on":
            continue
        if factors["spec_type"]:
            continue  # an MTP run measures without this term entirely
        if not parsed.get("arena_mib") or factors["ngl"] != 99:
            continue
        mask, swa_mask, hidden = arena_terms(record, charge_moe=False)
        tokens = min(factors["ctx"], factors["ubatch"])
        excess = (parsed["arena_mib"] - (mask + swa_mask + hidden)) * 1024**2
        if excess <= 0:
            continue
        points.append((parsed.get("arch"), parsed["n_embd"],
                       excess / tokens / parsed["n_embd"]))
    if not points:
        raise ValueError("no mainline tensor-split hybrid cells")
    rates = [r for _, _, r in points]
    detail = ", ".join(
        f"{arch} {rate:.1f}/unit"
        for arch, rate in sorted({(a, round(r, 1)) for a, _, r in points})
    )
    consensus(rates, "MTP embedded compute")
    return round(st.median(rates)), (
        f"{len(points)} mainline tensor-split hybrid cells: {detail}. Linear in "
        "batch — qwen35moe measures 28.3, 56.5, 113.0 and 226.1 MiB at ub 256, "
        "512, 1024 and 2048, a constant rate. The same models under layer split "
        "show 0.02 MiB, so this belongs to the split mode rather than to the "
        "hybrid placement alone."
    )


_TENSOR_BASE: dict[str, int] = {}
_BASE_OFFSET: dict[str, int] = {}


def derive_baseline_offset(rows: list[dict]) -> tuple[int, str]:
    """Per-architecture correction to the process baseline.

    `PROCESS_BASE_BYTES` plus a per-layer term plus a flat MoE allowance is the
    whole model, and it leaves a residual that is *architecture-shaped*: qwen35
    holds 297 MiB more than it predicts and qwen35moe 75, while gemma3 holds 47
    less. Two models of the same family show it and two of near-identical shape
    do not, so it is not size, and a long list of other causes has been ruled
    out by measurement — see FINDINGS.md.

    Modelled as an offset per architecture rather than explained. That is
    honest about what is known: the residual is reproducible, it is keyed on
    something the architecture string captures, and leaving it uncharged
    under-reserves qwen35 by nearly twice.
    """
    from collections import defaultdict as _dd
    import json as _json
    constants = _json.loads(TUNING_JSON.read_text())["constants"]
    per_layer = constants["PROCESS_BASE_BYTES_PER_LAYER"]["value"]
    flat = constants["PROCESS_BASE_BYTES"]["value"]
    moe = constants["PROCESS_BASE_BYTES_MOE"]["value"]
    dev = constants["PROCESS_BASE_BYTES_PER_DEVICE"]["value"]
    pinned = constants["PINNED_EXTRA_BYTES"]["value"]

    by_arch: dict[str, list[float]] = _dd(list)
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["ngl"] != 99 or not factors["gpus"] or factors["spec_type"]:
            continue
        if factors["n_cpu_moe"] or not factors["served"] or factors["bench"]:
            continue
        if factors["parallel"] != 1 or not parsed.get("arena_mib"):
            continue
        if (factors["split"] or "layer") != "layer" or not parsed.get("n_layer"):
            continue
        # Flash attention off is kept, under its own key. Pooling it with
        # flash attention on is what put lfm2's offset at both 35 and 169 MiB,
        # but excluding it left every such cell uncorrected — and while the
        # bulk of the effect is the per-token arena rate, a flat baseline shift
        # remains underneath it, small everywhere but lfm2 at +131 MiB.
        # mainline only. ik's residual against the same model runs from -264 to
        # +120 MiB where mainline's is -0 to +24, so the two binaries do not
        # share a baseline. They are now separated by the key rather than by
        # excluding one: grouped per runtime, ik's resident cells are as
        # consistent as mainline's — spreads of 62 to 100 MiB against the same
        # architectures — and simply sit 24 to 192 MiB higher. Excluding them
        # left every ik configuration with no correction at all.
        owned = (record["rss"].get("rss_anon_kb", 0)
                 + record["rss"].get("rss_shmem_kb", 0)) * 1024
        mask, swa_mask, hidden = arena_terms(record)
        cards = len(factors["gpus"].split(","))
        # ik does not replicate masks across cards at any count, so including
        # its cells means the multiplier can no longer be read off the card
        # count alone.
        copies = 4 if cards > 1 and factors["runtime"] != "ik" else 1
        # The per-token flash-attention term is a separate constant, derived
        # just above. Leaving it in the residual would charge it twice: once
        # in the arena and again in the baseline, where it would also stop
        # being flat and break the group.
        no_fa = 0
        if factors["flash_attn"] != "on":
            rate = _NO_FA_RATES.get(variant_key(record),
                                    max(_NO_FA_RATES.values(), default=0))
            no_fa = rate * min(factors["ctx"], factors["ubatch"])
        modelled = (flat + parsed["n_layer"] * per_layer
                    + (moe if parsed.get("n_expert") else 0))
        residual = (owned - (copies * (mask + swa_mask) + hidden) * 1024**2
                    - no_fa - pinned - dev * (cards - 1) - modelled)
        by_arch[variant_key(record, with_environment=True)].append(residual)

    if not by_arch:
        raise ValueError("no resident served cells")
    # No `consensus` call here, deliberately. That guard exists to stop a
    # *median* from hiding a disagreement, and this reduces by `max`: a maximum
    # bounds a spread rather than concealing it, and erring high on a baseline
    # is the safe direction. The spread is reported in the evidence instead, so
    # a wide one is visible rather than silently averaged.
    spreads = {a: (min(g) / 1024**2, max(g) / 1024**2) for a, g in by_arch.items()}
    # Negative offsets are charged too. The earlier rule kept only positive
    # ones, reasoning that a negative residual means the baseline already
    # over-covers and shaving it trades a safe over-prediction for a risk. That
    # does not survive two objections. The reduction is `max` — the *least*
    # negative residual — so subtracting it leaves every measured cell still
    # over-predicted; and an over-prediction is only safe while it stays inside
    # the band the rolling correction can travel. gemma3 sat at 0.78 against a
    # floor of 0.8, which no amount of observation can pull back, so the
    # "safe" direction had become the unreachable one.
    _BASE_OFFSET.clear()
    _BASE_OFFSET.update({a: round(max(g)) for a, g in by_arch.items()})
    detail = ", ".join(
        f"{a} {hi:+.0f}" + (f" (spans {lo:+.0f})" if hi - lo > 32 else "")
        for a, (lo, hi) in sorted(spreads.items())
    )
    return round(max(max(g) for g in by_arch.values())), (
        f"residual over the layer-count baseline, per architecture, across "
        f"{sum(len(g) for g in by_arch.values())} resident served cells: "
        f"{detail} MiB. Negative offsets are charged as well as positive ones: "
        "the reduction is the maximum, so subtracting it leaves every measured "
        "cell still over-predicted, and an over-prediction past the rolling "
        "correction's floor is unreachable rather than safe."
    )


def derive_tensor_split_baseline(rows: list[dict]) -> tuple[int, str]:
    """Host baseline a tensor split costs beyond a layer split.

    Measured on every model that ran both, at the same context, batch, slot
    count and card count: between 96 and 184 MiB more. The estimator had no
    term for it, so every tensor-split service was under-predicted by that
    much — and tensor split is what the operator runs for several of them.
    """
    from collections import defaultdict as _dd
    pairs: dict[tuple, dict[str, list[float]]] = _dd(lambda: _dd(list))
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["ngl"] != 99 or factors["gpus"] != "0,1" or factors["spec_type"]:
            continue
        if factors["n_cpu_moe"] or not factors["served"] or factors["bench"]:
            continue
        if factors["parallel"] != 1 or not parsed.get("arena_mib"):
            continue
        owned = (record["rss"].get("rss_anon_kb", 0)
                 + record["rss"].get("rss_shmem_kb", 0)) * 1024
        mask, swa_mask, hidden = arena_terms(record)
        split = factors["split"] or "layer"
        copies = 4 if split == "layer" else 1
        base = (owned - (copies * (mask + swa_mask) + hidden) * 1024**2) / 1024**2
        key = (record["provenance"]["model_key"], factors["ctx"], factors["ubatch"],
               parsed.get("arch"))
        pairs[key][split].append(base)

    deltas, detail = [], []
    by_arch: dict[str, list[float]] = _dd(list)
    for key, group in pairs.items():
        if "layer" not in group or "tensor" not in group:
            continue
        delta = st.median(group["tensor"]) - st.median(group["layer"])
        deltas.append(delta)
        by_arch[key[3]].append(delta)
        detail.append(f"{key[0].split('/')[-1][:18]} {delta:+.0f}")
    if not deltas:
        raise ValueError("no model ran both split modes at matching settings")
    for arch, group in by_arch.items():
        consensus(group, f"tensor-split baseline for {arch}", tolerance=0.20)
    _TENSOR_BASE.clear()
    _TENSOR_BASE.update({a: round(max(g) * 1024**2) for a, g in by_arch.items()})
    return round(max(deltas) * 1024**2), (
        f"{len(deltas)} models measured under both split modes at matching "
        f"context, batch, slots and cards: {'; '.join(sorted(detail))} MiB. "
        "Per architecture, since the spread across all of them is wider than "
        "any one of them is internally."
    )


def variant_key(record: dict, with_environment: bool = False) -> str:
    """The architecture, plus the distinctions that split one arch string.

    `gemma4` covers three models whose host terms differ by more than the
    rolling correction can travel: a mixture of experts, a dense model, and an
    E-variant. Both discriminators are ones `host_buffer` already applies —
    `has_experts` and `compute_buffer::is_gemma_e_variant` — so a key built
    from them is one the estimator can construct at lookup time.
    """
    parsed = record["parsed"]
    key = str(parsed.get("arch"))
    if parsed.get("n_expert"):
        key += "+moe"
    if "E4B" in record["provenance"]["model_key"]:
        # Stands in for the per-layer embedding tensor the records do not
        # carry, which is what the estimator keys on.
        key += "+e"
    # Only where the caller asks, which is the baseline offset alone. It
    # differs by runtime (ik sits 24 to 192 MiB above mainline on the same
    # architecture) and by flash attention, which shifts it by +21 to +33 MiB
    # on most architectures and +131 on lfm2 — on top of the per-token arena
    # rate, which is a separate term.
    #
    # The flash-attention *rates* must not be keyed this way: ik is excluded
    # from that derivation, so an ik-suffixed key would have no row and would
    # inherit the table's worst rate as its default.
    if with_environment and record["factors"]["runtime"] == "ik":
        key += "@ik"
    if with_environment and record["factors"]["flash_attn"] != "on":
        key += "@nofa"
    return key


def derive_no_flash_attn_rates(rows: list[dict]) -> str:
    """Extra pinned bytes per batch token when flash attention is off.

    The single constant this replaces was chosen as a representative value
    because the excess "is not uniform across architectures", and that is
    right — but the non-uniformity is a clean per-architecture rate rather
    than noise, which a sweep across context makes visible and a single
    context cannot.

    What the residual over the modelled arena does *not* do is scale with
    context: gemma-3-27B is 64 MiB out at ctx 8192, 32768, and 131072 alike,
    and 256 MiB out at ubatch 2048 in every one of them. So it is a per-batch
    -token term, the same shape as the quantised-cache rate, at 128 KiB per
    token on the sliding-window models against 32 KiB on the rest.

    The rate is per token *per stream*. Every cell here runs one slot, so the
    figure derived is the per-slot one; Qwen3-4B at four slots measures a
    quarter of it, flat across three contexts, which is why `pinned_graph_bytes`
    divides by the stream count rather than the table carrying it.

    ik_llama is excluded and keeps the small default: its fa-off arena is
    already modelled to within a megabyte, since it sizes masks against the
    whole cache and the widened element is the whole story there.
    """
    from collections import defaultdict as _dd
    by_arch: dict[str, list[float]] = _dd(list)
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["flash_attn"] == "on" or not parsed.get("arena_mib"):
            continue
        if factors["ngl"] != 99 or not factors["gpus"] or factors["spec_type"]:
            continue
        # One slot only: the term divides by the stream count, so pooling slot
        # counts would fit a quarter-sized rate against a full-sized one.
        if factors["runtime"] == "ik" or factors["parallel"] != 1:
            continue
        mask, swa, hidden = arena_terms(record)
        cards = len((factors["gpus"] or "0").split(","))
        # A hybrid does not replicate the mask across cards — the same
        # exception the quantised-cache rate needs, and leaving it out here
        # would put every hybrid architecture's rate wildly negative and drop
        # it from the table, where it would then inherit the *worst* rate as
        # its default.
        hybrid = bool(factors["n_cpu_moe"])
        copies = 4 if cards > 1 and (factors["split"] or "layer") == "layer" \
            and not hybrid else 1
        residual = parsed["arena_mib"] - (copies * (mask + swa) + hidden)
        tokens = min(factors["ctx"], factors["ubatch"])
        # `arena_terms` models the mask at its widened four-byte element and
        # nothing else, so the residual over it is the whole term rather than
        # an excess over one already charged.
        by_arch[variant_key(record)].append(residual * 1024**2 / tokens)
    if not by_arch:
        raise ValueError("no mainline cells with flash attention off")
    for arch, group in by_arch.items():
        # 4 KiB per token: below that the term is under a megabyte at the
        # largest batch measured, which is not worth splitting a group over.
        consensus(group, f"no-flash-attention rate for {arch}",
                  tolerance=0.10, absolute_floor=4096.0)
    _NO_FA_RATES.clear()
    # Negative rates mean the current constant over-charges that architecture;
    # clamping at zero keeps the term from *subtracting* pinned memory, which
    # no mechanism supports.
    _NO_FA_RATES.update({a: max(0, round(max(g))) for a, g in by_arch.items()})
    return (
        f"{sum(len(g) for g in by_arch.values())} mainline cells with flash "
        f"attention off across {len(by_arch)} architectures. The residual over "
        f"the modelled arena is flat in context and proportional to batch "
        f"tokens, so it is a per-token rate: "
        + ", ".join(f"{a} {v / 1024:.0f} KiB" for a, v in sorted(_NO_FA_RATES.items()))
        + ". ik_llama is excluded — its fa-off arena is already modelled to "
        "within a megabyte."
    )


def derive_quantised_cache_bytes(rows: list[dict]) -> tuple[int, str]:
    """Extra pinned bytes per batch token when the KV cache is quantised.

    Paired cells differing in nothing but `cache_type_*` show the arena larger
    with a quantised cache, in all 117 pairs measured, always positive and
    scaling exactly with batch — 1.28 MiB at ub 512 against 5.12 at 2048 on the
    same model.

    The per-copy rate varies by architecture and is not predicted by head
    count, head width, or layer count: 160 bytes per token on non-sliding-window
    models, 328 on sliding-window ones, 532, 2621, and 6144 on deepseek4. Since
    the mechanism is not identified, the worst observed rate is charged to all
    of them. Doing so costs 12 MiB at the largest batch measured, which is
    cheap insurance against an under-prediction whose size is not understood.
    """
    from collections import defaultdict as _dd
    paired: dict[tuple, dict[str, float]] = _dd(dict)
    archs: dict[str, str] = {}
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["ngl"] != 99 or not factors["gpus"] or factors["spec_type"]:
            continue
        if not parsed.get("arena_mib"):
            continue
        key = (record["provenance"]["model_key"], factors["ctx"], factors["ubatch"],
               factors["parallel"], bool(factors["kv_unified"]),
               factors["split"] or "-", factors["gpus"], factors["flash_attn"],
               bool(factors["served"]), bool(factors["n_cpu_moe"]))
        paired[key][factors["kv_type"]] = parsed["arena_mib"]
        archs[record["provenance"]["model_key"]] = parsed.get("arch", "?")

    rates = []
    by_arch: dict[str, list[float]] = _dd(list)
    for key, pair in paired.items():
        if len(pair) != 2:
            continue
        tokens = min(key[1], key[2])
        cards = len(key[6].split(","))
        # Divide out the layer-split replication, so the rate is per copy — but
        # a hybrid does not replicate, and treating one as though it did put
        # qwen35moe's rate at both 133 and 532.
        hybrid = key[9]
        copies = 4 if cards > 1 and key[5] == "layer" and not hybrid else 1
        rate = (pair["q8_0"] - pair["f16"]) * 1024**2 / tokens / copies
        rates.append(rate)
        by_arch[archs[key[0]]].append(rate)
    if not rates:
        raise ValueError("no cells differing only in cache type")
    # Per architecture, because they differ by a factor of forty and charging
    # the worst to all costs ~3 MiB of over-prediction on every quantised cell.
    for arch, group in by_arch.items():
        consensus(group, f"quantised-cache rate for {arch}", tolerance=0.05)
    _QUANT_RATES.clear()
    _QUANT_RATES.update({a: round(max(g)) for a, g in by_arch.items()})
    worst = max(rates)
    return round(worst), (
        f"{len(rates)} pairs differing in nothing but the cache type, every one "
        f"showing the arena larger when it is quantised. Per-copy rates run "
        f"{min(rates):.0f} to {worst:.0f} bytes per batch token and are not "
        "predicted by head count, head width or layer count, so the worst is "
        "charged to all — 12 MiB at the largest batch measured. Scaling with "
        "batch is exact, and dividing out the layer-split replication is what "
        "makes the two rates per model collapse to one."
    )


def derive_gemma_e_per_layer_token(rows: list[dict]) -> tuple[int, str]:
    """The E-variant's per-layer embedding input, in bytes per layer per token."""
    residuals, controls = [], []
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if parsed.get("arch") != "gemma4" or not parsed.get("arena_mib"):
            continue
        if factors["ngl"] != 99 or not factors["gpus"] or factors["flash_attn"] != "on":
            continue
        if factors["spec_type"]:
            continue
        # f16 only. A quantised cache shifts this residual — 1087 and 1278
        # against f16's steady 1025-1028 — so the arena model is missing a
        # term that depends on the cache type, and pooling the two would
        # attribute that term to the E-variant.
        if factors["kv_type"] != "f16":
            continue
        mask, swa_mask, hidden = arena_terms(record)
        cards = len(factors["gpus"].split(","))
        copies = 4 if cards > 1 and (factors["split"] or "layer") == "layer" else 1
        residual = parsed["arena_mib"] - (copies * (mask + swa_mask) + hidden)
        tokens = min(factors["ctx"], factors["ubatch"])
        per = residual * 1024**2 / (parsed["n_layer"] * tokens)
        # The two populations are ~1028 and ~3 bytes per layer per token, so
        # the boundary is nowhere near either. A cell between them is not an
        # E-variant reading and not a control; it is a sign the filter is
        # wrong, and `consensus` will say so rather than averaging it in.
        (residuals if per > 500 else controls).append(per)
    if not residuals:
        raise ValueError("no gemma4 E-variant cells")
    consensus(residuals, "gemma E-variant per-layer term")
    return round(st.median(residuals)), (
        f"{len(residuals)} E-variant cells at {st.median(residuals):.0f} B/layer/token, "
        f"against {len(controls)} same-architecture control cells at "
        f"{st.median(controls) if controls else 0:.0f}. "
        f"{round(st.median(residuals))} is {round(st.median(residuals) / 4)} f32 elements."
    )


def derive_per_device_bytes(rows: list[dict]) -> tuple[int, str]:
    """Host cost of each visible CUDA device beyond the first."""
    by_model: dict[str, dict[int, int]] = defaultdict(dict)
    for record in rows:
        factors = record["factors"]
        if not record["factors"]["label"].startswith(("devices-", "offload-ngl0")):
            continue
        if factors["ngl"] != 0:
            continue
        cards = len(factors["gpus"].split(",")) if factors["gpus"] else 0
        owned = record["rss"].get("rss_anon_kb", 0) + record["rss"].get("rss_shmem_kb", 0)
        by_model[record["provenance"]["model_key"]][cards] = owned
    deltas, detail = [], []
    for model, points in by_model.items():
        if 1 in points and 2 in points:
            delta = (points[2] - points[1]) * 1024
            deltas.append(delta)
            detail.append(f"{model.split('/')[-1][:24]} {delta / 1024**2:.0f} MiB")
    if not deltas:
        raise ValueError("no paired one-card/two-card device-scaling cells")
    consensus(deltas, "per-device host cost")
    return round(st.median(deltas)), (
        "measured with placement pinned to the CPU so only the CUDA context "
        f"count varies: {'; '.join(detail)} going from one card to two."
    )


def derive_layer_split_copies(rows: list[dict]) -> tuple[int, str]:
    """How many times mainline replicates the masks under layer split."""
    multiples, singles = [], []
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["runtime"] != "mainline" or not parsed.get("arena_mib"):
            continue
        if factors["ngl"] != 99 or not factors["gpus"] or factors["flash_attn"] != "on":
            continue
        if parsed.get("arch") == "gemma4":
            continue  # carries its own per-layer term; excluded to keep this clean
        if factors["n_cpu_moe"]:
            continue  # a hybrid does not replicate the masks — that is the point
        mask, swa_mask, hidden = arena_terms(record)
        if mask + swa_mask <= 0:
            continue
        k = (parsed["arena_mib"] - hidden) / (mask + swa_mask)
        cards = len(factors["gpus"].split(","))
        if cards > 1 and (factors["split"] or "layer") == "layer":
            multiples.append(k)
        else:
            singles.append(k)
    if not multiples:
        raise ValueError("no mainline layer-split cells")
    consensus(multiples, "layer-split mask multiple")
    return round(st.median(multiples)), (
        f"{len(multiples)} mainline layer-split cells at {st.median(multiples):.2f}, "
        f"against {len(singles)} single-card and tensor-split cells at "
        f"{st.median(singles):.2f}. Flat across context, batch, slot count and "
        "cache mode; ik is 1.00 at either card count."
    )


def derive_offload_min_batch(rows: list[dict]) -> tuple[int, str]:
    """The batch threshold at which ik moves its MoE ops off the CPU."""
    crossings = []
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if factors["runtime"] != "ik" or not parsed.get("arena_mib"):
            continue
        experts, used = parsed.get("n_expert") or 0, parsed.get("n_expert_used") or 0
        if not experts or not used:
            continue
        tokens = min(factors["ctx"], factors["ubatch"])
        if factors["flash_attn"] != "on" or factors["ngl"] != 99:
            continue
        mask, swa_mask, hidden = arena_terms(record, charge_moe=False)
        # Whether the term is there at all, judged from the measurement rather
        # than from the threshold being solved for. A present term is tens of
        # MiB; an absent one leaves hundredths.
        excess_per_token = (parsed["arena_mib"] - (mask + swa_mask + hidden)) * 1024**2 / tokens
        crossings.append((tokens * used / experts, excess_per_token > 1024))
    on = [ratio for ratio, present in crossings if present]
    off = [ratio for ratio, present in crossings if not present]
    if not on or not off:
        raise ValueError("the threshold is not bracketed by the dataset")
    return round(min(off)), (
        f"bracketed to ({max(on):.0f}, {min(off):.0f}] by {len(crossings)} ik MoE "
        "cells across two models with different expert counts; the term collapses "
        "from tens of MiB to hundredths exactly at the predicted crossing."
    )


# What kind of thing each constant is. Declared rather than inferred: an
# earlier version read the evidence text and guessed, which filed a structural
# fact under "reachable" and missed two constants whose review note happened
# to be lowercase. A constant absent from here is an error, so adding one
# forces the question of what justifies it.
KINDS = {
    # Read from llama.cpp's source or arithmetic over the graph; not fitted.
    "KV_CACHE_PAD": "structural",
    # A runtime's documented default, mirrored so the reservation and the
    # runtime's own cap are the same number.
    "DEFAULT_CACHE_RAM_MB": "policy",
    "DEFAULT_UBATCH": "policy",
    # Measured, but with a spread wide enough that the value is chosen so
    # every model lands inside the rolling correction's [0.8, 1.5] clamp
    # rather than to minimise error against any one of them.
    "NO_FLASH_ATTN_COMPUTE_HEAD_FACTOR": "reachable",
    "PROCESS_BASE_BYTES": "reachable",
    "PROCESS_BASE_BYTES_PER_LAYER": "reachable",
    "PROCESS_BASE_BYTES_MOE": "reachable",
    "PINNED_EXTRA_BYTES": "reachable",
    "NO_FLASH_ATTN_BYTES_PER_TOKEN": "reachable",
    "DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16": "reachable",
    # Has data, but the fit is contested and the value is held.
    "MTP_COMPUTE_MIB": "review",
    "DRAFT_MODEL_COMPUTE_MIB": "review",
    "MTP_HOST_BYTES_EMBEDDED": "review",
    "MTP_HOST_BYTES_SEPARATE_DRAFT": "review",
}

def derive_mtp_embedded_compute(rows: list[dict]) -> tuple[int, str]:
    """The GPU compute an embedded MTP head costs, from paired cells.

    Fitted against the *driver* delta between a with-MTP cell and its
    without-MTP twin at identical settings, not against llama.cpp's own
    `[spec]` line — which reports the MTP context alone and comes to roughly a
    quarter of what the process actually takes.
    """
    pairs = _mtp_pairs(rows, draft=False)
    if len(pairs) < 2:
        raise ValueError("fewer than two embedded-MTP pairs")
    # Two contexts per model give a slope; the intercept is the constant.
    # Grouped by slot count as well as model: `parallel` divides the KV budget
    # per stream, so it changes the MTP context's size. Fitting a slope across
    # points that differ in it measures neither.
    by_model: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for model, ctx, delta, record in pairs:
        by_model[(model, record["factors"]["parallel"])].append((ctx, delta))
    bases, detail = [], []
    for (model, slots), points in by_model.items():
        points.sort()
        if len(points) < 2:
            continue
        (c1, d1), (c2, d2) = points[0], points[-1]
        slope = (d2 - d1) / (c2 - c1)
        base = d1 - slope * c1
        bases.append(base)
        detail.append(f"{model.split('/')[-1][:22]} np{slots} {base:.0f} MiB")
    if not bases:
        raise ValueError("no model has two contexts")
    # The larger of the two, so neither model is under-reserved.
    value = round(max(bases))
    return value, (
        f"{len(pairs)} paired with/without cells across {len(by_model)} models, "
        f"fitted against the driver delta: {'; '.join(detail)}. The larger base "
        "is taken so neither model is under-reserved. Replaces a value fitted "
        "against llama.cpp's own [spec] line, which reports roughly a quarter "
        "of the delta the process actually shows."
    )


def mtp_slot_scaling(rows: list[dict]) -> str:
    """Does the MTP overhead depend on the slot count, at a fixed context?

    This is the question `MTP_COMPUTE_MIB` was held under review for, and the
    first campaign could not answer it: every one-slot pair sat at ctx 32768 or
    65536 and the only four-slot pair at 131072, so slots and context were
    confounded and the four-slot pair's much larger delta had two candidate
    causes.

    Reported rather than fitted. A flat series across slots says the earlier
    "slot dependence" was the longer context and the constant can be fitted on
    context alone; a rising one says the model needs a slot term before any
    value is trustworthy.
    """
    by_key: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for model, ctx, delta, record in _mtp_pairs(rows, draft=False) + \
            _mtp_pairs(rows, draft=True):
        by_key[(model.split("/")[-1][:24], ctx)][record["factors"]["parallel"]] = delta
    series = {k: dict(sorted(v.items())) for k, v in sorted(by_key.items())
              if len(v) > 1}
    if not series:
        return "no fixed-context slot series measured"
    return "; ".join(f"{model} ctx {ctx}: " +
                     ", ".join(f"np{n} {d} MiB" for n, d in points.items())
                     for (model, ctx), points in series.items())


_MTP_SLOTS: list[str] = []
_IK_RATES: dict[str, int] = {}
_QUANT_RATES: dict[str, int] = {}
_NO_FA_RATES: dict[str, int] = {}


def _record_ik_rates(by_arch: dict[str, float]) -> None:
    """Stash the per-architecture rates for `emit` to write out."""
    _IK_RATES.clear()
    _IK_RATES.update({a: round(r) for a, r in by_arch.items()})


def _mtp_pairs(rows: list[dict], draft: bool) -> list[tuple[str, int, int, dict]]:
    """With/without MTP cells matched on every factor but the MTP flag."""
    def identity(record: dict) -> tuple:
        f = record["factors"]
        return (record["provenance"]["model_key"], f["ctx"], f["parallel"],
                bool(f["kv_unified"]), f["split"] or "-", f["gpus"], f["ubatch"],
                f["kv_type"])

    grouped: dict[tuple, dict[bool, dict]] = defaultdict(dict)
    for record in rows:
        if record["parsed"].get("arch"):
            grouped[identity(record)][bool(record["factors"]["spec_type"])] = record
    out = []
    for key, pair in grouped.items():
        if len(pair) != 2:
            continue
        on, off = pair[True], pair[False]
        if bool(on["factors"]["draft"]) != draft:
            continue
        if not on["rss"].get("gpu_used_mib") or not off["rss"].get("gpu_used_mib"):
            continue
        # Both halves must come from the same sitting. Cell identity ignores
        # the label, so a freshly-measured cell can pair with one recorded
        # hours earlier under different machine state — which produced a
        # *negative* delta once, a with-MTP process apparently using less VRAM
        # than without. Repeats taken back to back reproduce to the megabyte,
        # so the machine is not noisy; the pairing was.
        apart = abs(_measured_at(on) - _measured_at(off))
        if apart > _SAME_SITTING_SECONDS:
            continue
        delta = on["rss"]["gpu_used_mib"] - off["rss"]["gpu_used_mib"]
        if delta <= 0:
            continue
        out.append((key[0], key[1], delta, on))
    return out


_SAME_SITTING_SECONDS = 3600


def _measured_at(record: dict) -> float:
    """When a record was taken, as a POSIX timestamp."""
    stamp = record["provenance"]["measured_at_utc"]
    return datetime.datetime.fromisoformat(stamp).timestamp()


# Headroom over the worst per-device figure measured. The reservation must
# cover configurations the dataset does not contain, and under-reserving OOMs
# a load where over-reserving only costs capacity, so the margin is generous.
# It also absorbs the batch-scaling constant the curve's form cannot carry —
# see the note in `derive_curve`.
CURVE_MARGIN = 1.6


def derive_curve(archs: tuple[str, ...], exclude_models: tuple[str, ...] = ()):
    """Fit one architecture's compute curve to what it actually needed.

    The target is llama.cpp's own `compute` column plus its `unaccounted`
    remainder — the CUDA context and whatever else sits outside its books —
    since that is what the packer must cover beyond weights and KV.

    Two exclusions matter. Tensor-split cells report a single fused `Meta()`
    device whose unaccounted remainder is an artifact of that representation,
    14 GiB of it, and would dominate any fit. And only the calibration batch
    is used, because the slope is scaled by batch at the point of use.
    """
    def deriver(rows: list[dict]) -> tuple[int, str]:
        by_ctx: dict[int, list[int]] = defaultdict(list)
        for record in rows:
            parsed, factors = record["parsed"], record["factors"]
            if parsed.get("arch") not in archs or not parsed.get("devices"):
                continue
            if any(m in record["provenance"]["model_key"] for m in exclude_models):
                continue
            if factors["spec_type"] or factors["flash_attn"] != "on":
                continue
            if factors["ubatch"] != 512 or (factors["split"] or "layer") != "layer":
                continue
            for device in parsed["devices"]:
                if device["compute_mib"] and not device["device"].startswith("Meta"):
                    by_ctx[factors["ctx"]].append(
                        (device["compute_mib"], device["unaccounted_mib"]))
        if len(by_ctx) < 2:
            raise ValueError(f"{'/'.join(archs)}: fewer than two contexts")
        # Fitted from the endpoints of the context sweep against the worst
        # per-device need at each.
        #
        # The curve's form limits what this can express. A reservation is
        # `base + slope * ctx * (ubatch / 512)`, so only the slope scales with
        # batch — but the measured compute has a constant part that scales with
        # batch too, and there is nowhere to put it. Forcing the slope to cover
        # it means dividing that constant by the *smallest* context in the
        # sweep, which made talkie's slope 580 and its reservation forty times
        # the measurement. The margin below absorbs it instead, which
        # over-reserves a little everywhere rather than absurdly in one place.
        points = sorted(
            (c, max(comp + unacc for comp, unacc in v)) for c, v in by_ctx.items())
        (c1, n1), (c2, n2) = points[0], points[-1]
        slope = (n2 - n1) / ((c2 - c1) / 1024) if c2 != c1 else 0.0
        base = n1 - slope * (c1 / 1024)
        # Cover every interior point too, not just the endpoints.
        base = max(base, max(n - slope * (c / 1024) for c, n in points))
        base = max(0, round(base * CURVE_MARGIN))
        slope = max(1, round(slope * CURVE_MARGIN))
        worst = max(comp + u for v in by_ctx.values() for comp, u in v)
        evidence = (
            f"base from the worst unaccounted remainder, which is flat in both "
            f"context and batch; slope from the compute buffer, which scales with "
            f"both. Fitted across "
            f"{sum(len(v) for v in by_ctx.values())} layer-split cells at ctx "
            f"{points[0][0]}-{points[-1][0]}, peaking at {worst} MiB, with "
            f"{int((CURVE_MARGIN - 1) * 100)}% headroom. Tensor-split cells are "
            f"excluded: they report one fused device whose unaccounted remainder "
            f"is an artifact of that representation.")
        return base, evidence, slope
    return deriver


def derive_deepseek4_curve(rows: list[dict]) -> tuple[int, str]:
    """deepseek4's compute-buffer base, over the regime this hardware reaches.

    The architecture cannot be run GPU-resident on 48 GiB of VRAM — the
    weights alone want 48.5 GiB on one card — so every measurable
    configuration is a high-offload hybrid. In that regime the buffer is flat.
    """
    points = []
    for record in rows:
        parsed, factors = record["parsed"], record["factors"]
        if parsed.get("arch") != "deepseek4" or not parsed.get("devices"):
            continue
        if factors["flash_attn"] != "on":
            continue
        device = parsed["devices"][0]
        points.append((factors["ctx"], device["compute_mib"] + device["unaccounted_mib"]))
    if len(points) < 3:
        raise ValueError("too few deepseek4 cells")
    contexts = sorted({c for c, _ in points})
    per_ctx = {c: st.median(v for cc, v in points if cc == c) for c in contexts}
    spread = max(per_ctx.values()) - min(per_ctx.values())
    value = round(max(per_ctx.values()) * 1.05 / 100) * 100
    return value, (
        f"{len(points)} cells over ctx {min(contexts)}-{max(contexts)}: the "
        f"compute buffer plus its unaccounted share moves {spread:.0f} MiB across "
        f"that range, i.e. it is flat. Base set {value} MiB with 5% headroom and "
        "the slope left nominal. LIMITATION: this architecture cannot be run "
        "GPU-resident on 48 GiB of VRAM, so the steep context scaling the "
        "previous slope encoded is untestable here and may be real on a larger "
        "machine."
    )


# Curves are derived separately: they live in the ordered table rather than
# among the scalars, and a deriver returns a base, a slope and its evidence.
# `derive_deepseek4_curve` below is superseded and kept only as the record of
# what the hold-out was protecting. Its argument was that the curve covers a
# 9.3 GiB residual at ctx 131072 that llama.cpp's own compute column, at 2.0
# GiB, does not describe. The sweep retires it: the total measured GPU
# footprint is 15496 MiB at ctx 8192 and 16425 at 131072, a difference of 929
# MiB and exactly the KV growth, so any residual beyond the compute column is
# *flat in context*. A flat residual cannot justify a context slope — it is
# equally present at 8192, where the old curve reserved 2428 MiB. deepseek4 is
# therefore fitted by the general deriver like everything else.
# Keyed by the entry each targets, not by architecture: `llama` and `qwen3`
# share the default curve, so it has to cover the worse of the two rather than
# be written twice, and gemma4 has a variant-guarded entry that must be fitted
# separately from the general one.
CURVE_DERIVERS = {
    # One entry covers gemma2, gemma3 and gemma4, so it is fitted over all of
    # them at once; the E-variant has its own entry and its model is held out.
    # Fitted like every other architecture now. Held out until the sweep could
    # answer the question: the worst per-device need is 2398 MiB at ctx 8192
    # and 2396 at 131072, so the curve's context slope described nothing. See
    # `derive_deepseek4_curve` below for what the hold-out was protecting and
    # why the measurements retire it.
    "deepseek4": derive_curve(("deepseek4",)),
    "gemma3": derive_curve(("gemma2", "gemma3", "gemma4"), exclude_models=("E4B",)),
    "laguna": derive_curve(("laguna",)),
    "lfm2": derive_curve(("lfm2",)),
    "qwen35": derive_curve(("qwen35",)),
    "qwen35moe": derive_curve(("qwen35moe",)),
    "talkie": derive_curve(("talkie",)),
    "default": derive_curve(("llama", "qwen3")),
}

# `derive_mtp_embedded_compute` is likewise held back. Its quantity — the
# driver delta between paired cells — is the right one, and it says the
# constant over-reserves by ~700 MiB at one slot. But the single `parallel = 4`
# pair shows a delta of 2892 MiB against the ~1100 the one-slot fit predicts,
# so there is a slot-count dependence the model does not carry. Lowering the
# constant without that term would under-reserve exactly the production
# configuration, which runs four slots.
DERIVERS = {
    "MAINLINE_TENSOR_MOE_BYTES_PER_NEMBD": derive_mainline_tensor_moe,
    "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN": derive_gemma_e_per_layer_token,
    "PROCESS_BASE_BYTES_PER_DEVICE": derive_per_device_bytes,
    "MAINLINE_LAYER_SPLIT_MASK_COPIES": derive_layer_split_copies,
    "IK_OP_OFFLOAD_MIN_BATCH": derive_offload_min_batch,
}


# Tables whose values may be negative, and which `build.rs` must therefore
# emit through `generate_signed_rate_table`.
SIGNED_TABLES = {"baseline_offset"}


def check_table_signs(document: dict) -> None:
    """Refuse to write a negative into a table read as unsigned.

    `build.rs` reads every table but the signed ones through `as_u64`, which
    turns a negative into the table's default rather than raising — the value
    vanishes with no error anywhere. That is how the negative baseline offsets
    would have failed had they gone out through the unsigned path, so the
    invariant is asserted here, where it can still be seen.
    """
    for name, table in document.items():
        if not isinstance(table, dict) or name in SIGNED_TABLES:
            continue
        negative = {k: v for k, v in (table.get("by_arch") or {}).items() if v < 0}
        if negative:
            raise Disagreement(
                f"{name} holds negative values {negative} but is read through "
                f"the unsigned path in build.rs, which would silently replace "
                f"each with the default. Emit it via "
                f"`generate_signed_rate_table` and add it to `SIGNED_TABLES`."
            )


def emit(rows: list[dict], path: Path, check: bool) -> int:
    """Regenerate `tuning.json`, or verify the committed one against the data.

    Constants with a deriver are recomputed and their evidence rewritten from
    what the dataset actually shows. Everything else keeps its declared value
    and reason — a policy default, a value chosen for reachability, or one
    held pending another run — because inventing a derivation for those would
    be exactly the dishonesty this is meant to prevent.
    """
    document = json.loads(path.read_text())
    constants = document["constants"]
    changed, failed = [], []

    # Runs for its side effect of recording the per-architecture rates, which
    # `emit` writes as a table rather than as one scalar.
    try:
        derive_ik_moe_per_nembd(rows)
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
        failed.append(f"ik MoE rates: cannot derive — {error}")
    try:
        _MTP_SLOTS.clear()
        _MTP_SLOTS.append(mtp_slot_scaling(rows))
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
        failed.append(f"MTP slot scaling: cannot report — {error}")
    try:
        derive_no_flash_attn_rates(rows)
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError, Disagreement) as error:
        failed.append(f"no-flash-attention rates: cannot derive — {error}")

    try:
        derive_baseline_offset(rows)
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
        failed.append(f"baseline offset: cannot derive — {error}")

    try:
        derive_tensor_split_baseline(rows)
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
        failed.append(f"tensor-split baseline: cannot derive — {error}")

    try:
        derive_quantised_cache_bytes(rows)
    except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
        failed.append(f"quantised-cache rates: cannot derive — {error}")

    for name, deriver in DERIVERS.items():
        entry = constants.get(name)
        if entry is None:
            failed.append(f"{name}: in DERIVERS but not in {path.name}")
            continue
        try:
            value, evidence = deriver(rows)
        except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
            failed.append(f"{name}: cannot derive — {error}")
            continue
        if entry["value"] != value:
            changed.append(f"{name}: {entry['value']} -> {value}")
        entry["value"] = value
        entry["evidence"] = evidence
        entry["kind"] = "derived"

    for arch, deriver in CURVE_DERIVERS.items():
        if arch == "default":
            entry = document["compute_buffer_curves"]["default"]
        else:
            # The *unguarded* entry for this architecture: a variant-guarded
            # one describes a different graph and is fitted separately.
            entry = next((e for e in document["compute_buffer_curves"]["entries"]
                          if arch in e["archs"] and not e.get("variant")), None)
        if entry is None:
            failed.append(f"{arch}: curve deriver has no matching entry")
            continue
        try:
            base, evidence, slope = deriver(rows)
        except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
            failed.append(f"{arch} curve: cannot derive — {error}")
            continue
        if entry["base_mib"] != base:
            changed.append(f"{arch} curve base: {entry['base_mib']} -> {base}")
        if entry.get("slope_mib_per_1k") != slope:
            changed.append(f"{arch} slope: {entry.get('slope_mib_per_1k')} -> {slope}")
        entry["base_mib"] = base
        entry["slope_mib_per_1k"] = slope
        entry["slope_scales_with_ubatch"] = True
        entry["evidence"] = evidence

    for name, entry in constants.items():
        if name in DERIVERS:
            continue
        kind = KINDS.get(name)
        if kind is None:
            failed.append(f"{name}: no declared kind — add it to KINDS")
            continue
        entry["kind"] = kind

    if _BASE_OFFSET:
        document["baseline_offset"] = {
            "$comment": "Per-architecture correction to the process baseline, "
                        "in bytes. The layer-count model leaves a residual that "
                        "is architecture-shaped and reproducible; this charges "
                        "it where it is positive. `default` is zero, since an "
                        "unmeasured architecture has no evidence either way.",
            "default": 0,
            "by_arch": dict(sorted(_BASE_OFFSET.items())),
        }
    if _TENSOR_BASE:
        document["tensor_split_baseline"] = {
            "$comment": "Extra host baseline bytes a tensor split costs beyond "
                        "a layer split, by architecture. `default` applies to "
                        "an architecture not listed.",
            "default": max(_TENSOR_BASE.values()),
            "by_arch": dict(sorted(_TENSOR_BASE.items())),
        }
    if _MTP_SLOTS:
        document["mtp_slot_scaling"] = {
            "$comment": "Reported, not fitted. The MTP overhead measured at a "
                        "fixed context across slot counts, which is what "
                        "separates a genuine slot term from the longer context "
                        "the first campaign confounded it with.",
            "observed": _MTP_SLOTS[0],
        }
    if _NO_FA_RATES:
        document["no_flash_attn_rates"] = {
            "$comment": "Extra pinned bytes per batch token when flash "
                        "attention is off, by architecture. The residual is "
                        "flat in context and proportional to batch, and the "
                        "rates differ fourfold between sliding-window models "
                        "and the rest. `default` applies to an architecture "
                        "not listed.",
            "default": max(_NO_FA_RATES.values()),
            "by_arch": dict(sorted(_NO_FA_RATES.items())),
        }
    if _QUANT_RATES:
        document["quantised_cache_rates"] = {
            "$comment": "Extra pinned bytes per batch token when the KV cache "
                        "is quantised, by architecture. They span a factor of "
                        "forty, so one value would either under-reserve "
                        "deepseek4 or over-reserve everything else by ~3 MiB. "
                        "`default` applies to an architecture not listed.",
            "default": max(_QUANT_RATES.values()),
            "by_arch": dict(sorted(_QUANT_RATES.items())),
        }
    if _IK_RATES:
        # Per architecture, because they differ and one number cannot serve
        # all three without either under-reserving the worst or over-reserving
        # the rest. The fallback is the worst seen, for an ik mixture of
        # experts this dataset has never measured.
        document["ik_moe_rates"] = {
            "$comment": "Bytes per batch token per unit of hidden size for "
                        "ik's CPU-resident MoE intermediates, by architecture. "
                        "`default` applies to an architecture not listed.",
            "default": max(_IK_RATES.values()),
            "by_arch": dict(sorted(_IK_RATES.items())),
        }
    document["constants"] = dict(sorted(constants.items()))
    document["measurements"] = len(rows)

    if check:
        current = json.loads(path.read_text())
        if current == document:
            print(f"{path.name} matches the dataset ({len(rows)} measurements)")
            return 0
        print(f"{path.name} does NOT match the dataset:")
        for line in changed:
            print(f"  {line}")
        for line in failed:
            print(f"  {line}")
        if not changed and not failed:
            print("  evidence text differs; re-run without --check to refresh")
        return 1

    check_table_signs(document)
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {path} from {len(rows)} measurements")
    for line in changed:
        print(f"  changed: {line}")
    for line in failed:
        print(f"  NOT DERIVED: {line}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
