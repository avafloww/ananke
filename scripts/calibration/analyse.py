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
    swa_mask = swa_rows * tokens * width if swa else 0
    # Two f32 hidden-state buffers on mainline, one on ik.
    hidden = (1 if ik else 2) * parsed["n_embd"] * tokens * 4
    # ik keeps its MoE op intermediates on the CPU below a batch threshold,
    # measured at 81 KiB per batch token — see FINDINGS.md.
    experts, used = parsed.get("n_expert") or 0, parsed.get("n_expert_used") or 0
    if charge_moe and ik and experts and used and tokens * used < 32 * experts:
        hidden += IK_MOE_BYTES_PER_TOKEN * tokens
    return mask / 1024**2, swa_mask / 1024**2, hidden / 1024**2


IK_MOE_BYTES_PER_TOKEN = 81 * 1024


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

def derive_ik_moe_per_nembd(rows: list[dict]) -> tuple[int, str]:
    """Bytes per batch token per unit of hidden size for ik's CPU-MoE buffers."""
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
        points.append((parsed["n_embd"], excess / parsed["n_embd"], parsed.get("arch")))
    if not points:
        raise ValueError("no ik MoE cells below the offload threshold")
    per_unit = st.median(p[1] for p in points)
    detail = ", ".join(
        f"{arch} {rate * embd:.0f} B/token at n_embd {embd} ({rate:.1f}/unit)"
        for embd, rate, arch in sorted({(p[0], round(p[1], 1), p[2]) for p in points})
    )
    return round(per_unit), (
        f"{len(points)} cells below the offload threshold: {detail}. "
        "Replaces a flat 81 KiB, which was this term evaluated at qwen35moe's "
        "hidden size and frozen."
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
        mask, swa_mask, hidden = arena_terms(record)
        cards = len(factors["gpus"].split(","))
        copies = 4 if cards > 1 and (factors["split"] or "layer") == "layer" else 1
        residual = parsed["arena_mib"] - (copies * (mask + swa_mask) + hidden)
        tokens = min(factors["ctx"], factors["ubatch"])
        per = residual * 1024**2 / (parsed["n_layer"] * tokens)
        (residuals if per > 100 else controls).append(per)
    if not residuals:
        raise ValueError("no gemma4 E-variant cells")
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
# Empty deliberately. `derive_deepseek4_curve` works and is kept below, but
# is not wired in: what it measures is llama.cpp's own `compute` attribution,
# and what the constant must cover is everything the packer needs beyond
# ananke's *own* weights and KV predictions. Those are different quantities,
# and the existing test asserts the curve covers a 9.3 GiB residual at
# ctx 131072 where the compute column reads 2.0 GiB. Until the end-to-end
# predicted-versus-measured check exists, reducing this constant would risk
# under-reserving at load, which is the one failure direction that OOMs.
CURVE_DERIVERS: dict[str, object] = {}

# `derive_mtp_embedded_compute` is likewise held back. Its quantity — the
# driver delta between paired cells — is the right one, and it says the
# constant over-reserves by ~700 MiB at one slot. But the single `parallel = 4`
# pair shows a delta of 2892 MiB against the ~1100 the one-slot fit predicts,
# so there is a slot-count dependence the model does not carry. Lowering the
# constant without that term would under-reserve exactly the production
# configuration, which runs four slots.
DERIVERS = {
    "IK_MOE_CPU_BYTES_PER_NEMBD": derive_ik_moe_per_nembd,
    "GEMMA_E_VARIANT_BYTES_PER_LAYER_TOKEN": derive_gemma_e_per_layer_token,
    "PROCESS_BASE_BYTES_PER_DEVICE": derive_per_device_bytes,
    "MAINLINE_LAYER_SPLIT_MASK_COPIES": derive_layer_split_copies,
    "IK_OP_OFFLOAD_MIN_BATCH": derive_offload_min_batch,
}


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
        entry = next((e for e in document["compute_buffer_curves"]["entries"]
                      if arch in e["archs"]), None)
        if entry is None:
            failed.append(f"{arch}: curve deriver has no matching entry")
            continue
        try:
            base, evidence = deriver(rows)
        except (ValueError, KeyError, ZeroDivisionError, StatisticsError) as error:
            failed.append(f"{arch} curve: cannot derive — {error}")
            continue
        if entry["base_mib"] != base:
            changed.append(f"{arch} curve base: {entry['base_mib']} -> {base}")
        if entry.get("slope_mib_per_1k") != 1:
            changed.append(f"{arch} curve slope: {entry.get('slope_mib_per_1k')} -> 1")
        entry["base_mib"] = base
        # Nominal rather than zero: the measurement says flat, and claiming
        # exactly zero growth asserts more than a flat measurement supports.
        entry["slope_mib_per_1k"] = 1
        entry["slope_scales_with_ubatch"] = False
        entry["evidence"] = evidence

    for name, entry in constants.items():
        if name in DERIVERS:
            continue
        kind = KINDS.get(name)
        if kind is None:
            failed.append(f"{name}: no declared kind — add it to KINDS")
            continue
        entry["kind"] = kind

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

    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {path} from {len(rows)} measurements")
    for line in changed:
        print(f"  changed: {line}")
    for line in failed:
        print(f"  NOT DERIVED: {line}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
