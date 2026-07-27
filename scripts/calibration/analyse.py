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
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data" / "measurements.ndjson"


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


def arena_terms(record: dict) -> tuple[float, float, float]:
    """The modelled arena, split into (mask, swa_mask, hidden) MiB.

    mainline sizes the KQ mask against one slot's share of the context unless
    the cache is unified; ik does not divide by slots at all. An interleaved
    SWA model carries a second mask sized to the window plus the batch, not to
    the window alone.
    """
    factors, parsed = record["factors"], record["parsed"]
    ctx, ubatch = factors["ctx"], factors["ubatch"]
    slots, unified = factors["parallel"], factors["kv_unified"]
    ik = factors["runtime"] == "ik"

    n_kv = ctx if (ik or unified or slots == 1) else ctx // slots
    n_kv = pad(n_kv)
    tokens = min(ctx, ubatch)
    width = 2 if factors["flash_attn"] == "on" else 4

    mask = n_kv * tokens * width
    swa = parsed.get("n_swa") or 0
    # mainline sizes the second mask to the window plus the batch; ik sizes it
    # to the whole context, which is why an SWA model costs it so much more.
    swa_rows = n_kv if ik else pad(swa + tokens)
    swa_mask = swa_rows * tokens * width if swa else 0
    # Two f32 hidden-state buffers on mainline, one on ik.
    hidden = (1 if ik else 2) * parsed["n_embd"] * tokens * 4
    return mask / 1024**2, swa_mask / 1024**2, hidden / 1024**2


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
        key = (parsed.get("arch", "?"), factors["runtime"], cards,
               factors["flash_attn"])
        groups[key].append((parsed["arena_mib"], mask, swa_mask, hidden, record))

    print(f"{'arch':12}{'runtime':9}{'cards':>6}{'fa':>4}{'n':>4}"
          f"{'K (mask multiple)':>20}{'residual MiB':>14}")
    for key in sorted(groups):
        arch, runtime, cards, fa = key
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
        print(f"{arch:12}{runtime:9}{cards:>6}{fa:>4}{len(rows_here):>4}"
              f"{k + spread:>20}{st.median(residuals):>14.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["arena"], nargs="?", default="arena")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    rows = load(args.data)
    print(f"{len(rows)} completed measurements\n")
    check_arena(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
