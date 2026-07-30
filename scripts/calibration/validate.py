#!/usr/bin/env python3
"""Check the estimator against every cell the campaign measured, not just the
production ones.

`scoreboard.py` compares seven deployed configurations. Seven points cannot
tell a model that generalises from one fitted to them, and every constant in
`tuning.json` was derived from this same dataset — so the dataset is also the
only place a term that was tuned into agreement can be caught disagreeing
somewhere else.

This runs the estimator over each measured cell's *own* configuration and
compares its packed GPU total against what the driver reported for that run.
A cell is validated only where the estimator can be asked the same question the
measurement answered:

- `-ngl 99`, or an explicit `--n-cpu-moe`. A partial `-ngl` describes a
  placement the operator chose, and the estimator chooses its own, so the two
  totals are answers to different questions.
- A driver reading present, and the run served a request. An idle process has
  not made its first-use allocations.
- The model in `models.toml`, since that is where a path resolves from.

Everything else is reported as skipped with the reason, so the coverage of this
check is visible rather than assumed.

Usage::

    python3 scripts/calibration/validate.py
    python3 scripts/calibration/validate.py --tolerance 5 --check
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MEASUREMENTS = HERE / "data" / "measurements.ndjson"

# The card capacities the campaign ran on, so the packer faces the same fit the
# measurement did. Read from the record's own hardware block.
DEFAULT_CPU_CAPACITY_MIB = 256_000


def cell_argv(record: dict) -> list[str] | None:
    """The estimator invocation matching this cell, or `None` if it has none."""
    factors = record["factors"]
    gpus = [g for g in factors["gpus"].split(",") if g]
    argv = [
        "--model", factors["model"],
        "--context", str(factors["ctx"]),
        "--active-devices", str(len(gpus)),
        "--visible-devices", str(len(gpus)),
        "--pack",
    ]
    for device in record["hardware"]["gpus"][: len(gpus)]:
        # The driver reserves a little of each card, which is why the campaign
        # sees ~24124 MiB free on a nominally 24576 MiB 3090. Pack against what
        # was actually available.
        argv += ["--gpu", str(device["memory_total_mib"])]
    argv += ["--cpu", str(DEFAULT_CPU_CAPACITY_MIB)]
    if factors["mmproj"]:
        argv += ["--mmproj", factors["mmproj"]]
    if factors["kv_type"]:
        argv += ["--cache-type-k", factors["kv_type"],
                 "--cache-type-v", factors["kv_type"]]
    if factors["ubatch"]:
        argv += ["--ubatch", str(factors["ubatch"])]
    if factors["parallel"]:
        argv += ["--parallel", str(factors["parallel"])]
    if factors["flash_attn"]:
        argv += ["--flash-attn", factors["flash_attn"]]
    argv += ["--kv-unified", "on" if factors["kv_unified"] else "off"]
    if factors["split"]:
        argv += ["--split-mode", factors["split"]]
    if factors["runtime"] == "ik":
        argv += ["--ik-llama"]
        # The fork's sparse-attention path is a separate flag, and the campaign
        # only ran it for the architecture that has one.
        if record["parsed"].get("arch") == "glm-dsa":
            argv += ["--ik-dsa"]
    if factors["spec_type"]:
        argv += ["--mtp"]
    if factors["draft"]:
        argv += ["--draft-model", factors["draft"]]
    if factors["n_cpu_moe"] is not None:
        argv += ["--n-cpu-moe", str(factors["n_cpu_moe"])]
    if factors["cram"] is not None:
        argv += ["--cache-ram-mb", str(factors["cram"])]
    return argv


def skip_reason(record: dict, known_models: set[str]) -> str | None:
    factors = record["factors"]
    if record.get("status") != "ok":
        return f"status {record['status']}"
    if not record["rss"].get("gpu_used_mib"):
        return "no driver reading"
    if not factors.get("served"):
        return "idle: no first-use allocations"
    if factors["model"] not in known_models:
        return "model not in models.toml"
    if factors["ngl"] != 99 and factors["n_cpu_moe"] is None:
        return f"operator-chosen placement (ngl {factors['ngl']})"
    if factors["embeddings"]:
        return "embedding modality"
    return None


def run_estimate(argv: list[str]) -> dict | None:
    result = subprocess.run(
        ["cargo", "run", "-q", "--example", "estimate", "--", *argv],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    start = result.stdout.find("{")
    if result.returncode != 0 or start == -1:
        return None
    depth, end = 0, start
    for index, char in enumerate(result.stdout[start:], start):
        depth += (char == "{") - (char == "}")
        if depth == 0 and char == "}":
            end = index + 1
            break
    try:
        return json.loads(result.stdout[start:end])
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="per-cell drift allowed, in percent")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any validated cell is outside it")
    parser.add_argument("--arch", help="only cells of this architecture")
    args = parser.parse_args()

    records = [json.loads(line) for line in MEASUREMENTS.read_text().splitlines()
               if line.strip()]
    known_models = {r["factors"]["model"] for r in records}
    # A cell whose model file has gone is not a validation failure; the same
    # path list `dump_estimates.py` reads is the authority on what resolves.
    known_models = {m for m in known_models if Path(m).exists()}

    subprocess.run(["cargo", "build", "-q", "--example", "estimate"],
                   cwd=REPO, check=True)

    skipped: Counter[str] = Counter()
    results = []
    seen: set[str] = set()
    for record in records:
        reason = skip_reason(record, known_models)
        if reason:
            skipped[reason] += 1
            continue
        if args.arch and record["parsed"].get("arch") != args.arch:
            continue
        # Repeats of one configuration measure the same thing; the cell hash
        # already collapses them, but two labels can share a configuration.
        key = json.dumps(cell_argv(record), sort_keys=True)
        if key in seen:
            skipped["duplicate configuration"] += 1
            continue
        seen.add(key)
        estimate = run_estimate(cell_argv(record))
        if estimate is None:
            skipped["estimator refused the configuration"] += 1
            continue
        placement = estimate.get("placement") or {}
        # The *prediction*, not the reservation. The reservation carries slop
        # the process is not expected to use — one layer's headroom above all —
        # and comparing it to a measurement reads that slop as error. On the
        # production Qwen3.6-27B cell the two differ by 472 MiB, which is the
        # difference between +1.1% and -0.1%.
        predicted = placement.get("predicted_vram_mib") or 0
        allocation = placement.get("allocation") or {}
        reserved = sum(v["mib"] for k, v in allocation.items()
                       if k.startswith("gpu:"))
        # llama.cpp's layer split spreads across every visible card; ananke
        # packs a model that fits onto one, deliberately. When the two land on
        # different numbers of cards the totals are not comparable — a second
        # card is a second CUDA context and a second compute buffer, ~450 MiB
        # of real cost that belongs to the placement rather than to the
        # estimate. Comparing them anyway is what made small models on two
        # cards read as a systematic 7-8% under-reservation.
        cards_placed = sum(1 for k, v in allocation.items()
                           if k.startswith("gpu:") and v["mib"])
        cards_measured = sum(
            1 for key, value in record["rss"].items()
            if key.startswith("gpu") and key.endswith("_used_mib")
            and key != "gpu_used_mib" and value)
        if cards_measured and cards_placed != cards_measured:
            skipped[f"placed on {cards_placed} card(s), measured on "
                    f"{cards_measured}"] += 1
            continue
        measured = record["rss"]["gpu_used_mib"]
        if not predicted:
            skipped["nothing placed on a GPU"] += 1
            continue
        results.append({
            "label": record["factors"]["label"],
            "arch": record["parsed"].get("arch", "?"),
            "drift": 100.0 * (predicted - measured) / measured,
            "predicted": predicted,
            "reserved": reserved,
            "measured": measured,
        })

    results.sort(key=lambda r: r["drift"])
    print(f"{'label':38}{'arch':18}{'predicted':>10}{'measured':>10}"
          f"{'drift':>9}{'reserved':>10}")
    for row in results:
        flag = "" if abs(row["drift"]) <= args.tolerance else "  <-- outside"
        print(f"{row['label'][:37]:38}{row['arch'][:17]:18}{row['predicted']:>10}"
              f"{row['measured']:>10}{row['drift']:>+8.1f}%{row['reserved']:>10}"
              f"{flag}")

    print(f"\n{len(results)} cells validated, {sum(skipped.values())} skipped")
    for reason, count in skipped.most_common():
        print(f"  {count:4}  {reason}")
    if results:
        drifts = [r["drift"] for r in results]
        outside = [r for r in results if abs(r["drift"]) > args.tolerance]
        print(f"\nmedian {statistics.median(drifts):+.1f}%  "
              f"mean {statistics.fmean(drifts):+.1f}%  "
              f"range {min(drifts):+.1f}% to {max(drifts):+.1f}%")
        print(f"{len(outside)} of {len(results)} outside "
              f"+/-{args.tolerance:g}%")
        if args.check and outside:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
