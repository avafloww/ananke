#!/usr/bin/env python3
"""Compare packed estimates against production nvidia-smi totals.

Runs :mod:`dump_estimates` and joins its per-device placement against the
``prod-*`` rows in ``data/measurements.ndjson``, printing the drift for each
model. This is the top-level pass/fail signal for the ±5% estimation campaign.

Usage::

    python3 scripts/calibration/scoreboard.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEASUREMENTS = HERE / "data" / "measurements.ndjson"

# Maps a `models.toml` entry name to the `factors.label` of its production cell.
PROD_LABELS = {
    "qwen3.6-35b-a3b": "prod-qwen36-35b-a3b",
    "qwen3.6-27b": "prod-qwen36-27b",
    "gemma-4-31b-it-qat": "prod-gemma4-31b-qat",
    "deepseek-v4-flash": "prod-dsv4f",
    "glm-5.2": "prod-glm52",
    "laguna-s-2.1-iq4-nl": "prod-laguna",
    "talkie-1930-13b-it": "prod-talkie",
}

TOLERANCE_PCT = 5.0


def production_totals() -> dict[str, int]:
    totals: dict[str, int] = {}
    with MEASUREMENTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            label = row["factors"].get("label", "")
            if label.startswith("prod-") and row["status"] == "ok":
                totals[label] = row["rss"]["gpu_used_mib"]
    return totals


def main() -> int:
    raw = subprocess.run(
        [sys.executable, str(HERE / "dump_estimates.py"), "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    estimates = {row["name"]: row for row in json.loads(raw)}
    prod = production_totals()

    print(f"{'Model':<24} {'Est GPU':>9} {'Prod GPU':>9} {'Drift':>8}")
    worst = 0.0
    for name, label in PROD_LABELS.items():
        est_row = estimates.get(name)
        measured = prod.get(label)
        if est_row is None or measured is None:
            print(f"{name:<24} {'-':>9} {'-':>9} {'missing':>8}")
            continue
        placement = est_row["placement"]
        est = sum(v for k, v in placement.items() if k.startswith("gpu:"))
        drift = 100.0 * (est - measured) / measured
        flag = "" if abs(drift) <= TOLERANCE_PCT else "  <-- FAIL"
        worst = max(worst, abs(drift))
        print(f"{name:<24} {est:>9} {measured:>9} {drift:>+7.1f}%{flag}")
    print(f"\nworst drift: {worst:.1f}% (tolerance {TOLERANCE_PCT}%)")
    return 0 if worst <= TOLERANCE_PCT else 1


if __name__ == "__main__":
    sys.exit(main())
