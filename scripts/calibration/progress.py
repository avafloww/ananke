"""Report how far the campaign has got, and whether it is still moving.

    python progress.py            # once
    python progress.py --watch    # refresh every 30s

Reads only committed and in-progress data files, so it is safe to run against
a live campaign — it never touches the GPUs or the running server.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def data_file(phase: str) -> Path | None:
    """The phase's data file, whatever suffix it was first written with.

    The early phases were named for what they measured rather than for their
    number, and renaming them now would strand the measurements.
    """
    exact = DATA / f"{phase}.ndjson"
    if exact.exists():
        return exact
    matches = sorted(DATA.glob(f"{phase}-*.ndjson"))
    return matches[0] if matches else None


def records(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # The campaign may be mid-write; a torn last line is expected.
                pass
    return out


def planned(phase: str) -> int:
    """How many cells the phase intends, regenerating the plan if need be."""
    saved = DATA / f"{phase}-plan.json"
    if saved.exists():
        try:
            return len(json.loads(saved.read_text()))
        except json.JSONDecodeError:
            pass
    result = subprocess.run([sys.executable, str(HERE / "plan.py"), phase],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    return len(json.loads(result.stdout))


def report(phases: list[str]) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    total_done = total_planned = 0
    latest: datetime.datetime | None = None
    print(f"{'phase':10} {'done':>12}  {'status':<28} last record")
    for phase in phases:
        rows = records(data_file(phase))
        want = planned(phase)
        # Only a completed measurement counts as done; the campaign retries
        # skips and load failures, so counting them would overstate progress.
        ok = sum(1 for r in rows if r.get("status", "ok") == "ok")
        total_done += ok
        total_planned += want
        counts = Counter(r.get("status", "ok") for r in rows if
                         r.get("status", "ok") != "ok")
        issues = ", ".join(f"{n}x {s}" for s, n in counts.most_common()) or "-"
        stamp = ""
        if rows:
            when = rows[-1].get("provenance", {}).get("measured_at_utc", "")
            stamp = when
            try:
                seen = datetime.datetime.fromisoformat(when)
                latest = max(latest, seen) if latest else seen
            except ValueError:
                pass
        bar = f"{ok}/{want}" if want else f"{ok}/?"
        print(f"{phase:10} {bar:>12}  {issues:<28} {stamp}")

    print(f"\ntotal {total_done}/{total_planned}"
          f" ({100 * total_done / total_planned:.0f}%)" if total_planned else "")
    if latest:
        idle = (now - latest).total_seconds() / 60
        state = "running" if idle < 45 else "STALLED or finished"
        print(f"last record {idle:.0f} min ago — {state}")
    alive = subprocess.run(["pgrep", "-af", "campaign.py"],
                           capture_output=True, text=True).stdout.strip()
    print(f"campaign process: {alive.splitlines()[0] if alive else 'not running'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", nargs="+",
                        default=["phase0", "phase1", "phase2", "phase2b",
                                 "phase3", "phase4", "phase5"])
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        report(args.phases)
        if not args.watch:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
