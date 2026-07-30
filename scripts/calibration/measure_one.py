#!/usr/bin/env python3
"""Measure a plan's cells without the pre-flight memory gate, one at a time.

`measure.py` refuses a cell whose *model file* does not fit in available memory
plus a headroom margin. That is the right check for a fully GPU-resident load,
and too strict for a heavily expert-offloaded one: GLM-5.2's file is 205 GiB but
its process peaks at 187 GiB of anonymous memory, because the GPU-resident share
never touches host RAM. The gate cannot tell the difference, so a cell that has
been measured before becomes unmeasurable.

This runs the same `measure()` the campaign uses and appends the same record,
with two differences: no fit gate, and a swap watchdog. If the kernel starts
swapping more than `--swap-limit-gib`, the run stops before the box begins to
thrash — the margin here is a gigabyte or two, so that is a real possibility
rather than a formality.

    python3 scripts/calibration/measure_one.py --plan plan.json --out data/measurements.ndjson
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_measure():
    spec = importlib.util.spec_from_file_location("measure", HERE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before executing: `@dataclass` resolves its own module out of
    # `sys.modules`, and raises on a module that is not there yet.
    sys.modules["measure"] = module
    spec.loader.exec_module(module)
    return module


def swap_used_gib() -> float:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        fields[key] = float(rest.strip().split()[0])
    return (fields["SwapTotal"] - fields["SwapFree"]) / 1024 / 1024


class SwapWatchdog:
    """Kill every `ik-llama-server`/`llama-server` if swap grows past a limit.

    Deliberately blunt: the alternative to stopping is a box that stops
    responding, and the measurement is worthless either way once it is paging.
    """

    def __init__(self, limit_gib: float) -> None:
        self.limit = limit_gib
        self.baseline = swap_used_gib()
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "SwapWatchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(2.0):
            grown = swap_used_gib() - self.baseline
            if grown > self.limit:
                self.tripped = True
                print(f"\n  swap grew {grown:.1f} GiB past the baseline — stopping "
                      f"the server before the box thrashes", flush=True)
                for name in ("ik-llama-server", "llama-server"):
                    subprocess.run(["pkill", "-f", name], check=False)
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path,
                        default=Path("/tmp/ananke-calibration"))
    parser.add_argument("--archive-dir", type=Path, default=HERE / "data" / "logs")
    parser.add_argument("--load-timeout", type=int, default=1800)
    parser.add_argument("--swap-limit-gib", type=float, default=4.0)
    args = parser.parse_args()

    measure = load_measure()
    cells = measure.load_plan(args.plan)
    done = measure.already_measured(args.out)
    for index, cell in enumerate(cells, start=1):
        if cell.cell_id in done:
            print(f"[{index}/{len(cells)}] skip {cell.label} (measured)", flush=True)
            continue
        print(f"[{index}/{len(cells)}] {cell.label} "
              f"(swap {swap_used_gib():.1f} GiB used) ...", end=" ", flush=True)
        started = time.monotonic()
        with SwapWatchdog(args.swap_limit_gib) as watchdog:
            result = measure.measure(cell, args.log_dir, measure.DEFAULT_PORT,
                                     args.load_timeout, args.archive_dir)
        if watchdog.tripped:
            print("aborted on swap growth; not recording", flush=True)
            return 1
        measure.append(args.out, result)
        print(f"{result.status} in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
