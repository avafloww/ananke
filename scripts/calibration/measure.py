"""Measure one llama-server configuration's host-memory decomposition.

Every memory constant in ananke's estimator should be traceable to rows this
module produces. It records a *decomposition* rather than a total, because the
decomposition is what the model is built from:

    arena     the graph allocator's buffer, as the loader logs it. Pinned
              (``CUDA_Host``) whenever a GPU is present, plain ``CPU`` otherwise.
    pinned    ``RssShmem - arena``. ``cudaMallocHost`` is accounted as shmem, so
              this is the pinned memory that is *not* the graph arena.
    baseline  ``RssAnon - CPU KV - (host weights, when anonymous)``. What the
              process holds beyond anything the model already explains.

Whether host weights are anonymous is read from the loader's own naming
(``CPU_Mapped model buffer size`` vs ``CPU model buffer size``) rather than
inferred from flags, because mainline and ik_llama disagree on it for identical
configurations.

Cells are keyed by a hash of their factors, and a cell already present in the
output is skipped, so a long campaign survives interruption and can be extended
later without redoing work.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PORT = 18099
HEALTH_POLL_SECONDS = 2
SETTLE_SECONDS = 3


@dataclass(frozen=True)
class Cell:
    """One measurable configuration.

    Every field is a factor that could plausibly move host memory, and every
    one is recorded, so a row is a complete description of the process that
    produced it. Adding a factor here automatically adds it to the output and
    to the cell identity.
    """

    label: str
    model: str
    runtime: str = "mainline"
    gpus: str = "0"
    ctx: int = 32768
    ubatch: int = 512
    batch: int | None = None
    parallel: int = 1
    ngl: int = 99
    split: str | None = None
    kv_type: str = "f16"
    kv_unified: bool = False
    flash_attn: str = "on"
    n_cpu_moe: int | None = None
    mmproj: str | None = None
    draft: str | None = None
    spec_type: str | None = None
    threads: int | None = None
    numa: str | None = None
    cram: int = 0
    no_mmap: bool = False
    rtr: bool = False
    thp: bool = False
    embeddings: bool = False
    served: bool = True
    soak: int = 0
    concurrency: int = 1
    extra: tuple[str, ...] = ()
    repeat: int = 0
    """Distinguishes otherwise identical cells.

    Repeats are how the noise floor gets measured, and the resume key would
    otherwise collapse them into one. It takes part in the cell identity and
    in nothing else — `argv` never sees it.
    """

    @property
    def cell_id(self) -> str:
        """Stable identity, so a rerun skips what has already been measured."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def argv(self, binary: str, port: int) -> list[str]:
        args = [
            binary,
            "-m", self.model,
            "-c", str(self.ctx),
            "-ub", str(self.ubatch),
            "-ngl", str(self.ngl),
            "-np", str(self.parallel),
            "-cram", str(self.cram),
            "-fa", self.flash_attn,
            "-ctk", self.kv_type,
            "-ctv", self.kv_type,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        # The two runtimes gate their buffer-size logging differently, and
        # without those lines there is nothing to measure.
        args += ["-lv", "5"] if self.runtime == "mainline" else ["--verbosity", "1"]
        optional = [
            ("-b", self.batch),
            ("--split-mode", self.split),
            ("--n-cpu-moe", self.n_cpu_moe),
            ("--mmproj", self.mmproj),
            ("-md", self.draft),
            ("--spec-type", self.spec_type),
            ("-t", self.threads),
            ("--numa", self.numa),
        ]
        for flag, value in optional:
            if value is not None:
                args += [flag, str(value)]
        for flag, on in [
            ("-kvu", self.kv_unified),
            ("--no-mmap", self.no_mmap),
            ("-rtr", self.rtr),
            ("--use-thp", self.thp),
            ("--embeddings", self.embeddings),
        ]:
            if on:
                args.append(flag)
        return args + list(self.extra)


@dataclass
class Measurement:
    """What one run of a cell yielded."""

    cell: Cell
    provenance: dict[str, str]
    parsed: dict[str, float | str]
    rss: dict[str, int]
    status: str = "ok"

    def row(self) -> dict[str, object]:
        row: dict[str, object] = {"cell": self.cell.cell_id, "status": self.status}
        row |= self.provenance
        row |= {k: _scalar(v) for k, v in dataclasses.asdict(self.cell).items()}
        row |= self.parsed
        row |= self.rss
        return row


def _scalar(value: object) -> object:
    """Flatten a factor for CSV, keeping empty distinguishable from zero."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (tuple, list)):
        return " ".join(str(v) for v in value)
    return value


# Buffer-size lines the loaders emit. Both forks are matched by one pattern
# each; the last occurrence wins because the loader logs a reserve pass first
# and then the real graph, with the same figure.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "arena_mib": re.compile(r"(?:CUDA_Host|CPU) compute buffer size *= *([0-9.]+)"),
    "out_buf_mib": re.compile(r"(?:CUDA_Host|CPU) +output buffer size *= *([0-9.]+)"),
    "cpu_kv_mib": re.compile(r"CPU KV buffer size *= *([0-9.]+)"),
    "cpu_model_mib": re.compile(r"CPU(?:_Mapped)? model buffer size *= *([0-9.]+)"),
}
_META = re.compile(r"(n_layer|n_embd|n_expert|n_expert_used|n_swa|n_vocab) *= *(\d+)")
_ARCH = re.compile(r"arch *= *([A-Za-z0-9_.-]+)")


def parse_log(text: str) -> dict[str, float | str]:
    """Pull the model's shape and its logged buffer sizes out of a load log."""
    parsed: dict[str, float | str] = {}
    for name, pattern in _PATTERNS.items():
        found = pattern.findall(text)
        parsed[name] = float(found[-1]) if found else 0.0
    # `CPU_Mapped` means the host-side weights are file-backed and so land in
    # RssFile; plain `CPU` means they were read into anonymous memory.
    parsed["cpu_model_mapped"] = "yes" if "CPU_Mapped model buffer" in text else "no"
    meta = {k: int(v) for k, v in _META.findall(text)}
    for key in ("n_layer", "n_embd", "n_expert", "n_expert_used", "n_swa", "n_vocab"):
        parsed[key] = meta.get(key, 0)
    arch = _ARCH.search(text)
    parsed["arch"] = arch.group(1) if arch else "?"
    return parsed


def read_rss(pid: int) -> dict[str, int]:
    """The `/proc/<pid>/status` resident-memory breakdown, in kB.

    The same three figures ananke's `ProcFs` reads, so a measurement here is
    directly comparable to what the daemon will observe in production.
    """
    fields = {"VmRSS": "rss_total_kb", "RssAnon": "rss_anon_kb",
              "RssFile": "rss_file_kb", "RssShmem": "rss_shmem_kb"}
    out = dict.fromkeys(fields.values(), 0)
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in fields:
            out[fields[key]] = int(line.split()[1])
    return out


def provenance(binary: str) -> dict[str, str]:
    """Facts that make a stale row identifiable later.

    A constant fitted from these measurements is specific to a driver, a build,
    and a machine; without recording them there is no way to tell whether an
    old row still describes the system.
    """
    driver = _run(["nvidia-smi", "--query-gpu=driver_version",
                   "--format=csv,noheader"]) or "none"
    resolved = shutil.which(binary) or binary
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": os.uname().nodename,
        "driver": driver.splitlines()[0].strip() if driver else "none",
        "binary": str(Path(resolved).resolve()),
        "ananke_rev": _run(["git", "rev-parse", "--short", "HEAD"]) or "?",
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _post(port: int, path: str, payload: dict, timeout: float) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # A failed probe still leaves the process measurable.


def _healthy(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def exercise(cell: Cell, port: int) -> None:
    """Put the server through enough work to allocate what it allocates lazily.

    Several host buffers are sized on first use rather than at load, so an idle
    process under-reports. `soak` goes further and grows the prompt across
    successive requests, which is the only way to reach terms that accumulate
    with use — the prompt cache and per-slot checkpoints — rather than at load.
    """
    if not cell.served:
        return
    if cell.embeddings:
        _post(port, "/v1/embeddings", {"input": "calibration probe"}, 120)
        return
    _post(port, "/v1/chat/completions",
          {"model": "m", "messages": [{"role": "user", "content": "Count to twenty."}],
           "max_tokens": 64}, 300)

    prompt = "Explain memory allocation."
    for i in range(cell.soak):
        prompt += f" Also cover point {i} in detail, at length, with examples."
        payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 256}
        # Overlapping requests touch slots beyond the first, which a strictly
        # serial probe never reaches.
        import threading
        threads = [threading.Thread(target=_post, args=(port, "/v1/chat/completions",
                                                        payload, 300))
                   for _ in range(cell.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def measure(cell: Cell, log_dir: Path, port: int, load_timeout: int) -> Measurement:
    binary = {"mainline": os.environ.get("MAINLINE_BIN", "llama-server"),
              "ik": os.environ.get("IK_BIN", "ik-llama-server")}[cell.runtime]
    prov = provenance(binary)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.cell_id}-{cell.label}.log"

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cell.gpus)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(cell.argv(binary, port), stdout=log_file,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + load_timeout
            while not _healthy(port):
                if proc.poll() is not None:
                    return Measurement(cell, prov, {}, {}, status="failed-to-load")
                if time.monotonic() > deadline:
                    _stop(proc)
                    return Measurement(cell, prov, {}, {}, status="timeout")
                time.sleep(HEALTH_POLL_SECONDS)

            exercise(cell, port)
            time.sleep(SETTLE_SECONDS)
            rss = read_rss(proc.pid)
            parsed = parse_log(log_path.read_text(errors="replace"))
        finally:
            _stop(proc)

    return Measurement(cell, prov, parsed, rss)


def _stop(proc: subprocess.Popen) -> None:
    """Stop by pid, never by pattern — a pattern can match the driving shell."""
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)


def already_measured(out: Path) -> set[str]:
    if not out.exists():
        return set()
    with out.open() as handle:
        return {row["cell"] for row in csv.DictReader(handle)}


def append(out: Path, measurement: Measurement) -> None:
    row = measurement.row()
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_cells(cells: list[Cell], out: Path, log_dir: Path, port: int,
              load_timeout: int) -> None:
    done = already_measured(out)
    for index, cell in enumerate(cells, start=1):
        prefix = f"[{index}/{len(cells)}]"
        if cell.cell_id in done:
            print(f"{prefix} skip {cell.label} ({cell.cell_id})", flush=True)
            continue
        print(f"{prefix} {cell.label} ...", end=" ", flush=True)
        measurement = measure(cell, log_dir, port, load_timeout)
        append(out, measurement)
        if measurement.status != "ok":
            print(f"{measurement.status}; see {log_dir}", flush=True)
            continue
        print(
            f"arena={measurement.parsed['arena_mib']:.2f} "
            f"anon={measurement.rss['rss_anon_kb'] // 1024} "
            f"shmem={measurement.rss['rss_shmem_kb'] // 1024} "
            f"file={measurement.rss['rss_file_kb'] // 1024} MiB",
            flush=True,
        )


def load_plan(path: Path) -> list[Cell]:
    """Read a campaign plan: a JSON list of objects, each a `Cell`'s fields."""
    raw = json.loads(path.read_text())
    return [Cell(**{**entry, "extra": tuple(entry.get("extra", []))}) for entry in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="JSON list of cells to run")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(os.environ.get("TMPDIR", "/tmp")) / "ananke-calibration")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--load-timeout", type=int, default=1800,
                        help="seconds to wait for a model to load; a 200 GiB "
                             "--no-mmap load takes minutes")
    args = parser.parse_args(argv)

    if not args.plan:
        parser.error("--plan is required")
    run_cells(load_plan(args.plan), args.out, args.log_dir, args.port, args.load_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
