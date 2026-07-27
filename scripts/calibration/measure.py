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
import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
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
    bench: bool = False
    """Drive the vendored coding-agent benchmark instead of one short request.

    A single request only reaches what the runtime allocates on first use. The
    benchmark runs a multi-turn loop with real coding prompts and feeds each
    real reply back in, so the context grows the way an agent's does and the
    prompt cache fills on representative tokens rather than on filler a cache
    or a drafter finds unnaturally easy.

    It does not cover everything. The loop is strictly sequential, so with
    `parallel > 1` only one slot is ever touched and per-slot state stays
    unallocated — `soak` with `concurrency` is what reaches those. Context also
    grows by only ~300 tokens a turn, so `bench_turns` has to be large before
    any claim about KV-driven growth at production context lengths.
    """
    bench_turns: int = 40
    verbose_log: bool = True
    """Whether to raise the loader's log verbosity.

    Needed to read the buffer sizes, but the tuning skill warns that verbose
    logging serialises graph ops. Growth runs turn it off: their subject is
    memory over time, and the arena is already known from the matching
    non-growth cell.
    """
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
        if self.verbose_log:
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

    hardware: dict[str, object] = dataclasses.field(default_factory=dict)

    def record(self) -> dict[str, object]:
        """One NDJSON record.

        Nested rather than flat: the hardware block, the per-device VRAM split,
        and the factor set all have their own shapes, and a fixed column set
        would force every one of them into a lowest common denominator — which
        is what previously capped the GPU breakdown at two devices and made
        adding a factor a schema migration.
        """
        return {
            "cell": self.cell.cell_id,
            "status": self.status,
            "provenance": self.provenance,
            "hardware": self.hardware,
            "factors": dataclasses.asdict(self.cell),
            "parsed": self.parsed,
            "rss": self.rss,
        }


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
# llama.cpp's own memory-breakdown row, which splits each device into
# model / context / compute. The compute column is what the estimator's
# per-architecture GPU curves are fitted against, so it has to be captured
# per device rather than as a total.
_BREAKDOWN = re.compile(
    r"- (CUDA\d+)[^|]*\|\s*\d+ = \d+ \+ \(\s*\d+ =\s*(\d+) \+\s*(\d+) \+\s*(\d+)\)"
)
MAX_GPUS = 2
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

    # Per-device VRAM split. Columns are fixed so the schema stays stable
    # whatever the machine; a device that did not report leaves zeros.
    devices = {name: (int(m), int(c), int(cp))
               for name, m, c, cp in _BREAKDOWN.findall(text)}
    for index in range(MAX_GPUS):
        model, kv, compute = devices.get(f"CUDA{index}", (0, 0, 0))
        parsed[f"gpu{index}_model_mib"] = model
        parsed[f"gpu{index}_kv_mib"] = kv
        parsed[f"gpu{index}_compute_mib"] = compute
    return parsed


class RssSampler:
    """Sample a process's resident memory on a fixed cadence, keeping peaks.

    ananke does not read `/proc` once; its snapshotter samples every two
    seconds and keeps *monotonic peaks*, which is what the rolling correction
    later divides by. A single snapshot therefore measures a different
    quantity than the daemon does, and misses anything transient — the pinned
    staging ring during a `--no-mmap` load, or growth part-way through a
    request. Matching the cadence makes a measurement here directly
    comparable to what the daemon will observe, and keeping the trace makes
    growth visible rather than merely suspected.
    """

    INTERVAL_SECONDS = 2.0

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.trace: list[tuple[float, dict[str, int]]] = []
        self.peak: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._start = time.monotonic()

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = read_rss(self.pid)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                return  # The process exited; the trace so far still stands.
            self.trace.append((time.monotonic() - self._start, sample))
            for key, value in sample.items():
                if value > self.peak.get(key, 0):
                    self.peak[key] = value
            self._stop.wait(self.INTERVAL_SECONDS)

    def growth(self) -> dict[str, int]:
        """Peak minus the first sample: what accumulated after startup."""
        if not self.trace:
            return {}
        first = self.trace[0][1]
        return {f"growth_{k}": self.peak.get(k, 0) - v for k, v in first.items()}


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


def hardware() -> dict[str, object]:
    """The machine, in enough detail to key a calibration curve on.

    Several terms are hardware-specific rather than universal: the CUDA
    runtime's host footprint scales with driver and device count, and CPU-side
    expert dequant with core count and memory topology. A constant fitted on
    one box is only transferable to another if you can tell the two apart, so
    the box is recorded alongside every measurement.
    """
    gpus = []
    query = _run(["nvidia-smi",
                  "--query-gpu=name,memory.total,compute_cap,driver_version",
                  "--format=csv,noheader,nounits"])
    for line in filter(None, query.splitlines()):
        name, total, cap, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "memory_total_mib": int(total),
                     "compute_capability": cap, "driver": driver})

    cpu: dict[str, object] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu["model"] = line.split(":", 1)[1].strip()
            break
    lscpu = _run(["lscpu"])
    for key, field in [("Core(s) per socket", "cores_per_socket"),
                       ("Socket(s)", "sockets"), ("CPU(s)", "threads"),
                       ("NUMA node(s)", "numa_nodes")]:
        for line in lscpu.splitlines():
            if line.startswith(key + ":"):
                cpu[field] = line.split(":", 1)[1].strip()
                break

    meminfo = Path("/proc/meminfo").read_text()
    total_kb = next((int(l.split()[1]) for l in meminfo.splitlines()
                     if l.startswith("MemTotal:")), 0)
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "gpus": gpus,
        "cpu": cpu,
        "mem_total_gib": round(total_kb / (1024 * 1024), 1),
        "kernel": os.uname().release,
        # The tuning skill's first sanity check: a `powersave` governor pins
        # cores to the base clock and silently halves CPU-bound throughput.
        "cpu_governor": governor.read_text().strip() if governor.exists() else "?",
        "transparent_hugepage": thp.read_text().strip() if thp.exists() else "?",
    }


def provenance(binary: str, cell: Cell | None = None) -> dict[str, str]:
    """Facts that make a stale row identifiable later.

    A constant fitted from these measurements is specific to a moment, a
    driver, a build, and a machine. Dates matter twice over: when the
    measurement was taken, and when the thing measured was *built* — a row
    from a llama.cpp of a given vintage stops describing the runtime the day
    someone bumps the pin, and the binary's own timestamp is what makes that
    visible without having to remember.
    """
    resolved = Path(shutil.which(binary) or binary)
    now = datetime.datetime.now(datetime.timezone.utc)
    record = {
        "measured_at_utc": now.isoformat(timespec="seconds"),
        "measured_at_local": now.astimezone().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "binary": str(resolved.resolve()) if resolved.exists() else str(resolved),
        "binary_built_at": _mtime(resolved),
        "ananke_rev": _run(["git", "rev-parse", "--short", "HEAD"]) or "?",
        "ananke_dirty": "yes" if _run(["git", "status", "--porcelain"]) else "no",
    }
    if cell is not None:
        record["model_file_at"] = _mtime(Path(cell.model))
    return record


def _mtime(path: Path) -> str:
    """A file's modification time, as an ISO 8601 UTC timestamp.

    Nix normalises store timestamps to the epoch for reproducibility, so a
    build date is not available for a store path — but the store hash in the
    recorded path *is* the build identity, and a more precise one than a date.
    Say that rather than record 1970.
    """
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return "?"
    if stamp < 86400 * 2:
        return "nix-store (build identity is the path hash)"
    return (datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
            .isoformat(timespec="seconds"))


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

    if cell.bench:
        _run_bench(cell, port)
        return

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


# The coding-agent benchmark, which drives a server with real prompts and real
# generated replies rather than repetitive filler. Realism is the part that
# cannot be synthesised: prompt-cache behaviour and generation both depend on
# what the tokens actually are. Vendored beside this module so a calibration
# run needs nothing outside the repository.
BENCH = Path(os.environ.get("CODING_BENCH", Path(__file__).parent / "coding_bench.py"))


def _run_bench(cell: Cell, port: int) -> None:
    if not BENCH.exists():
        print(f"    (bench not found at {BENCH}; falling back to soak)", flush=True)
        return
    subprocess.run(
        [sys.executable, str(BENCH), "--url", f"http://127.0.0.1:{port}",
         "--turns", str(cell.bench_turns), "--model", "m"],
        capture_output=True, text=True, timeout=3600, check=False,
    )


def available_gib() -> float:
    """Host memory the kernel says is available, in GiB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def model_gib(cell: Cell) -> float:
    """On-disk size of the model, shards included, in GiB."""
    first = Path(cell.model)
    if not first.exists():
        return 0.0
    stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first.name)
    if stem == first.name:
        return first.stat().st_size / (1024**3)
    shards = sorted(first.parent.glob(f"{stem}-*-of-*.gguf"))
    return sum(s.stat().st_size for s in shards) / (1024**3)


def fits(cell: Cell, headroom_gib: float) -> bool:
    """Whether a cell can run without risking the machine.

    An unattended sweep that pushes the box into swap or the OOM killer costs
    far more than the row it was trying to collect, so a cell that cannot fit
    with headroom to spare is skipped and recorded as skipped. `--no-mmap`
    charges the whole model to anonymous memory; a mapped load can rely on
    page cache being reclaimable and needs only the host-resident share, which
    is not known ahead of time — so the conservative figure is used for both.
    """
    return model_gib(cell) + headroom_gib <= available_gib()


def measure(cell: Cell, log_dir: Path, port: int, load_timeout: int) -> Measurement:
    binary = {"mainline": os.environ.get("MAINLINE_BIN", "llama-server"),
              "ik": os.environ.get("IK_BIN", "ik-llama-server")}[cell.runtime]
    prov = provenance(binary, cell)
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
                    return Measurement(cell, prov, {}, {}, status="failed-to-load",
                                   hardware=hardware())
                if time.monotonic() > deadline:
                    _stop(proc)
                    return Measurement(cell, prov, {}, {}, status="timeout",
                                       hardware=hardware())
                time.sleep(HEALTH_POLL_SECONDS)

            with RssSampler(proc.pid) as sampler:
                exercise(cell, port)
                time.sleep(SETTLE_SECONDS)
                final = read_rss(proc.pid)
            # Report the peak, because that is the figure ananke's snapshotter
            # keeps and the correction divides by; the final reading and the
            # growth since startup ride alongside it.
            rss = dict(sampler.peak or final)
            rss |= {f"final_{k}": v for k, v in final.items()}
            rss |= sampler.growth()
            rss["samples"] = len(sampler.trace)
            parsed = parse_log(log_path.read_text(errors="replace"))
            if cell.bench:
                _write_trace(log_dir / f"{cell.cell_id}-{cell.label}-trace.csv", sampler)
        finally:
            _stop(proc)

    return Measurement(cell, prov, parsed, rss, hardware=hardware())


def _write_trace(path: Path, sampler: RssSampler) -> None:
    """Persist the full time series for a growth run, not just its summary."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_seconds", "rss_total_kb", "rss_anon_kb",
                         "rss_file_kb", "rss_shmem_kb"])
        for elapsed, sample in sampler.trace:
            writer.writerow([f"{elapsed:.1f}", sample["rss_total_kb"],
                             sample["rss_anon_kb"], sample["rss_file_kb"],
                             sample["rss_shmem_kb"]])


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
    seen = set()
    for line in out.read_text().splitlines():
        if line.strip():
            seen.add(json.loads(line)["cell"])
    return seen


def append(out: Path, measurement: Measurement) -> None:
    """Append one NDJSON record.

    Self-describing per line, so a run that adds a factor or a parsed field
    appends happily beside older records instead of needing a schema
    migration — and an analysis reads what each record actually carries.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(measurement.record(), default=str) + "\n")


def run_cells(cells: list[Cell], out: Path, log_dir: Path, port: int,
              load_timeout: int, headroom_gib: float = 30.0) -> None:
    done = already_measured(out)
    for index, cell in enumerate(cells, start=1):
        prefix = f"[{index}/{len(cells)}]"
        if cell.cell_id in done:
            print(f"{prefix} skip {cell.label} ({cell.cell_id})", flush=True)
            continue
        if not fits(cell, headroom_gib):
            print(f"{prefix} skip {cell.label}: needs "
                  f"{model_gib(cell):.0f} GiB + {headroom_gib:.0f} headroom, "
                  f"{available_gib():.0f} available", flush=True)
            append(out, Measurement(cell, provenance("true", cell), {}, {},
                                    status="skipped-insufficient-memory",
                                    hardware=hardware()))
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
    parser.add_argument("--headroom-gib", type=float, default=30.0,
                        help="host memory to leave free; a cell needing more "
                             "than the remainder is skipped rather than risking "
                             "the machine")
    parser.add_argument("--load-timeout", type=int, default=1800,
                        help="seconds to wait for a model to load; a 200 GiB "
                             "--no-mmap load takes minutes")
    args = parser.parse_args(argv)

    if not args.plan:
        parser.error("--plan is required")
    run_cells(load_plan(args.plan), args.out, args.log_dir, args.port,
              args.load_timeout, args.headroom_gib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
