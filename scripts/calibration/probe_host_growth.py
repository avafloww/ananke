"""Attribute host-memory growth that a single measurement cannot separate.

`measure.py` samples one process once per configuration. That is the right
shape for fitting constants, but it cannot tell a term that is allocated once
from one that accumulates with use — both look like "this model holds more
than the model predicts". These probes vary one thing at a time against a
fresh server and report the series, which is what separates them.

Three questions, three subcommands:

    python probe_host_growth.py maps    <model> [<model> ...]
    python probe_host_growth.py growth  <model> [<model> ...]
    python probe_host_growth.py prefill <model> [<model> ...]

`maps` aggregates /proc/<pid>/smaps anonymous pages by mapping, so a residual
can be attributed to the heap, a library arena, or a pinned CUDA allocation
rather than inferred from a total. `growth` repeats an identical request with
the prompt cache disabled and enabled, which is what distinguishes an
unbounded leak from the cache filling to its cap. `prefill` varies prompt
length and generation length independently, which is what located the
one-time step on the batched-prefill path.

What they established, in FINDINGS.md under "The per-model residual is a
first-request step":

- There is no leak. At `-cram 0` the footprint steps once on the first
  request and is flat forever after.
- The growth at the default `-cram` is the prompt cache, with a step that
  tracks the model's KV state — ~300 MiB for Qwen3.6-27B against ~14 for the
  sliding-window gemma-3-27B — and it stops at the cap.
- The one-time step is triggered by a *batched prefill*, not by generation,
  and saturates by a 64-token prompt.

Run these against an idle machine, one at a time: they load real models and
read real memory, and anything else resident moves the numbers.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLM_DIR = Path(os.environ.get("LLM_DIR", "/mnt/ssd0/ai/llm"))
PORT = int(os.environ.get("PROBE_PORT", "8399"))

# A deliberately small registry: these probes answer "what shape is this
# term", which needs a handful of contrasting models rather than the whole
# library. `plan.py` owns the full registry for the campaign proper.
MODELS = {
    "qwen36-27b": "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf",
    "gemma3-27b": "mlabonne/gemma-3-27b-it-abliterated-GGUF/gemma-3-27b-it-abliterated.q4_k_m.gguf",
    "magidonia-24b": "bartowski/TheDrummer_Magidonia-24B-v4.3-GGUF/TheDrummer_Magidonia-24B-v4.3-Q5_K_M.gguf",
    "qwen3-4b": "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf",
    "talkie-13b": "mradermacher/talkie-1930-13b-it-hf-GGUF/talkie-1930-13b-it-hf.Q6_K.gguf",
    "gemma4-31b-qat": "unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
    "gemma4-e4b": "unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q5_K_XL.gguf",
    "qwen36-35b-a3b": "unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
}
MAX_CTX = {"talkie-13b": 2048}


def rss(pid: int) -> dict[str, int]:
    """The three resident counters that matter, in MiB.

    `RssShmem` is not incidental: `cudaMallocHost` is accounted as shmem, so
    reading `RssAnon` alone misses the whole pinned graph arena.
    """
    wanted = ("RssAnon", "RssShmem", "RssFile")
    return {k: int(v.split()[0]) // 1024
            for k, v in (l.split(":") for l in open(f"/proc/{pid}/status") if ":" in l)
            if k in wanted}


def smaps_anon(pid: int) -> dict[str, int]:
    """Anonymous pages per mapping, in KiB, keyed by the mapping's name."""
    by: dict[str, int] = collections.defaultdict(int)
    name = "[anon]"
    for line in open(f"/proc/{pid}/smaps"):
        head = line.split()
        if head and "-" in head[0] and ":" not in head[0]:
            name = head[5] if len(head) > 5 else "[anon]"
        elif line.startswith("Anonymous:"):
            by[name] += int(line.split()[1])
    return dict(by)


class Server:
    """One llama-server, started and stopped around a probe."""

    def __init__(self, model: str, cram: int = 0, **flags):
        self.model = model
        self.argv = [
            "llama-server", "-m", str(LLM_DIR / MODELS[model]),
            "-c", str(MAX_CTX.get(model, 32768)), "-ub", "512", "-ngl", "99",
            "-fa", "on", "-np", "1", "--port", str(PORT), "--host", "127.0.0.1",
            "-lv", "5", "-cram", str(cram),
        ]
        for key, value in flags.items():
            self.argv += [("-" if len(key) <= 2 else "--") + key, str(value)]

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv, stdout=open(f"/tmp/probe-{self.model}.log", "w"),
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        for _ in range(1800):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
                return self
            except Exception:
                if self.proc.poll() is not None:
                    raise SystemExit(f"{self.model} died; see /tmp/probe-{self.model}.log")
        raise SystemExit(f"{self.model} never became healthy")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except Exception:
            self.proc.kill()
        # The next server binds the same port, and llama-server does not set
        # SO_REUSEADDR — without a pause the successor silently loses the bind
        # and the probe measures the wrong process.
        time.sleep(3)

    def complete(self, prompt: str, n_predict: int) -> None:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                           "cache_prompt": False}).encode()
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion", body,
            {"Content-Type": "application/json"}), timeout=900).read()


def probe_maps(models: list[str]) -> None:
    """Where the residual lives, by mapping."""
    out = {}
    for model in models:
        with Server(model) as server:
            server.complete("Write a haiku about memory.", 32)
            time.sleep(2)
            out[model] = {"rss": rss(server.proc.pid),
                          "maps": {k: v for k, v in smaps_anon(server.proc.pid).items()
                                   if v > 2048}}
            print(f"{model:16} {out[model]['rss']}", flush=True)
    if len(models) == 2:
        a, b = (out[m]["maps"] for m in models)
        print(f"\n{'delta MiB':>10}{'':4}mapping   ({models[0]} minus {models[1]})")
        for key in sorted(set(a) | set(b),
                          key=lambda k: a.get(k, 0) - b.get(k, 0)):
            delta = (a.get(key, 0) - b.get(key, 0)) / 1024
            if abs(delta) > 5:
                print(f"{delta:>10.0f}    {key}")


def probe_growth(models: list[str]) -> None:
    """Does it accumulate with use, and is the prompt cache why?"""
    for model in models:
        for cram in (0, 8192):
            with Server(model, cram=cram) as server:
                series = [rss(server.proc.pid)["RssAnon"]]
                for _ in range(6):
                    server.complete("Count upward, one number per line.", 16)
                    time.sleep(1)
                    series.append(rss(server.proc.pid)["RssAnon"])
            print(f"{model:16} cram={cram:<5} RssAnon MiB: {series}", flush=True)


def probe_prefill(models: list[str]) -> None:
    """Is the one-time step sized by the prompt or by the generation?"""
    points = ((1, 8), (8, 8), (64, 8), (400, 8), (1, 64), (1, 256), (8, 256))
    for model in models:
        for words, n_predict in points:
            with Server(model) as server:
                before = rss(server.proc.pid)["RssAnon"]
                server.complete(" ".join(["word"] * words), n_predict)
                time.sleep(2)
                after = rss(server.proc.pid)["RssAnon"]
            print(f"{model:16} prompt_words={words:<5} n_predict={n_predict:<5} "
                  f"{before} -> {after}  step={after - before}", flush=True)


PROBES = {"maps": probe_maps, "growth": probe_growth, "prefill": probe_prefill}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in PROBES:
        print(__doc__)
        print(f"probes: {', '.join(PROBES)}\nmodels: {', '.join(MODELS)}")
        return 2
    models = sys.argv[2:] or list(MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"unknown models: {unknown}")
        return 2
    PROBES[sys.argv[1]](models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
