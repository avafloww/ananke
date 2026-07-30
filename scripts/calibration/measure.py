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
import dataclasses
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Bumped whenever a record's shape changes in a way an analysis must notice.
# 1: the original flat CSV-era rows. 2: nested NDJSON with hardware and traces.
# 3: generic per-device breakdown (tensor-split included), per-process GPU
#    memory, model identity, first-occurrence metadata, retained log tails.
SCHEMA = 3

DATA = Path(__file__).parent / "data"
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
    purpose: tuple[str, ...] = ()
    """What questions this configuration answers.

    Several questions want the same process measured — the per-model baseline
    and the context curve overlap at their shared point, for instance. Purpose
    is a tag rather than a schedule: it does not take part in the cell's
    identity, so one measurement serves every question that asked for it.
    """
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
    embeddings: bool = False
    served: bool = True
    probe_tokens: int = 64
    """How many tokens the warm-up probe generates.

    Memory does not depend on this — the first-request step is identical at
    `n_predict` 8, 4096, and 12288 — so it is a timing knob, not a factor.
    `probe_prompt_tokens` is the one that matters.
    """
    probe_prompt_tokens: int = 4
    """How long the warm-up probe's *prompt* is, in tokens.

    This is what moves host memory. llama.cpp's server takes a context
    checkpoint while decoding a prompt, spaced by `--checkpoint-min-step`
    (8192 tokens), so a prompt below that captures one checkpoint and a prompt
    of a few tokens captures only part of one: the step measures 11 MiB at one
    token against 274 at sixty-four, and 431 once past the spacing.

    The default is four, matching the "Count to twenty." probe the campaign
    was measured with, so existing cells keep their identity and their
    meaning.
    """
    """How many tokens the first request asks for.

    Serving a first request allocates host memory that an idle process has not
    — measured from -2 MiB to +238 across models, and predicted by neither
    vocabulary, model size, nor architecture. Every served cell in the campaign
    used the same 64-token probe, so the request itself was never a variable.
    Varying it is what separates a per-request allocation from a per-model one.
    """
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
        """Stable identity, so a rerun skips what has already been measured.

        The label and the purpose tags are deliberately excluded: they name
        the cell and say why it was wanted, but two cells with the same flags
        are the same measurement whatever they are called, and measuring one
        configuration twice under two names is pure waste.

        Fields still at their default are excluded too, and that is what makes
        the schema extensible. Hashing every field means adding or removing
        one changes the identity of *every* cell ever measured, so the harness
        stops recognising its own dataset and re-measures all of it. That trap
        is why a prompt-length knob was folded into `probe_tokens` rather than
        given its own field, and why a dead `thp` field was kept: both were
        cheaper than a full re-measure. Excluding defaults costs nothing —
        two cells differing in a field still hash differently, since one of
        them is not at the default — and a new defaulted field is free.
        """
        defaults = {f.name: f.default for f in dataclasses.fields(self)}
        fields = {k: v for k, v in dataclasses.asdict(self).items()
                  if k not in ("label", "purpose") and v != defaults.get(k)}
        payload = json.dumps(fields, sort_keys=True, default=str)
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
    tail: str = ""
    log: str = ""

    hardware: dict[str, object] = dataclasses.field(default_factory=dict)
    trace: list[dict[str, object]] = dataclasses.field(default_factory=list)
    checkpoints: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def record(self) -> dict[str, object]:
        """One NDJSON record.

        Nested rather than flat: the hardware block, the per-device VRAM split,
        and the factor set all have their own shapes, and a fixed column set
        would force every one of them into a lowest common denominator — which
        is what previously capped the GPU breakdown at two devices and made
        adding a factor a schema migration.
        """
        return {
            "schema": SCHEMA,
            "cell": self.cell.cell_id,
            "status": self.status,
            "provenance": self.provenance,
            "hardware": self.hardware,
            "factors": dataclasses.asdict(self.cell),
            "parsed": self.parsed,
            "rss": self.rss,
            "log_tail": self.tail,
            "log": self.log,
            # The full time series, not just its summary: growth is a shape,
            # and a peak alone cannot distinguish "allocated on first use" from
            # "still climbing when we stopped looking".
            "trace": self.trace,
            # Memory against tokens, which a time series alone cannot give.
            "checkpoints": self.checkpoints,
        }


# Buffer-size lines the loaders emit. Both forks are matched by one pattern
# each; the last occurrence wins because the loader logs a reserve pass first
# and then the real graph, with the same figure.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "arena_mib": re.compile(r"(?:CUDA_Host|CPU) compute buffer size *= *([0-9.]+)"),
    "out_buf_mib": re.compile(r"(?:CUDA_Host|CPU) +output buffer size *= *([0-9.]+)"),
    "cpu_kv_mib": re.compile(r"CPU KV buffer size *= *([0-9.]+)"),
    "cpu_model_mib": re.compile(r"CPU(?:_Mapped)? model buffer size *= *([0-9.]+)"),
}
_META = re.compile(
    r"(n_layer|n_embd|n_expert|n_expert_used|n_swa|n_vocab|n_head_kv|n_head"
    r"|n_embd_head_k|n_embd_head_v|n_ctx_train|n_ff"
    # The recurrent state's whole shape. `n_embd_r` (the rolling convolution
    # state) and `n_embd_s` (the SSM state) are computed by llama.cpp from
    # these, and both are GGUF metadata — which is what makes the RS term
    # predictable from the model file rather than something to measure.
    r"|ssm_d_conv|ssm_d_inner|ssm_d_state|ssm_n_group|ssm_dt_rank"
    # `n_layer` is the layer span the contexts cover, which already excludes
    # the MTP head's trailing block — llama.cpp reports the full block count as
    # `n_layer_all`, deliberately *not* captured here because this parser
    # already spells a repeated key `<key>_all` and the two would collide.
    # `nextn_predict_layers` carries the same difference.
    r"|n_group_used) *= *(\d+)")
_META_KEYS = ("n_layer", "n_embd", "n_expert", "n_expert_used", "n_swa", "n_vocab",
              "n_head", "n_head_kv", "n_embd_head_k", "n_embd_head_v", "n_ctx_train",
              "n_ff", "ssm_d_conv", "ssm_d_inner", "ssm_d_state", "ssm_n_group",
              "ssm_dt_rank", "n_group_used")
# Every numeric GGUF metadata key the loader echoes. The `print_info:` block
# above covers only the hyperparameters llama.cpp keeps in `hparams`; keys it
# reads straight into an architecture-specific path — `lfm2.shortconv.l_cache`
# sizes LFM2's rolling state and is printed nowhere else — are only recoverable
# from the key/value dump.
_GGUF_KV = re.compile(
    r"llama_model_loader: - kv +\d+: +([A-Za-z0-9_.]+) +[a-z0-9]+ += +(-?\d+)\s*$",
    re.MULTILINE)
# The embedded MTP head's depth, which sizes the modelled KV term. It is a
# metadata key rather than one of llama.cpp's `n_* = ` summary lines, so it
# needs its own pattern.
_NEXTN = re.compile(r"\.nextn_predict_layers[^=]*= *(\d+)")
# The tensor `compute_buffer::is_gemma_e_variant` keys on. Recorded so the
# analysis can use the same discriminator the estimator does, rather than a
# filename proxy that silently disagrees the moment an E-variant ships under
# another name.
_PER_LAYER_EMBD = re.compile(r"per_layer_token_embd\.weight")
# llama.cpp's own figure for an MTP context. It was the stated calibration
# source for the constants in estimator/mtp.rs, and it is reported *per
# context* — flat across slot counts while the real cost scales with them —
# so it is recorded to keep that discrepancy visible rather than to fit to.
_MTP = re.compile(r"estimated memory usage of MTP context is *([0-9.]+)")
# llama.cpp's own memory-breakdown row, which splits each device into
# model / context / compute. The compute column is what the estimator's
# per-architecture GPU curves are fitted against, so it has to be captured
# per device rather than as a total.
# A device row carries every column, and the device is *not* always named
# `CUDA<n>`: under `--split-mode tensor` llama.cpp fuses the cards and reports
# a single `Meta()` device. Keying on the CUDA name silently recorded zeros for
# every tensor-split run — which is to say for every production configuration.
# Every separator tolerates padding: the columns are right-aligned, so a value
# with fewer digits than its column is preceded by more spaces. A literal single
# space after the first `=` silently dropped any row whose free-memory figure was
# narrower than its neighbour's — one card of a two-card breakdown, which left
# the surviving row misaligned against the other card's driver reading and turned
# one cell's compute target into 2042 MiB against a true 1032.
_BREAKDOWN = re.compile(
    r"- (.+?) +\| *(\d+) *= *(\d+) *\+ *\( *(\d+) *= *(\d+) *\+ *(\d+) *\+ *(\d+)\)"
    r" *\+ *(\d+)")
# The host row has no total/free and no unaccounted column.
_BREAKDOWN_HOST = re.compile(
    r"- Host +\| *(\d+) = *(\d+) \+ *(\d+) \+ *(\d+)")
MAX_GPUS = 4
_ARCH = re.compile(r"arch *= *([A-Za-z0-9_.-]+)")

# A server creates more than one context: the main one, a sliding-window
# sibling on an interleaved-SWA model, and an MTP draft context under
# `--spec-type draft-mtp`. Each prints its own memory pools and its own
# compute reserve, and a whole-log `findall` collapses them — which is how the
# MTP context's compute buffer stayed invisible while its cost was being
# fitted as an opaque constant. So the log is segmented into contexts first
# (each ends with its `graph nodes` line) and every figure is attributed to
# the context that allocated it.
_CONTEXT_END = re.compile(
    r"^.*(?:sched_reserve|llama_init_from_model): graph nodes.*$"
    # `graph splits` follows `graph nodes` on the next line, so a boundary that
    # stopped at the latter attributed every split count to the *following*
    # context.
    r"(?:\n^.*: graph splits.*$)?", re.MULTILINE)
# The attention cache's own summary line: the physical total across devices,
# the cell count (per sequence), the layer count that actually allocates, the
# sequence count, and the K/V split with the cache types in play. Together
# these say exactly what llama.cpp sized, so a modelled KV can be checked
# term by term instead of only in aggregate.
_KV_POOL = re.compile(
    r"llama_kv_cache: size = *([0-9.]+) MiB *\( *(\d+) cells, *(\d+) layers, "
    r"*(\d+)/(\d+) seqs\), K \(([a-z0-9_]+)\): *([0-9.]+) MiB, "
    r"V \(([a-z0-9_]+)\): *([0-9.]+) MiB")
# The recurrent module's equivalent. `rs_seq` is the speculative rollback
# depth: the state is replicated `n_seq × (rs_seq + 1)` times, and `rs_seq` is
# non-zero only under speculative decoding.
_RS_POOL = re.compile(
    r"llama_memory_recurrent: size = *([0-9.]+) MiB *\( *(\d+) cells, *(\d+) layers, "
    r"*(\d+) seqs *(\d+) rs_seq\), R \([a-z0-9_]+\): *([0-9.]+) MiB, "
    r"S \([a-z0-9_]+\): *([0-9.]+) MiB")
# Per-device buffer lines, whatever the loader called the stage. `Meta()` is
# the fused device a tensor split reports, and its figure is ONE card's share.
# `llm_load_tensors` is ik_llama's spelling and it omits the word `model`
# entirely — `CUDA0 buffer size = 6992.89` — so a pattern demanding the kind
# recorded nothing at all for the fork, which is most of what runs in
# production. The kind is taken from the stage when the line does not name one.
_DEV_BUFFER = re.compile(
    r"(load_tensors|llm_load_tensors|llama_kv_cache(?:_init)?|llama_memory_recurrent"
    r"|sched_reserve|llama_init_from_model|llama_context): +([A-Za-z0-9_()]+) +"
    r"(?:(model|KV|RS|compute|output) +)?buffer size *= *([0-9.]+)")
_GRAPH_SHAPE = re.compile(r"graph (nodes|splits) += *(\d+)")
# What a vision projector costs, as llama.cpp's own accounting states it. The
# `fit_params_target` line is the whole per-device figure — the projector's
# weights *and* its CLIP graph buffer — so pairing it with the summed tensor
# sizes below isolates the graph term, which the estimator was modelling as
# zero. The two are printed on different lines and neither is derivable from
# the mmproj file's size (that includes GGUF framing).
_MMPROJ_RESERVED = re.compile(
    r"adding ([0-9.]+) MiB to fit_params_target for device ([A-Za-z0-9_()]+)")
_CLIP_TENSOR = re.compile(r"clip_model_loader: tensor\[\d+\]:.*tensor_size=(\d+)")
# The two vision settings that differ between the projectors measured, and so
# the first candidates for what the graph buffer scales with. Recorded rather
# than used: three cells across two configurations cannot distinguish a rate
# in either of them from a constant.
_CLIP_IMAGE_SIZE = re.compile(r"image size = (\d+) x (\d+)")
_CLIP_MERGE = re.compile(r"n_merge: *(\d+)")


def parse_log(text: str) -> dict[str, float | str]:
    """Pull the model's shape and its logged buffer sizes out of a load log."""
    parsed: dict[str, float | str] = {}
    for name, pattern in _PATTERNS.items():
        found = [float(v) for v in pattern.findall(text)]
        parsed[name] = found[-1] if found else 0.0
        # A cell with `-md` or `--mmproj` loads two models, so a key can appear
        # more than once with genuinely different values. Keep every occurrence
        # so a later reader can tell the target's figure from the draft's
        # rather than inheriting whichever this pass happened to pick.
        if len(found) > 1:
            parsed[f"{name}_all"] = found
    # `CPU_Mapped` means the host-side weights are file-backed and so land in
    # RssFile; plain `CPU` means they were read into anonymous memory.
    parsed["cpu_model_mapped"] = "yes" if "CPU_Mapped model buffer" in text else "no"
    # First occurrence, not last: the target model loads before any draft or
    # projector, and last-wins recorded the *draft's* shape for exactly the
    # MTP cells whose target shape the constants are fitted against.
    meta: dict[str, list[int]] = {}
    for key, value in _META.findall(text):
        meta.setdefault(key, []).append(int(value))
    for key in _META_KEYS:
        found = meta.get(key, [])
        parsed[key] = found[0] if found else 0
        if len(set(found)) > 1:
            parsed[f"{key}_all"] = found
    gguf_kv: dict[str, int] = {}
    for key, value in _GGUF_KV.findall(text):
        gguf_kv.setdefault(key, int(value))
    parsed["gguf_kv"] = gguf_kv
    parsed["contexts"] = _parse_contexts(text)
    reserved = {device: float(mib) for mib, device in _MMPROJ_RESERVED.findall(text)}
    if reserved:
        parsed["mmproj_reserved_mib"] = reserved
        parsed["mmproj_tensor_bytes"] = sum(
            int(n) for n in _CLIP_TENSOR.findall(text))
        image = _CLIP_IMAGE_SIZE.search(text)
        if image:
            parsed["clip_image_size"] = int(image.group(1))
        merge = _CLIP_MERGE.search(text)
        if merge:
            parsed["clip_n_merge"] = int(merge.group(1))
    mtp = _MTP.search(text)
    parsed["mtp_context_mib"] = float(mtp.group(1)) if mtp else 0.0
    nextn = _NEXTN.search(text)
    parsed["nextn_predict_layers"] = int(nextn.group(1)) if nextn else 0
    parsed["per_layer_token_embd"] = bool(_PER_LAYER_EMBD.search(text))
    arch = _ARCH.search(text)
    parsed["arch"] = arch.group(1) if arch else "?"

    # Every device row, in the order the loader printed them, with every
    # column. `unaccounted` is the difference between what the driver reports
    # for the process and what llama.cpp can attribute — the term the GPU
    # compute-buffer bases carry as a margin, and previously discarded.
    # Only the *last* table. Recent builds print one at the parameter-fitting
    # stage, before anything is allocated, and another once the context exists;
    # the first is a projection and its rows would misalign `devices[index]`
    # against the cards. Relying on the fit-stage rows' negative `unaccounted`
    # to exclude them worked by accident and stopped working for 16 cells whose
    # projection happened to come out positive.
    tables = text.split("memory breakdown [MiB]")
    final = tables[-1] if len(tables) > 1 else text
    devices = []
    for name, total, free, own, model, kv, compute, unacc in _BREAKDOWN.findall(final):
        devices.append({"device": name.strip(), "total_mib": int(total),
                        "free_mib": int(free), "self_mib": int(own),
                        "model_mib": int(model), "kv_mib": int(kv),
                        "compute_mib": int(compute),
                        "unaccounted_mib": int(unacc)})
    parsed["devices"] = devices
    host = _BREAKDOWN_HOST.search(final)
    if host:
        parsed["host_breakdown"] = {
            "self_mib": int(host.group(1)), "model_mib": int(host.group(2)),
            "kv_mib": int(host.group(3)), "compute_mib": int(host.group(4))}
    # Flat mirrors of the first MAX_GPUS device rows, kept because they are
    # convenient to fit against; `devices` above is the authoritative list.
    for index in range(MAX_GPUS):
        entry = devices[index] if index < len(devices) else {}
        for column in ("model", "kv", "compute", "unaccounted", "self"):
            parsed[f"gpu{index}_{column}_mib"] = entry.get(f"{column}_mib", 0)
    return parsed


def _parse_contexts(text: str) -> list[dict[str, object]]:
    """One entry per context the server created, in creation order.

    A context is everything the loader printed up to and including its
    `graph nodes` line. The first segment of a run belongs to llama.cpp's
    parameter-fitting dry run — it reports the same shape with no weights
    loaded — so segments are kept whole rather than merged, and a reader picks
    the one it wants by the pools it holds.
    """
    contexts: list[dict[str, object]] = []
    start = 0
    for boundary in _CONTEXT_END.finditer(text):
        segment = text[start:boundary.end()]
        start = boundary.end()
        buffers: dict[str, dict[str, float]] = {}
        for stage, device, kind, mib in _DEV_BUFFER.findall(segment):
            if not kind:
                kind = "model" if stage.endswith("load_tensors") else "compute"
            buffers.setdefault(device, {})[kind.lower()] = float(mib)
        entry: dict[str, object] = {"buffers": buffers}
        kv_pools = []
        for (total, cells, layers, seqs, seqs_max, k_type, k_mib,
             v_type, v_mib) in _KV_POOL.findall(segment):
            kv_pools.append({"total_mib": float(total), "cells": int(cells),
                             "layers": int(layers), "seqs": int(seqs),
                             "seqs_max": int(seqs_max), "k_type": k_type,
                             "k_mib": float(k_mib), "v_type": v_type,
                             "v_mib": float(v_mib)})
        entry["kv_pools"] = kv_pools
        rs = _RS_POOL.search(segment)
        if rs:
            entry["rs_pool"] = {
                "total_mib": float(rs.group(1)), "cells": int(rs.group(2)),
                "layers": int(rs.group(3)), "seqs": int(rs.group(4)),
                "rs_seq": int(rs.group(5)), "r_mib": float(rs.group(6)),
                "s_mib": float(rs.group(7))}
        for name, value in _GRAPH_SHAPE.findall(segment):
            entry[f"graph_{name}"] = int(value)
        contexts.append(entry)
    return contexts


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
        self.trace: list[dict[str, object]] = []
        self.peak: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.started_at = time.monotonic()
        self._start = self.started_at

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
                per_device = read_gpu_per_device(self.pid)
                if per_device:
                    sample["gpu_used_mib"] = sum(per_device.values())
                    for index, mib in per_device.items():
                        sample[f"gpu{index}_used_mib"] = mib
            except (FileNotFoundError, ProcessLookupError, ValueError):
                return  # The process exited; the trace so far still stands.
            self.trace.append({
                "t_seconds": round(time.monotonic() - self._start, 1),
                "at_utc": datetime.datetime.now(datetime.timezone.utc)
                          .isoformat(timespec="seconds"),
                **sample,
            })
            for key, value in sample.items():
                # `key not in peak` first: a metric that is legitimately zero
                # for the whole run — RssShmem with no CUDA, so no pinned
                # allocations — never exceeds the default and would otherwise
                # be absent entirely rather than present as zero.
                if key not in self.peak or value > self.peak[key]:
                    self.peak[key] = value
            self._stop.wait(self.INTERVAL_SECONDS)

    def growth(self) -> dict[str, int]:
        """Peak minus the first sample: what accumulated after startup."""
        if not self.trace:
            return {}
        first = self.trace[0]
        return {f"growth_{k}": self.peak.get(k, 0) - v
                for k, v in first.items() if isinstance(v, int)}


def read_gpu_mib(pid: int) -> int | None:
    """Total per-process VRAM as the driver reports it."""
    per_device = read_gpu_per_device(pid)
    return sum(per_device.values()) if per_device else None


def read_gpu_per_device(pid: int) -> dict[int, int]:
    """Per-process VRAM, split by card.

    llama.cpp's own breakdown attributes what it allocated; the driver counts
    the CUDA context and everything else besides, and the GPU compute-buffer
    bases in `estimator/compute_buffer.rs` are defined against the driver's
    figure. ik_llama does not print mainline's breakdown table at all, so for
    every ik cell this is the *only* per-device source — summing the cards
    would leave the fork's placement unmeasurable.
    """
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    index_of = _gpu_index_by_uuid()
    used: dict[int, int] = {}
    for line in apps.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) == pid:
            index = index_of.get(parts[1])
            if index is not None:
                used[index] = used.get(index, 0) + int(parts[2])
    return used


def _gpu_index_by_uuid() -> dict[str, int]:
    """Map each card's UUID to its index; the app query reports only UUIDs."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    mapping = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            mapping[parts[1]] = int(parts[0])
    return mapping


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
    machine, and a binary. The timestamp is what lets a later reader place a
    row in time — against a llama.cpp pin bump, a driver update, or a change
    to the box — without having to remember when anything happened.
    """
    resolved = Path(shutil.which(binary) or binary)
    now = datetime.datetime.now(datetime.timezone.utc)
    record = {
        "measured_at_utc": now.isoformat(timespec="seconds"),
        "measured_at_local": now.astimezone().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "binary": str(resolved.resolve()) if resolved.exists() else str(resolved),
        "ananke_rev": _run(["git", "rev-parse", "--short", "HEAD"]) or "?",
        "runtime_version": _binary_version(resolved),
        "runtime_sha256": _sha256(resolved),
        "ananke_dirty": "yes" if _run(["git", "status", "--porcelain"]) else "no",
    }
    if cell is not None:
        record["model_file_at"] = _mtime(Path(cell.model))
        record |= model_identity(Path(cell.model))
    return record


def model_identity(path: Path) -> dict[str, str]:
    """What identifies a model to a reader who does not have this machine.

    `factors.model` is an absolute path under whatever the operator set as the
    model directory, which is useless for joining one contributor's rows to
    another's. The repo-and-file suffix, the byte total across shards, and the
    quant string are all portable.
    """
    parts = path.parts
    key = "/".join(parts[-3:]) if len(parts) >= 3 else path.name
    quant = "?"
    for token in re.split(r"[-.]", path.stem):
        if re.fullmatch(r"(UD_)?(I?Q\d[A-Z0-9_]*|BF16|F16|F32)", token, re.I):
            quant = token
    return {"model_key": key, "model_quant": quant,
            "model_bytes": str(model_bytes(path))}


def _binary_version(path: Path) -> str:
    """What the server reports about itself.

    Custom forks report `version: 0 (unknown)` and a nix build normalises the
    binary's mtime to the epoch, so neither identifies anything. The hash
    beside this does: it cannot be mapped to an upstream commit, but it
    answers "was this the same binary", which is what separates drift in the
    runtime from error in a fit, and what tells a contributor's build from
    ours. Recorded anyway because a non-nix build does report a version.
    """
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "?"
    text = (out.stdout + out.stderr).strip().splitlines()
    return text[0][:200] if text else "?"


def _sha256(path: Path) -> str:
    """The binary's hash: an exact identity even when it reports build 0."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return "?"


def _mtime(path: Path) -> str:
    """A file's modification time, as an ISO 8601 UTC timestamp."""
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return "?"
    return (datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
            .isoformat(timespec="seconds"))


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _post(port: int, path: str, payload: dict, timeout: float) -> dict | None:
    """Send a request, returning the decoded body when there is one.

    The body carries the server's own token accounting, which is what ties a
    memory reading to the work that produced it.
    """
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        raw = urllib.request.urlopen(request, timeout=timeout).read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None  # A failed probe still leaves the process measurable.
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def wait_for_port(port: int, timeout: float = 180.0) -> bool:
    """Block until nothing holds the port.

    Stopping a server is not the same as the port being free: the previous
    listener's socket can outlive it, and ik_llama's server does not set
    SO_REUSEADDR, so it loses the bind and exits instead of retrying. That
    reads downstream as a load failure, which is how a whole run of ik cells
    can fail for a reason that has nothing to do with the model.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Deliberately without SO_REUSEADDR: the question is whether the
            # next server can bind, not whether this process could.
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            time.sleep(1.0)
        finally:
            probe.close()
    return False


def _healthy(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def exercise(cell: Cell, port: int, pid: int = 0) -> list[dict[str, object]]:
    """Put the server through enough work to allocate what it allocates lazily.

    Several host buffers are sized on first use rather than at load, so an idle
    process under-reports. `soak` goes further and grows the prompt across
    successive requests, which is the only way to reach terms that accumulate
    with use — the prompt cache and per-slot checkpoints — rather than at load.
    """
    if not cell.served:
        return []
    if cell.embeddings:
        return _exercise_embeddings(cell, port, pid)
    # One word per token, near enough: what matters is which side of the
    # checkpoint spacing the prompt falls on, not the exact count.
    prompt = ("Count to twenty." if cell.probe_prompt_tokens <= 4
              else " ".join(["word"] * cell.probe_prompt_tokens))
    _post(port, "/v1/chat/completions",
          {"model": "m", "messages": [{"role": "user", "content": prompt}],
           "max_tokens": cell.probe_tokens}, 600)


    if cell.bench:
        return run_growth(cell, port, pid)


    prompt = "Explain memory allocation."
    for i in range(cell.soak):
        prompt += f" Also cover point {i} in detail, at length, with examples."
        payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 256}
        # Overlapping requests touch slots beyond the first, which a strictly
        # serial probe never reaches.
        threads = [threading.Thread(target=_post, args=(port, "/v1/chat/completions",
                                                        payload, 300))
                   for _ in range(cell.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return []


# The coding-agent benchmark, which drives a server with real prompts and real
# generated replies rather than repetitive filler. Realism is the part that
# cannot be synthesised: prompt-cache behaviour and generation both depend on
# what the tokens actually are. Vendored beside this module so a calibration
# run needs nothing outside the repository.
BENCH = Path(os.environ.get("CODING_BENCH", Path(__file__).parent / "coding_bench.py"))


def _exercise_embeddings(cell: Cell, port: int,
                         pid: int) -> list[dict[str, object]]:
    """An embedding model has no generation, so requests drive it instead.

    The equivalent growth question for this modality is whether repeated
    embedding calls accumulate anything, so a growth cell issues many and
    checkpoints the same way rather than being skipped.
    """
    rounds = cell.bench_turns if cell.bench else 1
    checkpoints: list[dict[str, object]] = []
    for index in range(rounds):
        _post(port, "/v1/embeddings",
              {"input": f"calibration probe {index} " + "token " * 64}, 120)
        if not cell.bench:
            continue
        checkpoint = {
            "turn": index + 1,
            "at_utc": datetime.datetime.now(datetime.timezone.utc)
                      .isoformat(timespec="seconds"),
            "prompt_tokens": 0, "completion_tokens": 0,
            "generated_tokens_total": 0, "kv_depth_tokens": 0,
            **read_rss(pid),
        }
        per_device = read_gpu_per_device(pid)
        if per_device:
            checkpoint["gpu_used_mib"] = sum(per_device.values())
        checkpoints.append(checkpoint)
    return checkpoints


def run_growth(cell: Cell, port: int, pid: int) -> list[dict[str, object]]:
    """Drive an agent-shaped conversation, checkpointing memory against tokens.

    The RSS sampler answers "did memory move"; it cannot answer "per what",
    because nothing in a time series says how many tokens were generated
    between two samples. A run whose footprint grows with generation and one
    whose footprint grows with wall-clock look identical on a clock.

    So each turn is a checkpoint: the conversation so far, the tokens the
    server reports for it, and the process's memory read immediately after.
    That makes growth fittable against cumulative tokens and against KV depth
    separately, and it makes a flat result a *measurement* of no growth
    rather than an absence of evidence.

    Replies are fed back in, so the context grows the way an agent's does and
    the prompt cache sees a real prefix rather than filler.
    """
    prompts = _bench_prompts()
    # `-cram` serialises prompts that have been *evicted* from a slot. One
    # strictly-growing conversation shares a prefix and never evicts anything,
    # so it would measure cram 0 and cram 8192 identically for a reason that
    # has nothing to do with the cache. Alternating distinct conversations is
    # what makes a slot's prompt get displaced and the cache get used.
    conversations = [[{"role": "system", "content": _bench_system() + marker}]
                     for marker in ("", " Prefer Rust.", " Prefer Python.",
                                    " Answer tersely.")] if cell.cram else \
        [[{"role": "system", "content": _bench_system()}]]
    checkpoints: list[dict[str, object]] = []
    generated = 0
    for turn in range(cell.bench_turns):
        messages = conversations[turn % len(conversations)]
        messages.append({"role": "user", "content": prompts[turn % len(prompts)]})
        body = _post(port, "/v1/chat/completions",
                     {"model": "m", "messages": messages,
                      "max_tokens": GROWTH_MAX_TOKENS}, 900)
        if body is None:
            break
        try:
            reply = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError):
            break
        messages.append({"role": "assistant", "content": reply or ""})
        generated += int(usage.get("completion_tokens", 0))
        rss = read_rss(pid)
        checkpoint = {
            "turn": turn + 1,
            "at_utc": datetime.datetime.now(datetime.timezone.utc)
                      .isoformat(timespec="seconds"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "generated_tokens_total": generated,
            # The prompt token count is the KV depth the server is holding,
            # which is the term that scales with context rather than with use.
            "kv_depth_tokens": int(usage.get("prompt_tokens", 0))
                               + int(usage.get("completion_tokens", 0)),
            **rss,
        }
        per_device = read_gpu_per_device(pid)
        if per_device:
            checkpoint["gpu_used_mib"] = sum(per_device.values())
            for index, mib in per_device.items():
                checkpoint[f"gpu{index}_used_mib"] = mib
        checkpoints.append(checkpoint)
        # Stop before the context wraps: past that point the server evicts and
        # the footprint stops being a function of what was generated.
        checkpoint["conversation"] = turn % len(conversations)
        if checkpoint["kv_depth_tokens"] > cell.ctx * 0.85:
            break
    return checkpoints


GROWTH_MAX_TOKENS = 512


def _bench_system() -> str:
    module = _bench_module()
    return getattr(module, "SYSTEM", "You are a helpful assistant.") if module \
        else "You are a helpful assistant."


def _bench_prompts() -> list[str]:
    module = _bench_module()
    prompts = getattr(module, "PROMPTS", None) if module else None
    return list(prompts) if prompts else [
        "Write a function to parse an ISO 8601 duration.",
        "Now add unit tests for it.",
        "Refactor it to avoid regular expressions.",
    ]


def _bench_module():
    """The vendored benchmark's prompts, so growth is driven by real work."""
    if not BENCH.exists():
        return None
    spec = importlib.util.spec_from_file_location("coding_bench", BENCH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


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


def model_bytes(first: Path) -> int:
    """Total size across every shard of a model, in bytes."""
    if not first.exists():
        return 0
    stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first.name)
    if stem == first.name:
        return first.stat().st_size
    shards = sorted(first.parent.glob(f"{stem}-*-of-*.gguf"))
    return sum(s.stat().st_size for s in shards)


def model_gib(cell: Cell) -> float:
    """On-disk size of the model, shards included, in GiB."""
    return model_bytes(Path(cell.model)) / (1024**3)


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


def archive_log(log_path: Path, archive_dir: Path) -> str:
    """Keep the load log alongside the record, compressed.

    The parsers here read four kinds of line. Everything else the loader
    prints — and everything a future question turns out to need — is only
    recoverable if the log itself survives, and a log left in a temporary
    directory does not. They compress to tens of KiB, which is a small price
    for making a record re-parseable rather than merely re-readable.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{log_path.stem}.log.gz"
    try:
        with log_path.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except OSError:
        return ""
    return target.name


def measure(cell: Cell, log_dir: Path, port: int, load_timeout: int,
            archive_dir: Path | None = None) -> Measurement:
    binary = {"mainline": os.environ.get("MAINLINE_BIN", "llama-server"),
              "ik": os.environ.get("IK_BIN", "ik-llama-server")}[cell.runtime]
    prov = provenance(binary, cell)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.cell_id}-{cell.label}.log"

    if not wait_for_port(port):
        return Measurement(cell, prov, {}, {}, status="port-busy",
                           hardware=hardware())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cell.gpus)
    with log_path.open("wb") as log_file:
        trace: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        proc = subprocess.Popen(cell.argv(binary, port), stdout=log_file,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            # Sampling starts at spawn, not at health: the load itself is where
            # the transients are — the pinned staging ring on a `--no-mmap`
            # load is gone by the time the server answers /health.
            with RssSampler(proc.pid) as sampler:
                deadline = time.monotonic() + load_timeout
                while not _healthy(port):
                    if proc.poll() is not None:
                        return Measurement(cell, prov, {}, {},
                                           status="failed-to-load",
                                           hardware=hardware(),
                                           tail=_tail(log_path),
                                           log=archive_log(log_path, archive_dir)
                                           if archive_dir else "")
                    if time.monotonic() > deadline:
                        _stop(proc)
                        return Measurement(cell, prov, {}, {}, status="timeout",
                                           hardware=hardware(),
                                           tail=_tail(log_path),
                                           log=archive_log(log_path, archive_dir)
                                           if archive_dir else "")
                    time.sleep(HEALTH_POLL_SECONDS)
                loaded_at = time.monotonic() - sampler.started_at
                checkpoints = exercise(cell, port, proc.pid)
                time.sleep(SETTLE_SECONDS)
                final = read_rss(proc.pid)
            # Report the peak, because that is the figure ananke's snapshotter
            # keeps and the correction divides by; the final reading and the
            # growth since startup ride alongside it.
            rss = dict(sampler.peak or final)
            rss |= {f"final_{k}": v for k, v in final.items()}
            rss |= sampler.growth()
            rss["samples"] = len(sampler.trace)
            rss["load_seconds"] = round(loaded_at, 1)
            trace = sampler.trace
        finally:
            _stop(proc)

    # After the stop, not before: llama.cpp prints its memory-breakdown table
    # while tearing the context down, so parsing a still-running server's log
    # silently loses every per-device figure.
    parsed = parse_log(log_path.read_text(errors="replace"))

    archived = archive_log(log_path, archive_dir) if archive_dir else ""
    return Measurement(cell, prov, parsed, rss, hardware=hardware(), trace=trace,
                       log=archived, checkpoints=checkpoints)


def _tail(path: Path, lines: int = 40) -> str:
    """The end of a failed run's log, so a bad record says why it is bad."""
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


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
        if not line.strip():
            continue
        record = json.loads(line)
        # A cell skipped because the box was momentarily full is not measured,
        # and a rerun with memory free must retry it rather than inherit the
        # skip forever. Load failures are likewise worth retrying; only a
        # completed measurement ends a cell.
        if record.get("status", "ok") == "ok":
            seen.add(record["cell"])
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
              load_timeout: int, headroom_gib: float = 30.0,
              archive_dir: Path | None = None) -> None:
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
        try:
            measurement = measure(cell, log_dir, port, load_timeout, archive_dir)
        except Exception as error:
            # One cell must never end the campaign. A crash here — a parse that
            # met an unexpected log, a counter absent for a configuration that
            # never allocates it — costs that cell and nothing else; the run
            # has hours of work behind it and the cell is retried on resume.
            print(f"ERROR {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            append(out, Measurement(cell, provenance("true", cell), {}, {},
                                    status="harness-error", hardware=hardware(),
                                    tail=traceback.format_exc()[-2000:]))
            continue
        append(out, measurement)
        if measurement.status != "ok":
            print(f"{measurement.status}; see {log_dir}", flush=True)
            continue
        print(
            f"arena={measurement.parsed.get('arena_mib', 0):.2f} "
            f"anon={measurement.rss.get('rss_anon_kb', 0) // 1024} "
            f"shmem={measurement.rss.get('rss_shmem_kb', 0) // 1024} "
            f"file={measurement.rss.get('rss_file_kb', 0) // 1024} MiB",
            flush=True,
        )


def reparse(out: Path, archive_dir: Path) -> int:
    """Rebuild every record's `parsed` block from its archived log.

    The logs are kept precisely so that a question the parser could not answer
    when a cell ran can still be answered later. Re-running the campaign to
    add a field would cost days of GPU time and would measure a different
    llama.cpp build; re-reading the logs costs seconds and changes nothing
    about what was observed.

    A record whose log is missing keeps the `parsed` block it already has,
    with `reparsed` left absent so an analysis can tell which rows carry the
    newer fields.
    """
    if not out.exists():
        print(f"{out}: no measurements to reparse", file=sys.stderr)
        return 1
    rewritten, skipped = 0, 0
    lines: list[str] = []
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        archived = archive_dir / record.get("log", "")
        # A cell that never loaded carries an empty `parsed` by construction,
        # and its log holds only the failure. Parsing it anyway replaces that
        # emptiness with a full block of zeros, which reads as a measurement
        # that found nothing rather than as no measurement at all.
        if record.get("status") != "ok":
            skipped += 1
            lines.append(json.dumps(record, default=str))
            continue
        if not record.get("log") or not archived.exists():
            skipped += 1
            lines.append(json.dumps(record, default=str))
            continue
        with gzip.open(archived, "rt", errors="replace") as handle:
            record["parsed"] = parse_log(handle.read())
        record["reparsed"] = True
        rewritten += 1
        lines.append(json.dumps(record, default=str))
    out.write_text("\n".join(lines) + "\n")
    print(f"reparsed {rewritten} records, {skipped} without an archived log")
    return 0


def retire_stale_builds(out: Path, tolerance: float = 0.02) -> int:
    """Mark rows a runtime upgrade invalidated, so every reader skips them.

    A cell id hashes the *factors*, and the runtime binary is not one of them.
    Upgrade llama.cpp and every existing row keeps describing a program that is
    no longer installed, with nothing to say so — and a constant fitted across
    the two is fitted to two programs.

    The evidence needed is one re-measurement per architecture. From this
    dataset's upgrade: six of seven production cells reproduce to the megabyte,
    and the seventh, GLM-5.2, moves 9.6% because ik's sparse-attention compute
    buffer shrank by a third. laguna on the same fork is unchanged, so the fork
    is not the unit that changed — `glm-dsa` is.

    So an (architecture, build) pair is retired only where a cell measured under
    both it and a later build disagrees. Being older is not evidence: three ik
    builds appear here and only one is shown to differ, and retiring on age
    alone would discard a whole batch-and-context sweep because a
    `nixos-rebuild` landed between running it and checking it.

    Retired rows keep their data and their archived log; only `status` changes,
    to `stale-runtime`. That is deliberately the same gate every consumer
    already applies — `ananke-calibrate`'s derivers and reports, and the
    estimator's integration tests, all take `status == "ok"` — so none of them
    needs to learn about builds, and one that forgets is looking at a status it
    understands rather than being silently wrong.
    """
    records = [json.loads(line) for line in out.read_text().splitlines()
               if line.strip()]
    readings: dict[tuple[str, str], dict[str, tuple[int, str]]] = {}
    for record in records:
        if record.get("status") != "ok" or not record["rss"].get("gpu_used_mib"):
            continue
        key = (record["parsed"].get("arch", "?"), record["cell"])
        readings.setdefault(key, {})[record["provenance"]["runtime_sha256"]] = (
            record["rss"]["gpu_used_mib"], record["provenance"]["measured_at_utc"])

    stale: set[tuple[str, str]] = set()
    for (arch, _cell), seen in readings.items():
        for build, (value, when) in seen.items():
            for other_build, (other, other_when) in seen.items():
                if build == other_build or not value:
                    continue
                if abs(other - value) / value > tolerance and when < other_when:
                    stale.add((arch, build))

    retired = 0
    for record in records:
        key = (record["parsed"].get("arch", "?"),
               record["provenance"]["runtime_sha256"])
        if record.get("status") == "ok" and key in stale:
            record["status"] = "stale-runtime"
            retired += 1
    out.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
    if stale:
        for arch, build in sorted(stale):
            print(f"retired {arch} rows measured under {build}")
    print(f"{retired} row(s) marked stale-runtime")
    return 0


def load_plan(path: Path) -> list[Cell]:
    """Read a campaign plan: a JSON list of objects, each a `Cell`'s fields."""
    raw = json.loads(path.read_text())
    return [Cell(**{**entry, "extra": tuple(entry.get("extra", []))}) for entry in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="JSON list of cells to run")
    parser.add_argument("--retire-stale-builds", action="store_true",
                        help="mark rows a runtime upgrade invalidated, proven "
                             "by a re-measured cell, as stale-runtime so every "
                             "reader skips them")
    parser.add_argument("--reparse", action="store_true",
                        help="re-derive every record's parsed block from its "
                             "archived log instead of measuring anything")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(os.environ.get("TMPDIR", "/tmp")) / "ananke-calibration")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--archive-dir", type=Path, default=DATA / "logs",
                        help="where to keep the gzipped load logs; they are "
                             "what makes a record re-parseable later")
    parser.add_argument("--headroom-gib", type=float, default=30.0,
                        help="host memory to leave free; a cell needing more "
                             "than the remainder is skipped rather than risking "
                             "the machine")
    parser.add_argument("--load-timeout", type=int, default=1800,
                        help="seconds to wait for a model to load; a 200 GiB "
                             "--no-mmap load takes minutes")
    args = parser.parse_args(argv)

    if args.retire_stale_builds:
        return retire_stale_builds(args.out)
    if args.reparse:
        return reparse(args.out, args.archive_dir)
    if not args.plan:
        parser.error("--plan is required")
    run_cells(load_plan(args.plan), args.out, args.log_dir, args.port,
              args.load_timeout, args.headroom_gib, args.archive_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
