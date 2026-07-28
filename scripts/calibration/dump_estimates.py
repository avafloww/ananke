#!/usr/bin/env python3
"""Dump estimator output for every model in the TOML config.

Runs ``cargo run --example estimate`` against each model defined in
``models.toml`` with the flags the example supports, then post-processes
the JSON to report the per-device allocation alongside the raw estimate.

Usage::

    python3 scripts/calibration/dump_estimates.py          # table output
    python3 scripts/calibration/dump_estimates.py --json   # JSON for diffing
    python3 scripts/calibration/dump_estimates.py --compare  # compare against measured GPU memory

The script reads ``models.toml`` (gitignored, copy from ``models.toml.example``)
for the list of models and their service configurations. Paths in the TOML are
relative to ``$LLM_DIR`` (default ``/mnt/ssd0/ai/llm``).

The repo root is inferred from this script's location, so it works from any
working directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LLM_DIR = Path(os.environ.get("LLM_DIR", "/mnt/ssd0/ai/llm"))
MODELS_TOML = Path(__file__).parent / "models.toml"


@dataclass
class ModelConfig:
    name: str
    model: str
    mmproj: str | None = None
    context: int = 4096
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    flash_attn: bool | None = None
    parallel: int | None = None
    kv_unified: bool | None = None
    mtp: bool = False
    draft_model: str | None = None
    split_mode: str = "layer"
    ubatch: int | None = None
    ik_llama: bool = False
    ik_dsa: bool = False
    n_cpu_moe: int | None = None
    expert_offload_auto: bool = False
    allow_fallback: bool = False
    active_devices: int = 2
    visible_devices: int = 2
    pack: bool = True
    gpu_capacity_mib: list[int] | None = None
    cpu_capacity_mib: int | None = None

    @classmethod
    def from_toml(cls, table: dict[str, Any]) -> "ModelConfig":
        return cls(
            name=table["name"],
            model=table["model"],
            mmproj=table.get("mmproj"),
            context=table.get("context", 4096),
            cache_type_k=table.get("cache_type_k"),
            cache_type_v=table.get("cache_type_v"),
            flash_attn=table.get("flash_attn"),
            parallel=table.get("parallel"),
            kv_unified=table.get("kv_unified"),
            mtp=table.get("mtp", False),
            draft_model=table.get("draft_model"),
            split_mode=table.get("split_mode", "layer"),
            ubatch=table.get("ubatch"),
            ik_llama=table.get("ik_llama", False),
            ik_dsa=table.get("ik_dsa", False),
            n_cpu_moe=table.get("n_cpu_moe"),
            expert_offload_auto=table.get("expert_offload_auto", False),
            allow_fallback=table.get("allow_fallback", False),
            active_devices=table.get("active_devices", 2),
            visible_devices=table.get("visible_devices", 2),
            pack=table.get("pack", True),
            gpu_capacity_mib=table.get("gpu_capacity_mib"),
            cpu_capacity_mib=table.get("cpu_capacity_mib"),
        )


def load_models() -> list[ModelConfig]:
    """Load model configs from models.toml."""
    if not MODELS_TOML.exists():
        print(
            f"error: {MODELS_TOML} not found. "
            f"Copy models.toml.example to models.toml and edit the paths.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    return [ModelConfig.from_toml(t) for t in data.get("model", [])]


def build_estimate_args(cfg: ModelConfig) -> list[str]:
    """Build the argument list for `cargo run --example estimate`."""
    model_path = LLM_DIR / cfg.model
    args = [
        "cargo", "run", "--example", "estimate", "--",
        "--model", str(model_path),
        "--context", str(cfg.context),
        "--active-devices", str(cfg.active_devices),
        "--visible-devices", str(cfg.visible_devices),
    ]
    if cfg.mmproj:
        args += ["--mmproj", str(LLM_DIR / cfg.mmproj)]
    if cfg.cache_type_k:
        args += ["--cache-type-k", cfg.cache_type_k]
    if cfg.cache_type_v:
        args += ["--cache-type-v", cfg.cache_type_v]
    if cfg.ubatch:
        args += ["--ubatch", str(cfg.ubatch)]
    if cfg.mtp:
        args += ["--mtp"]
    if cfg.draft_model:
        args += ["--draft-model", str(LLM_DIR / cfg.draft_model)]
    if cfg.allow_fallback:
        args += ["--allow-fallback"]
    if cfg.parallel:
        args += ["--parallel", str(cfg.parallel)]
    if cfg.flash_attn is not None:
        args += ["--flash-attn", "on" if cfg.flash_attn else "off"]
    if cfg.kv_unified is not None:
        args += ["--kv-unified", "on" if cfg.kv_unified else "off"]
    if cfg.split_mode != "layer":
        args += ["--split-mode", cfg.split_mode]
    if cfg.ik_llama:
        args += ["--ik-llama"]
    if cfg.ik_dsa:
        args += ["--ik-dsa"]
    # Manual expert offload takes precedence over auto.
    if cfg.n_cpu_moe is not None:
        args += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    elif cfg.expert_offload_auto:
        args += ["--host-resident-experts"]
    if cfg.pack:
        gpu_caps = cfg.gpu_capacity_mib or [24000, 24000]
        cpu_cap = str(cfg.cpu_capacity_mib or 256000)
        args += ["--pack"]
        for cap in gpu_caps:
            args += ["--gpu", str(cap)]
        args += ["--cpu", cpu_cap]
    return args


def run_estimate(cfg: ModelConfig) -> dict[str, Any]:
    """Run the estimate example for one model and return parsed JSON."""
    args = build_estimate_args(cfg)
    result = subprocess.run(
        args, cwd=REPO, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "stdout": result.stdout.strip()}

    raw = result.stdout
    start = raw.find("{")
    if start == -1:
        return {"error": "no JSON in output", "stdout": raw[:500]}
    # Find the matching closing brace by counting depth.
    depth = 0
    end = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse: {e}", "stdout": raw[start:start + 200]}


def post_process(est: dict[str, Any], cfg: ModelConfig) -> dict[str, Any]:
    """Extract a compact summary from the estimate output."""
    extras: dict[str, Any] = {}
    extras["gpu_vram_mib"] = est.get("gpu_vram_mib", 0)
    extras["gpu_vram_gib"] = round(est.get("gpu_vram_mib", 0) / 1024, 2)
    extras["compute_buffer_mb"] = est.get("compute_buffer_mb", 0)
    extras["kv_total_mib"] = est.get("kv_total_mib", 0)
    extras["mtp_mib"] = est.get("mtp_mib", 0)
    extras["weights_gib"] = est.get("weights_gib", 0)
    extras["host_overhead_mib"] = est.get("host_overhead_mib", 0)
    extras["host_cache_mib"] = est.get("host_cache_mib", 0)
    extras["architecture"] = est.get("architecture", "?")
    expert_bytes = est.get("expert_total_bytes") or 0
    extras["expert_total_mib"] = round(expert_bytes / (1024 * 1024))

    placement = est.get("placement", {})
    if "error" in placement:
        extras["placement_error"] = placement["error"]
    else:
        alloc = placement.get("allocation", {})
        extras["placement"] = {}
        for dev, info in sorted(alloc.items()):
            extras["placement"][dev] = info["mib"]
        extras["expert_offload_mib"] = placement.get("expert_offload_mib", 0)
        extras["expert_offload_layers"] = placement.get("expert_offload_layers", 0)

    extras["config"] = {
        "context": cfg.context,
        "parallel": cfg.parallel,
        "split": cfg.split_mode,
        "cache_type_k": cfg.cache_type_k,
        "flash_attn": cfg.flash_attn,
        "mtp": cfg.mtp,
        "ik_llama": cfg.ik_llama,
        "ik_dsa": cfg.ik_dsa,
        "n_cpu_moe": cfg.n_cpu_moe,
        "expert_offload_auto": cfg.expert_offload_auto,
        "active_devices": cfg.active_devices,
    }

    return extras


def print_table(results: list[dict[str, Any]]) -> None:
    """Print results as a readable table."""
    print(
        f"\n{'model':<30} {'arch':<12} {'weights':>8} {'cb/dev':>7} "
        f"{'kv':>8} {'mtp':>8} {'gpu_vram':>10} {'host_oh':>8}"
    )
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error'][:80]}")
            continue
        print(
            f"{r.get('name','?'):<30} {r.get('architecture','?'):<12} "
            f"{r.get('weights_gib', 0):>7.1f}G "
            f"{r.get('compute_buffer_mb', 0):>6}M "
            f"{r.get('kv_total_mib', 0):>7}M "
            f"{r.get('mtp_mib', 0):>7}M "
            f"{r.get('gpu_vram_gib', 0):>8.1f}G "
            f"{r.get('host_overhead_mib', 0):>7}M"
        )

    # Packer allocation detail.
    print(f"\n{'model':<28} {'gpu0':>10} {'gpu1':>10} {'cpu':>10} {'offload':>10} {'layers':>7}")
    print("-" * 82)
    for r in results:
        if "error" in r:
            continue
        name = r.get("name", "?")
        p = r.get("placement", {})
        if isinstance(p, dict):
            g0 = p.get("gpu:0", 0)
            g1 = p.get("gpu:1", 0)
            cpu = p.get("cpu", 0)
        else:
            g0 = g1 = cpu = 0
        off = r.get("expert_offload_mib", 0)
        off_l = r.get("expert_offload_layers", 0)
        if err := r.get("placement_error"):
            print(f"{name:<28} PLACEMENT ERROR: {err[:50]}")
        else:
            print(f"{name:<28} {g0:>9}M {g1:>9}M {cpu:>9}M {off:>9}M {off_l:>7}")


def main() -> int:
    emit_json = "--json" in sys.argv
    do_compare = "--compare" in sys.argv

    models = load_models()
    if not models:
        print("error: no models found in models.toml", file=sys.stderr)
        return 1

    # Build the estimate example once.
    print("Building estimate example...", file=sys.stderr)
    build = subprocess.run(
        ["cargo", "build", "--example", "estimate"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    if build.returncode != 0:
        print(f"Build failed:\n{build.stderr}", file=sys.stderr)
        return 1

    results = []
    for cfg in models:
        print(f"  {cfg.name}...", file=sys.stderr)
        est = run_estimate(cfg)
        if "error" in est:
            results.append({"name": cfg.name, "error": est["error"]})
            continue
        extras = post_process(est, cfg)
        results.append({"name": cfg.name, **extras})

    if emit_json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
