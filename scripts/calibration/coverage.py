"""Report where the dataset is too thin to have measured what it claims.

A term measured at a single point in some axis looks flat in that axis. That
is not a hypothetical: it produced four wrong constants in this campaign.

- The flash-attention cost read as an inconsistent baseline shift, because
  seventeen of its nineteen cells sat at one context and one batch. Swept, it
  is a clean per-token rate.
- The shared-cache window mask was three copies, from a sweep taken entirely
  at ubatch 512 where a 1024-token window spans two batches. At 2048 it is
  two.
- A separate MTP draft's compute was context-independent, from cells at two
  contexts that happened to agree.
- The MTP overhead's slot dependence was confounded with context, because
  every one-slot pair sat at one context and the only four-slot pair at
  another.

Each was found by accident, late, after the constant had been in use. This
turns the audit into something that runs: for every regime the estimator
models, how many distinct points exist along the axes that regime's rule
depends on. A regime measured at one point is reported, whether or not
anybody currently suspects it.

    python coverage.py            # report
    python coverage.py --check    # exit non-zero if a modelled regime is thin

`--check` is what CI runs. It fails on a regime that both feeds a constant and
has one point in an axis, which is the configuration that has been wrong every
time so far.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data" / "measurements.ndjson"

# Each regime names the axes its rule depends on. A rule that scales with the
# batch cannot be checked by cells at one batch, however many there are, which
# is why this is per-axis rather than a cell count.
REGIMES: dict[str, dict] = {
    "flash attention off": {
        "select": lambda f: f["flash_attn"] != "on",
        "axes": ("ctx", "ubatch", "gpus"),
        "constant": "no_flash_attn_rates",
    },
    "quantised KV": {
        "select": lambda f: f["kv_type"] != "f16",
        "axes": ("ctx", "ubatch"),
        "constant": "quantised_cache_rates, quantised KV compute",
    },
    "tensor split": {
        "select": lambda f: f["split"] == "tensor",
        "axes": ("ctx", "ubatch", "gpus"),
        "constant": "tensor_split_baseline",
    },
    "shared KV cache": {
        "select": lambda f: f["kv_unified"] and f["parallel"] > 1,
        "axes": ("ctx", "ubatch"),
        "constant": "the window-mask count",
    },
    "multiple slots": {
        "select": lambda f: f["parallel"] > 1,
        "axes": ("ctx", "ubatch", "parallel"),
        "constant": "mask streams, MTP KV",
    },
    "concurrent requests": {
        "select": lambda f: (f.get("concurrency") or 1) > 1,
        "axes": ("concurrency", "ctx"),
        "constant": "per_slot_host_bytes",
    },
    "checkpointed prompt": {
        "select": lambda f: (f.get("probe_prompt_tokens") or 4) >= 8192,
        "axes": ("ctx", "gpus"),
        "constant": "checkpoint_headroom_bytes",
    },
    "MTP": {
        "select": lambda f: bool(f["spec_type"]),
        "axes": ("ctx", "parallel"),
        "constant": "MTP_COMPUTE_MIB, MTP host bytes",
    },
    "ik_llama": {
        "select": lambda f: f["runtime"] == "ik",
        "axes": ("ctx", "ubatch", "gpus"),
        "constant": "ik_moe_rates, baseline @ik",
    },
    "hybrid": {
        "select": lambda f: bool(f["n_cpu_moe"]),
        "axes": ("ctx", "ubatch"),
        "constant": "expert offload placement",
    },
    "single card": {
        "select": lambda f: len((f["gpus"] or "0").split(",")) == 1,
        "axes": ("ctx", "ubatch"),
        "constant": "the mask-copy rule at one copy",
    },
    "no mmap": {
        "select": lambda f: bool(f.get("no_mmap")),
        "axes": ("ctx", "ubatch"),
        "constant": "host_peak's RssFile discriminator",
    },
}


def rows() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def audit() -> list[tuple[str, str, int, int, str]]:
    """One entry per regime: cells, and the thinnest axis it depends on."""
    measured = [r for r in rows()
                if r.get("status") == "ok" and (r.get("parsed") or {}).get("arena_mib")]
    out = []
    for name, spec in REGIMES.items():
        group = [r for r in measured if spec["select"](r["factors"])]
        if not group:
            out.append((name, spec["constant"], 0, 0, "never measured"))
            continue
        thinnest, count = min(
            ((axis, len({r["factors"].get(axis) for r in group})) for axis in spec["axes"]),
            key=lambda pair: pair[1])
        out.append((name, spec["constant"], len(group), count, thinnest))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if a modelled regime has one point "
                             "in an axis its rule depends on")
    args = parser.parse_args()

    thin = []
    print(f"{'regime':22}{'cells':>7}{'thinnest axis':>26}   constant")
    for name, constant, cells, points, axis in sorted(audit()):
        mark = "  <-- one point" if cells and points < 2 else ""
        print(f"{name:22}{cells:>7}{f'{axis} ({points})':>26}   {constant}{mark}")
        if cells and points < 2:
            thin.append((name, axis, constant))

    if not args.check:
        return 0
    if thin:
        print("\nregimes measured at a single point in an axis their rule depends on:")
        for name, axis, constant in thin:
            print(f"  {name}: one distinct {axis}, and {constant} is fitted from it")
        print("\nA rule that is wrong in that axis is invisible at one point. Add a "
              "second before trusting the constant.")
        return 1
    print("\nevery modelled regime varies in the axes its rule depends on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
