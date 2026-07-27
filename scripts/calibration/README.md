# Memory calibration

ananke predicts how much GPU and host memory a llama.cpp service will need
before it starts one. Those predictions come from constants —
`ananke/src/estimator/host_buffer.rs`, `compute_buffer.rs`, `mtp.rs` — and a
constant nobody can point at data for is a guess with a decimal point.

This directory measures real `llama-server` processes so every one of those
constants has a dataset behind it, and so a later reader can refit them
without re-deriving the method.

## Running it

```sh
python campaign.py                      # every phase, in order, committing as it goes
python campaign.py --phases phase4      # one phase
python campaign.py --dry-run            # cell counts only
python measure.py --out data/mine.ndjson --plan myplan.json   # ad hoc
```

The campaign is meant to be left alone for hours. It commits each phase's data
before starting the next, skips cells that will not fit in memory, and resumes
where it stopped — a rerun re-measures only what did not complete.

**Nothing else may use the GPUs while it runs.** A second process changes both
the free-memory figures and the timings.

## What the phases are for

| phase | asks |
|---|---|
| `phase0` | the noise floor — repeat one cell, so a later difference is known to be signal |
| `phase1` | which factors move the host baseline at all, on the cheapest model |
| `phase2` | whether the per-model term follows layers, hidden size, or nothing — seven models |
| `phase2b` | the slopes: two contexts, two batch sizes, flash attention on and off |
| `phase3` | the same on ik_llama, which sizes its graph by different rules |
| `phase4` | terms with a switch rather than a curve — MTP, mmproj, offload regimes, `-rtr`, huge pages, embeddings, growth under an agent workload |
| `phase5` | the operator's real service configurations, **held out of every fit** |

Phase 5 is the honesty check. Everything else is in-sample.

## Contributing measurements from another machine

The constants are hardware-dependent in places, so records from a different
box are useful rather than merely redundant. Three things to set:

```sh
export LLM_DIR=/path/to/your/gguf/collection   # default /mnt/ssd0/ai/llm
export MAINLINE_BIN=llama-server               # default: found on PATH
export IK_BIN=ik-llama-server
```

Then edit `MODELS` in `plan.py` to name the models you actually have. Each
entry needs a `key`, a path relative to `LLM_DIR`, and the runtimes it can be
served with. **`n_cpu_moe` is tuned for 2x24 GiB** — recompute it for your
cards, or the hybrid cells will not be measuring the regime they claim to.

Records carry the hardware they were taken on, so per-hardware curves can be
fitted without separating the files. They join across machines on
`provenance.model_key` (repo-and-file, not the absolute path), which is why
that field exists.

If a model will not fit, the cell is recorded with
`status: "skipped-insufficient-memory"` rather than dropped silently, and a
later rerun retries it.

## The record format

One NDJSON object per measurement, in `data/<phase>.ndjson`, with the load log
kept gzipped alongside in `data/logs/`.

- `schema` — bump it in `measure.py` whenever the shape changes. `1` was flat
  CSV-era rows; `2` added nesting, hardware, and traces; `3` added the generic
  per-device breakdown, per-process GPU memory, and model identity.
- `provenance` — when, where, which binary, which model, which ananke revision.
- `hardware` — GPUs, driver, CPU, cores, RAM, kernel, THP, governor.
- `factors` — the complete `Cell`; a row is a full description of the process
  that produced it.
- `parsed` — what the loader logged: buffer sizes, model shape, and `devices`,
  the per-device `total/free/self/model/context/compute/unaccounted` split.
- `rss` — `/proc` at the peak, sampled on ananke's own two-second cadence,
  plus the final reading, the growth since spawn, and per-process VRAM.
- `trace` — every sample, each with an absolute timestamp.
- `log` — the gzipped log's filename, so anything these parsers missed is
  still recoverable.

## Gotchas worth knowing before you trust a number

- **The memory-breakdown table is printed at shutdown.** Parse the log after
  the process exits, not while it is serving.
- **Under `--split-mode tensor` the device is named `Meta()`**, not `CUDA0`.
  Matching on the CUDA name records zeros for every production configuration.
- **A draft model or vision projector loads after the target**, so a metadata
  key appears twice with different values. First occurrence is the target's.
- **`cudaMallocHost` is accounted as `RssShmem`**, not `RssAnon`. Read
  `RssAnon + RssShmem`, and not `VmRSS`, which includes the mapped GGUF.
- **Verbose logging serialises graph ops**, so growth runs turn it off and
  take the arena from the matching non-growth cell.
- **Check the CPU governor.** `powersave` on `acpi-cpufreq` halves CPU-bound
  throughput; it is recorded in `hardware` for exactly this reason.

## What this does not cover

Single-process llama.cpp services only. Sustained multi-hour growth, heavy
concurrent multi-slot load, and failure modes that appear only under real
traffic are out of scope — `phase4`'s bench cells reach toward the first two
but do not settle them.
