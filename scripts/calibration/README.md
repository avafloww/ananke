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
python campaign.py                  # every cell, cheapest order, committing as it goes
python campaign.py --dry-run        # print the schedule and stop
python campaign.py --only laguna    # cells whose label matches a substring
python progress.py --watch          # how far it has got
```

The campaign is meant to be left alone for hours. It commits every fifteen
minutes, skips cells that will not fit in memory, and resumes where it
stopped — a rerun re-measures only what did not complete.

**Nothing else may use the GPUs while it runs.** A second process changes both
the free-memory figures and the timings.

## Questions, and why they are not a schedule

Each cell is tagged with the questions it answers:

| tag | asks |
|---|---|
| `noise` | the noise floor — repeat one cell, so a later difference is known to be signal |
| `factor-screen` | which factors move the host baseline at all, on the cheapest model |
| `model-baseline` | whether the per-model term follows layers, hidden size, or nothing |
| `curves` | the slopes: three contexts, two batch sizes, flash attention on and off |
| `fork` | the same on ik_llama, which sizes its graph by different rules |
| `switches` | terms with a switch rather than a curve — MTP, mmproj, offload regimes, `-rtr`, huge pages, embeddings, and growth under an agent workload |
| `mtp-slots` | the MTP overhead against slot count at a fixed context, which the first campaign confounded with context |
| `flash-attention` | flash attention off across both context and batch, which one point could not distinguish from a baseline shift |
| `slot-batch` | the slot rules at a second batch size, since both feed terms that scale with the batch |
| `concurrency` | the per-slot cost, across architectures rather than the one that has a series |
| `holdout` | the operator's real service configurations, **held out of every fit** |

`holdout` is the honesty check. Everything else is in-sample.

The last four tags exist because of a mistake worth not repeating. A term
measured at a single point in some axis looks flat in that axis, and three
constants were wrong for exactly that reason — the flash-attention cost read
as an inconsistent baseline shift, the shared-cache window mask as a fixed
three copies, and a separate MTP draft's compute as context-independent. Each
was fitted from cells that all sat at one context, or one batch, or one slot
count. When adding a phase, vary the axis the rule is *about*, and one more.

These are tags, not passes. Running one pass per question would reload every
model once per question, and reloading is the expensive move here — the 205
GiB production quant cannot share the page cache with anything else, so each
revisit pays full disk cost. `plan.all_cells` merges every question into a
single list ordered so that consecutive cells disturb as little as possible:
all of a model's work happens while its weights are hot, and models run
smallest first, because the largest evicts everything behind it on the way
past. A configuration wanted by two questions is measured once and tagged
with both.

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

One NDJSON object per measurement, in `data/measurements.ndjson`, with the
load log kept gzipped alongside in `data/logs/`. Records from earlier,
narrower schemas are kept in `data/legacy/` rather than merged, since they
lack fields the current ones carry.

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
- `checkpoints` — for growth cells, one entry per turn: generated tokens, KV
  depth, and memory read immediately after, so growth is fittable against
  tokens rather than against the clock.
- `factors.purpose` — which questions wanted this cell.
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

See `PLAN.md` for the complete specification: every constant this campaign
derives, the cells that determine it, the analysis protocol, and an explicit
list of what it does not establish.

## Architectures this campaign cannot calibrate

`compute_buffer.rs` carries tuning curves for nineteen architectures. The
model library reaches nine of them. These ten have **no model here at all**,
so their curves are inherited from whoever first fitted them and are not
re-derivable from this dataset:

`deepseek2`, `gemma2`, `glm4moe`, `gpt-oss`, `jamba`, `llama4`, `mamba`,
`mixtral`, `qwen3moe`, `qwen3vlmoe`

Fixing this means adding models, not rearranging phases — a VL-MoE quant for
`qwen3vlmoe`, a Mamba or Jamba model for the recurrent curves, and so on. Any
claim that this campaign validates the estimator applies to the nine measured
architectures and not to these.

## What this does not cover

Single-process llama.cpp services only. Sustained multi-hour growth, heavy
concurrent multi-slot load, and failure modes that appear only under real
traffic are out of scope — `phase4`'s bench cells reach toward the first two
but do not settle them.

## Probes

`measure.py` samples one process once per configuration, which cannot separate
a term allocated once from one that accumulates with use. `probe_host_growth.py`
varies one thing at a time against a fresh server and reports the series:

```sh
python probe_host_growth.py maps    qwen36-27b gemma3-27b   # where it lives
python probe_host_growth.py growth  qwen36-27b              # leak or prompt cache
python probe_host_growth.py prefill qwen36-27b              # what triggers it
```

These load real models and read real memory, so run them one at a time against
an idle machine. What they settled is in FINDINGS.md under "The per-model
residual is a first-request step".
