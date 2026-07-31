# Calibration plan

The complete specification of what this campaign measures, why, and which
constant each measurement is meant to determine. Written so that a reader who
has never seen the machine can judge whether the sampling is sufficient, and
so that someone with different hardware can repeat it.

## The problem

ananke decides, before starting a `llama-server`, how much GPU and host memory
that server will need. It uses the answer to pack models onto cards and to
reserve host memory. Under-predict and the load OOMs; over-predict and
capacity is wasted, or a service is refused placement it could have had.

The prediction is a sum of modelled terms plus tuning constants. The constants
are the part with no first-principles derivation — they stand in for
allocations llama.cpp makes that we can only observe. Today most of them come
from one or two ad-hoc measurements against whichever model was at hand. This
campaign replaces that with a dataset.

A rolling correction (`ananke/src/supervise/rolling.rs`) later scales the
prediction by observed reality, but only within `[0.8, 1.5]`. A constant that
puts a model outside that band cannot be recovered by any amount of
observation, which is why "reachability" matters more than closeness.

## The constants to be derived

### `crates/estimate/src/host_buffer.rs`

| constant | current | what it stands for |
|---|---|---|
| `PINNED_EXTRA_BYTES` | 16 MiB | slack between the arena llama.cpp reports and the pinned memory actually resident |
| `PROCESS_BASE_BYTES` | 112 MiB | fixed per-process host cost: CUDA runtime, tokenizer, sampler, graph metadata |
| `PROCESS_BASE_BYTES_PER_LAYER` | 3.4 MiB | the part of that base which scales with layer count |
| `PROCESS_BASE_BYTES_MOE` | 200 MiB | flat additional allowance for mixture-of-experts models |
| `IK_MOE_CPU_BYTES_PER_TOKEN` | 81 KiB | ik_llama's CPU-resident MoE op buffers, per batch token |
| `IK_OP_OFFLOAD_MIN_BATCH` | 32 | the batch threshold below which ik keeps those ops on the CPU |
| `NO_FLASH_ATTN_BYTES_PER_TOKEN` | 8 KiB | extra KQ mask width when flash attention is off |
| `MTP_HOST_BYTES_SEPARATE_DRAFT` | 512 MiB | host cost of a separate draft GGUF |
| `MTP_HOST_BYTES_EMBEDDED` | 224 MiB | host cost of an embedded MTP head |
| `DEFAULT_CACHE_RAM_MB` | 8192 | the `-cram` prompt-cache reservation |

The arena itself is modelled, not fitted: KQ mask at
`n_kv x min(ctx, ubatch) x (fa ? 2 : 4)`, a second window-sized mask on
interleaved-SWA models, and hidden-state buffers of `n_embd x n_tokens` f32.
The two runtimes size it by **different rules** — mainline uses
`n_kv = ctx / parallel` and two hidden buffers; ik uses `n_kv = ctx` and one.
The campaign must confirm both laws, not just fit a constant.

### `crates/estimate/src/compute_buffer.rs`

Per-architecture GPU curves of the form `base + slope * (ctx / 1024)` MiB per
device. `deepseek4`'s slope is additionally linear in `ubatch`, because its
NSA indexer scores every query token against the whole context.

Architectures reachable with the models in this library: `gemma3`, `gemma4`
(and its E-variant), `qwen35`, `qwen35moe`, `laguna`, `deepseek4`, `glm-dsa`,
`lfm2`, `talkie`, and the llama-family default.

Architectures with a curve but **no model here**: `deepseek2`, `gemma2`,
`glm4moe`, `gpt-oss`, `jamba`, `llama4`, `mamba`, `mixtral`, `qwen3moe`,
`qwen3vlmoe`. Their constants stay inherited and unverified. This is a stated
limitation, not an oversight.

### `crates/estimate/src/mtp.rs`

`MTP_COMPUTE_MIB` (1700), `DRAFT_MODEL_COMPUTE_MIB` (300), and the KV formula
`nextn x head_count_kv x (key_length + value_length) x 2 x context`.

## What is measured, per cell

One cell is one `llama-server` process, started with a specific flag set,
health-checked, exercised, then stopped. Recorded:

- **From the load log**: every buffer size llama.cpp prints, the model's shape
  (`n_layer`, `n_embd`, `n_head_kv`, `n_embd_head_k/v`, `n_expert`, `n_swa`,
  `n_vocab`, `n_ctx_train`), the architecture, and the full per-device memory
  breakdown — `total / free / self / model / context / compute / unaccounted`
  — plus the `Host` row. The log itself is kept gzipped so anything the
  parsers missed is recoverable.
- **From `/proc`**: `RssAnon + RssShmem` (owned) and `RssFile` (mapped),
  sampled every 2 seconds from process spawn, keeping monotonic peaks. The
  2-second cadence matches ananke's own snapshotter, so a reading here is
  directly comparable to what the daemon will observe.
- **From the driver**: per-process VRAM per card, via
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`. This is the only
  per-device source for ik, which does not print mainline's breakdown table.
- **For growth cells**: a checkpoint per turn — generated tokens, KV depth,
  and memory read immediately after — so growth is fittable against tokens
  rather than against wall-clock.
- **Provenance**: UTC and local timestamp, hostname, binary path, ananke
  revision and dirty flag, model key/quant/bytes.
- **Hardware**: GPUs and driver, CPU model and core counts, NUMA, RAM, kernel,
  THP mode, CPU governor.

Records are NDJSON, one object per cell, schema-versioned.

## Measurement hazards already found and handled

Each of these silently corrupted data before it was caught:

1. The memory-breakdown table is printed **at shutdown**; parsing the log
   while the server was still serving lost every per-device figure.
2. Under `--split-mode tensor` llama.cpp fuses the cards and names the device
   `Meta()`, not `CUDA0`. Matching on the CUDA name recorded zeros for every
   production configuration.
3. A draft model or projector loads **after** the target, so metadata keys
   appear twice. Last-occurrence recorded the draft's shape.
4. ik_llama's server does not set `SO_REUSEADDR` and loses the bind to the
   previous server's lingering socket, exiting in a way that reads as a load
   failure. The harness now waits for the port to be bindable.
5. `cudaMallocHost` is accounted as **RssShmem**, not RssAnon.
6. `RssFile` is the mapped GGUF, mapped with `MAP_POPULATE`, so `VmRSS`
   massively overstates owned memory on a hybrid.

## The schedule

496 cells across 13 models, ~8 hours. Counts below are generated by
the sweep generator; regenerate the full list with
`cargo run -p ananke-calibrate --bin plan -- all`.

Cells are ordered so consecutive ones disturb as little as possible: all of a
model's work runs while its weights are hot in the page cache, and models run
smallest first because the largest (205 GiB) evicts everything behind it.

Questions are **tags on cells**, not separate passes. A configuration wanted
by two questions is measured once and tagged with both, which is why the tag
counts below sum to more than 496.

| tag | cells | determines |
|---|---|---|
| `model-baseline` | 176 | `PROCESS_BASE_BYTES`, `_PER_LAYER`, `_MOE`, `PINNED_EXTRA_BYTES` |
| `fork` | 104 | the same on ik, whose graph-sizing law differs |
| `curves` | 94 | every `compute_buffer.rs` curve; confirms the arena law |
| `factor-screen` | 64 | which factors move the baseline at all |
| `switches` | 32 | MTP, mmproj, offload regimes, `-rtr`/THP, `-cram`, embeddings, growth |
| `interactions` | 24 | whether curve slopes survive q8_0, `np` 4, and tensor split |
| `interior` | 14 | the points that turn two-sample lines into measured relationships |
| `device-scaling` | 12 | per-visible-device host cost, and thread-count dependence |
| `holdout` | 7 | production configurations |
| `replication` | 6 | noise in the regimes the floor never visited |
| `noise` | 5 | the noise floor |
| `concurrency` | 3 | per-slot state, which no other cell allocates |

### Axis values

```
gpus        {none, 0, 0+1}      split   {layer, tensor}
kv_type     {f16, q8_0}         fa      {on, off}
ctx         {512 … 524288}      ubatch  {256, 512, 1024, 2048}
parallel    {1, 2, 4}           ngl     {0, 18, 99}
runtime     {mainline, ik}      cram    {0, 8192}
served      {yes, no}           rtr/thp {on, off}
```

Not a full cross — that would be millions of cells. Each block varies the
axes its question needs and holds the rest at ctx 32768 / ub 512 / fa on /
np 1 / layer split.

### Per-model deviations, and why

- **dsv4f**: no tensor split — mainline rejects it for `deepseek4` at load.
- **glm52**: f16 only (`-dsa` refuses a quantised cache), always with
  `-mla 1 -dsa -amb 512 --no-mmap -t 24`, which is how it is served.
- **lfm2-embed**: single GPU, `--embeddings`; spreading a 350M model changes
  the baseline the cell exists to isolate.
- **talkie**: curve at 512/1024/2048, its native context, rather than
  extrapolating past what it was trained on.
- **hybrids on one card**: their own `n_cpu_moe`, because a value sized for
  two cards keeps too many layers resident and aborts the load.

## Reproducibility on other hardware

Three environment variables (`LLM_DIR`, `MAINLINE_BIN`, `IK_BIN`) and the
`Model` registry in `calibration/crates/calibrate/src/plan/library.rs` are the whole
machine-specific surface.
Records join across machines on `provenance.model_key` (repo-and-file, not
absolute path). `n_cpu_moe` values are tuned for 2x24 GiB and must be
recomputed elsewhere; single-GPU variants have their own field because a
hybrid sized for two cards aborts on one.

Cells that will not fit are recorded as `skipped-insufficient-memory` rather
than dropped, and retried on a later run.

## Analysis protocol

Accuracy is reported by **leave-one-model-out cross-validation**, not by the
holdout. For each shared constant, fit without one model and predict that
model's cells; repeat for every model. This costs no extra measurement and is
a genuine out-of-sample estimate, which the holdout is not — the holdout's
models are also in the fitting set, so any accuracy figure from it alone is
optimistic.

Cross-validation is impossible for the five constants that have exactly one
model per architecture (`laguna`, `glm-dsa`, `deepseek4`, `talkie`, `lfm2`).
Those are **model-specific fits**, not architecture constants, and must be
labelled as such wherever they are quoted.

## Known limitations

Stated so that nobody later mistakes this campaign for more than it is.

- **Portability is unverified, and unverifiable from one machine.** Every cell
  runs on the same box, so any hardware sensitivity is perfectly confounded
  with the constant itself. The `device-scaling` cells detect whether a
  per-visible-device cost exists and bound it; they cannot establish
  invariance across GPU generations, VRAM sizes, or card counts beyond two.
  The defensible claim is "measured on this hardware, portability bounded but
  unverified". Closing this needs contributed data, which is why the record
  format carries hardware as regressors rather than as documentation.
- **The estimator has nowhere to put a hardware term.** `host_overhead_bytes`
  and `tuning_for` take model shape and service flags only; every constant is
  a compiled scalar. So portability depends on the constants being genuinely
  hardware-invariant — a stronger requirement than the campaign can test. If
  `device-scaling` finds a per-device increment, the estimator needs a new
  input, not a new constant.
- **Ten architectures have curves but no model here**, listed above. Their
  constants stay inherited.
- **Multi-service co-residency is not measured.** Every cell runs alone, while
  the daemon's actual job is packing several servers onto the same cards.
  Fragmentation and contention between concurrent servers are unobserved.
- **Vision is not measured.** The mmproj cells load the projector and send
  text; the vision graph allocates on the first image request, which is not
  sent, because image inference currently segfaults on the QAT build.
- **Growth runs ten turns**, roughly five thousand generated tokens. Enough to
  separate "grows per token" from "allocates once on first use"; not enough to
  characterise a slow leak.
- **The runtime cannot be versioned.** Both forks report `version: 0
  (unknown)` and nix normalises the binary mtime to the epoch. Records carry
  the binary's sha256 and store path instead, which identify the build exactly
  but cannot be mapped to an upstream commit — so cross-machine pooling can
  tell "different build" from "same build", but not how they differ.
- **Single-process llama.cpp services only.** No vLLM, no concurrent
  multi-service load beyond the three `concurrency` cells.
