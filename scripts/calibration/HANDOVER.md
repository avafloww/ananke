# Estimation parity handover

Status as of 2026-07-30, after the §6–§10 bundle the previous handover described
landed. It records where the campaign stands, what is now proven about
llama.cpp's allocation semantics, and what is left — with the derivation recipe
for each remaining item.

## Where it stands

Two checks, and they measure different things.

`scoreboard.py` compares the seven deployed configurations. Every one is inside
±5%, worst 3.2%:

| Model | Reserved | Measured | Drift |
|---|---|---|---|
| qwen3.6-35b-a3b | 14993 | 15080 | -0.6% |
| qwen3.6-27b | 39514 | 39072 | +1.1% |
| gemma-4-31b-it-qat | 43312 | 43594 | -0.6% |
| deepseek-v4-flash | 17620 | 17182 | +2.5% |
| glm-5.2 | 37477 | 38708 | -3.2% |
| laguna-s-2.1-iq4-nl | 38557 | 39570 | -2.6% |
| talkie-1930-13b-it | 13203 | 12798 | +3.2% |

`validate.py` compares *every* measured cell — 212 of 583 are comparable, and
the rest are reported with the reason. Median -1.8%, mean -0.9%, 73 outside ±5%.
That is the honest number: seven points cannot distinguish a model that
generalises from one fitted to them, and every constant comes from this dataset.

    python3 scripts/calibration/scoreboard.py
    python3 scripts/calibration/validate.py

`validate.py` compares the **prediction**, `scoreboard.py` the **reservation**.
The reservation deliberately carries slop; on the 27B they differ by 472 MiB,
which is the difference between +1.1% and -0.1%. Neither is wrong — one is the
accuracy metric, the other the safety one.

Per-architecture, from `validate.py`:

| arch | n | median | worst | outside |
|---|---|---|---|---|
| glm-dsa | 11 | +20.2% | +37.1% | 10/11 |
| qwen35moe | 41 | +2.1% | -31.8% | 18/41 |
| laguna | 27 | -4.4% | -20.8% | 13/27 |
| gemma4 | 24 | -4.5% | -12.9% | 7/24 |
| gemma3 | 16 | -5.5% | -6.9% | 9/16 |
| qwen3 | 29 | -4.6% | -6.5% | 13/29 |
| qwen35 | 24 | +0.0% | +7.0% | 1/24 |
| deepseek4 | 11 | +0.1% | +5.5% | 1/11 |
| gemma4-assistant | 7 | -3.9% | -4.3% | 0/7 |
| llama | 13 | +0.5% | +2.6% | 0/13 |
| talkie | 7 | +0.3% | +0.8% | 0/7 |

## What changed, and what it taught

Every fix below replaced a fitted constant with something read from the model
file or from llama.cpp's own accounting. That is the pattern worth continuing:
in each case the constant had been fitted to a quantity the runtime was already
telling us.

- **Recurrent state** (`estimator/recurrent.rs`). The GGUF does expose the SSM
  dimensions — `ssm.conv_kernel`, `ssm.inner_size`, `ssm.state_size`,
  `ssm.group_count` — and llama.cpp sizes both state tensors from exactly those.
  The measured constant it replaced was the tensor-split *share*, never scaled by
  slot count, and had absorbed the speculative rollback replication.
  `analyse.py` now holds the formula to all 13 measured pools, R and S apart.
- **Tensor-split compute** (`compute_buffer::tensor_split_per_device`). Charged
  per spanned GPU, not divided: llama.cpp builds the same graph on each device,
  and the reported compute column reads identically on one card and two. The
  value is `ubatch × (K·n_embd·4 + 2·n_kv + q·context) + shadow`, where the
  2-byte term is the f16 KQ mask read off the graph — which is why every
  architecture measured grows by exactly 1.00 MiB per 1024 cache tokens at
  ubatch 512.
- **MTP** (`estimator/mtp.rs`). `[spec] estimated memory usage of MTP context is
  N MiB` is the draft cache's physical size plus **one** device's compute share,
  exactly, on all 13 MTP cells. Three fitted constants went.
- **Sparse-attention indexer** (`estimator/moe/mla.rs`). A second narrow cache,
  672 MiB unaccounted on GLM-5.2. Which layers index is in the *tensor table* —
  an indexing layer carries `blk.N.indexer.*` weights.
- **mmproj CLIP graph** (`MMPROJ_GRAPH_BYTES`). llama.cpp reports it;
  subtracting the summed clip tensor sizes isolates it at 140–248 MiB.
- **Unfused attention** (`no_flash_attn_score_bytes`). One f32 per (head, cache
  token, batch token) — *paired* against each cell's flash-attention-on sibling.
  Unpaired, the old derivation saw 1.07–1.88 and doubled it for safety.
- **Tied output head** (`NonLayer::tied_head_bytes`). A tied-embedding model
  keeps a second, sharded, GPU-resident copy of its table under a tensor split
  spanning >1 card. The single-GPU cells are what make this legible: at one card
  tensor and layer report *identical* weights, so it is sharding that causes it,
  not the split mode.

Tooling: `measure.py --reparse` rebuilds every record's `parsed` block from the
archived logs, which is how the parser gained per-context memory pools, ik's
buffer lines, and the mmproj figures without re-running a single cell. It also
repairs records parsed by older parser versions. **Reach for it first** — the
logs answer far more than the parser reads.

## Established allocation semantics

Grounded in measured logs under `data/logs/`; do not re-litigate.

1. **`Meta()` breakdown columns report ONE GPU's share** of a tensor split for
   `model`, `kv`, and `compute`. `total`, `free`, `self`, and `unaccounted` are
   summed across cards, so the last is *not* a per-device figure — 23 GiB of it
   on the gemma4 production cell.
2. **`--n-cpu-moe N` offloads expert tensors of N layers; attention stays on GPU
   for every layer.**
3. **The recurrent module is replicated `parallel × (rs_seq + 1)`.** It scales
   with `--parallel`, *not* the stream count — `--kv-unified` collapses the
   attention cache and leaves this alone.
4. **Embedded MTP is one unified context**, whose cache covers the whole context
   budget however it is divided. Not per-slot.
5. **Tensor-split compute is per device, not divided.** Identical on one card
   and on two at every context measured.
6. **At one GPU a tensor split allocates exactly what a layer split does.** Every
   tensor/layer difference in the set appears only once sharding happens.
7. **The per-device CUDA shadow is 330–470 MiB**, derived per architecture from
   single-device cells where `unaccounted` still means something.
8. **ik_llama prints no memory-breakdown table.** `analyse.py`'s
   `table_less_compute` recovers the same quantity from the driver total less
   weights and context; it reproduces the GLM-5.2 production cell to 110 MiB.

## What is left

Ordered by size of error.

### 1. glm-dsa's GPU compute curve is held, not fitted (+20% median, +37% worst)

`compute_buffer_curves` gives it `base 3700 + slope 15` by hand. Its real
per-device figures are now recorded in `tuning.json` under
`table_less_compute_observations`: 901, 986, 1332, and 1968 MiB at ctx 8192,
32768, 65536, 131072 at ubatch 512, against 6370 at the production ubatch 2048
and ctx 131072.

The curve's affine-in-batch shape cannot carry that. The service runs `-amb
512`, which chunks ik's attention, so the graph changes regime above ubatch 512
and the context and batch terms couple: below the threshold the per-device figure
is `≈830 + 8.9 × ctx/1024`; above it, three points fit `≈1447 + 17.8 bytes ×
ubatch × ctx` — two parameters on three points, which is fitting noise.

The held value reproduces production and over-reserves the low-batch cells.
Fitting the affine shape anyway does the reverse (glm-5.2 goes to +7.8%).

**Recipe:** sweep ctx 32768/65536/131072 at ubatch 1024 and 2048, and one cell at
`-amb 2048` to confirm the threshold is `-amb` and not a fixed batch. Then model
two regimes explicitly, keyed on `ubatch > amb`. Do not fit until then.

### 2. ik_llama cells are excluded from every compute curve

`derive_curve` skips cells with no breakdown table. That is deliberate — pooling
them moved four architectures' curves at once and took glm-5.2 from -3.2% to
+7.8%, because the two runtimes build different graphs for one architecture.

But it means laguna's curve is fitted from *mainline* cells while laguna runs on
ik in production, and qwen35moe's likewise. Their ik cells' figures are in
`table_less_compute_observations`.

**Recipe:** give the curve table a runtime dimension (an `ik` variant beside the
existing `gemma_e` one), then fit each runtime's cells separately. The variant
machinery already exists in `curve_entry` and `variant_of`; the estimator knows
`inputs.ik_llama`.

### 3. qwen35moe's worst cells are ik at ubatch ≥ 1024 (-31.8%)

`curve-qwen36-35b-a3b-ik-c32768-ub2048` predicts 4678 against 6864;
`ikmoe-qwen36-35b-a3b-ub1024` 4678 against 5806. The same batch-regime shape as
item 1, and the same fix — item 2 is a prerequisite.

### 4. The layer-split compute curve is the last fitted *shape*

gemma3 (-5.5%) and qwen3 (-4.6%) medians sit just outside, and the curve is an
affine fit with pooled architecture groups rather than a model. The tensor path
shows a hyperparameter model is reachable; the layer split needs its own
decomposition, and it is not the same one — its residual is *not* proportional to
batch (dividing by ubatch makes `K` range 15–126 within one architecture), so it
has a large flat term the tensor graph does not.

**Recipe:** the layer split adds, over the tensor model, the output logits buffer
on the head device and the split-boundary copies. Fit
`flat(arch) + ubatch × (K·n_embd·4 + 2·n_kv)` against the per-device columns of
layer cells, head and non-head devices separately — `derive_curve` already
distinguishes them in `_tensor_intermediates`' shape.

### 5. gemma4-E4B's tensor-split extra is 790 MiB against a 525 MiB table (-12.9%)

The tied-head copy explains 525 of it. The remaining ~265 MiB is unexplained and
specific to the E-variant, whose `per_layer_token_embd.weight` stack has its own
CPU path. One model, one configuration.

**Recipe:** compare `gemma4-e4b` 1-GPU vs 2-GPU tensor `Meta() model buffer size`
against the per-layer stack's size divided by the layer count — the hypothesis
worth testing first is that a slice of the stack becomes GPU-resident per card.

### 6. `MMPROJ_GRAPH_BYTES` is a flat maximum over two vision configurations

248 MiB, from image 1472/merge 2; gemma-4's 768/merge 3 takes 140. Two points
cannot distinguish a scaling in image size from one in the merge factor.

**Recipe:** measure a third vision configuration — any mmproj with a different
image size *or* merge factor — and the rate becomes derivable. The parser already
records `clip_image_size` and `clip_n_merge` for exactly this.

### 7. `validate.py --check` is not in CI

It takes ~15 minutes (one estimator invocation per cell) and needs the model
files, so it cannot run on a hosted runner. It is the check that would have
caught every item above.

## Conventions worth keeping

- **Nothing hardcoded.** Every number in `tuning.json` has a deriver in
  `analyse.py` or a `kind` saying why not (`structural`, `policy`, `reachable`).
  `analyse.py emit --check` is CI's guard and it compares evidence text too, so a
  constant cannot drift from the data that justifies it.
- **Prefer a formula over a fit, and a paired measurement over an absolute one.**
  Both the recurrent-state and unfused-attention fixes came from noticing that a
  fitted constant was standing in for something the runtime states directly.
- **Check the model against every measurement, not the aggregate.** The recurrent
  formula is held to R and S separately because a model wrong in both directions
  can still land on the right total — which is exactly how the constant it
  replaced reproduced Qwen3.6-27B's one-slot reading while being twofold out on
  its two-slot one.
- **Compare like placements.** 101 cells are skipped by `validate.py` because
  ananke packs a small model onto one card where the campaign measured it across
  two. That is a difference in placement, not in estimation, and reading it as
  error cost a spurious 7–8% systematic.
- The committing skill applies: propose each commit before landing it.
