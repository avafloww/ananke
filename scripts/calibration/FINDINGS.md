# Findings

Live notes from the campaign, written as results land. Each entry says what
was measured, on how much data, and what is still unresolved — so that a
conclusion drawn early is not mistaken later for one the full dataset
supports.

Reproduce any of these with `python analyse.py` against
`data/measurements.ndjson`.

## The arena law is exact, including for sliding-window models

At 193 measurements the modelled arena reproduces the measured
`CUDA_Host compute buffer size` to within 0.1 MiB everywhere except two cases,
both isolated below:

| arch | runtime | cards | n | mask multiple | residual |
|---|---|---|---|---|---|
| qwen3 | mainline | 1 | 38 | 1.00 ±0.02 | 0.0 MiB |
| qwen3 | ik | 1 | 4 | 1.00 ±0.00 | 0.0 MiB |
| qwen3 | ik | 2 | 8 | 1.00 ±0.00 | 0.0 MiB |
| gemma3 | mainline | 1 | 8 | 1.00 ±0.00 | 0.1 MiB |
| gemma3 | ik | 2 | 7 | 1.00 ±0.00 | 0.0 MiB |
| lfm2 | mainline | 1 | 9 | 1.00 ±0.00 | 0.0 MiB |
| llama | ik | 1 | 2 | 1.00 ±0.00 | 0.0 MiB |
| talkie | mainline | 1 | 1 | 1.01 | 0.0 MiB |

gemma3 is the significant row: an interleaved-SWA model, exact on both
runtimes, which confirms the second-mask term — window-plus-batch on mainline,
full-context on ik — rather than leaving it assumed.

That distinction was worth a mistake. The first version of `analyse.py`
applied mainline's window sizing to ik and produced K = 1.91 for gemma3 on ik,
which reads exactly like a fork multiplier and is not one. `CONTRIBUTING.md`
had the rule right; the re-implementation had it wrong. A "multiplier" that
appears in only one architecture is more likely a missing term than a law.

`mask = pad(n_kv) x min(ctx, ubatch) x (fa ? 2 : 4)`, plus two f32
`n_embd x n_tokens` hidden-state buffers on mainline and one on ik, with
`n_kv = ctx / parallel` on mainline unless the cache is unified, and `n_kv =
ctx` on ik regardless. That is arithmetic over ggml's graph, not a fit, and it
holds.

## mainline replicates the mask four times under LAYER split only

| placement | runtime | flash attn | mask multiple |
|---|---|---|---|
| one card | either | on | 1.00 |
| two cards, **tensor** split | mainline | on | **1.00** |
| two cards, **layer** split | mainline | on | **4.00** |
| two cards, layer split | mainline | off | 4.25 (non-SWA) / 4.9 (SWA) |
| two cards, either split | ik | on/off | 1.00 |

Confirmed on qwen3, gemma3, gemma4, and llama. Flat across context
(8192-65536), batch (512-2048), slot count, and cache mode.

**This corrects an earlier entry in this file.** It first read as "four times
on two cards", because the analysis keyed on card count. Under
`--split-mode tensor` llama.cpp fuses the cards into one `Meta()` device and
the mask is *not* replicated; it is the split mode that decides, not how many
cards are present. Keying on cards attributed a layer-split effect to every
two-card run — including tensor-split ones, where it is absent.

The distinction matters for which services are affected. In the operator's
config the Qwen 3.6 models and gemma-4-31B-QAT run `devices.split = "tensor"`
and are **not** affected. The hybrids that run layer-split — laguna, dsv4f,
glm-5.2 — are, and they are also the ones at long context.

`CONTRIBUTING.md`'s "extra GPU, layer-split: +24 MiB" is right about *when*
(it says layer-split) and wrong about *what*: it is a multiplier on a term
that scales with `n_kv x n_tokens`, not a constant. It was calibrated at
ctx 8192, where `3 x 8 MiB` happens to equal 24 MiB.

| context | true excess | flat 24 MiB says |
|---|---|---|
| 8192 | 24 MiB | 24 MiB |
| 32768 | 96 MiB | 24 MiB |
| 65536 | 192 MiB | 24 MiB |
| 32768 @ ub 2048 | 384 MiB | 24 MiB |

**RESOLVED**: it is not about the number of cards. With placement pinned to
the CPU (`-ngl 0`) and only the count of visible CUDA devices varying, the
arena does not move at all — 54.52 MiB on gemma3 at one card and at two,
52.11 MiB on the 35B at one and at two. Replication happens when layers are
*split across* devices, not when devices are merely present. So the
multiplier belongs to layer-split placement, and an operator with more cards
inherits the same 4x rather than a larger one — though whether it grows beyond
two cards under layer split is still unmeasured here, since only two are
available.

## RESOLVED: the gemma4 E-variant carries a per-layer input buffer

Charging the mainline law leaves gemma-4-E4B with a residual that is flat in
context and grows with batch. Dividing it by `n_layer x n_tokens` gives a
constant:

| model | layers | cards | ctx | ubatch | residual | bytes / layer / token |
|---|---|---|---|---|---|---|
| gemma-4-26B-A4B | 30 | 1 | 32768 | 512 | **0.02 MiB** | ~0 |
| gemma-4-E4B | 42 | 1 | 32768 | 512 | 21.02 MiB | 1025 |
| gemma-4-E4B | 42 | 2 | 8192 | 512 | 21.09 MiB | 1028 |
| gemma-4-E4B | 42 | 2 | 32768 | 512 | 21.09 MiB | 1028 |
| gemma-4-E4B | 42 | 2 | 65536 | 512 | 21.09 MiB | 1028 |
| gemma-4-E4B | 42 | 2 | 32768 | 2048 | 84.34 MiB | 1028 |

**1028 bytes is 257 x 4** — an f32 buffer of 257 elements per layer per token,
which is the E-variant's per-layer embedding input. gemma-4-26B-A4B is the
control: same architecture string, not an E-variant, residual 0.02 MiB.

So the term is `n_layer x 257 x 4 x n_tokens`, applies to E-variants only, and
is independent of context and card count. `compute_buffer.rs` already
special-cases the E-variant on the GPU side with its own curve; the host model
has no such term, so a gemma-4-E4B service is under-reserved on the host by
~21 MiB at ub 512 and ~84 MiB at ub 2048.

Small in absolute terms. What matters is that it has a discovered shape rather
than a fitted fudge, and the control model rules out the alternatives.

## MTP: the constant was calibrated against the wrong number

qwen3.6-27B, embedded MTP head, measured at two contexts:

| quantity | ctx 32768, np 1 | ctx 131072, np 4 |
|---|---|---|
| llama.cpp's own `[spec] estimated memory usage of MTP context` | 258 MiB | 708 MiB |
| **measured GPU delta** (MTP minus no-MTP, same config) | — | **+2892 MiB** |
| ananke's model (KV at nextn=1, plus `MTP_COMPUTE_MIB` 1700) | — | 2212 MiB |

Two problems.

**The log line is not the cost.** `CONTRIBUTING.md` says `MTP_COMPUTE_MIB` was
calibrated against that `[spec]` line. The line reports the MTP *context* —
708 MiB — while the process actually takes 2892 MiB more VRAM in the same
configuration. Calibrating against it measures a quantity four times smaller
than the one being reserved for. The driver delta between a with-MTP and a
no-MTP cell is the right target, which is why those cells are paired.

**The constant is the wrong size anyway.** Fitting llama.cpp's own figure over
the two contexts gives `108 MiB + 4800 bytes/token`; `MTP_COMPUTE_MIB` is
1700 MiB. Against the driver delta, ananke under-predicts by ~680 MiB at
ctx 131072 — a ratio of 1.31, inside the rolling correction's [0.8, 1.5] band,
so recoverable but not free.

The per-token slope is also unexplained: 4800 bytes/token measured against
4096 from the modelled `nextn x head_count_kv x (key_length + value_length) x
2` at nextn = 1.

**Two points, one model.** qwen3.6-35B-A3B runs embedded MTP with 2 kv heads
against the 27B's 4, which is the factor the formula multiplies by and the
thing one model cannot check. It is measuring now. gemma-4-31B-QAT's
separate-draft cells are paired the same way and give the other MTP shape.

## A visible CUDA device costs ~20 MiB of host memory, and the estimator has nowhere to put it

Placement pinned to the CPU, so nothing varies but how many CUDA contexts get
initialised:

| model | no CUDA | one card | two cards | first card | each extra |
|---|---|---|---|---|---|
| gemma3-27b | 3364 MiB | 3447 MiB | 3467 MiB | +83 MiB | **+20 MiB** |
| qwen3.6-35B-A3B | 1000 MiB | 1067 MiB | 1087 MiB | +67 MiB | **+20 MiB** |

Two models of very different size and shape agree on the increment. The first
device costs 67-83 MiB — CUDA runtime initialisation — and each additional one
costs a further 20 MiB.

`PROCESS_BASE_BYTES` is a compiled 112 MiB with no hardware input at all, so
this increment is currently folded into a constant fitted on a two-card box. A
four-card operator inherits it wrong by ~40 MiB and an eight-card operator by
~120 MiB. Small, but it is exactly the kind of hidden hardware dependence that
makes a constant unportable, and it is the one term this campaign can measure
*because* the cells pin placement and vary only device count.

Acting on it means threading device count into `EstimatorInputs` — the
estimator takes model shape and service flags today and nothing about the
machine. That is a real change and belongs in its own commit, after the
campaign.

**Also measured here**: a run with no CUDA visible has a much larger arena —
199 MiB against 54.5 for gemma3, 104 against 52 for the 35B, 112 against 42
for qwen3-4b — because the intermediates a GPU run offloads to the device have
to live on the host instead. That confirms the existing note in
`CONTRIBUTING.md` rather than changing it.

## CONFIRMED: ik's CPU-MoE term and its batch threshold

Sweeping ubatch 256 / 512 / 1024 / 2048 on two ik hybrids, and subtracting the
modelled mask and hidden buffers:

| model | experts/used | threshold | ub | above? | excess | bytes/token |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 256 / 8 | 1024 | 256 | no | 20.26 MiB | **82 985** |
| Qwen3.6-35B-A3B | 256 / 8 | 1024 | 512 | no | 40.51 MiB | **82 964** |
| Qwen3.6-35B-A3B | 256 / 8 | 1024 | 1024 | yes | 0.03 MiB | 31 |
| Qwen3.6-35B-A3B | 256 / 8 | 1024 | 2048 | yes | 0.05 MiB | 26 |
| Laguna-S-2.1 | 256 / 10 | 819 | 256 | no | 40.25 MiB | 164 864 |
| Laguna-S-2.1 | 256 / 10 | 819 | 512 | no | 54.01 MiB | 110 612 |
| Laguna-S-2.1 | 256 / 10 | 819 | 1024 | yes | 0.01 MiB | 10 |
| Laguna-S-2.1 | 256 / 10 | 819 | 2048 | yes | 0.02 MiB | 10 |

`IK_MOE_CPU_BYTES_PER_TOKEN` is 81 KiB = **82 944 bytes**. Qwen3.6-35B-A3B
measures 82 985 and 82 964 — within **0.05%**, at two batch sizes.

`IK_OP_OFFLOAD_MIN_BATCH` is 32, predicting the term switches off at
`n_tokens x n_used >= 32 x n_expert`. That is ub 1024 for the Qwen (8 experts
used) and ub 819 for laguna (10 used). **Both switch off exactly there** — the
excess collapses from tens of MiB to hundredths. Two models with different
expert counts, and the predicted crossing is right for both.

The review judged both constants underdetermined and warned the campaign
*loosened* the constraint on the threshold, because the original curve block
straddled it with only ub 512 and 2048. The interior points at 256 and 1024
were added in response, and they are what makes this a confirmation rather
than a bracket.

**Laguna's per-token figure does not match** — 164 864 and 110 612 against the
Qwen's flat 82 985. It is not a constant across batch, so it is unlikely to be
the MoE term misbehaving; the likelier explanation is that the modelled *SWA
second mask* for ik is wrong for this model, which would pollute the excess
this subtraction leaves behind. Laguna is the only SWA model in this pair. The
threshold still lands exactly where predicted, which is evidence the MoE term
itself is behaving.

## deepseek4's VRAM is flat in context, where the curve says it should climb

The production hybrid (`--n-cpu-moe 40` of 43 layers, layer split, two cards):

| ctx | ubatch | driver total | llama.cpp attributes | difference |
|---|---|---|---|---|
| 8192 | 512 | 16 260 MiB | 15 496 | 764 |
| 32768 | 512 | 17 566 MiB | 16 811 | 755 |
| 65536 | 512 | 16 706 MiB | 15 947 | 759 |
| 131072 | 512 | 17 182 MiB | 16 425 | 757 |
| 32768 | 1024 | 16 692 MiB | 15 930 | 762 |
| 32768 | 2048 | 17 206 MiB | 16 446 | 760 |

Total VRAM moves by about 1.4 GiB, non-monotonically, across a 16x range of
context — it is flat, not rising. The per-device `compute` column is likewise
~1976-2019 MiB at every context and every batch.

`compute_buffer.rs` gives `deepseek4` a slope of 66 MiB per 1024 tokens of
context at ub 512, which between ctx 8192 and 131072 predicts roughly **8 GiB**
more compute buffer. Nothing of the sort appears.

This over-reserves rather than under-reserves, so it does not crash — it
refuses the model room it could have used, which on a hybrid means fewer
expert layers on the GPU and a slower service.

**The architecture explains it, and the constant's own claim fails a direct
test.** `CONTRIBUTING.md` describes the NSA indexer as scoring every one of the
`ubatch` query tokens against the whole context, giving a residual of
`k x ubatch x ctx`. The model's metadata says otherwise:

    deepseek4.attention.indexer.top_k = 512

The indexer selects a fixed top-512 keys, so its working set is bounded by
`top_k` rather than by context — sparse attention doing exactly what sparse
attention is for. And at fixed ctx 32768, quadrupling ubatch from 512 to 2048
moves the per-device compute buffer from 2019 to 2001 MiB. A `k x ubatch x
ctx` term must quadruple there. It does not move at all.

That test does not depend on the hybrid regime: quadrupling the batch would
quadruple the term wherever the layers live. So the slope is not merely being
applied outside its fitted regime — the relationship it encodes is not present
in this build.

Whether it ever was is a separate question. The operator updated llama.cpp
during this campaign, and an older build that materialised the full score
matrix instead of tiling it would have shown exactly the original scaling.
This is precisely what the recorded binary hash exists to disambiguate, and
why "the constant was right once" and "the constant is right now" are
different claims.

**Also measured**: the gap between what the driver reports and what llama.cpp
attributes is **757-764 MiB**, flat across every context and batch on two
cards — the CUDA context and whatever else sits outside llama.cpp's own
accounting. It is one of the things the per-architecture compute bases quietly
absorb, and it is now separately measured rather than folded in.

## The ik CPU-MoE "constant" is a hidden-size term frozen at one model

Measured on three ik MoE models, flash attention on, two cards, ub 512,
ctx 32768 — below the offload threshold in every case:

| model | n_embd | bytes/token | / n_embd | vs the constant |
|---|---|---|---|---|
| qwen35moe | 2048 | 82 964 | 40.5 | **1.00x** |
| laguna | 3072 | 164 864 | 53.7 | 1.99x |
| glm-dsa | 6144 | 263 168 | 42.8 | **3.17x** |

`2048 x 40.5 = 82 944`, which is `IK_MOE_CPU_BYTES_PER_TOKEN` exactly. The
constant is not a constant: it is a hidden-size-proportional term evaluated at
qwen35moe's `n_embd` and hard-coded. For GLM-5.2, whose hidden size is three
times larger, the true figure is 3.17x the constant — an under-reservation of
~88 MiB of host memory at ub 512.

Two caveats, both real. Three models cannot separate `n_embd` from other
quantities that co-vary with it. And laguna sits at 53.7 rather than ~41; it
is also the model whose ik sliding-window mask this analysis is known to model
wrongly, and that error lands in exactly this residual, so its deviation is
more likely mine than the runtime's.

## Arena model: confirmed exact for glm-dsa

Above the MoE threshold, where nothing contaminates the figure, glm-dsa's
residual is **0.0 MiB** at ctx 32768 and 131072 and at ub 1024 and 2048.

Getting there needed two corrections to the analysis. The sparse path
allocates **three masks and does not halve them** for MLA — measured at
exactly 6.00 half-width units, consistently, which is 3 full-width. An earlier
pass charged three *half-width* masks and produced a 3.38x multiple that
looked like a fork law.

## Growth is allocation on first use, not a leak — for most models

Per-turn checkpoints separate the two directly:

| model | shape |
|---|---|
| gemma3-27b | 352 -> 1521 MiB over three turns, then flat (+0/+1 for turns 4-10) |
| qwen3-4b, `-cram 8192` | sawtooth, 251 -> 2212 MiB net over 40 turns |
| qwen3-4b, `-cram 0` | 240 -> 287 MiB over *more* tokens |
| gemma-4-31B-QAT | still climbing at turn 10, decelerating; **under investigation** |

The cram pair is the clean result: with the cache enabled the process grew
1.96 GiB over an agent session; with it disabled and more tokens generated it
grew 47 MiB. The prompt cache does fill with use. That is only visible because
the growth driver alternates distinct conversations — a single growing one
shares its prefix and never evicts, so it would have measured the two
identically.

`CONTRIBUTING.md` says the prompt cache is deliberately excluded from the
rolling correction's base because it allocates nothing at load. That is true
at load and false thereafter, which is worth knowing when the correction reads
a host observation taken from a server that has been working.

## What the validation test found

`ananke/tests/estimator_matches_measurements.rs` replays the dataset through
the estimator. It caught three things a hand-written analysis had missed.

**The layer-split mask multiple does not apply to hybrids.** Every hybrid
measures 1.00 — deepseek4, laguna, qwen35moe — against 4.00 on every
fully-resident model. The discriminator is placement, not mixture-of-experts:
gemma-4-26B-A4B is a resident MoE and measures 4.00. The estimator was
charging 4x to hybrids and over-predicting their arena by ~384 MiB until this
test failed on it.

**CORRECTED — it is tensor split, not hybrids.** This entry first read
"mainline hybrids carry a CPU-resident MoE term", which was the wrong
attribution: it came from a sweep whose deduplication key dropped `gpus` and
`split`, so one cell stood in for a group that was not uniform. Breaking the
same data down by placement:

| placement | excess over mask + hidden |
|---|---|
| layer split, one or two cards | **0.02 MiB** |
| tensor split | **56.5 MiB** (qwen35moe), **94.0 MiB** (laguna) |

So a mainline hybrid under *layer* split is modelled essentially exactly, and
the shortfall belongs to **hybrid plus tensor split**, where it is flat across
ctx 8192-65536. Per unit of hidden size it comes to 56.5 and 62.7 bytes per
batch token on the two models — close, but two models is not a law.

Whether it scales with batch is unmeasured: the campaign ran no tensor-split
hybrid above ub 512. Cells for that are queued. Until then it is an
under-prediction of 56-94 MiB on a configuration the operator does run, which
is the direction that OOMs, so it is worth closing rather than noting.

**The reachability claim is not currently true.** Comparing owned host memory
against the modelled overhead on fully-resident models: ratios span 0.75-2.80
with a median of 1.11, and **44 of 301 cells sit outside the rolling
correction's [0.8, 1.5]**, mostly under-predicted — the direction that OOMs.
The constants marked `reachable` in `tuning.json` are chosen to make that band
hold, and for about one cell in seven they do not. The test records the count
as a ceiling that may only fall; closing the gap needs the host baseline
refitted, not the threshold raised.

Arena accuracy by architecture, worst case across the dataset:

| architecture | worst error |
|---|---|
| lfm2 | 0.05 MiB |
| llama, qwen3, talkie | 0.28 MiB |
| glm-dsa | 0.50 MiB |
| gemma3 | 0.54 MiB |
| qwen35 | 0.86 MiB |
| qwen35moe | 1.93 MiB |

All eight are under two mebibytes and five under half of one. That is what
"the arena is arithmetic, not a fit" means once every term it needs is
present — and getting there took three terms found after the campaign
finished: the quantised cache, the shared-cache window masks, and keying ik's
MoE rate on the device count.

## The process baseline is not predictable from model shape

Measured baseline — owned host memory less the arena and the per-device cost —
for every fully-resident model:

| model | layers | n_embd | measured base |
|---|---|---|---|
| LFM2.5-Embedding-350M | 16 | 1024 | 203 MiB |
| gemma-4-26B-A4B | 30 | 2816 | 348 MiB |
| Qwen3-4B | 36 | 2560 | 246 MiB |
| talkie-13B | 40 | 5120 | 209 MiB |
| Magidonia-24B | 40 | 5120 | 261 MiB |
| gemma-4-E4B | 42 | 2560 | 363 MiB |
| gemma-4-31B-QAT | 60 | 5376 | 367 MiB |
| gemma-3-27b | 62 | 5376 | 264 MiB |
| **Qwen3.6-27B** | 65 | 5120 | **600 MiB** |

Two models of nearly identical shape — gemma-3-27b at 62 layers and 5376
hidden, Qwen3.6-27B at 65 and 5120 — differ by **2.3x**. Two models of
identical shape, talkie and Magidonia at 40 layers and 5120 hidden, differ by
25%.

A least-squares fit over the nine gives an intercept of 84 MiB, 9.3 MiB per
layer, **minus** 48 KiB per unit of hidden size, and a 152 MiB residual. The
negative width coefficient is not physical; it is what a fit does when its
regressors correlate at 0.74 and the thing being fitted does not depend on
them. Adding a width term makes the model worse, not better.

A grid search for the constants that put the most cells inside the correction
band lands at 165 MiB base and 2.2 MiB per layer, leaving 33 of 297 outside
against the current 36. That is not enough of an improvement to justify moving
three constants onto a fit this unstable, so they stay.

**What is actually true**: 44 of 301 cells sit outside `[0.8, 1.5]`, 25 of
them one model — Qwen3.6-27B, under-predicted by up to 2.97x. Its 763 MiB of
anonymous memory has no CPU-resident weights behind it (`cpu_model_mib` is 0)
and its vocabulary is *smaller* than gemma-3-27b's, so neither explains the
gap. It carries an embedded MTP head, which is the obvious suspect and is not
confirmed.

**Why this is recorded rather than fixed**: the rolling correction clamps at
1.5, so a service needing 2.97x stays under-reserved whatever it observes. For
*host* memory on a 251 GiB machine that has no practical consequence, which is
why it has never been noticed. It would matter on a smaller host, and it would
matter a great deal if the same were true of VRAM. Closing it properly needs
a per-architecture baseline term, which is a change to the model's shape and
not a retune of its constants.

## The two remaining arena ceilings, located

**gemma3, 12.08 MiB — RESOLVED.** It appears only when several slots *share*
one cache, and is flat across context, which is the signature of whole extra
masks. Three window masks fit exactly where one was modelled. gemma4 shows the
same three in the same configuration.

An earlier attempt gated this on slot count alone and was contradicted by a
hardware sweep measuring gemma-4-31B-QAT at four slots with a *per-slot* cache
and one mask. That was read as gemma3 and gemma4 disagreeing and written up as
an unresolvable contradiction — wrongly. They are different architectures, so
one could not have been evidence about the other in the first place; and in
fact they agree, once the condition is the shared cache rather than the slot
count. gemma3's worst arena error falls from 12.08 MiB to **0.70**.

**glm-dsa, 44.99 MiB — RESOLVED, and it was conflated.** The rate measured
28.0 per unit of hidden size on one card against 42.8 on two. That was blamed
on the expert offload differing (96 against 92), then on card count, then
dismissed by pointing at qwen35moe — whose rate does not vary with cards, and
which is a different architecture and therefore no evidence either way.

Checking properly: both offload values exceed glm's 79 layers, and both load
logs report `offloaded 80/80 layers to GPU`, so the placement is identical and
the card count is the only difference. Two further cells settle the shape.
At **ub 1024, above the MoE offload threshold where the term vanishes
entirely, one card and two cards measure 216.01 MiB and both match the model
exactly** — so the card dependence lives inside the MoE term rather than
beside it. And the gap scales with tokens: 22.25 MiB at ub 256 against 44.49
at ub 512, exactly double.

The rate is therefore keyed on architecture *and* device count. qwen35moe is
41 on both, glm-dsa 28 and 43, laguna 36 and 54 — so the card dependence is
real on two of three, and the one that lacks it would have been the wrong
model to generalise from. glm-dsa's worst arena error falls from **44.99 MiB
to 0.50**.

## A stale constant in the analysis, and what it cost

`analyse.py` went on using the flat 81 KiB per token for ik's MoE term after
the campaign had shown it scales with hidden size and the estimator had been
changed to match. Every residual this file computed for an ik mixture of
experts was inflated by the difference.

Correcting it moved glm-dsa's two-card fit from a 3.38x multiple with a 136
MiB residual to **1.00 with -0.5**, and turned what looked like an unexplained
fork behaviour into an exact one. The lesson is narrow and worth stating: the
analysis and the estimator now share `tuning.json` as their source of truth,
but only the Rust side reads it mechanically. The Python side copies two
values by hand, and those copies are the one place a drift can still hide.

## The GPU curves under-reserved, in two ways that OOM

Validating `compute_buffer::default_for` against llama.cpp's own per-device
figure found 12 of 320 cells reserving *less* than the runtime took.

**Flash attention off materialises the score matrix.** With it on, the scores
are consumed tile by tile and never exist whole; without it the graph holds
`n_head x ctx x n_tokens` f32 entries. The curves modelled none of this:
qwen3-4b at ub 2048 reserved 956 MiB against **9464 taken**, a tenfold
shortfall. Normalised by `n_head x ctx x n_tokens x 4` the measurement is
1.07-1.88 across nine architectures, so a factor of 2 covers all of them.
deepseek4 sits at 0.49 because MLA shares a latent across heads.

**The context term scales with batch on every architecture, not just
deepseek4.** `CONTRIBUTING.md` said most compute buffers are "effectively
independent of `--ubatch-size`", which is close enough at the calibration
batch and wrong away from it: Magidonia measures 388 MiB at ub 512 and 1552 at
2048, Qwen3.6-27B 290 and 1160 — exactly fourfold in both cases. Every curve
now scales its slope with batch, which is a no-op at the default 512.

Together these take the under-reserving cells from 12 to 1, and raising
deepseek4's base by 100 MiB — the safe direction, and not a resolution of its
review — takes it to none.

**What is left is over-reservation, and it is large.** Worst headroom per
architecture now runs from 3.8x on llama to **27.5x on laguna**. That does not
OOM; it refuses a model room it could have used, which on a hybrid means fewer
expert layers on the GPU and a slower service. The bases were set to cover the
worst case of a family and the compute column is often far below them. These
are recorded per architecture as ratchets, and they are the clearest remaining
argument for fitting the curves to this dataset rather than carrying inherited
numbers.

## The baseline's residual is structure, not noise

The process baseline was reported as a per-model constant with a wide spread.
That reading came from taking a median over each model's cells, which
collapsed exactly the variance that mattered. Qwen3.6-27B's cells, broken out:

| factor | effect on the measured baseline |
|---|---|
| `served` against idle | **+21 to +238 MiB** — allocation on first request |
| `parallel` 1 -> 2 -> 4 | 590 -> 754 -> 1082 MiB on this model |
| tensor against layer split | +180 MiB |

Its cells span 278-1082 MiB with a standard deviation of 183. The "600 MiB
baseline" was a median across configurations that differ by a factor of four.

Neither effect is uniform enough to model as it stands. First-use allocation
runs from -2 MiB on talkie to +238 on Qwen3.6-27B; the per-slot cost runs from
-19 to +48 across models. But they are *structure*: the model has no term for
either, so the residual gets attributed to a per-model constant that then has
to be wide enough to absorb them.

Qwen3.6-27B remains the largest outlier on both axes, and is the only
non-hybrid model with an embedded MTP head — one point, so a term for it would
be fitted to itself.

## Over-reservation was mostly a measurement artifact

Reported at up to 27x. The comparison was wrong in two ways: it took
llama.cpp's `compute` column alone, omitting the `unaccounted` remainder a
reservation must also cover, and it read that column from whichever device
came first — which under tensor split is the fused `Meta()`, whose figures are
not a per-device quantity at all.

Comparing like with like, the real headroom is **1.9x to 3.2x**, which is
about the 60% margin plus the batch-scaling constant the curve's form cannot
express. The hybrids are no worse than anything else, which the artifact had
also obscured — laguna and qwen35moe read as the two worst offenders at 20 and
28x and are in fact 2.0 and 2.1.

Counting the remainder also exposed two genuine under-reservations that the
compute column alone had hidden, on Magidonia at ub 2048 and on deepseek4 at
ctx 8192. Both are closed.

## The derivers now refuse to average over a disagreement

Ten conclusions in this campaign were drawn by pooling across a factor that
turned out to matter, and every one produced a plausible law. The common
mechanism is that a median says nothing about the spread behind it: the median
of a bimodal set describes none of its members and looks exactly like the
median of a tight one.

`analyse.py` now reduces measurements through `consensus`, which raises rather
than returns when the values span more than 15% of their median. A wide spread
is treated as *a failure to have grouped properly*, not as noise to average
over.

Turning it on found four more instances immediately, all in derivers written
earlier in this same campaign:

| constant | what it was pooling | spread |
|---|---|---|
| layer-split mask multiple | hybrids, which do not replicate | 1.0 to 5.3 |
| mainline tensor MoE rate | MTP cells, which lack the term | 0.02 to 62.8 |
| gemma E-variant term | cells whose residual is neither | 103 to 1278 |
| ik MoE rate | three architectures at once | 28 to 54 |

Fixing those exposed two more. The 5.27 outlier in the mask multiple was
gemma3 at four slots with a shared cache — the three-window-mask rule that had
been added to the estimator and never to this file, the same drift that had
left the ik MoE rate stale here. And the E-variant residual turned out to
depend on the **cache type**: 1025-1028 at f16 against 1087-1278 at q8_0, so
the arena model is missing a term that varies with it, and pooling the two
attributed that term to the E-variant.

Two disagreements are carried deliberately, with the reason recorded at the
override: glm-dsa and laguna both measure a different ik MoE rate on one card
than on two, and the larger is taken because over-reserving does not OOM.

## RESOLVED: a quantised KV cache costs pinned host memory

The consensus guard surfaced this: the gemma E-variant term measured 1025
bytes per layer per token at f16 and 1278 at q8_0, so something in the arena
depended on the cache type and was being charged to the E-variant instead.

Isolating it — pairs differing in nothing but `cache_type_*` — the arena is
larger with a quantised cache in **all 117 pairs**, always positive, scaling
exactly with batch (1.28 MiB at ub 512 against 5.12 at 2048 on one model).

The rate is per architecture and spans a factor of forty:

| architecture | bytes per batch token |
|---|---|
| lfm2 | 61 |
| llama, qwen3, talkie | 164 |
| gemma3, laguna | 328 |
| qwen35moe | 532 |
| qwen35 | 543 |
| gemma4 | 2621 |
| deepseek4 | 6144 |

Charging the worst to all of them cost about 3 MiB of over-prediction on every
quantised cell, so it is a table. The mechanism is not identified — the rate is
not predicted by head count, head width or layer count — but it is bounded,
measured, and now charged.

Two conflations were caught getting here, both by the guard. The rate appeared
to take two values per model until the layer-split replication was divided
out, and qwen35moe still showed 133 and 532 until hybrids were excluded from
that divisor, since a hybrid does not replicate.

Every architecture's worst arena error improved: gemma3 0.70 to **0.54**,
qwen35 1.13 to **0.86**, llama, qwen3 and talkie 0.36 to **0.28**.

The validation test now passes the cache type through. It had been leaving it
unset, which made the whole term unreachable from the check that exists to
catch exactly this.

## RESOLVED: a tensor split costs host baseline

Chasing Qwen3.6-27B's baseline turned up a general term instead. Every model
that ran under both split modes at matching context, batch, slots and cards
holds more host memory under tensor split:

| architecture | extra |
|---|---|
| qwen3 | 100 MiB |
| talkie | 108 |
| gemma3 | 127 |
| llama | 130 |
| gemma4 | 154 |
| qwen35 | 184 |

The estimator had no term for it, so every tensor-split service was
under-predicted by that much — and several of the operator's are tensor-split.
Charging it takes the out-of-band cells from 42 to **33** and the median ratio
from 1.16 to **1.06**.

## What first-use allocation is not

Serving a first request allocates host memory an idle process has not: -2 MiB
on talkie, +238 on Qwen3.6-27B, deterministic per model. Ruled out as causes,
each by measurement rather than argument:

- **The request itself.** Probes asking for 16, 64, 256 and 1024 tokens all
  produce the same figure — 433 MiB on gemma3 and 778 on Qwen3.6-27B, flat.
- **Vocabulary.** gemma3 has a larger one and allocates a quarter as much.
- **Tokenizer size.** Magidonia has *more* BPE merges than Qwen3.6-27B —
  269 443 against 247 587 — and allocates 268 MiB against 611.
- **Model size, layer count, architecture family.** gemma3 and Qwen3.6-27B are
  within 1% on layers and hidden size and differ fourfold.
- **CPU-resident weights and any logged buffer.** gemma3's CPU buffer is
  *larger*. Nothing in either load log accounts for the difference.

So it is untracked heap that varies per model for a reason not visible in the
GGUF, the log, or the request. The band test now judges against served cells,
since a reservation is made before the first request but has to cover the
state after it — judging against an idle process counts a required
over-prediction as an error.

### Charged as a per-architecture offset

Since the cause is not visible but the residual is reproducible, it is charged
rather than explained: `baseline_offset` in `tuning.json`, keyed on the
architecture and derived by `analyse.py` alongside everything else. Three
things had to be separated out of that derivation first, each surfaced by the
`consensus` guard refusing to reduce a group that did not agree:

- **Flash attention off is its own regime.** It costs 30 to 254 MiB of host
  residual beyond the arena term, and inconsistently — gemma3's scales with
  batch and qwen3's does not. Pooling it put lfm2's offset at both 35 and 169
  MiB. Excluded, and unmodelled: the estimator over-reserves there rather than
  guessing.
- **ik_llama does not share the baseline.** Its residuals run -264 to +120 MiB
  where mainline's run -0 to +24. Excluded.
- **One architecture string can cover several models.** `gemma4` is three:
  the mixture of experts is over-covered by 66 MiB, the dense model needs 17,
  and the E-variant 107. The key therefore carries a `+moe`/`+e` suffix — both
  are distinctions the estimator already reads, so it is a key it can build.

The deriver reduces by `max`, not median, and so deliberately does *not* call
`consensus`: a maximum bounds a spread rather than concealing it, and erring
high on a baseline is the safe direction. The spread is reported in the
evidence string instead, which is how the E-variant's remaining batch
dependence (92 MiB at ub 512, 170 at 2048) stays visible.

Out-of-band cells fall from 33 of 211 to 5. The five that remain are each a
regime the offset is derived without: two flash-attention-off cells, two
`parallel = 4` slot splits, and one gemma3 cell at 0.78.

## deepseek4's context slope was not real

The curve charged 1900 MiB plus 66 per 1k of context, scaling with batch, on
the premise that the NSA lightning indexer scores every query token against
the whole context. The sweep measures the primary device's compute buffer at
**1976 MiB at ctx 8192, 32768, 65536, and 131072** — identical to the
megabyte across a sixteenfold range — and 1976 / 1984 / 2001 across ubatch
512 / 1024 / 2048. `indexer.top_k` is 512, a fixed working set, so once the
context passes it the indexer scores against a fixed-size selection and the
buffer stops growing.

The premise was not invented, only misplaced. The `k x ubatch x ctx` term is
real on the *secondary* device — 202, 223, 279, 327 MiB at those four contexts,
scaling cleanly with batch (223 / 448 / 904 for ub 512 / 1024 / 2048) — but at
about 1 MiB per 1k, not 66. At ctx 131072 the old curve reserved 10348 MiB
against the 2396 the runtime took.

The hold-out that kept this out of the deriver argued the curve was covering a
9.3 GiB residual the compute column does not describe. Total measured GPU
footprint is 15496 MiB at ctx 8192 and 16425 at 131072 — a difference of 929
MiB, exactly the KV growth — so any such residual is flat in context, equally
present at 8192 where the curve reserved 2428 MiB. Whatever it is, a context
slope is not its remedy.

Fitted by the general deriver, the curve becomes 3842 + 1 per 1k: a *higher*
base than before at every context under 30k, and a sixth of the reservation at
131072.

## The per-model residual is a first-request step, and it is bounded

Attributing it rather than correlating against it settles what a long list of
ruled-out variables could not. `/proc/<pid>/smaps` puts the whole difference
between Qwen3.6-27B and gemma3-27B in `[heap]` (+171 MiB) and an unnamed
anonymous mapping (+150 MiB) — ordinary allocator memory, no CUDA or library
arena — and nothing in either load log accounts for it. Qwen3.6-27B's logged
compute buffers are in fact the *smaller* of the two.

Watching it over successive identical requests says what it is:

| model | `-cram 0` | `-cram 8192` |
|---|---|---|
| Qwen3.6-27B | 306 → 556, then flat | 306 → 555 → 856 → 856 → 856 → 1157 → 1458 |
| gemma-3-27B | 226 → 238, then flat | 226 → 238 → 252 → 252 → 266 → 279 |
| Magidonia-24B | 220 → 237, then flat | 220 → 237 → 241 → 245 → 248 → 252 |
| Qwen3-4B | 212 → 219, then flat | 212 → 219 → 223 → 226 → 229 → 233 |

Two separate effects, and only one of them is the residual:

- **The prompt cache grows with use and stops at its cap**, which is what
  `-cram` documents. The step size tracks the model's KV state — ~300 MiB for
  Qwen3.6-27B against ~14 for the sliding-window gemma3 — so it is a per-model
  quantity, but a *reserved* one. ananke passes `-cram` explicitly and the
  packer charges it as slop, which is the right treatment: at `-cram 0` the
  growth is gone entirely.
- **A one-time step on the first request**, present at `-cram 0` and flat
  forever after. This is the residual the baseline offset charges: +250 MiB for
  Qwen3.6-27B against +7 to +17 for the other three. It is bounded and it does
  not accumulate, so a constant is the right shape for it after all.

The campaign measures at `-cram 0` and every cell in the baseline group serves
exactly two requests, so neither effect is confounded with the derivation. The
offset stands.

The campaign's own `growth-cram0` and `growth-cram8192` cells corroborate it
on a third model and a different harness: Qwen3-4B holds 246, 247, … 249 MiB
across nine checkpoints at `-cram 0`, and climbs 257 → 359 → 459 → 541 → 739
at `-cram 8192` before falling back to 601 — a cache being evicted against its
cap, which is not a shape a leak has. That data was in the dataset before this
investigation; what was missing was the comparison.

### The step is on the prefill path, and it saturates

Varying prompt length and generation length independently against a fresh
server, Qwen3.6-27B at `-cram 0`:

| prompt tokens | 1 | 8 | 64 | 400 |
|---|---|---|---|---|
| step MiB | 11 | 250 | 274 | 273 |

Generation length does nothing at all — `n_predict` 8, 64, and 256 against a
one-token prompt all step 11 MiB. Only the prompt moves it, and it plateaus by
64 tokens. So it is scratch allocated the first time a *batched prefill* runs,
sized by something that saturates well below the 512-token ubatch, and a
one-token prompt never triggers it because that path decodes rather than
prefills.

This is why a constant is the right shape: the term is bounded, it saturates,
and every real service prefills. The harness probe sends a four-token prompt,
so every campaign cell crosses the threshold and the offset is measured at or
above saturation.

Two hypotheses for the mechanism are ruled out by measurement. It is not the
output logits buffer: the estimator predicts 242 MiB for Qwen3.6-27B and 256
for gemma-3-27B, and their steps are 273 and 12. It is not tied embeddings
routing the output head to the CPU backend: Qwen3.6-27B and Magidonia-24B both
carry a separate `output.weight` and step 273 and 17, while gemma-3-27B and
Qwen3-4B both tie and step 12 and 7. The mechanism is still unidentified; what
is established is its shape, its bound, and its trigger.

## Flash attention off is a per-token rate, not a baseline shift

It was recorded as a baseline effect — "30 to 254 MiB, inconsistently" — on a
dataset where seventeen of nineteen cells sat at one context and one batch. A
sweep across both axes shows it is neither inconsistent nor a baseline term:
the process baseline does not move at all (gemma-3-27B holds 289, 290, 289 MiB
of anonymous memory at ctx 8192, 32768, and 131072 with flash attention off),
and the whole effect is in the pinned arena.

The residual over the modelled arena is **flat in context and proportional to
batch tokens**. gemma-3-27B is 64 MiB over at all three contexts and 256 MiB
over at ubatch 2048 in each of them — 128 KiB per batch token either way. So
it is a per-token rate, the same shape as the quantised-cache rate, and it
lands on near-exact powers of two:

| | KiB per batch token |
|---|---|
| gemma4 (dense) | 256 |
| talkie | 160 |
| gemma3, gemma4+moe | 128 |
| gemma4+e | 106 |
| llama, qwen3, qwen35 | 32 |
| lfm2 | 4 |
| ik_llama, any architecture | 0 |

ik is excluded and charged nothing: its fa-off arena is already modelled to
within a megabyte, because it sizes masks against the whole cache and the
widened four-byte element is the entire story there.

The single 8 KiB constant this replaces was chosen as "a representative value
rather than a law", and its own evidence noted the non-uniformity (4.25x on
non-SWA models against 4.8-4.9x on sliding-window ones). That reading was
correct; what it lacked was the second axis that turns the non-uniformity into
a per-architecture rate.

## The design hole flash attention exposed, audited everywhere else

Flash-attention-off spent the first campaign recorded as an inconsistent
baseline shift because seventeen of its nineteen cells sat at one context and
one batch. A rule that is wrong in its *batch* dependence is invisible at one
batch size, so every other modelled regime was audited the same way — cells,
distinct contexts, distinct batches, and distinct (context, batch) pairs:

| regime | cells | ctx x ub points |
|---|---|---|
| quantised KV | 142 | 7 |
| tensor split | 142 | 11 |
| ik_llama | 95 | 8 |
| hybrid | 121 | 9 |
| `--kv-unified` | 14 | **5, all at one ubatch** |
| `parallel > 1` | 55 | **7, all at one ubatch** |
| `-rtr`, `--numa` | 1 each | 1 |

The two slot regimes have the same hole, and both feed rules that multiply
terms which scale with the batch: the stream division that sizes the KQ mask,
and the three window masks an interleaved-SWA model builds when slots share
one cache. The `slot-batch` phase measures them at a second batch size.

`-rtr` and `--numa` are single points and stay that way deliberately — neither
feeds a modelled constant. `RollingBase::host_peak` decides from the measured
`RssFile` rather than from either flag, precisely because their effect cannot
be predicted from the flags. `thp` is in the cell schema and has never been
varied at all.

## Open

- The single-card gemma4 sample is one distinct configuration; the baseline
  block fixes ctx at 32768 and the curve block runs two cards, so single-card
  context variation is thin for every model except qwen3-4b.
- Nothing here says anything about portability. Every row is one machine.
- The baseline offset is charged, not understood. A machine with a different
  CUDA runtime may well need different numbers, and nothing in the derivation
  would notice.
