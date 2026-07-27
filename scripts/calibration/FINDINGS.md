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
| lfm2, llama, qwen3, talkie | < 0.5 MiB |
| qwen35 | 1.1 MiB |
| qwen35moe | 2.5 MiB |
| gemma3 | 12.1 MiB |
| glm-dsa | 45.0 MiB |

The first five are the sense in which "the arena is arithmetic, not a fit"
holds. The last two are open.

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

**glm-dsa, 44.99 MiB — narrowed to one architecture's card-count effect.**
The ik CPU-MoE rate is now per architecture, because the three ik mixtures
measured differ by a third: qwen35moe 41, glm-dsa 43, laguna 54. A single
constant either under-reserved laguna or over-reserved the other two, and the
median that was there did both.

What remains is *within* glm-dsa: 28.0 per unit of hidden size on one card
against 42.8 on two, each figure exact across every context and batch measured.
Both its expert-offload values exceed its layer count, so the placement is
identical and the card count is the only difference. That was previously
dismissed by pointing at qwen35moe, whose rate does not vary with card count —
but qwen35moe is a different architecture and says nothing about this one. Two
cells on the one-card side, so it is measured rather than explained, and it
over-predicts, which costs capacity rather than risking a load.

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

## Open

- The single-card gemma4 sample is one distinct configuration; the baseline
  block fixes ctx at 32768 and the curve block runs two cards, so single-card
  context variation is thin for every model except qwen3-4b.
- Nothing here says anything about portability. Every row is one machine.
