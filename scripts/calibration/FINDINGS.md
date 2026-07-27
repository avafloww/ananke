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

## mainline allocates the mask four times on two cards; ik does not

| runtime | cards | flash attn | mask multiple |
|---|---|---|---|
| mainline | 1 | on | 1.00 |
| mainline | 2 | on | **4.00** |
| mainline | 2 | off | **4.25** |
| ik | 1 or 2 | on/off | 1.00 |

Flat across context (8192-65536), batch (512-2048), slots (1 and 4), and
unified vs per-slot cache. ik is unaffected at either card count, so this is a
mainline property, not a consequence of having two GPUs.

**This contradicts a constant currently in the tree.** `CONTRIBUTING.md`
records "extra GPU, layer-split: +24 MiB" — calibrated at ctx 8192, where
`3 x 8 MiB` happens to equal 24 MiB. It is not a constant; it is a multiplier
on a term that scales with `n_kv x n_tokens`:

| context | true excess | flat 24 MiB says |
|---|---|---|
| 8192 | 24 MiB | 24 MiB |
| 32768 | 96 MiB | 24 MiB |
| 65536 | 192 MiB | 24 MiB |
| 32768 @ ub 2048 | 384 MiB | 24 MiB |

At production contexts a two-card mainline service is under-reserved by more
than a gigabyte of pinned host memory. This is the failure mode the campaign
exists to catch — a term fitted at one point and used everywhere — and it sat
in the part of the model treated as *derived* rather than fitted, where nobody
was checking it.

The `fa=off` excess is larger than 4x and **not uniform** — 4.25 on qwen3,
4.94 on gemma3, 4.80 on gemma4 — so it is not a single extra buffer. It is
measured on only one or two cells per architecture so far; the interior
fa-off points added at ub 2048 are what will give it a shape.

**Unresolved**: whether the multiple is 4 because there are two cards, or for
some other reason. The `device-scaling` cells pin placement and vary only the
count of visible CUDA devices, which is what separates those. Do not act on
this until they and the remaining eleven models agree.

## gemma4 carries a batch-scaling term the model lacks

Charging the mainline law leaves gemma-4-E4B with a residual that is flat in
context and grows with batch:

| ctx | ubatch | residual |
|---|---|---|
| 8192 | 512 | 25.09 MiB |
| 32768 | 512 | 25.09 MiB |
| 65536 | 512 | 25.09 MiB |
| 32768 | 2048 | 124.34 MiB |

Context-independence rules out a mask — including the interleaved-SWA second
mask, which is modelled and, at `n_swa = 512`, is far too small to account for
this. A term flat in context and rising with batch points at a per-token input
buffer.

**Hypothesis**: the E-variant's per-layer embedding input, which would scale
as `n_layer x d x n_tokens`. At 42 layers and 512 tokens, ~25 MiB implies
`d x 4 bytes` of about 1.2 KiB per layer per token.

**One prediction has now held.** gemma3 is interleaved-SWA but not an
E-variant, and it comes out exact on one card. So the gemma4 residual is not
the SWA term — that term is confirmed correct — and the E-variant explanation
survives its first test.

**Still to test**: gemma-4-26B-A4B (30 layers) and gemma-4-31B-QAT (60
layers). Both are gemma4 and neither is an E-variant, so if the residual is
E-specific they should come out at 1.00; if it tracks their layer counts
instead, the term is per-layer and general to gemma4. Either result is
informative. Both are still to run.

## Open

- The single-card gemma4 sample is one distinct configuration; the baseline
  block fixes ctx at 32768 and the curve block runs two cards, so single-card
  context variation is thin for every model except qwen3-4b.
- Nothing here says anything about portability. Every row is one machine.
