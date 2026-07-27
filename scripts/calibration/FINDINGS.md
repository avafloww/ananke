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

**Unresolved**: whether the multiple is 4 *because* there are two cards under
layer split, or a fixed replication factor. The `device-scaling` cells, which
pin placement and vary only the count of visible CUDA devices, are what
separate those, and they have not run yet. A four-card operator inherits
whichever answer is right.

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

## Open

- The single-card gemma4 sample is one distinct configuration; the baseline
  block fixes ctx at 32768 and the curve block runs two cards, so single-card
  context variation is thin for every model except qwen3-4b.
- Nothing here says anything about portability. Every row is one machine.
