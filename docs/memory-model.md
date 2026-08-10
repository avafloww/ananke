# The memory model

What ananke predicts a llama.cpp service will hold, and how each term was arrived at. The code is `crates/estimate/`; this is the model those functions implement.

The tuned constants themselves live in `crates/tuning/tuning.json`, each with its own evidence attached. [`calibration/README.md`](../calibration/README.md) is how to re-derive them.

## The compute buffer

Per-device graph memory, beyond weights and KV. One fitted linear model covers every architecture: `crates/estimate/src/compute_model.rs` evaluates it, its design columns dimensionally normalised so architectures of different width share coefficients, and `tuning.json`'s `compute_model` section carries one coefficient set per (runtime, split, architecture, variant). `compute_buffer::per_device_for` adds the head card's extra on top and the unfused-attention score matrix where flash attention is off.

Adding an architecture therefore does not mean writing a curve. Measure it into the dataset and re-run the campaign's `fit`; the coefficients follow. [`calibration/README.md`](../calibration/README.md) is that loop. An architecture nobody has measured falls to the pooled default, which is what the `coverage` gate is for.

Two things worth knowing before measuring one:

- Read the residual as `used - gpu_weights - kv_total`, where `gpu_weights` excludes the token-embedding table — llama.cpp keeps it on the CPU. A residual that trends with context means `kv_per_token` is wrong, not the compute model.
- Most architectures' compute buffers are near-independent of `--ubatch-size`, but a minority scale with it: a sparse-attention indexer scores every one of the `ubatch` query tokens against the whole context, so its residual goes as `ubatch x context`. Sweep a second ubatch before trusting an extrapolation.

## Host-side memory

The curve above is VRAM. The host side is modelled separately in `crates/estimate/src/host_buffer.rs`, and the two are not interchangeable: the `Cpu` slot must never be charged `Estimate::buffers.compute_mb`, which describes a device and was never measured against a host backend.

What llama-server actually holds in host RAM, beyond weights and the CPU's KV share:

- **The pinned graph arena.** ggml pins every graph *input* tensor to the CPU backend, and when a GPU is present that backend's buffer type is swapped for the device's host buffer type — `cudaMallocHost`, page-locked and unswappable. llama.cpp logs it as `CUDA_Host compute buffer size`, not `CPU`. Three measured components, all scaling with the batch: the KQ mask at `n_kv × min(context, ubatch) × (fa ? 2 : 4)`, a second window-sized mask on an interleaved-SWA model (sized `n_swa + n_tokens`, not the window alone), and two `n_embd × n_tokens` f32 hidden-state inputs. That last term is easy to miss and *dominates at short contexts* — the token embeddings stay on the CPU backend, so the embedding lookup and the split-boundary copy both land here.
- **The process baseline.** CUDA runtime host allocations, tokenizer, sampler state, graph metadata. Measured at `112 MiB + 3.4 MiB × n_layer` against *serving* processes (an idle one reads ~26 MiB lower) across three models; the layer count predicts it better than the hidden size does.
- **The prompt cache.** `-cram`, default 8192 MiB of serialized evicted prompts. ananke passes the flag explicitly so the reservation and the runtime's cap are the same number.

The two runtimes size the arena by **different rules**, so calibrate each:

| | mainline | ik_llama |
|---|---|---|
| mask width | `ctx / parallel`, padded | `ctx` — `-np` does not divide it |
| SWA second mask | window-sized (`n_swa + n_tokens`) | full context |
| MLA | half-width mask (`deepseek4`; `deepseek2` unmeasured) | — |
| sparse-attention indexer | one extra mask (`glm-dsa`) | two extra, keyed on `-dsa` |
| hidden-state buffers | two `n_embd x n_tokens` f32 | one |
| extra GPU, layer-split | +24 MiB | none |
| CPU-resident MoE ops | — | 81-161 KiB/token, only below `n_tokens x n_used >= 32 x n_expert` |

Same model and flags at ctx 32768 / ub 512: mainline 18 MiB, ik 37 MiB.

Calibrate along **two** axes, not one. The obvious sweep is context × batch; the one that is easy to forget is **how much of the model is on the GPU**, and a `-ngl 99`-only sweep measures the regime where the host side matters least. The arena turns out to be offload-independent (18.01 MiB at `-ngl 99`, `-ngl 18`, and `-ngl 0` alike) and the host side grows through the CPU's KV share instead — but that is a finding, not an assumption to start from. A service with no GPU visible at all is different again: nothing is pinned, and the CPU compute buffer swells to hold the op intermediates a GPU run offloads to the device (measured 88 MiB against 18).

Sweep *real* hybrids, not a small model pushed to the CPU with `-ngl`. Expert offload (`--n-cpu-moe`) is a different mechanism, and a mixture of experts has a much larger process baseline than its layer count suggests — a 41-layer MoE was measured holding more than a 65-layer dense model, which is why the baseline carries a flat MoE allowance. Note that Laguna and other ik_llama-only architectures cannot be measured with a mainline build at all; keep the fork constant across a sweep or the fork's own differences will be read as model differences.

**Measure both forks before trusting a host model.** Qwen3.6-35B-A3B under `--n-cpu-moe 40`, one GPU, no `--no-mmap` on either:

| runtime | load log | owned (anon+shmem) | mapped (`RssFile`) |
|---|---|---|---|
| mainline | `CPU_Mapped model buffer size = 24771` | 465 MiB | 24935 MiB |
| ik_llama | `CPU buffer size` | 23759 MiB | 164 MiB |

Nothing in the configuration distinguishes those runs, yet the same bytes land in different counters. The divergence is specific to the expert-offload path — at `-ngl 0` the same ik build maps normally, and honours `--no-mmap` when given it — so it cannot be predicted from flags at all. That is why `RollingBase::host_peak` decides from the measured `RssFile` rather than from `mmap`/`-rtr`: inferring from flags takes ik's 62.3 GiB owned against a 6.6 GiB base, a ratio of 9 clamped to 1.5, and over-reserves a host slot tens of GiB wide by half. Mainline's `RssFile` came to the host weight total plus ~150-230 MiB of shared libraries in every configuration tried, which is what makes it a usable discriminator.

**Aim for reachability, not closeness.** Where a term cannot be predicted well — the process baseline varies 400-546 MiB across three MoEs that all have 256 experts, with layer count running the wrong way — pick the constant so that every known model lands inside the rolling correction's `[0.8, 1.5]` clamp of reality. That band is the whole distance the correction can travel, so a "closer" constant that pushes one model outside it is strictly worse: no amount of observation can bring that service back.

The arena's *shape* is read from llama.cpp's graph construction; the baseline is a fit. Calibrate the baseline against a process that has **served a request** — measuring at idle understates it by ~26 MiB of first-use scratch. Recent builds also print a `memory breakdown` table splitting each device into model / context / compute, which is easier to read than the individual buffer lines. To re-check either:

```bash
# One run per (context, ubatch) point. Read the arena from the load log —
# note -lv 5, without which recent builds omit the buffer lines entirely:
llama-server -m <model> -c <ctx> -ub <ub> -ngl 99 -fa on -lv 5 2>&1 \
  | grep "CUDA_Host compute buffer size"
# and the host footprint from /proc once it is serving:
grep -E "^(RssAnon|RssShmem|RssFile|VmRSS):" /proc/<pid>/status
```

**Read `RssAnon + RssShmem`, not `RssAnon`.** `cudaMallocHost` is accounted as *shmem*: growing the arena from 18 MiB to 72 MiB moves `RssShmem` by exactly that and leaves `RssAnon` flat. And **not `VmRSS`** — `RssFile` is the mapped GGUF, which llama.cpp maps with `MAP_POPULATE` and then unmaps only outside the host-resident tensor span, so a hybrid run leaves nearly the whole file resident as clean, reclaimable pages. That is what `crates/supervise/src/rolling.rs` compares against a weights-excluded base.

Two gotchas. `ik_llama`'s `-rtr` forces `--no-mmap`, so a repacked model's weights are anonymous and *do* appear in the owned figure. And the prompt cache allocates nothing at load — `-cram 0` and `-cram 4096` measure identically on a fresh server — so it is reserved but deliberately kept out of the rolling correction's base, or every observation would read as a large over-reservation.

## Multi-token prediction (MTP / NextN) overhead

When a service sets `spec_type = "draft-mtp"`, llama.cpp enables multi-token-prediction speculative decoding. For models that ship an embedded MTP head (`{arch}.nextn_predict_layers > 0` — e.g. Qwen 3.6's `qwen35` and `qwen35moe`), this needs *no separate draft model*: llama.cpp creates a second context against the same target model whose KV cache covers only the trailing `nextn_predict_layers` block(s) — the dense-attention MTP head — using the draft cache types (f16 by default, independent of `--cache-type-*`). No extra weights load, because the nextn-layer tensors are resident regardless.

`crates/estimate/src/mtp.rs` models this as `nextn × head_count_kv × (key_length + value_length) × 2 (f16) × context` for the KV term, plus a roughly constant `MTP_COMPUTE_MIB` compute buffer. The estimator computes it once in `estimate_with_summary` (architecture-independent — it reads the metadata directly), stores it on `Estimate::mtp.bytes`, and the packer reserves it as a single lump on the primary GPU (`Packer::seed_mtp_overhead`). The compute constant is calibrated against llama.cpp's own `[spec] estimated memory usage of MTP context is N MiB` log line; re-derive it the same way (run `llama-server … --spec-type draft-mtp`, read the figure, subtract the modelled KV) if a new MTP arch lands with a materially different curve.

Some families ship the MTP head as a **separate draft GGUF** instead of embedding it (e.g. Gemma 4's `gemma4-assistant`, a 4-block model loaded via `-md`). Set `draft_model = "…/mtp-head.gguf"` alongside `spec_type = "draft-mtp"`; the validator requires `spec_type` whenever `draft_model` is set, and the estimator reads the draft file in `estimate_with_summary` and passes its summary to `mtp_overhead_bytes`. The draft's attention layers *share the target model's KV cache* (the load log shows `llama_kv_cache: layer 3: sharing with layer 59`), so there is no context-scaling KV term — the overhead is just the draft's GPU-resident weights (everything but the CPU-side `token_embd.weight`) plus a small `DRAFT_MODEL_COMPUTE_MIB` buffer. That constant is calibrated against the production 2×3090 Gemma 4 run: the estimator landed within ~10 MiB of the measured 40858 MiB peak. Because the cache keys on the `model` and `mmproj` paths but not the draft path, `draft_model` is folded into `EstimatorInputs::config_fingerprint` so swapping the draft GGUF invalidates a stale estimate.

The above is mainline's MTP. The estimator does not model ik_llama's: it sets speculation only from `spec_type = "draft-mtp"`, and ik takes `mtp:n_max=4,p_min=0.5`, so an ik service running MTP is reserved as though it were not. One ik-specific behaviour is modelled — `drop_mtp_head_blocks` subtracts an embedded head's trailing blocks, which ik does not load. ik added separate-draft MTP for DeepSeek 4 in [ik_llama#2216](https://github.com/ikawrakow/ik_llama.cpp/pull/2216) on 2026-08-01; measure one before running it.

MTP composes with `parallel > 1` and `mmproj` — both are supported by current llama.cpp, including image inference, so there is deliberately no validator rejection of those combinations. Note that `parallel > 1` with a non-unified KV splits the `-c` budget across slots, so each request's effective context is `context / parallel`; raise `context` if every slot needs the full window.
