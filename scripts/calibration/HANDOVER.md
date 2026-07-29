# Estimation parity handover

Status as of 2026-07-30. This document records where the ±5% estimation campaign
stands, what is proven about llama.cpp's allocation semantics, and every known
remaining gap between the estimator and production, with the derivation recipe
for each fix. It exists so the work can be picked up without re-deriving any of
the analysis below.

## Scoreboard (production nvidia-smi totals vs packed estimate)

| Model | Est GPU | Prod GPU | Drift | Notes |
|---|---|---|---|---|
| qwen3.6-35b-a3b | 16009M | 15080M | **+6.2%** | only outlier; decomposition below |
| qwen3.6-27b | 38069M | 39072M | -2.6% | |
| gemma-4-31b-it-qat | 43711M | 43594M | +0.3% | see §5 — passes largely by luck |
| deepseek-v4-flash | 17620M | 17182M | +2.5% | |
| glm-5.2 | 36788M | 38708M | -5.0% | at edge |
| laguna-s-2.1-iq4-nl | 38557M | 39570M | -2.6% | |
| talkie-1930-13b-it | 13203M | 12798M | +3.2% | |

Reproduce: `python3 scripts/calibration/dump_estimates.py --json` and compare
`gpu:0 + gpu:1` against `rss.gpu_used_mib` of the `prod-*` rows in
`scripts/calibration/data/measurements.ndjson`.

**Important:** the current scoreboard's defining feature is that several of the
passing models pass by *coincidental compensation* — an over-predicted term in
one place masking an under-predicted term elsewhere. Fixing any single term
honestly can regress the total; this is expected and documented below. Do not
"revert to the working number" — land the whole bundle for a model instead.

## Committed so far

- `e5c5d23` — gemma4 KV fix: when `kv_unified=true` and `parallel>1`, llama.cpp
  does **not** unify the SWA cache; each slot gets its own window-sized cache.
  `compute_kv_per_token` now multiplies SWA window cells by `parallel` under
  kv_unified. Verified against `prod-gemma4-31b-qat` log:
  `llama_kv_cache_iswa: creating SWA KV cache, size = 4608 cells` =
  4 slots × padded window 1024 (unified non-SWA cache = 240128 cells).
  Took gemma4-qat from -5.3% to +0.3%.

## Uncommitted (in working tree, verified mechanically but regresses 35b alone)

1. `ananke/src/allocator/placement/sharded/mod.rs` —
   `sharded_expert_offload` now honours `OffloadMode::Layers(n)` as
   `n.min(total)` instead of offloading all expert layers regardless of `n`.
2. `ananke/src/estimator/mtp.rs` — embedded-MTP overhead no longer multiplies
   KV by slot count (llama.cpp creates **one** unified draft context), and the
   KV slope is computed from GGUF metadata (`nextn × head_count_kv ×
   (key_length+value_length) × 2 f16`) instead of the global-max
   `MTP_KV_SLOPE_BYTES_PER_TOKEN=4471` constant (which was fitted on qwen35's 4
   KV heads and over-charges qwen35moe's 2 KV heads by ~2×).

Each change is independently verified against production evidence (below), but
the 35b total moves +6.2% → -13.8% because the old errors were cancelling.
They must land with the RS/CB fixes in §6–§7.

## Established allocation semantics (verified from production logs)

Do not re-litigate these; they are grounded in measured logs under
`scripts/calibration/data/logs/`.

1. **`Meta()` breakdown columns report ONE GPU's share** of the tensor-split
   (sharded) allocation. Physical total = column × n_gpus. Verified on
   gemma4-31b-qat (2-GPU tensor, no offload): `model_mib` 8616 ≈ 16471/2
   (1-GPU measurement of the same model); and on prod-qwen36-35b-a3b where
   `kv_mib` 2971 = (K 2720 + RS 251) and 2720 = 5440/2.
2. **`--n-cpu-moe N` offloads expert tensors of N layers; attention stays on
   GPU for all layers.** Verified: prod-qwen36-35b-a3b physical GPU model
   2×1431 = 2862 MiB ≈ attention-all-41-layers (1837) + 1 layer's experts
   (568) + output head (242) + nextn tensors (~215). `CPU_Mapped model buffer
   size = 24771.72 MiB` is the 40 offloaded layers' experts. The old comment in
   `sharded/mod.rs` claimed tensor split moves ALL expert tensors to CPU
   regardless of N — that is wrong for manual `n_cpu_moe`.
3. **Hybrid SSM state (RS) scales by streams and by spec copies.** RS =
   `(n_max+1) × n_seq × rs_per_seq`. The `llama_memory_recurrent` size line
   carries `( N cells, L layers, S seqs R rs_seq)`. Without MTP (`0 rs_seq`):
   qwen35 = 149.62 MiB/seq, qwen35moe = 62.81 MiB/seq (mtpslot-none RS buffer
   lines). With `--spec-type draft-mtp` (`3 rs_seq`, n_max=3): production
   27b RS = 1197 MiB = 149.62 × 2 seqs × 4 ✓; 35b RS = 502.5 MiB =
   62.81 × 2 × 4 ✓. Nothing else explains the 4× — it is the spec-verify
   depth copies (llama spec log: `n_max=3`).
4. **Embedded MTP is one unified context.** Its GPU cost = KV
   (`nextn × kv_heads × (kl+vl) × f16 × ctx`, sharded) + its own compute
   buffer (sharded) — qwen35moe prod: KV 512×2 + compute 296×2 = 1616 MiB;
   llama.cpp's own line says `[spec] estimated memory usage of MTP context is
   1320.02 MiB` (= share KV 512 + share compute 296 + host 512? close to
   1024+296: the log prints one share). It is NOT per-slot KV.
5. **Tensor-split compute also shards.** Per-GPU share is in the clean
   `compute_mib` breakdown column (only `unaccounted_mib` is polluted by the
   Meta-fused report). Clean per-arch data points (compute_mib share):
   qwen35moe 88@ctx32K → 1332@ctx524K; qwen35 154@32K → 1664@360K;
   gemma4 212@32K → 418@240K. The packer's sharded path currently charges the
   compute-buffer curve **once, shared**, which is wrong in general but passes
   by accident wherever the curve value ≈ production's per-GPU share × n_gpus
   (27b: curve 3278 ≈ prod total 3328 ✓) or where overshoot is masked.
6. **Compute-buffer curves are layer-split semantics** (fit of
   `max (compute+unaccounted)/n_gpus` on layer-split cells) and extrapolate
   badly past the cell range (≤65536): the shared gemma2/3/4 slope=11 yields
   3083 MiB at ctx 240000 where gemma4 production uses 418/GPU. A gemma4-only
   fit of its own cells (584@8K, 680@32K, ~808@64K) gives
   `base 205 + base_batch 347 + slope 4 × ctx/1024` = 1488 at 240000 — still
   layer-split semantics.
7. **The ~315 MiB/GPU CUDA-runtime shadow** (unaccounted header in 1-GPU
   cells) is already absorbed in the CB curves via the compute+unaccounted
   fit. The 2-GPU layer-split rows show the same ≈310-340/GPU.

## Full decomposition of the remaining outlier (qwen3.6-35b-a3b)

Production physical totals (2-GPU tensor, ctx 524288, np 2, `--n-cpu-moe 40`,
embedded MTP, q8_0 KV, 1 mmproj):

- model: 2862 (attn 1837 + 1-layer experts 568 + output 242 + nextn ~215)
- K+V: 5440 + RS: 502.5 = 5942
- compute: 1332 × 2 = 2664
- MTP: KV 1024 + compute 592 = 1616
- mmproj: 1284 (CUDA0; file bytes 857 — the ~1.5× is CLIP graph buffers)
- CUDA0 AllReduce staging: 248; AR pipeline: 64; CUDA runtime: ~400
- **total ≈ 15080** ✓

Estimator (uncommitted state) charges: model 3504 (attn 1837 + experts 568 +
mmproj 857 + output 242) + KV 5471 + compute 1370 + MTP 2478 ≈ 13002.

Per-term ledger vs production:

| Term | Est | Prod | Δ | Fix |
|---|---|---|---|---|
| RS/SSM state | 62.5 | 502.5 | -440 | §6 |
| compute | 1370 | 2664 | -1294 | §7 |
| model | 3504 | 4146 | -642 | §9 (mmproj expansion, nextn) |
| MTP | 2478 | 1616 | +862 | §8 (overhead constants) |
| **total** | **13002** | **15080** | **-13.8%** | |

Applying RS + CB fixes alone: ≈ 14770 (-2.1%); then §8/§9 for the rest.

## §6. RS/SSM state model (not landed)

Three bugs in the current handling:

1. `hybrid.rs` folds `ssm_state_per_slot` into `kv_per_token/context` once —
   it never multiplies by `streams()`.
2. `tuning.json ssm_state_per_slot_bytes` stores the **breakdown share**
   (one GPU), not the physical per-seq state: qwen35 78643200 (75 MiB) and
   qwen35moe 32768000 (31.25 MiB) are exactly half of the physical 149.62 and
   62.81 MiB (all mtpslot cells are 2-GPU tensor, so `kv_mib` = share).
3. No spec multiplier: with `--spec-type draft-mtp` the RS buffer is
   `(n_max+1)×` (measured exactly 4.0 on both production models).

Fix recipe:

- In `analyse.py`'s SSM deriver, multiply the measured per-slot slope by the
  cell's GPU count (or parse the RS buffer lines — cleaner: add an `rs_mib`
  field to the measurement parser; the lines are already in every log:
  `Meta() RS buffer size = N MiB` and `size = X MiB (N cells, L layers,
  S seqs R rs_seq)`).
- Derive the spec multiplier = 4 from the two production cells
  (`1197/(2×149.62)` and `502.5/(2×62.81)`); this equals
  `spec n_max + 1` (llama.cpp default n_max=3). Gate on `inputs.mtp` for
  embedded MTP; the separate-draft path (gemma4) does not use the recurrent
  cache for drafting.
- `hybrid.rs`: `ssm_state_per_slot × streams × (mtp ? spec_multiplier : 1)`.
- Beware double-counting in `mtp.rs`: the current
  `MTP_PER_SLOT_OVERHEAD_MIB=367` / `MTP_BASE_OVERHEAD_MIB=584` constants were
  fitted from mtpslot mtp-vs-none deltas that *include* the spec-inflated RS.
  Once RS moves to hybrid.rs with the ×4 multiplier, re-derive those constants
  per arch from the same deltas minus the RS term (see §8) — this is §6+§8
  together or the RS gets charged twice.

## §7. Tensor-split compute buffer (not landed)

The sharded packer charges the CB curve once (shared). For the 35b the true
production sharded compute is 2664 (1332/GPU) against its curve value 1370 —
nearly exactly 2×. For the 27b the curve value 3278 already ≈ the true total
3328. For gemma4-qat the curve 3083 heavily overshoots the true 836.

The per-arch truth is sharded compute per GPU, derivable from the clean
`compute_mib` tensor-breakdown column:

- qwen35moe: 88@32K, 1332@524K → base ≈ 7, slope ≈ 2.53/1K
- qwen35: 154@32K, 1664@360K → base ≈ 10, slope ≈ 4.5/1K
- gemma4: 212@32K, 418@240K → base ≈ 180, slope ≈ 1/1K

Recipe: derive a `tensor_compute_curves` section in `analyse.py` from tensor
cells' `compute_mib` (same affine `base + base_batch + slope×ctx/1024` shape,
same max-cover convention as the layer curves), emit it, and in the sharded
packer path charge `curve × n_spanned_gpus` distributed by
`tensor_split_weights`. Do **not** touch the layer-split curve path — it is
correct.

This changes gemma4-qat to 43711 - 3083 + ~828 = 41456 (≈ -4.9%, boundary);
the proper companion is the gemma4-only layer-curve refit (§10) so that both
split modes are honest rather than each relying on the other's error.

## §8. MTP overhead constants (not landed)

After §6+§7: `mtp.rs` uses `raw_kv (metadata) + intercept 136 +
per_slot_overhead 367 × slots + base 584`. The 367/584 were max-fitted on
qwen35 (4 KV heads) and include the spec-inflated RS, which §6 now charges
separately. Production 35b MTP = 1616 vs estimate 2478 (+53%). Direction is
safe (over-reserve), but for parity re-derive per arch from
`mtpslot-*-mtp-npN vs -none-npN` deltas net of the RS term:

- 27b deltas (kv col, total): 225@np1, 449@np2, 898@np4
- 35b deltas: 94@np1, 189@np2, 377@np4

These deltas = MTP-context KV + compute (share) at ctx 32768. Raw KV (metadata)
at ctx32768: 27b = 1×4×512×2×32768 = 128 MiB; 35b = 1×2×512×2×32768 = 64 MiB.
Overhead/ctx-independent remainder per slot: 27b ≈ 225-128 = 97 share
(149.6-ish total?); 35b ≈ 94-64 = 30/slot share. The RS alone is +75 (share,
27b) / +31 (share, 35b) per slot per spec multiplier — consistent with the
delta decomposition. Re-derive constants as:
`per_slot_overhead = per_arch_delta − raw_kv_share − rs_spec_share`, base =
intercept, and keep the max(across arches) policy for safety.

## §9. Model-side residuals (not landed)

- **mmproj CUDA graph expansion**: gemma4-qat mmproj file = 857 MiB; llama.cpp
  reserves 1284 MiB on CUDA0 (`[mtmd] adding 1283.89 MiB to fit_params_target`).
  One data point — expand the CLIP across 2-3 models before deriving a
  multiplier. Do not hardcode 1.5×.
- **nextn tensor accounting** in the expert-offload path (~215 MiB for
  qwen35moe): the MTP head tensors of the kept layer are currently being
  shuffled with the expert offload instead of staying pinned like llama.cpp
  keeps them.

## §10. gemma4 layer-curve slope (not landed)

The shared `[gemma2, gemma3, gemma4]` curve (base 213, base_batch 292, slope
11) fits badly at gemma4's high-ctx extrapolation. Add a gemma4-only entry
ahead of the shared one in `analyse.py`'s curve grouping, fit from gemma4's
own clean layer-split cells (584@8K, 680@32K, ~808@64K ⇒ base 205,
base_batch 347, slope 4). Left unchanged: the shared curve for gemma2/3.

## Verification checklist for the bundle

After landing §6–§9:

- [ ] qwen3.6-35b-a3b within ±5% (expected ≈ -2%)
- [ ] qwen3.6-27b within ±5% (expected ≈ +0.4% after RS×streams×4 +
      CB-curve swap)
- [ ] gemma-4-31b-it-qat within ±5% (expected ≈ -4.9% after §7+§10; verify RS
      constant isn't double-charged for it — gemma4 doesn't route through
      `kv_for_hybrid`)
- [ ] laguna / glm / deepseek / talkie unchanged (all layer-split or ik_llama)
- [ ] `python3 scripts/calibration/analyse.py emit --check` passes after
      regenerating `tuning.json`
- [ ] `cargo test --workspace --all-features` and clippy clean
- [ ] Every new constant carries an `evidence` string naming the cells it was
      derived from

## Where to look first when resuming

1. `scripts/calibration/analyse.py` — emit the tensor compute curves
   (§7), fix the SSM deriver's share→physical factor (§6), emit the spec
   multiplier (§6), split the gemma4 layer curve (§10).
2. `ananke/src/estimator/hybrid.rs` — streams × spec-multiplier RS.
3. `ananke/src/allocator/placement/sharded/mod.rs` — charge tensor CB per
   spanned GPU (§7).
4. `ananke/src/estimator/mtp.rs` — shrink §8 constants per arch after RS
   separation.
5. Re-run the scoreboard loop at the top of this file after each step. When a
   term-fix regresses a model that "passed", prefer the decomposition tables
   above over the raw percentage — the goal is correct terms, and the totals
   follow.

The committing skill applies: propose each commit before landing it.
