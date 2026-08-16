ananke oversubscribes GPU memory by dynamically managing which models are active:

- **llama.cpp Services**: VRAM usage is determined by the model size and `n_gpu_layers`. ananke uses an internal GGUF-aware estimator to track usage. No allocation mode is needed.
- **Command Services**: Support two allocation modes via `[service.allocation]`:
  - `static`: Reserves a fixed amount of memory (`reserve_gb`) — host RAM for a cpu-only service, VRAM otherwise. The pre-rename `vram_gb` spelling is still accepted.
  - `dynamic`: Operates within a range (`min_reserve_gb` to `max_reserve_gb`).

For a GPU-placed service the daemon picks, in both modes, the GPU with the most available headroom (subject to `gpu_allow`), preferring one whose free capacity satisfies the upper bound (`reserve_gb` for `static`, `max_reserve_gb` for `dynamic`) so dynamic services have room to grow. The picked GPU id is exported to the spawned child as `CUDA_VISIBLE_DEVICES`, and is also available as the `{gpu_ids}` placeholder in `command` argv. A containerized service instead sets `container.gpu_device`, and ananke injects a CDI device per picked GPU so those are the only ones the container sees. A `placement = "cpu-only"` service skips the pick entirely — its reservation is host RAM, and `{gpu_ids}` substitutes to the empty string.

```toml
[service.allocation]
mode = "dynamic"         # "static" or "dynamic" (command services only)
reserve_gb = 44              # static: fixed reservation in GiB
min_reserve_gb = 2.0         # dynamic: minimum reservation in GiB
max_reserve_gb = 12.0        # dynamic: maximum reservation in GiB
min_borrower_runtime = "60s" # dynamic: balloon resolver grace period
```

**Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.
