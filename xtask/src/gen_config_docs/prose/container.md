Either template can run inside a Docker or Podman container. Add a `[service.container]` block to the service. Without the block the service runs as an ordinary host process and nothing about it changes.

The block does not replace the template; it relocates it. A `llama-cpp` service still gets its placement-derived llama-server argv, and a `command` service still supplies its own. What changes is where that argv runs: as the container's command rather than as a child of the daemon.

```toml
[[service]]
name = "qwen3.6-35b-ninfer"
template = "command"
port = 8205
command = [
  "ninfer-serve", "/artifacts/qwen3_6_35b_a3b.ninfer",
  "--host", "${listen_host}",
  "--port", "${listen_port}",
]
allocation = { mode = "static", reserve_gb = 26 }
health = { http = "/health", timeout = "10m" }

[service.container]
runtime = "docker"
image = "ninfer:local"
network = "host"
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/ai/ninfer", target = "/artifacts", read_only = true },
]
```

#### Prerequisites

ananke drives the runtime's CLI. It never talks to the Engine or Libpod socket, and it neither pulls nor builds images. It starts and stops the image named by the service; the image must already be in the local store before the service starts:

```sh
docker pull vllm/vllm-openai:v0.26.0
podman build -t ninfer:local ~/src/ninfer
```

A missing image surfaces as a start failure carrying the runtime's own message, which distinguishes an unknown tag, a registry that needs credentials, and an image that was never built.

The daemon needs the selected runtime binary on its `$PATH` (or an explicit `runtime_executable`) and permission to use it.

Selecting Podman changes the `runtime` field and nothing else about the service. It does not make the two runtimes interchangeable in general: ananke owns the lifecycle-flag differences, but CDI support, rootless cgroup layouts, and SELinux relabelling all vary by runtime and version, and a capability the runtime lacks fails with an error rather than silently degrading.

#### Conflicting fields

Two existing fields are rejected alongside `container`, because each replaces the same boundary the container does:

- `launcher` (llama-cpp) wraps the host execution boundary, which is what the container now is.
- `shutdown_command` (command) exists so a wrapper script can stop something signals cannot reach. Native runtime cleanup replaces it.

#### Networking

The service endpoint has two sides. The placeholders resolve to the side the workload should bind, not the side ananke connects to.

Under `network = "bridge"`, ananke publishes `127.0.0.1:<private_port>` onto `<container_port>` and nothing else. Inside the container, `${listen_host}` resolves to `0.0.0.0` and `${listen_port}` to `<container_port>`: a workload that bound loopback inside its own network namespace would be unreachable from the host side of the publication. Because ananke can only guarantee reachability it was told about, a bridge-networked `command` service must consume both `${listen_host}` and `${listen_port}` in its argv or environment; one that hardcodes its own address is rejected at config load. Use host networking for a command whose listener cannot be parameterised.

Under `network = "host"`, there is no publication and no translation. `${listen_host}` resolves to `127.0.0.1` and `${listen_port}` to the private port, exactly as for a host process.

`${port}` remains accepted everywhere as an alias for `${listen_port}`, so pre-container command templates keep working. `${host_port}` is always the host-side private port, for the rare command that needs both numbers.

llama.cpp takes the same resolved endpoint, but as a typed renderer input rather than a placeholder: ananke emits `--host`/`--port` itself, so a containerized llama.cpp service needs no configuration for this.

#### Environment and secrets

`env_inherit` copies the daemon's whole environment into a host process. That is reasonable for a child process and wrong for a container, so it governs host processes only. A container's environment is exactly what the service declares:

- `env` on the service, and `container.env` layered over it, for explicit values.
- `container.env_passthrough` for names forwarded from the daemon's own environment, e.g. `["HF_TOKEN"]`.

Passthrough values are read at launch and never leave the runtime invocation: the management API and launch previews show the variable's name only, and nothing expands a secret into a persisted command line or a log line.

#### Mounts and path translation

Nothing inside the container can see a host path that was not mounted. For `command` services the service arranges this: write container paths in the argv, and mount whatever backs them. Paths are not guessed.

For `llama-cpp`, ananke generates the argv, so it also translates the four path fields it owns (`model`, `mmproj`, `draft_model`, and `chat_template_file`) through the configured mounts. Matching is lexical and component-aware, longest source prefix wins, and there is no symlink resolution. A typed path no mount covers is a start failure naming the exact path and field, rather than a container that comes up and cannot find its weights. Paths embedded in `extra_args` stay opaque and must already be container paths.

#### GPUs

`gpu_device` is a CDI template expanded once per GPU that ananke's placement selected, so the container sees the same devices the allocator reserved for it. `nvidia.com/gpu=${id}` with GPUs 0 and 2 selected yields `--device nvidia.com/gpu=0 --device nvidia.com/gpu=2`.

Leaving it unset injects nothing. If the runtime cannot satisfy the requested CDI device, the start fails with that error. ananke does not fall back to bind-mounting `/dev/nvidia*`, which would expose every GPU on the machine to the container.

#### Memory attribution

ananke reads the container's cgroup back from its host PID after start and attributes every process in that cgroup to the service. This needs no configuration and holds regardless of runtime, cgroup driver, or rootless layout, because it is the path the container actually landed in rather than one predicted from a slice name. The host PID's process descendants are attributed too, so the two sources cover each other.

#### Lifecycle and recovery

Containers can outlive both ananke and the CLI process that started them, so the lifecycle is conservative and every step is recoverable:

1. A durable launch intent is written before anything is created, carrying the owner, the service and run identity, the exact runtime executable, and the generated name and labels.
2. The container is created, not started, and its ID is attached to the intent.
3. The ownership row is committed. Only then is the container started: a container is never running without a persisted row that says so.
4. Logs are followed by a separate `logs --follow` process, and the exit status comes from an independent `wait`.
5. On drain, `kill --signal TERM`, escalating to `KILL` on the existing `drain_timeout`. Final logs are drained before the container is removed.

Each run gets its own container named `ananke-<service>-<run-id>`, removed along with any anonymous volumes the runtime created for it. A stable per-service name would be easier to read, but a single failed cleanup would then block the service permanently: the next `create` would fail because the name is already in use. A per-run name makes a stray container inert rather than blocking, and the startup sweep bounds how long one can persist.

`--rm` is deliberately not used. Auto-removal races the log follower and the exit-status read, and it destroys the evidence a crashed daemon needs, so removal is always an explicit final step and is idempotent.

At startup ananke reconciles what it finds against those records. Every container it creates is labelled with a stable per-installation owner UUID as well as its service and run, and cleanup requires both a persisted identity and a matching owner label, not a name prefix alone. This keeps a second ananke, or an unrelated container that shares a name, from being removed. A container whose ownership cannot be resolved blocks only its own service from reprovisioning, preserving the evidence, while every other service starts normally.

#### What happens when ananke stops

A native child dies with the daemon: the spawner sets `PR_SET_PDEATHSIG`, and the kernel honours it however the daemon died. A container has no equivalent: it belongs to the runtime, not to ananke's process tree. Its survival is therefore handled explicitly rather than assumed.

On a signal ananke can catch (`SIGTERM`, `SIGINT`, `SIGQUIT`, as sent by `systemctl stop`, a restart, a reboot, or Ctrl-C), every service drains: TERM, then KILL after `drain_timeout`, then removal. Containers do not outlive it.

If that drain overruns `shutdown_timeout`, the daemon exits rather than hanging. Before it exits, it removes every container still carrying this installation's owner label. The sweep exists for this case; after a drain that finished, it finds nothing.

On `SIGKILL`, the OOM killer, or a power loss, no userspace code runs: that is what those events mean. Containers keep running and keep the VRAM they reserved. Two things bound the exposure:

- Startup reconciliation removes them when ananke comes back, so the exposure lasts as long as the daemon is down. Running under a supervisor that restarts it keeps that to seconds.
- The allocator reads real free VRAM from NVML, not just its own pledge book, so a surviving container makes ananke conservative rather than wrong: it declines to place, or evicts, instead of over-committing onto memory something else is holding.

An orphan is therefore a capacity problem until reconciliation clears it, not a correctness one. To make containers die with the daemon under `SIGKILL` as well, that guarantee has to come from outside ananke, for example a cgroup the supervisor tears down, because no process can clean up after a signal it cannot receive.

#### Logs

Neither `docker logs` nor `podman logs` documents a stream framing that survives the CLI boundary, so container output is stored and served as a single `combined` stream rather than being mislabelled as stdout. Host processes are unaffected and keep their split `stdout`/`stderr`. Filtering for `both` includes combined lines; asking for `stdout` or `stderr` will not match a container's output.

#### A containerized llama.cpp service

```toml
[[service]]
name = "muse-glimmer"
template = "llama-cpp"
port = 8202
model = "/home/philpax/ai/muse-glimmer/models/muse-glimmer-30B-kquant-dynamic.gguf"
mmproj = "/home/philpax/ai/muse-glimmer/models/mmproj-kquant.gguf"
draft_model = "/home/philpax/ai/muse-glimmer/models/dflash-kquant.gguf"
spec_type = "draft-dflash"
context = 262144
health = { http = "/health", timeout = "5m" }

[service.container]
runtime = "docker"
image = "ghcr.io/ggml-org/llama.cpp:server-cuda"
network = "host"
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/ai/muse-glimmer/models", target = "/models", read_only = true },
]
```

Only the generated flags follow the image; the image's own ENTRYPOINT runs them. `llama_server` is not consulted here: it answers where llama-server lives on the host, and its `llama-server` default is a `$PATH` lookup that means nothing inside an image. The official llama.cpp image ships `/app/llama-server` with `/app` off `$PATH`, so an entrypoint derived from that default would fail to resolve. An image that needs something other than its declared ENTRYPOINT says so with `container.entrypoint`.

The three model paths are translated through the mount before creation, becoming `-m /models/muse-glimmer-30B-kquant-dynamic.gguf` and so on.

#### A containerized OpenAI-compatible server

vLLM's image declares its own entrypoint, so the command supplies arguments only:

```toml
[[service]]
name = "diffusiongemma-26b-a4b"
template = "command"
port = 8200
command = [
  "--model", "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
  "--max-model-len", "131072",
  "--gpu-memory-utilization", "0.715",
  "--host", "${listen_host}",
  "--port", "${listen_port}",
]
env = { VLLM_NO_USAGE_STATS = "1" }
allocation = { mode = "static", reserve_gb = 23 }
health = { http = "/health", timeout = "10m" }
openai_proxy = { upstream_model = "nvidia/diffusiongemma-26B-A4B-it-NVFP4" }

[service.container]
runtime = "docker"
image = "vllm/vllm-openai:v0.26.0"
network = "bridge"
container_port = 8000
ipc = "host"
env_passthrough = ["HF_TOKEN"]
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/.cache/huggingface", target = "/root/.cache/huggingface" },
]
```

An image whose CMD is to be replaced instead takes the executable as its first argument (`command = ["ninfer-serve", "/artifacts/model.ninfer", ...]`), with the same block otherwise. The only field that differs between the Docker and Podman versions of either example is `runtime`.

#### Field Reference
