Used for arbitrary binaries. Only `name`, `template`, `port`, and `command` are required. To run the command inside Docker or Podman, add a `[service.container]` block rather than wrapping it in a script — see [Container Workloads](#container-workloads).

```toml
[[service]]
name = "comfyui"            # required
template = "command"        # required
port = 8188                 # required
command = ["/bin/bash", "start_comfy.sh", "--port", "${port}"] # required
lifecycle = "on_demand"

[service.allocation]
mode = "dynamic"
min_reserve_gb = 2.0
max_reserve_gb = 12.0

[service.health]
http = "/system_stats"
timeout = "30s"
```

#### Field Reference

#### Placeholders

The following placeholders are substituted in `command` and `shutdown_command` argv entries (and in `env` values):

- `${port}` - the private loopback port assigned by ananke.
- `${gpu_ids}` - comma-separated NVML index list ananke picked for this service.
- `${reserve_mb}` - the reservation in MiB, on whichever device the service was placed. Still accepted under its former name `${vram_mb}`.
- `${model}` - model path (llama-cpp only; empty for command services).
- `${name}` - service name.
- `${listen_host}` / `${listen_port}` - the interface and port the workload should bind. For a host process these are `127.0.0.1` and the private port, so `${listen_port}` and `${port}` agree; they differ only for a bridge-networked container (see [Container Workloads](#container-workloads)).
- `${host_port}` - the host-side private port, for the rare command that needs both sides of a bridge publication.

A placeholder is `${name}`, and only `${` and `$$` mean anything to the substituter. A bare `{` never does, so an argument carrying JSON, a Jinja template, or a Python format string passes through exactly as written — vLLM's `--diffusion-config '{"canvas_length": 256}'` needs no escaping at all. `$$` is a literal `$`, which is how a literal `${port}` is written (`$${port}`); a `$` before anything else is itself, since arguments carry bare dollars far more often than they carry `${`.

An unknown name is an error, so a typo surfaces at config load rather than reaching the argv. So does `${` with no closing brace.

> **Changed in 0.3.0.** Placeholders were previously written `{name}`. That form could not be told apart from an argument's own braces without guessing, which made JSON arguments fail and silently rewrote Jinja templates. Rename each placeholder to `${name}`; nothing else needs escaping. A config still using `{name}` will not error — the text is now literal — so check that a service's rendered command still contains the value you expect, via `GET /api/services/{name}/command` or the launch-command panel.

The child also inherits `CUDA_VISIBLE_DEVICES` set to the picked GPU id(s). A containerized service does not need to forward it by hand: set `container.gpu_device` and ananke injects exactly the GPUs it picked.

The `shutdown_command` field is for external processes that cannot stop via signal alone. ananke runs it after the drain pipeline completes, ensuring a clean exit for a service that doesn't respond to SIGTERM. A container service has no use for it — native runtime cleanup replaces it, and the two are rejected together.

#### OpenAI Proxy

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy - the same as before.

```toml
[[service]]
name = "qwen3.6-27b-vllm"
template = "command"
port = 8210
command = ["/srv/vllm/qwen36_27b.sh", "${port}"]
lifecycle = "on_demand"
idle_timeout = "10m"

[service.allocation]
mode = "static"
reserve_gb = 44

[service.devices]
placement = "gpu-only"
placement_override = { "gpu:0" = 22000, "gpu:1" = 22000 }

[service.health]
http = "/health"

[service.openai_proxy]
upstream_model = "qwen3.6-27b-autoround"
```

When this is set:

- The service appears in `GET /v1/models` under its `name` (here, `qwen3.6-27b-vllm`).
- `POST /v1/chat/completions` (and `/v1/completions`, `/v1/embeddings`) addressed to `qwen3.6-27b-vllm` are routed to this service. ananke ensures the command is started, then forwards the body to the service's private loopback port.
- Before forwarding, ananke rewrites the JSON `model` field to `upstream_model` (here, `qwen3.6-27b-autoround`) - the name vLLM was started with via `--served-model-name`. Clients address ananke's name; the upstream sees its own name.

The rewrite happens *after* `[service.filters]` is applied, so `openai_proxy.upstream_model` overrides any `filters.set_params.model`. Filters can still strip or set other JSON keys (see [Request Filters](#request-filters)).

#### Embedding Services

By default every service is registered as a chat model. Pooling-only embedding models (Jina v5, BGE, E5, LFM2.5, …) opt in by setting `modality = "embedding"` on the service. The proxy itself is endpoint-agnostic - it already routes `POST /v1/embeddings` by `model` field.

What the modality does depends on the template:

- On `llama-cpp` services it passes `--embeddings` to llama-server, enabling its embeddings endpoint. The pooling strategy comes from the GGUF's `{arch}.pooling_type`.
- On `command` services the wrapped server is expected to speak `/v1/embeddings` natively; nothing extra is passed.

Beyond that the field is a typed declaration: clients filter on it through `/v1/models` and `/api/services`, and the frontend renders a teal `embedding` badge next to the service name (mirroring the purple `vision` badge for llama.cpp services with an `mmproj`).

```toml
[[service]]
name = "jina-embeddings-v5-text-small-retrieval-vllm"
template = "command"
port = 8211
modality = "embedding"
command = ["/srv/vllm/jina_embed_v5_small.sh", "${port}"]
lifecycle = "on_demand"
idle_timeout = "30m"

[service.allocation]
mode = "static"
reserve_gb = 7

[service.devices]
placement = "gpu-only"
placement_override = { "gpu:1" = 7000 }

[service.health]
http = "/health"

[service.openai_proxy]
upstream_model = "jina-embeddings-v5-text-small-retrieval"
```

Valid values are `"chat"` (the default) and `"embedding"`; any other string is a hard config error rather than a silent fall-back. The field is elided from `/v1/models` and `/api/services` JSON when it equals `"chat"`, so chat-only deployments see byte-identical wire output to what they shipped before this field landed.

Once registered, hit the endpoint as you would any OpenAI embedding API:

```sh
curl -s -X POST http://localhost:7070/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-embeddings-v5-text-small-retrieval-vllm",
    "input": ["the quick brown fox", "lorem ipsum dolor sit amet"]
  }' | jq '{model, dim: (.data[0].embedding | length), n: (.data | length)}'
```

ananke ensures the upstream container is started (cold-starting it on first request if needed), rewrites `model` to `upstream_model`, and relays the embedding vectors back unchanged. The static VRAM pledge is held only while the service is running; on-demand services drain back to idle after `idle_timeout` elapses with no traffic.
