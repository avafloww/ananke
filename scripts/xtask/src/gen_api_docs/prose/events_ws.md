The `/api/events` WebSocket delivers a system-wide stream of daemon events. Connect with a standard WebSocket handshake to `ws://<host>:7071/api/events`.

An optional `?service=<name>` query parameter filters events to a single service. Events that don't carry a service field (`config_reloaded`, `overflow`) are always delivered.

### Frame format

Each frame is a text message containing a single JSON object with a `type` tag. The `at_ms` field (millisecond UNIX timestamp) is present on every variant except `overflow`.

### Variants

#### `state_changed`

Emitted when a service transitions between states.

```json
{
  "type": "state_changed",
  "service": "demo",
  "from": "idle",
  "to": "starting",
  "at_ms": 1700000000000
}
```

#### `allocation_changed`

Emitted when a service's per-device memory pledge changes.

```json
{
  "type": "allocation_changed",
  "service": "demo",
  "reservations": {
    "gpu:0": 8589934592
  },
  "at_ms": 1700000000000
}
```

#### `config_reloaded`

Emitted when the daemon's config file is reloaded. `changed_services` lists service names whose config was modified.

```json
{
  "type": "config_reloaded",
  "at_ms": 1700000000000,
  "changed_services": ["demo", "qwen"]
}
```

#### `estimator_drift`

Emitted when one of a service's rolling estimator corrections moves by more than 5%. Each service carries one correction per memory pool, learned independently from that pool's own observation, and `class` says which this event is about: `"vram"` (the observed NVML peak over the reservation's GPU slots) or `"host"` (the observed RSS peak, less the GPU-resident share of the model mapping, over the reservation's CPU slot).

```json
{
  "type": "estimator_drift",
  "service": "demo",
  "class": "vram",
  "rolling_mean": 1.05,
  "at_ms": 1700000000000
}
```

#### `auto_restarted`

Emitted when a `Running` service is drained and respawned by its auto-restart policy (see [auto-restart](configuration.md#auto-restart)). `trigger` is `"error_rate"`, `"periodic"`, `"ttft_stall"` (the service produced no response frame within the stall timeout while a request was in flight), `"generation_stall"` (the child's `/metrics` progress counters stayed flat for the stall timeout while requests were in flight), or `"spec_collapse"` (a run that previously accepted speculative draft tokens stopped accepting any); `detail` is a human-readable reason such as the observed error rate and window.

```json
{
  "type": "auto_restarted",
  "service": "demo",
  "trigger": "error_rate",
  "detail": "error rate 100% (24/24 requests over 120s) ≥ threshold 50%",
  "at_ms": 1700000000000
}
```

#### `overflow`

Emitted when the event bus drops events because a subscriber fell behind.

```json
{
  "type": "overflow",
  "dropped": 42
}
```

### Heartbeat

The server sends a WebSocket Ping frame every 30 seconds. Clients should respond with Pong to keep intermediaries from closing the connection.
