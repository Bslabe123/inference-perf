# Load Generation Configuration

Configuration for the `load:` block, which defines the request pattern the
benchmark drives at the model server: the load type, the per-stage schedule, and
the worker pool that issues requests.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md). For how
the load generator behaves at runtime (scheduling, stage lifecycle), see
[docs/loadgen.md](../../../docs/loadgen.md).

Schema: [`config.py`](./config.py).

## Load types

`load.type` selects the request pattern. The type also determines which stage
shape `load.stages` must use (see [Stages](#stages)).

| `type` | Pattern | Stage type |
| --- | --- | --- |
| `constant` (default) | Fixed request rate (QPS) per stage | `StandardLoadStage` |
| `poisson` | Poisson-distributed arrivals at the target rate | `StandardLoadStage` |
| `trace_replay` | Replays request timings from a trace file | `StandardLoadStage` |
| `concurrent` | Fixed in-flight concurrency, not a rate | `ConcurrentLoadStage` |
| `trace_session_replay` | Replays multi-turn sessions from a trace | `TraceSessionReplayLoadStage` |

`trace_session_replay` is documented in detail in
[docs/otel_trace_replay.md](../../../docs/otel_trace_replay.md); the rest are
covered below.

## Top-level `load` fields

<!-- FIELDS: LoadConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `constant` | Load pattern (see Load types). |
| `interval` | float | `1.0` | Seconds between request batches within a stage. |
| `stages` | list[StandardLoadStage] or list[ConcurrentLoadStage] or list[TraceSessionReplayLoadStage] | `[]` | Ordered load stages; shape depends on type. |
| `sweep` | SweepConfig | `null` | Auto-derive stages from a saturation search (see Sweeps). |
| `num_workers` | int | CPU core count | Number of worker processes (see Worker model). |
| `worker_max_concurrency` | int | `100` | Max in-flight requests per worker. |
| `worker_max_tcp_connections` | int | `2500` | Max TCP connections per worker. |
| `trace` | TraceConfig | `null` | Trace source for trace_replay / trace_session_replay. |
| `circuit_breakers` | list[str] | `[]` | Named circuit breakers to abort a run on breach (see docs/goodput.md). |
| `request_timeout` | float | `null` | Per-request timeout in seconds. |
| `lora_traffic_split` | list[MultiLoRAConfig] | `null` | MultiLoRA traffic weights (see LoRA traffic split). |
| `base_seed` | int | per-run timestamp (ms) | Base random seed; set explicitly for reproducible runs. |

<!-- /FIELDS -->

## Stages

Each entry in `load.stages` is one phase of the run. Stages execute in order.
The required fields depend on `load.type`.

### Standard stages (`constant`, `poisson`, `trace_replay`)

<!-- FIELDS: StandardLoadStage: rate, duration -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `rate` | float > 0 | (required) | Request rate (QPS) |
| `duration` | int > 0 | (required) | Duration in seconds |

<!-- /FIELDS -->

```yaml
load:
  type: constant
  stages:
    - rate: 1
      duration: 30
    - rate: 5
      duration: 30
```

### Concurrent stages (`concurrent`)

A concurrent stage holds a fixed number of in-flight requests rather than a
rate. `rate`/`duration` are set internally at runtime and must not be configured.

<!-- FIELDS: ConcurrentLoadStage: num_requests, concurrency_level -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_requests` | int > 0 | (required) | Number of requests to send |
| `concurrency_level` | int > 0 | (required) | Concurrency level |

<!-- /FIELDS -->

```yaml
load:
  type: concurrent
  stages:
    - concurrency_level: 8
      num_requests: 400
```

### Trace session replay stages (`trace_session_replay`)

<!-- FIELDS: TraceSessionReplayLoadStage -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `concurrent_sessions` | int ≥ 0 | (required) | Maximum number of sessions active simultaneously. 0 = all sessions active at once (stress test mode). N > 0 = at most N sessions active; when one completes, next is activated. |
| `session_rate` | float > 0 | `null` | Sessions to start per second (optional, omit for no rate limit) |
| `num_sessions` | int > 0 | `null` | Number of sessions to run in this stage. Draws the next N sessions from the corpus. None = all remaining sessions. |
| `timeout` | float > 0 | `null` | Wall-clock safety limit in seconds. If exceeded, in-flight sessions are cancelled and stage exits as FAILED. Optional. |

<!-- /FIELDS -->

See [docs/otel_trace_replay.md](../../../docs/otel_trace_replay.md) for the full
workflow and trace format.

## Worker & multiprocessing model

`num_workers` controls how the load is driven:

- **`num_workers: 0`**: single-process mode. Requests are issued from one
  asyncio event loop in the main process. Simplest, and best for low rates and
  debugging.
- **`num_workers: N > 0`** (default is the host's CPU core count): `N` worker
  **processes**, each running its own asyncio event loop. Use this to push rates
  a single process can't sustain.

> These are OS processes (`multiprocessing.Process`), **not** threads. Earlier
> docs called them "worker threads"; that was inaccurate.

Per worker:

- `worker_max_concurrency` caps the number of requests a single worker keeps in
  flight (a semaphore). Total in-flight ≈ `num_workers × worker_max_concurrency`.
- `worker_max_tcp_connections` caps that worker's TCP connection pool.

## Load sweeps

Instead of hand-writing stages, a sweep runs a preprocessing phase that searches
for the service's saturation point and generates stages around it. Only valid
with `constant` / `poisson` (not `concurrent` or `trace_session_replay`).

<!-- FIELDS: SweepConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | (required) | linear (evenly spaced 1 to saturation) or geometric (clustered near saturation). |
| `num_requests` | int | `2000` | Requests used to probe saturation. |
| `timeout` | float | `60` | Seconds to run the saturation probe. |
| `num_stages` | int | `5` | Number of stages to generate. |
| `stage_duration` | int | `180` | Duration of each generated stage. |
| `saturation_percentile` | float | `95` | Percentile of sampled rates taken as the saturation point. |

<!-- /FIELDS -->

```yaml
load:
  type: constant
  interval: 15
  sweep:
    type: linear
    timeout: 60
    num_stages: 5
    stage_duration: 180
    saturation_percentile: 95
```

## LoRA traffic split

Splits traffic across multiple LoRA adapters. Weights must sum to `1.0`.

<!-- FIELDS: MultiLoRAConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | (required) | Adapter name as registered on the model server. |
| `split` | float | (required) | Fraction of traffic routed to this adapter. |

<!-- /FIELDS -->

```yaml
load:
  lora_traffic_split:
    - name: adapter_1
      split: 0.5
    - name: adapter_2
      split: 0.5
```
