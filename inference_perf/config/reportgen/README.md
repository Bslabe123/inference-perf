# Reporting Configuration

Configuration for the `report:` block, which controls which reports the benchmark
generates after a run and at what granularity (summary, per-stage, per-request,
and per-adapter breakdowns). Each report family is independently toggleable, so
you can keep the high-level summary cheap while opting into verbose per-request
output only when needed.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md). For the
shape and contents of the generated report files, see
[docs/reports.md](../../../docs/reports.md). For goodput constraints and how they
are evaluated, see [docs/goodput.md](../../../docs/goodput.md).

Schema: [`config.py`](./config.py).

## Top-level `report` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `request_lifecycle` | object | enabled (all defaults) | Latency/throughput metrics derived from request lifecycle timings (see [request_lifecycle](#request_lifecycle)). |
| `prometheus` | object | enabled (all defaults) | Server-side metrics scraped from the model server's Prometheus endpoint (see [prometheus](#prometheus)). Set to `null` to disable. |
| `session_lifecycle` | object | enabled (all defaults) | Multi-turn session metrics for session-based load (see [session_lifecycle](#session_lifecycle)). |
| `goodput` | object | `null` (disabled) | Goodput constraints; when set, requests are scored against per-metric thresholds (see [goodput](#goodput)). |

## request_lifecycle

Metrics computed from per-request timing (latency, time-to-first-token,
throughput, etc.).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool | `true` | Generate the high-level aggregate summary across the whole run. |
| `per_stage` | bool | `true` | Include a breakdown per load stage. |
| `per_request` | bool | `false` | Emit detailed per-request records (verbose). |
| `per_adapter` | bool | `true` | Group metrics by LoRA adapter. |
| `per_adapter_stage` | bool | `false` | Group metrics by adapter and stage. |
| `percentiles` | list[float] | `[0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]` | Percentiles to compute for latency/throughput distributions. |

## prometheus

Server-side metrics scraped from the model server. Set the whole `prometheus`
key to `null` to skip Prometheus reporting entirely.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool | `true` | Include the aggregate Prometheus metrics summary. |
| `per_stage` | bool | `false` | Include a Prometheus breakdown per load stage. |

## session_lifecycle

Metrics for multi-turn session load (for example `trace_session_replay`).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool | `true` | Generate the aggregate session summary. |
| `per_stage` | bool | `true` | Include a breakdown per load stage. |
| `per_session` | bool | `false` | Emit detailed per-session records (verbose). |

## goodput

Defines pass/fail constraints used to compute goodput. Disabled by default
(`goodput` is `null`); set it to enable scoring. See
[docs/goodput.md](../../../docs/goodput.md) for the metric names and semantics.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `constraints` | dict[str, float] | `{}` | Map of metric name to threshold value; a request counts as "good" when it satisfies all constraints. |

## Example

```yaml
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: false
    per_adapter: false
    per_adapter_stage: false
    percentiles: [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
  prometheus:
    summary: true
    per_stage: false
  session_lifecycle:
    summary: true
    per_stage: true
    per_session: false
  goodput:
    constraints:
      time_per_output_token: 0.05
      time_to_first_token: 1.0
```
