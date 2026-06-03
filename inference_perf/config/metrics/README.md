# Metrics Configuration

Configuration for the `metrics:` block, which selects the server-side metrics
client the benchmark uses to scrape model-server performance metrics (for
example vLLM Prometheus counters) alongside the client-side request stats that
inference-perf records on its own.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md). For the
definitions of the metrics themselves (throughput, latency, formulas), see
[docs/metrics.md](../../../docs/metrics.md).

Schema: [`config.py`](./config.py).

## Metrics client types

`metrics.type` selects the backend used to collect server-side metrics.

| `type` | Behavior | Requires `prometheus` block |
| --- | --- | --- |
| `default` | No external metrics client; the benchmark relies only on client-side request stats. | No |
| `prometheus` | Queries a Prometheus endpoint for server-reported metrics and folds them into the report. | Yes |

## Top-level `metrics` fields

<!-- FIELDS: MetricsClientConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | (required) | Metrics client backend: 'default' or 'prometheus'. |
| `prometheus` | PrometheusClientConfig | `null` | Prometheus client settings; required when type is 'prometheus'. |

<!-- /FIELDS -->

## Prometheus config

The `prometheus` sub-block configures the Prometheus query client (schema:
[`../client/server_metrics/config.py`](../client/server_metrics/config.py)).
Exactly one of `url` or `google_managed` must be set: pick a direct Prometheus
URL, or enable Google Managed Prometheus (which is queried through the GMP API
instead of a URL). Setting both, or neither, is a validation error.

<!-- FIELDS: PrometheusClientConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `scrape_interval` | int | `15` | Metrics scrape interval in seconds; should match the server's scrape interval. |
| `url` | URL | `null` | Prometheus server URL to query (for example http://localhost:9090). Mutually exclusive with google_managed. |
| `filters` | list[str] | `[]` | Metric names to collect; empty collects the default set. |
| `google_managed` | bool | `false` | Query Google Managed Prometheus (GMP) via its API instead of a URL. Mutually exclusive with url. |

<!-- /FIELDS -->

When `google_managed: true`, the run requires Application Default Credentials
with the `roles/monitoring.viewer` role. See the "Google Managed Prometheus
(GMP) Requirements" section in [docs/config.md](../../../docs/config.md) for the
full permission and environment setup.

## When to use

- **`default`**: use when you only need inference-perf's own client-side request
  metrics (throughput, TTFT, end-to-end latency) and have no Prometheus endpoint
  to scrape. Simplest setup, no external dependency.
- **`prometheus` with `url`**: use when the model server (or a sidecar) exposes a
  reachable Prometheus endpoint and you want server-reported metrics in the
  report. Standard self-hosted Prometheus.
- **`prometheus` with `google_managed: true`**: use on GKE or GCE where metrics
  flow into Google Managed Prometheus and there is no direct Prometheus URL to
  hit.

## Examples

Direct Prometheus URL:

```yaml
metrics:
  type: prometheus
  prometheus:
    url: "http://localhost:9090"
    scrape_interval: 15
    filters: []
```

Google Managed Prometheus:

```yaml
metrics:
  type: prometheus
  prometheus:
    google_managed: true
    scrape_interval: 15
```

Client-side metrics only:

```yaml
metrics:
  type: default
```
