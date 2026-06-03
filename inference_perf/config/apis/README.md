# API Configuration

Configuration for the `api:` block, which defines how the benchmark talks to the
model server: the API surface it targets (text completion vs. chat), whether
responses stream, custom HTTP headers, per-request SLO evaluation, and structured
output. When SLO headers are present, each request is checked for SLO compliance
and SLO-related metrics are reported.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md).

Schema: [`config.py`](./config.py).

## API types

`api.type` selects which API surface the benchmark calls.

| `type` | Endpoint | When to use |
| --- | --- | --- |
| `completion` (default) | Text completion API | Default. Works with minimal server config; use for raw prompt/completion benchmarking. |
| `chat` | Chat completions API | Use for chat-formatted (multi-turn / role-based) workloads. May require extra server configuration (chat template, etc.). |

## Top-level `api` fields

<!-- FIELDS: APIConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `completion` | API surface to target (completion or chat). |
| `streaming` | bool | `false` | Stream the response. Required to measure TTFT, ITL, and TPOT. |
| `headers` | dict | `null` | Custom HTTP headers sent with every request. |
| `slo_unit` | str | `null` | Unit for SLO header values (e.g. ms, s). |
| `slo_tpot_header` | str | `null` | Name of the header carrying the per-request TPOT SLO. |
| `slo_ttft_header` | str | `null` | Name of the header carrying the per-request TTFT SLO. |
| `response_format` | ResponseFormat | `null` | Structured-output spec. |

<!-- /FIELDS -->

### When to set each field

- **`streaming`**: set to `true` whenever you need token-timing metrics (TTFT,
  ITL, TPOT). Leave `false` for end-to-end latency only.
- **`headers`**: use for routing hints (e.g. model selection, routing strategy)
  or to carry SLO values that the SLO headers below reference.
- **`slo_unit` / `slo_tpot_header` / `slo_ttft_header`**: set these together when
  you want per-request SLO compliance evaluated and reported. The header names
  must match the keys you put in `headers`.

## Response format

`api.response_format` requests structured output from the server (vLLM/OpenAI
compatible). See the
[vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

<!-- FIELDS: ResponseFormat -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `json_schema` | json_schema (constrain output to a schema) or json_object (any valid JSON object). |
| `name` | str | `"structured_output"` | Name for the JSON schema (used when type is json_schema). |
| `json_schema` | dict | `null` | The JSON schema the output must conform to (used when type is json_schema). |

<!-- /FIELDS -->

When to use each `type`:

- **`json_schema`**: constrain output to a specific schema. Provide `json_schema`
  (and optionally `name`).
- **`json_object`**: require any syntactically valid JSON object without
  constraining its shape. `name` and `json_schema` are ignored.

## Example

```yaml
api:
  type: completion             # completion is default since chat may require extra server config
  streaming: true              # enable streaming for TTFT, ITL, and TPOT metrics
  headers:                     # custom HTTP headers
    x-inference-model: llama
    x-routing-strategy: round-robin
    x-slo-tpot-ms: "2"
    x-slo-ttft-ms: "1000"
  slo_unit: "ms"               # SLO unit (e.g. ms, s)
  slo_tpot_header: "x-slo-tpot-ms"   # header carrying the TPOT SLO
  slo_ttft_header: "x-slo-ttft-ms"   # header carrying the TTFT SLO
```
