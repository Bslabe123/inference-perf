# Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration Structure](#configuration-structure) (per-block reference)
3. [Full Configuration Examples](#full-configuration-examples)
4. [Advanced Use Cases](#advanced-use-cases)
5. [Google Managed Prometheus (GMP) Requirements](#google-managed-prometheus-gmp-requirements)

## Overview

This document provides complete documentation for all configuration options available in the Kubernetes Inference Performance Benchmark tool.

## Configuration Structure

A benchmark is configured through a single YAML file (or equivalent CLI flags). Each top-level key configures one subsystem. The reference for each block below is generated from that block's colocated `README.md`, which lives next to its schema under [`inference_perf/config/`](../inference_perf/config/). Edit those READMEs (not this region) and run `pdm run update:config-doc`.

> `circuit_breakers` is documented separately in [goodput.md](./goodput.md).

<!-- BEGIN GENERATED: config subsystems. Do not edit by hand; edit the per-subdirectory README.md files and run `pdm run update:config-doc`. -->

Jump to a block:

- [API (`api`)](#api-api)
- [Data Generation (`data`)](#data-generation-data)
- [Load Generation (`load`)](#load-generation-load)
- [Model Server (`server`)](#model-server-server)
- [Metrics (`metrics`)](#metrics-metrics)
- [Reporting (`report`)](#reporting-report)
- [Storage (`storage`)](#storage-storage)
- [Tokenizer (`tokenizer`)](#tokenizer-tokenizer)

## API (`api`)

Configuration for the `api:` block, which defines how the benchmark talks to the
model server: the API surface it targets (text completion vs. chat), whether
responses stream, custom HTTP headers, per-request SLO evaluation, and structured
output. When SLO headers are present, each request is checked for SLO compliance
and SLO-related metrics are reported.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md).

Schema: [`config.py`](../inference_perf/config/apis/config.py).

### API types

`api.type` selects which API surface the benchmark calls.

| `type` | Endpoint | When to use |
| --- | --- | --- |
| `completion` (default) | Text completion API | Default. Works with minimal server config; use for raw prompt/completion benchmarking. |
| `chat` | Chat completions API | Use for chat-formatted (multi-turn / role-based) workloads. May require extra server configuration (chat template, etc.). |

### Top-level `api` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `completion` | API surface to target (see [API types](#api-types)). |
| `streaming` | bool | `false` | Stream the response. Required to measure TTFT, ITL, and TPOT. |
| `headers` | dict[str, str] | `null` | Custom HTTP headers sent with every request. |
| `slo_unit` | str | `null` | Unit for SLO header values (e.g. `ms`, `s`). |
| `slo_tpot_header` | str | `null` | Name of the header carrying the per-request TPOT SLO. |
| `slo_ttft_header` | str | `null` | Name of the header carrying the per-request TTFT SLO. |
| `response_format` | object | `null` | Structured-output spec (see [Response format](#response-format)). |

#### When to set each field

- **`streaming`**: set to `true` whenever you need token-timing metrics (TTFT,
  ITL, TPOT). Leave `false` for end-to-end latency only.
- **`headers`**: use for routing hints (e.g. model selection, routing strategy)
  or to carry SLO values that the SLO headers below reference.
- **`slo_unit` / `slo_tpot_header` / `slo_ttft_header`**: set these together when
  you want per-request SLO compliance evaluated and reported. The header names
  must match the keys you put in `headers`.

### Response format

`api.response_format` requests structured output from the server (vLLM/OpenAI
compatible). See the
[vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `json_schema` | `json_schema` (constrain output to a schema) or `json_object` (any valid JSON object). |
| `name` | str | `structured_output` | Name for the JSON schema (used when `type` is `json_schema`). |
| `json_schema` | dict | `null` | The JSON schema the output must conform to (used when `type` is `json_schema`). |

When to use each `type`:

- **`json_schema`**: constrain output to a specific schema. Provide `json_schema`
  (and optionally `name`).
- **`json_object`**: require any syntactically valid JSON object without
  constraining its shape. `name` and `json_schema` are ignored.

### Example

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


## Data Generation (`data`)

Configuration for the `data:` block, which defines what requests the benchmark
sends to the model server: where prompts come from (synthetic, a HuggingFace
dataset, or a replayed trace), how long inputs and outputs are, and any
multimodal media (images, video, audio) attached to each request.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md). For the
two replay data types, see
[docs/conversation_replay.md](conversation_replay.md) and
[docs/otel_trace_replay.md](otel_trace_replay.md).

Schemas: [`config.py`](../inference_perf/config/datagen/config.py) (data types, distributions),
[`multimodal.py`](../inference_perf/config/datagen/multimodal.py) (media), [`replay.py`](../inference_perf/config/datagen/replay.py) (trace and
conversation replay).

### Data types

`data.type` selects where prompts come from. The chosen type determines which of
the other `data` fields apply.

| `type` | Source | When to use |
| --- | --- | --- |
| `mock` (default) | Built-in placeholder prompts | Smoke tests and plumbing checks; no dataset or distribution needed. |
| `synthetic` | Tokens generated from `input_distribution` / `output_distribution` | Control input/output token lengths precisely; the only type (with `shared_prefix`) that takes a `multimodal` block. |
| `random` | Randomly sampled tokens, optionally driven by a trace file | Distribution-controlled load, or replay request lengths/timings from an Azure-format trace via `trace`. |
| `shared_prefix` | Generated groups that share a system-prompt prefix | Exercise prefix caching / KV reuse; see [`shared_prefix`](#shared-prefix). |
| `shareGPT` | The ShareGPT dataset on disk | Realistic conversational prompts; set `path` to the downloaded dataset. |
| `cnn_dailymail` | CNN/DailyMail dataset | Summarization-style prompts; set `path`. |
| `infinity_instruct` | Infinity-Instruct dataset | Instruction-following prompts; set `path`. |
| `billsum_conversations` | BillSum conversations dataset | Long-document / legal-summary prompts; set `path`. |
| `conversation_replay` | In-memory synthetic multi-turn conversations | Agentic / multi-turn benchmarks generated from distributions; see [`conversation_replay`](#conversation-replay). |
| `otel_trace_replay` | Recorded OpenTelemetry trace spans | Replay real recorded session traffic; see [`otel_trace_replay`](#otel-trace-replay). |

### Top-level `data` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `mock` | Data generation type (see [Data types](#data-types)). |
| `path` | str | `null` | On-disk dataset path. Required for `shareGPT`, `cnn_dailymail`, `infinity_instruct`, and `billsum_conversations`. |
| `input_distribution` | object | `null` | Prompt-length distribution. Used by `synthetic` / `random` (see [Distributions](#token-length-distributions)). |
| `output_distribution` | object | `null` | Generation-length distribution. Used by `synthetic` / `random`. |
| `shared_prefix` | object | `null` | Settings for the `shared_prefix` type (see [`shared_prefix`](#shared-prefix)). |
| `multimodal` | object | `null` | Media to attach per request. Valid for `synthetic` and `shared_prefix` (see [Multimodal](#multimodal-data-generation)). |
| `trace` | object | `null` | Trace file source. Supported for the `random` type (see [`trace`](#trace-file-random-type)). |
| `otel_trace_replay` | object | `null` | OTel trace replay settings (see [`otel_trace_replay`](#otel-trace-replay)). |
| `conversation_replay` | object | `null` | Conversation replay settings (see [`conversation_replay`](#conversation-replay)). |

```yaml
data:
  type: synthetic
  input_distribution:
    min: 10
    max: 100
    mean: 50
    std_dev: 10
    total_count: 100
  output_distribution:
    min: 10
    max: 100
    mean: 50
    std_dev: 10
    total_count: 100
```

### Token-length distributions

`input_distribution`, `output_distribution`, and the `*_len` fields under
`shared_prefix` and `conversation_replay` all share the same `Distribution`
model. It controls how token counts are sampled per request.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `normal` | Sampling distribution: `normal`, `skew_normal`, `lognormal`, `uniform`, `poisson`, or `fixed`. |
| `min` | int | `10` | Lower clamp on sampled length. Must be `<= max`. |
| `max` | int | `1024` | Upper clamp on sampled length. |
| `mean` | float | `512` | Distribution mean. |
| `std_dev` | float | `200` | Standard deviation. Mutually exclusive with `variance`. |
| `variance` | float | `null` | Alternative to `std_dev` (`std_dev = sqrt(variance)`). Setting both is an error. |
| `skew` | float | `0.0` | Shape parameter, only used when `type: skew_normal`. |
| `total_count` | int | `null` | Total number of prompts to generate from this distribution. |

```yaml
data:
  type: synthetic
  input_distribution:
    type: lognormal
    min: 10
    max: 1024
    mean: 256
    std_dev: 64
    total_count: 500
  output_distribution:
    type: uniform
    min: 16
    max: 256
    total_count: 500
```

### Shared prefix

For `type: shared_prefix`. Generates `num_groups` groups, each sharing one
system-prompt prefix, with `num_prompts_per_group` distinct questions per group.
The `*_len` fields accept either a plain integer or an inline `Distribution`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_groups` | int | `10` | Number of shared-prefix groups (alias `num_unique_system_prompts`). |
| `num_prompts_per_group` | int | `10` | Distinct questions per group (alias `num_users_per_system_prompt`). |
| `system_prompt_len` | int or Distribution | `100` | Shared prefix length in tokens. |
| `question_len` | int or Distribution | `50` | Question length in tokens. |
| `output_len` | int or Distribution | `50` | Output length in tokens. |
| `seed` | int | `null` | Random seed for deterministic generation. |
| `enable_multi_turn_chat` | bool | `false` | Generate multi-turn chats instead of single-turn prompts. |
| `multimodal` | object | `null` | Media to attach per request (see [Multimodal](#multimodal-data-generation)). |
| `question_distribution` | object | `null` | Legacy: distribution for question lengths. Prefer an inline distribution on `question_len`. |
| `output_distribution` | object | `null` | Legacy: distribution for output lengths. Prefer an inline distribution on `output_len`. |

Specifying both an inline `Distribution` on `question_len` and the legacy
`question_distribution` (or the `output_len` pair) is rejected at config load.

```yaml
data:
  type: shared_prefix
  shared_prefix:
    num_groups: 10
    num_prompts_per_group: 10
    system_prompt_len: 100
    question_len: { type: normal, min: 10, max: 1024, mean: 50, std_dev: 5 }
    output_len: 50
```

### Trace file (random type)

For `type: random`, `trace` replays request lengths and timings from a trace
file instead of sampling distributions.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | str | (required) | Path to the trace file. |
| `format` | enum | `AzurePublicDataset` | Trace format. Currently only `AzurePublicDataset`. |

```yaml
data:
  type: random
  trace:
    file: ./traces/azure_llm_inference.csv
    format: AzurePublicDataset
```

### Multimodal data generation

For VLMs, the `synthetic` and `shared_prefix` data types accept an optional
`multimodal` block that produces images, video, and/or audio alongside the text
prompt. Each of the three modalities is independently optional.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | object | `null` | Image generation settings (see below). |
| `video` | object | `null` | Video generation settings (see below). |
| `audio` | object | `null` | Audio generation settings (see below). |

#### Common per-modality fields

`image`, `video`, and `audio` all share these:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | Distribution | `null` | Distribution of the number of media items per request. |
| `insertion_point` | float or Distribution | `null` | Placement within the text prompt: float in `[0.0, 1.0]` (`0`=start, `1`=end), or a `Distribution` to sample from. |

#### Image

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `resolutions` | resolution or list[WeightedResolution] | `null` | A single resolution or a weighted list. A resolution is a preset string (`4k`, `1080p`, `720p`, `360p`) or an explicit `{ height, width }`. |
| `representation` | enum | `png` | Wire encoding: `png` (lossless) or `jpeg` (lossy, smaller). |

A `WeightedResolution` is `{ resolution, weight }` where `weight` (default `1.0`)
is the relative selection frequency.

#### Video

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `profiles` | profile or list[WeightedVideoProfile] | `null` | A single `VideoProfile` or a weighted list. A `VideoProfile` is `{ resolution, frames }` (`frames` is required). |
| `representation` | enum | `mp4` | Wire-format strategy: `mp4`, `png_frames`, or `jpeg_frames` (see [Wire formats](#wire-formats)). |

A `WeightedVideoProfile` is `{ profile, weight }` (`weight` default `1.0`).

#### Audio

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `durations` | float or list[WeightedDuration] | `null` | A single clip length in seconds or a weighted list. A `WeightedDuration` is `{ duration, weight }` (`weight` default `1.0`). |

```yaml
data:
  type: synthetic
  input_distribution: { min: 10, max: 100, mean: 50, std_dev: 10, total_count: 100 }
  output_distribution: { min: 10, max: 100, mean: 50, std_dev: 10, total_count: 100 }
  multimodal:
    image:
      count: { type: uniform, min: 1, max: 2, mean: 1.5 }  # items per request
      insertion_point: 0.0                                  # 0=prefix, 1=suffix, or a Distribution
      resolutions:
        - resolution: "1080p"
          weight: 0.8
        - resolution: "4k"
          weight: 0.2
    video:
      count: { type: fixed, min: 1, max: 1, mean: 1 }
      insertion_point: 1.0
      profiles:
        - profile: { resolution: "720p", frames: 32 }
          weight: 1.0
    audio:
      count: { type: fixed, min: 1, max: 1, mean: 1 }
      insertion_point: 0.5
      durations:
        - duration: 5.0
          weight: 1.0
```

The reportgen output adds `throughput.{images,videos,audios}_per_sec`,
`request_size_bytes`, and per-modality distribution blocks
(`image.{count,pixels,bytes,aspect_ratio}`,
`video.{count,frames,pixels,bytes,aspect_ratio}`, `audio.{count,seconds,bytes}`)
to `summary_lifecycle_metrics.json`.

#### Wire formats

- **Images** are PNG by default. Set `image.representation: jpeg` for
  JPEG-encoded payloads (smaller, lossy): useful when the target VLM expects
  JPEG or when you want wire-size closer to real client traffic.
- **Videos** carry one `video.representation` value:
  - `mp4` (default): one `video_url` block carrying an MP4 blob. Measures the
    full pipeline including server-side decode.
  - `png_frames`: emit `frames` x PNG `image_url` blocks at one insertion point.
    No server-side MP4 decode dependency; useful for prefix-cache benchmarks and
    servers that do not accept `video_url`.
  - `jpeg_frames`: same as `png_frames` but with JPEG-encoded frames. Smaller
    wire payload, matches client pipelines that pre-extract and JPEG-compress
    frames before sending to the model server.
- **Audio** is 16-bit mono WAV at 16 kHz (not configurable today).

#### Picking values

Multimodal config is passed to the model server **as-is** with no model-aware
validation, so the right starting point is the model's spec sheet. Out-of-range
payloads typically fail at the wire (counted in the report's `failures`) or get
silently downsized server-side, so per-modality byte/pixel numbers reflect what
was sent, not what the model processed.

- **Per-request media count**: align with vLLM's `--limit-mm-per-prompt` (e.g.
  `image=4,video=2,audio=2`). Sending more items than the server allows fails at
  the wire and shows up in the report's `failures` count.
- **Image resolutions**: stay within the vision encoder's pixel cap (Qwen2-VL
  `min_pixels`/`max_pixels`, LLaVA fixed 336/672, Pixtral tile caps, etc.).
  Above-cap images get downsized server-side.
- **Video frames**: most VLMs sample to a fixed `num_frames` budget (often
  8/16/32). Sending more frames than the server samples wastes wire bytes
  without changing the workload the model sees. If the server does not accept
  `video_url`, switch to `representation: png_frames` or `jpeg_frames`.
- **Audio durations**: most audio-capable models cap clips around 30 s
  (Qwen2-Audio, Whisper-style chunking). Longer clips fail or get truncated.
- **Effective context length**: images and audio consume context tokens. Large
  media plus a long text prompt can exceed `--max-model-len` and fail.

When in doubt: start small, watch the `failures` count in the report, and ramp
up resolutions/counts/durations once the success path is solid.

### Conversation replay

For `type: conversation_replay`. Generates synthetic multi-turn conversations
in-memory from distributions: each conversation has a two-part system prompt
(shared prefix plus a per-conversation dynamic suffix) and a sequence of
user/assistant turns. Full workflow and examples:
[docs/conversation_replay.md](conversation_replay.md).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int | `42` | Random seed for deterministic generation. |
| `num_conversations` | int > 0 | `200` | Number of conversation blueprints to generate. |
| `shared_system_prompt_len` | int >= 0 | `8359` | Fixed shared system-prompt length in tokens. |
| `dynamic_system_prompt_len` | Distribution | `null` | Per-conversation dynamic system-prompt length. |
| `turns_per_conversation` | Distribution | `null` | Number of turns per conversation. |
| `input_tokens_per_turn` | Distribution | `null` | Input tokens per turn. |
| `output_tokens_per_turn` | Distribution | `null` | Output tokens per turn. |
| `tool_call_latency_sec` | Distribution | `null` | Per-turn tool-execution latency (seconds). When set, each turn sleeps the sampled duration after inference, holding the session lock so the GPU serves other conversations. Omit for pure GPU throughput. `min`/`max` are whole seconds; `mean`/`std_dev` may be fractional. |
| `max_model_len` | int | `null` | Maximum model context length in tokens. |

### OTel trace replay

For `type: otel_trace_replay`. Replays recorded OpenTelemetry trace spans as
sessions. Exactly one trace source (`trace_directory`, `trace_files`, or
`hf_dataset_path`) must be provided. Full workflow and trace format:
[docs/otel_trace_replay.md](otel_trace_replay.md).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `trace_directory` | str | `null` | Directory of OTel JSON trace files. |
| `trace_files` | list[str] | `null` | Specific OTel JSON trace file paths. |
| `hf_dataset_path` | str or object | `null` | HuggingFace dataset: `"user/dataset"`, or `{ path, revision, split }`. |
| `use_static_model` | bool | `false` | Route all requests to a single static model. |
| `static_model_name` | str | `""` | Static model name. Required when `use_static_model` is `true`. |
| `model_mapping` | object | `null` | Map recorded model names to target models. |
| `default_max_tokens` | int > 0 | `1000` | `max_tokens` used when the trace does not specify one. |
| `inject_random_session_id` | bool | `false` | Inject a random string into unique segments to invalidate KV-cache between sessions. |
| `duplicate_sessions_target` | int > 0 | `null` | Duplicate existing sessions up to this count. `null` = no duplication. |
| `max_wait_ms` | int >= 0 | `15000` | Cap on inter-event wait time in ms, to avoid reproducing unusually long tool/agent execution times. |
| `include_errors` | bool | `true` | Include spans with error status. |
| `skip_invalid_files` | bool | `false` | Skip invalid trace files instead of failing. |


## Load Generation (`load`)

Configuration for the `load:` block, which defines the request pattern the
benchmark drives at the model server: the load type, the per-stage schedule, and
the worker pool that issues requests.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md). For how
the load generator behaves at runtime (scheduling, stage lifecycle), see
[docs/loadgen.md](loadgen.md).

Schema: [`config.py`](../inference_perf/config/loadgen/config.py).

### Load types

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
[docs/otel_trace_replay.md](otel_trace_replay.md); the rest are
covered below.

### Top-level `load` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `constant` | Load pattern (see [Load types](#load-types)). |
| `interval` | float | `1.0` | Seconds between request batches within a stage. |
| `stages` | list | `[]` | Ordered load stages; shape depends on `type`. |
| `sweep` | object | `null` | Auto-derive stages from a saturation search (see [Sweeps](#load-sweeps)). |
| `num_workers` | int | CPU core count | Number of worker **processes** (see [Worker model](#worker--multiprocessing-model)). |
| `worker_max_concurrency` | int | `100` | Max in-flight requests per worker. |
| `worker_max_tcp_connections` | int | `2500` | Max TCP connections per worker. |
| `trace` | object | `null` | Trace source for `trace_replay` / `trace_session_replay`. |
| `circuit_breakers` | list[str] | `[]` | Named circuit breakers to abort a run on breach (see [docs/goodput.md](goodput.md)). |
| `request_timeout` | float | `null` | Per-request timeout in seconds. |
| `lora_traffic_split` | list | `null` | MultiLoRA traffic weights (see [LoRA traffic split](#lora-traffic-split)). |
| `base_seed` | int | current time (ms) | Base random seed; set explicitly for reproducible runs. |

### Stages

Each entry in `load.stages` is one phase of the run. Stages execute in order.
The required fields depend on `load.type`.

#### Standard stages (`constant`, `poisson`, `trace_replay`)

| Field | Type | Description |
| --- | --- | --- |
| `rate` | float > 0 | Target request rate in QPS. |
| `duration` | int > 0 | Seconds to hold this rate. |

```yaml
load:
  type: constant
  stages:
    - rate: 1
      duration: 30
    - rate: 5
      duration: 30
```

#### Concurrent stages (`concurrent`)

A concurrent stage holds a fixed number of in-flight requests rather than a
rate. `rate`/`duration` are set internally at runtime and must not be configured.

| Field | Type | Description |
| --- | --- | --- |
| `num_requests` | int > 0 | Total requests to send in the stage. |
| `concurrency_level` | int > 0 | Number of requests kept in flight at once. |

```yaml
load:
  type: concurrent
  stages:
    - concurrency_level: 8
      num_requests: 400
```

#### Trace session replay stages (`trace_session_replay`)

| Field | Type | Description |
| --- | --- | --- |
| `concurrent_sessions` | int ≥ 0 | Max sessions active at once. `0` = all at once (stress mode). |
| `session_rate` | float > 0, optional | Sessions started per second. Must not exceed `concurrent_sessions`. |
| `num_sessions` | int > 0, optional | Sessions drawn from the corpus this stage; `null` = all remaining. |
| `timeout` | float > 0, optional | Wall-clock safety limit; on breach, in-flight sessions are cancelled and the stage is marked `FAILED`. |

See [docs/otel_trace_replay.md](otel_trace_replay.md) for the full
workflow and trace format.

### Worker & multiprocessing model

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

### Load sweeps

Instead of hand-writing stages, a sweep runs a preprocessing phase that searches
for the service's saturation point and generates stages around it. Only valid
with `constant` / `poisson` (not `concurrent` or `trace_session_replay`).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | (required) | `linear` (evenly spaced 1 to saturation) or `geometric` (clustered near saturation). |
| `num_requests` | int | `2000` | Requests used to probe saturation. |
| `timeout` | float | `60` | Seconds to run the saturation probe. |
| `num_stages` | int | `5` | Number of stages to generate. |
| `stage_duration` | int | `180` | Duration of each generated stage. |
| `saturation_percentile` | float | `95` | Percentile of sampled rates taken as the saturation point. |

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

### LoRA traffic split

Splits traffic across multiple LoRA adapters. Weights must sum to `1.0`.

| Field | Type | Description |
| --- | --- | --- |
| `name` | str | Adapter name as registered on the model server. |
| `split` | float | Fraction of traffic routed to this adapter. |

```yaml
load:
  lora_traffic_split:
    - name: adapter_1
      split: 0.5
    - name: adapter_2
      split: 0.5
```


## Model Server (`server`)

Configuration for the `server:` block, which defines how the benchmark connects
to the model serving backend: the server flavor, the endpoint, the model to
target, and the credentials used to reach it.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md).

Schema: [`config.py`](../inference_perf/config/client/modelserver/config.py).

### Server types

`server.type` selects the backend flavor the client speaks to.

| `type` | Backend |
| --- | --- |
| `vllm` (default) | vLLM OpenAI-compatible server |
| `sglang` | SGLang server |
| `tgi` | Text Generation Inference server |
| `mock` | In-process mock server for testing without a real backend |

#### When to use

- **`vllm`**: Default. Use for benchmarking a vLLM deployment over its
  OpenAI-compatible endpoint.
- **`sglang`**: Use when driving an SGLang server.
- **`tgi`**: Use when driving a Hugging Face Text Generation Inference server.
- **`mock`**: Use for local development and CI, when you want to exercise the
  load generator and reporting paths without standing up a real backend.

### Top-level `server` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `vllm` | Backend flavor (see [Server types](#server-types)). |
| `model_name` | str | `null` | Model identifier the server should serve (for example, a Hugging Face repo ID). |
| `base_url` | str | (required) | Server endpoint, including scheme, host, and port. |
| `ignore_eos` | bool | `true` | Whether to ignore End-of-Sequence tokens so generation runs to the requested length. |
| `api_key` | str | `null` | API key for authenticated endpoints. |
| `cert_path` | str | `null` | Path to a client TLS certificate for mutual-TLS endpoints. |
| `key_path` | str | `null` | Path to the private key paired with `cert_path`. |

`base_url` is the only required field. `cert_path` and `key_path` are used
together to present a client certificate to endpoints that require mutual TLS.

### Example

```yaml
server:
  type: vllm
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  base_url: "http://0.0.0.0:8000"
  ignore_eos: true
  api_key: ""
```


## Metrics (`metrics`)

Configuration for the `metrics:` block, which selects the server-side metrics
client the benchmark uses to scrape model-server performance metrics (for
example vLLM Prometheus counters) alongside the client-side request stats that
inference-perf records on its own.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md). For the
definitions of the metrics themselves (throughput, latency, formulas), see
[docs/metrics.md](metrics.md).

Schema: [`config.py`](../inference_perf/config/metrics/config.py).

### Metrics client types

`metrics.type` selects the backend used to collect server-side metrics.

| `type` | Behavior | Requires `prometheus` block |
| --- | --- | --- |
| `default` | No external metrics client; the benchmark relies only on client-side request stats. | No |
| `prometheus` | Queries a Prometheus endpoint for server-reported metrics and folds them into the report. | Yes |

### Top-level `metrics` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | (required) | Metrics client: `default` or `prometheus` (see [client types](#metrics-client-types)). |
| `prometheus` | object | `null` | Prometheus client settings; required when `type: prometheus` (see [Prometheus config](#prometheus-config)). |

### Prometheus config

The `prometheus` sub-block configures the Prometheus query client (schema:
[`../client/server_metrics/config.py`](../inference_perf/config/client/server_metrics/config.py)).
Exactly one of `url` or `google_managed` must be set: pick a direct Prometheus
URL, or enable Google Managed Prometheus (which is queried through the GMP API
instead of a URL). Setting both, or neither, is a validation error.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | URL | `null` | Prometheus server URL to query (for example `http://localhost:9090`). Mutually exclusive with `google_managed`. |
| `scrape_interval` | int | `15` | Metrics scrape interval in seconds; should match the server's scrape interval. |
| `google_managed` | bool | `false` | Query Google Managed Prometheus (GMP) via its API instead of a URL. Mutually exclusive with `url`. |
| `filters` | list[str] | `[]` | Metric names to collect; empty collects the default set. |

When `google_managed: true`, the run requires Application Default Credentials
with the `roles/monitoring.viewer` role. See the "Google Managed Prometheus
(GMP) Requirements" section in [docs/config.md](config.md) for the
full permission and environment setup.

### When to use

- **`default`**: use when you only need inference-perf's own client-side request
  metrics (throughput, TTFT, end-to-end latency) and have no Prometheus endpoint
  to scrape. Simplest setup, no external dependency.
- **`prometheus` with `url`**: use when the model server (or a sidecar) exposes a
  reachable Prometheus endpoint and you want server-reported metrics in the
  report. Standard self-hosted Prometheus.
- **`prometheus` with `google_managed: true`**: use on GKE or GCE where metrics
  flow into Google Managed Prometheus and there is no direct Prometheus URL to
  hit.

### Examples

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


## Reporting (`report`)

Configuration for the `report:` block, which controls which reports the benchmark
generates after a run and at what granularity (summary, per-stage, per-request,
and per-adapter breakdowns). Each report family is independently toggleable, so
you can keep the high-level summary cheap while opting into verbose per-request
output only when needed.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md). For the
shape and contents of the generated report files, see
[docs/reports.md](reports.md). For goodput constraints and how they
are evaluated, see [docs/goodput.md](goodput.md).

Schema: [`config.py`](../inference_perf/config/reportgen/config.py).

### Top-level `report` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `request_lifecycle` | object | enabled (all defaults) | Latency/throughput metrics derived from request lifecycle timings (see [request_lifecycle](#request_lifecycle)). |
| `prometheus` | object | enabled (all defaults) | Server-side metrics scraped from the model server's Prometheus endpoint (see [prometheus](#prometheus)). Set to `null` to disable. |
| `session_lifecycle` | object | enabled (all defaults) | Multi-turn session metrics for session-based load (see [session_lifecycle](#session_lifecycle)). |
| `goodput` | object | `null` (disabled) | Goodput constraints; when set, requests are scored against per-metric thresholds (see [goodput](#goodput)). |

### request_lifecycle

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

### prometheus

Server-side metrics scraped from the model server. Set the whole `prometheus`
key to `null` to skip Prometheus reporting entirely.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool | `true` | Include the aggregate Prometheus metrics summary. |
| `per_stage` | bool | `false` | Include a Prometheus breakdown per load stage. |

### session_lifecycle

Metrics for multi-turn session load (for example `trace_session_replay`).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool | `true` | Generate the aggregate session summary. |
| `per_stage` | bool | `true` | Include a breakdown per load stage. |
| `per_session` | bool | `false` | Emit detailed per-session records (verbose). |

### goodput

Defines pass/fail constraints used to compute goodput. Disabled by default
(`goodput` is `null`); set it to enable scoring. See
[docs/goodput.md](goodput.md) for the metric names and semantics.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `constraints` | dict[str, float] | `{}` | Map of metric name to threshold value; a request counts as "good" when it satisfies all constraints. |

### Example

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


## Storage (`storage`)

Configuration for the `storage:` block, which controls where benchmark report
files are written. Reports can be persisted to a local directory, to Google
Cloud Storage (GCS), or to an S3 / S3-compatible object store. Multiple backends
can be enabled at once; local storage is always available.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md).

Schema: [`config.py`](../inference_perf/config/client/filestorage/config.py).

### Backends

| Key | Backend | Enabled by default |
| --- | --- | --- |
| `local_storage` | Local filesystem directory | Yes (always on) |
| `google_cloud_storage` | Google Cloud Storage bucket | No (`null` unless set) |
| `simple_storage_service` | AWS S3 or S3-compatible store | No (`null` unless set) |

All three backends share two base fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | `reports-{timestamp}` | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted `reports-YYYYMMDD-HHMMSS`. |
| `report_file_prefix` | str | `null` | Optional prefix prepended to each report filename. |

### `local_storage`

Writes reports to a directory on the machine running the benchmark. This backend
is always active, so reports are written locally even when no other backend is
configured.

**When to use:** default for local runs and quick iteration; no external
credentials or buckets required.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | `reports-{timestamp}` | Local directory to write reports into. |
| `report_file_prefix` | str | `null` | Optional filename prefix. |

```yaml
storage:
  local_storage:
    path: "reports-{timestamp}"
    report_file_prefix: null
```

### `google_cloud_storage`

Uploads reports to a GCS bucket. `bucket_name` is required; `path` acts as a key
prefix within the bucket.

**When to use:** runs on GCP, or when reports should be centralized in a GCS
bucket for sharing and retention.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `bucket_name` | str | required | Target GCS bucket. |
| `path` | str | `reports-{timestamp}` | Key prefix within the bucket. |
| `report_file_prefix` | str | `null` | Optional filename prefix. |

```yaml
storage:
  google_cloud_storage:
    bucket_name: "your-bucket-name"
    path: "reports-{timestamp}"
    report_file_prefix: null
```

### `simple_storage_service`

Uploads reports to AWS S3 or any S3-compatible object store. `bucket_name` is
required; the remaining fields target custom endpoints and addressing schemes.

**When to use:** runs on AWS, or against an S3-compatible store (set
`endpoint_url`). Use `addressing_style: path` for stores that do not support
virtual-hosted-style buckets.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `bucket_name` | str | required | Target S3 bucket. |
| `path` | str | `reports-{timestamp}` | Key prefix within the bucket. |
| `report_file_prefix` | str | `null` | Optional filename prefix. |
| `endpoint_url` | str | `null` | Custom endpoint URL for S3-compatible stores. |
| `region_name` | str | `null` | AWS region name. |
| `addressing_style` | enum | `null` | Bucket addressing: `auto`, `virtual`, or `path`. |

```yaml
storage:
  simple_storage_service:
    bucket_name: "your-bucket-name"
    path: "reports-{timestamp}"
    report_file_prefix: null
    endpoint_url: null
    region_name: null
    addressing_style: null
```


## Tokenizer (`tokenizer`)

Configuration for the `tokenizer:` block, which selects the tokenizer used for
token counting, exact-length prompt construction, and any data generator that
needs to encode or decode text. The tokenizer drives input/output token metrics
and lets synthetic generators hit precise prompt lengths.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](config.md).

Schema: [`config.py`](../inference_perf/config/utils/config.py).

### When a tokenizer is required

For most runs the tokenizer is **inferred** from the model server: the `vllm`,
`sglang`, and `tgi` clients derive it from `server.model_name`, so an explicit
`tokenizer:` block is optional. Set the block to override that choice (a
different tokenizer than the served model) or to supply credentials for a gated
model.

Some data generators require a usable tokenizer (inferred or explicit) and raise
at startup if one is unavailable:

- `random`
- `synthetic`
- `shared_prefix`
- `multimodal`
- `conversation_replay`
- `cnn_dailymail`
- `shareGPT` (for both `completion` and `chat` APIs)

### `CustomTokenizerConfig` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `pretrained_model_name_or_path` | str | `null` | HuggingFace model ID or local path the tokenizer is loaded from. Required to activate an explicit tokenizer; if unset, the tokenizer is inferred from the model server. |
| `trust_remote_code` | bool | `null` | Allow loading custom tokenizer code shipped with the model repo. Leave unset (treated as off) unless the model requires it. |
| `token` | str | `null` | HuggingFace access token for private or gated models. |

### Example

```yaml
tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
  trust_remote_code: true
  token: ""
```

<!-- END GENERATED: config subsystems -->

## Full Configuration Examples

### Minimal Configuration

```yaml
data:
  type: shareGPT
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api: 
  type: chat
server:
  type: vllm
  model_name: HuggingFaceTB/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
```

### Advanced Configuration

```yaml
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api: 
  type: completion
server:
  type: vllm
  model_name: HuggingFaceTB/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
data:
  type: random
  input_distribution:
    min: 10             # min length of the synthetic prompts
    max: 100            # max length of the synthetic prompts
    mean: 50            # mean length of the synthetic prompts
    std_dev: 10         # standard deviation of the length of the synthetic prompts
    total_count: 100    # total number of prompts to generate to fit the above mentioned distribution constraints
  output_distribution:
    min: 10             # min length of the output to be generated
    max: 100            # max length of the output to be generated
    mean: 50            # mean length of the output to be generated
    std_dev: 10         # standard deviation of the length of the output to be generated
    total_count: 100    # total number of output lengths to generate to fit the above mentioned distribution constraints
metrics:
  type: prometheus
  prometheus:
    url: http://localhost:9090
    scrape_interval: 15
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
  prometheus:
    summary: true
    per_stage: true
```

### To Run Inference Perf Offline

```yaml
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api:
  type: chat
server:
  type: vllm
  model_name: ./models/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: ./models/SmolLM2-135M-Instruct
data:
  type: shareGPT
  path: ./data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json # path to the downloaded shareGPT dataset
metrics:
  type: prometheus
  prometheus:
    url: http://localhost:9090
    scrape_interval: 15
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: false
  prometheus:
    summary: true
    per_stage: true
```

## Advanced Use Cases

### OpenTelemetry Trace Replay

Replay real-world LLM workloads captured as OpenTelemetry traces. This feature enables benchmarking with production traffic patterns, including complex dependency graphs, multi-turn conversations, and agent workflows.

#### Overview

OTel trace replay reconstructs the original call graph from trace files, preserving:
- **Sequential dependencies** — requests that must wait for predecessors
- **Parallel fan-outs** — concurrent requests with no dependencies
- **Shared-prefix patterns** — requests sharing common message history (KV-cache opportunities)
- **Output-aware replay** — substitutes recorded assistant messages with actual generated text for realistic growing-context behavior

#### How It Works

1. **Trace → Replay Graph**: Each trace file is converted to a directed acyclic graph (DAG) where:
   - LLM spans become nodes
   - Dependencies are inferred from message content (assistant messages matching predecessor outputs)
   - Timing gaps between calls are preserved as `wait_ms` delays

2. **Session-Based Execution**: Each trace file represents one *session*. The load generator controls:
   - How many sessions run concurrently (`concurrent_sessions`)
   - How many sessions to process per stage (`num_sessions`)
   - Optional rate limiting for session starts (`session_rate`)

3. **Output Substitution**: When a request depends on a predecessor's output, the recorded assistant message is replaced at runtime with the actual generated text, ensuring realistic KV-cache behavior for multi-turn conversations and agent chains.

#### Configuration

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    # Source — specify one:
    trace_files:                                  # List of specific trace files
      - "path/to/trace1.json"
      - "path/to/trace2.json"
    trace_directory: "path/to/traces/"            # OR: all .json files in directory

    # Model configuration
    use_static_model: true                        # Override recorded model names
    static_model_name: "my-model"                 # Model to use for all requests
    model_mapping:                                # OR: remap per recorded name
      "gpt-4": "my-model"
      "gpt-3.5-turbo": "my-other-model"

    # Generation parameters
    default_max_tokens: 1000                      # Fallback if output tokens are set to 0 in the otel file

    # Error handling
    include_errors: false                         # Skip spans with error status, that is, status != 0 (default)
    skip_invalid_files: true                      # Skip unparseable trace files during replay

load:
  type: trace_session_replay                      # Required for otel_trace_replay
  stages:
    - concurrent_sessions: 4                      # Max sessions active simultaneously
      num_sessions: 20                            # Run 20 sessions in this stage
      session_rate: 2.0                           # Optional: start max 2 sessions/sec
      timeout: 300                                # Optional: stage timeout in seconds
  num_workers: 4                                  # Worker processes
  worker_max_concurrency: 10                      # Max concurrent requests per worker
```

#### Stage Configuration

**`concurrent_sessions`** (required): Controls session-level concurrency
- `0` = unlimited (all sessions active at once, stress test mode)
- `N > 0` = at most N sessions active; when one completes, the next starts

**`num_sessions`** (optional): Number of sessions to run in this stage
- If omitted, runs all remaining sessions in the corpus
- Stages advance through the corpus sequentially (like standard load stages)

**`session_rate`** (optional): Rate limit for starting new sessions
- Omit for no rate limiting
- Useful for controlled ramp-up scenarios

**`timeout`** (optional): Wall-clock safety limit
- If exceeded, in-flight sessions are cancelled and stage exits as FAILED

#### Trace File Format

Traces must be JSON files with a `spans` array. Each LLM span requires:

```json
{
  "span_id": "unique-id",
  "trace_id": "trace-id",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T00:00:01Z",
  "name": "chat gpt-4",
  "attributes": {
    "gen_ai.request.model": "gpt-4",
    "gen_ai.input.messages": "[{\"role\":\"user\",\"content\":\"hello\"}]",
    "gen_ai.output.text": "hi there",
    "gen_ai.usage.prompt_tokens": 10,
    "gen_ai.usage.completion_tokens": 5
  }
}
```

Token counts are read from `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens` (also accepts `input_tokens` / `output_tokens`). If absent, a 4 chars/token estimate is used.

#### Example Configurations

**Simple Sequential Replay:**
```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_files:
      - "examples/otel/test_traces/simple/simple_chain.json"
    use_static_model: true
    static_model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
    default_max_tokens: 100

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 1  # One session at a time
  num_workers: 4
  worker_max_concurrency: 10

server:
  type: vllm
  base_url: "http://localhost:8000"
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

**Multi-Stage with Rate Limiting:**
```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "examples/otel/test_traces/advanced"
    use_static_model: true
    static_model_name: "my-model"

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 2
      num_sessions: 10
      session_rate: 1.0      # Warm-up: 2 concurrent, 1/sec start rate
    - concurrent_sessions: 5
      num_sessions: 20
      session_rate: 2.0      # Ramp-up: 5 concurrent, 2/sec start rate
    - concurrent_sessions: 10
                             # Final stage: 10 concurrent, all remaining sessions
  num_workers: 8
  worker_max_concurrency: 20
```

#### Use Cases

- **Production Traffic Replay**: Benchmark with real user interaction patterns
- **Agent Workflow Testing**: Replay complex multi-step agent traces with tool calls
- **Multi-Turn Conversation Analysis**: Test KV-cache efficiency with realistic conversation flows
- **Dependency Graph Validation**: Verify server behavior under complex request dependencies
- **Comparative Analysis**: Replay the same traces against different model configurations

#### Architecture Notes

Unlike standard data generators that produce independent requests, OTel trace replay operates at the session granularity. Each session is a complete trace file with an internal dependency graph. The load generator:

1. Activates sessions according to `concurrent_sessions` limit
2. Dispatches all events for a session immediately (workers handle internal parallelism)
3. Each event blocks until its predecessors complete and outputs are available
4. Tracks session completion and starts new sessions as slots become available

This design preserves the causal structure of the original workload while allowing the load generator to control session-level concurrency and throughput.

## Google Managed Prometheus (GMP) Requirements

When setting `google_managed: true`, `inference-perf` queries the GMP API directly. You must configure [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) in your environment with sufficient permissions.

1. **Required Permissions**
   The identity used by ADC must have the Monitoring Viewer role:
   * `roles/monitoring.viewer`

2. **Environment Configuration**
   * **GKE Cluster:** Ensure the Pod is running with [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) enabled and linked to a Google Service Account (GSA) with the required role.
   * **GCE VM:** Ensure the [VM's attached Service Account](https://cloud.google.com/compute/docs/access/service-accounts#associating_a_service_account_to_an_instance) has the required role.
   * **Local Development:** Authenticate using your user credentials:
     ```bash
     gcloud auth application-default login
     ```
     > **Note:** Your personal user account must have the `monitoring.viewer` role on the target GCP project.

**Common Error:**
Failing to configure these permissions will result in API errors similar to:
```text
ERROR - error executing query: 403 Client Error: Forbidden for url: [https://monitoring.googleapis.com/v1/projects/](https://monitoring.googleapis.com/v1/projects/)...
```
