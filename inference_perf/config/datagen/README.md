# Data Generation Configuration

Configuration for the `data:` block, which defines what requests the benchmark
sends to the model server: where prompts come from (synthetic, a HuggingFace
dataset, or a replayed trace), how long inputs and outputs are, and any
multimodal media (images, video, audio) attached to each request.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md). For the
two replay data types, see
[docs/conversation_replay.md](../../../docs/conversation_replay.md) and
[docs/otel_trace_replay.md](../../../docs/otel_trace_replay.md).

Schemas: [`config.py`](./config.py) (data types, distributions),
[`multimodal.py`](./multimodal.py) (media), [`replay.py`](./replay.py) (trace and
conversation replay).

## Data types

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

## Top-level `data` fields

<!-- FIELDS: DataConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `mock` | Data generation type (see Data types). |
| `path` | str | `null` | On-disk dataset path. Required for shareGPT, cnn_dailymail, infinity_instruct, and billsum_conversations. |
| `input_distribution` | Distribution | `null` | Prompt-length distribution. Used by synthetic / random. |
| `output_distribution` | Distribution | `null` | Generation-length distribution. Used by synthetic / random. |
| `shared_prefix` | SharedPrefix | `null` | Settings for the shared_prefix type. |
| `multimodal` | SyntheticMultimodalDatagenConfig | `null` | Media to attach per request. Valid for synthetic and shared_prefix. |
| `trace` | TraceConfig | `null` | Trace file source. Supported for the random type. |
| `otel_trace_replay` | OTelTraceReplayConfig | `null` | OTel trace replay settings. |
| `conversation_replay` | ConversationReplayConfig | `null` | Conversation replay settings. |

<!-- /FIELDS -->

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

## Token-length distributions

`input_distribution`, `output_distribution`, and the `*_len` fields under
`shared_prefix` and `conversation_replay` all share the same `Distribution`
model. It controls how token counts are sampled per request.

<!-- FIELDS: Distribution -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `min` | int | `10` | Lower clamp on sampled length. Must be <= max. |
| `max` | int | `1024` | Upper clamp on sampled length. |
| `mean` | float | `512` | Distribution mean. |
| `std_dev` | float | `200` | Standard deviation. Mutually exclusive with variance. |
| `total_count` | int | `null` | Total number of prompts to generate from this distribution. |
| `type` | enum | `normal` | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `variance` | float | `null` | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `skew` | float | `0.0` | Shape parameter, only used when type is skew_normal. |

<!-- /FIELDS -->

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

## Shared prefix

For `type: shared_prefix`. Generates `num_groups` groups, each sharing one
system-prompt prefix, with `num_prompts_per_group` distinct questions per group.
The `*_len` fields accept either a plain integer or an inline `Distribution`.

<!-- FIELDS: SharedPrefix -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_groups` | int | `10` | Number of shared-prefix groups (alias num_unique_system_prompts). |
| `num_prompts_per_group` | int | `10` | Distinct questions per group (alias num_users_per_system_prompt). |
| `system_prompt_len` | int or Distribution | `100` | Shared prefix length in tokens. |
| `question_len` | int or Distribution | `50` | Question length in tokens. |
| `output_len` | int or Distribution | `50` | Output length in tokens. |
| `seed` | int | `null` | Random seed for deterministic generation. |
| `question_distribution` | Distribution | `null` | Legacy: distribution for question lengths. Prefer an inline distribution on question_len. |
| `output_distribution` | Distribution | `null` | Legacy: distribution for output lengths. Prefer an inline distribution on output_len. |
| `enable_multi_turn_chat` | bool | `false` | Generate multi-turn chats instead of single-turn prompts. |
| `multimodal` | SyntheticMultimodalDatagenConfig | `null` | Media to attach per request. |

<!-- /FIELDS -->

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

## Trace file (random type)

For `type: random`, `trace` replays request lengths and timings from a trace
file instead of sampling distributions.

<!-- FIELDS: TraceConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | str | (required) | Path to the trace file. |
| `format` | enum | `AzurePublicDataset` | Trace format. Currently only AzurePublicDataset. |

<!-- /FIELDS -->

```yaml
data:
  type: random
  trace:
    file: ./traces/azure_llm_inference.csv
    format: AzurePublicDataset
```

## Multimodal data generation

For VLMs, the `synthetic` and `shared_prefix` data types accept an optional
`multimodal` block that produces images, video, and/or audio alongside the text
prompt. Each of the three modalities is independently optional.

<!-- FIELDS: SyntheticMultimodalDatagenConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | ImageDatagenConfig | `null` | Image generation settings. |
| `video` | VideoDatagenConfig | `null` | Video generation settings. |
| `audio` | AudioDatagenConfig | `null` | Audio generation settings. |

<!-- /FIELDS -->

### Common per-modality fields

`image`, `video`, and `audio` all share these:

<!-- FIELDS: MediaDatagenConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | Distribution | `null` | Distribution of the number of media items to generate per request. |
| `insertion_point` | float or Distribution | `null` | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |

<!-- /FIELDS -->

### Image

<!-- FIELDS: ImageDatagenConfig: resolutions, representation -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `resolutions` | enum or Resolution or list[WeightedResolution] | `null` | Resolution or list of weighted resolutions for generated images. |
| `representation` | enum | `png` | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |

<!-- /FIELDS -->

A `WeightedResolution` is `{ resolution, weight }` where `weight` (default `1.0`)
is the relative selection frequency. A resolution is a preset string (`4k`,
`1080p`, `720p`, `360p`) or an explicit `{ height, width }`.

### Video

<!-- FIELDS: VideoDatagenConfig: profiles, representation -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `profiles` | VideoProfile or list[WeightedVideoProfile] | `null` | Video profile or list of weighted video profiles for generated videos. |
| `representation` | enum | `mp4` | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |

<!-- /FIELDS -->

A `VideoProfile` is `{ resolution, frames }` (`frames` is required). A
`WeightedVideoProfile` is `{ profile, weight }` (`weight` default `1.0`).

### Audio

<!-- FIELDS: AudioDatagenConfig: durations -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `durations` | float or list[WeightedDuration] | `null` | Duration or list of weighted durations for generated audio clips. |

<!-- /FIELDS -->

A `WeightedDuration` is `{ duration, weight }` (`weight` default `1.0`).

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

### Wire formats

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

### Picking values

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

## Conversation replay

For `type: conversation_replay`. Generates synthetic multi-turn conversations
in-memory from distributions: each conversation has a two-part system prompt
(shared prefix plus a per-conversation dynamic suffix) and a sequence of
user/assistant turns. Full workflow and examples:
[docs/conversation_replay.md](../../../docs/conversation_replay.md).

<!-- FIELDS: ConversationReplayConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int | `42` | Random seed for deterministic generation |
| `num_conversations` | int > 0 | `200` | Number of conversation blueprints to generate |
| `shared_system_prompt_len` | int ≥ 0 | `8359` | Fixed shared system prompt length in tokens |
| `dynamic_system_prompt_len` | Distribution | `null` | Per-conversation dynamic system prompt length distribution |
| `turns_per_conversation` | Distribution | `null` | Number of turns per conversation distribution |
| `input_tokens_per_turn` | Distribution | `null` | Input tokens per turn distribution |
| `output_tokens_per_turn` | Distribution | `null` | Output tokens per turn distribution |
| `tool_call_latency_sec` | Distribution | `null` | Per-turn tool execution latency distribution in seconds. When set, each turn sleeps for the sampled duration after model inference completes and before the next turn begins, simulating tool call round-trips. The sleep holds the session lock so the GPU is free to serve other concurrent conversations — correctly modelling offline agentic workloads. Omit for pure GPU throughput measurement. Values are in seconds; min/max are whole seconds, mean/std_dev may be fractional. |
| `max_model_len` | int | `null` | Maximum model context length in tokens |

<!-- /FIELDS -->

## OTel trace replay

For `type: otel_trace_replay`. Replays recorded OpenTelemetry trace spans as
sessions. Exactly one trace source (`trace_directory`, `trace_files`, or
`hf_dataset_path`) must be provided. Full workflow and trace format:
[docs/otel_trace_replay.md](../../../docs/otel_trace_replay.md).

<!-- FIELDS: OTelTraceReplayConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `use_static_model` | bool | `false` | Use a single static model for all requests |
| `static_model_name` | str | `""` | Static model name (required if use_static_model=True) |
| `model_mapping` | dict | `null` | Map recorded model names to target models |
| `default_max_tokens` | int > 0 | `1000` | Default max_tokens if not specified in trace |
| `inject_random_session_id` | bool | `false` | Inject random string into unique segments to invalidate KV-cache between sessions |
| `duplicate_sessions_target` | int > 0 | `null` | Target number of sessions to reach by duplicating existing sessions. If None, no duplication occurs. |
| `max_wait_ms` | int ≥ 0 | `15000` | Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace. |
| `include_errors` | bool | `true` | Include spans with error status |
| `skip_invalid_files` | bool | `false` | Skip invalid trace files instead of failing |
| `trace_directory` | str | `null` | Directory containing OTel JSON trace files |
| `trace_files` | list[str] | `null` | List of paths to specific OTel JSON trace files |
| `hf_dataset_path` | str or dict | `null` | HuggingFace dataset path. Can be:   - String: 'username/dataset-name'   - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'} |

<!-- /FIELDS -->
