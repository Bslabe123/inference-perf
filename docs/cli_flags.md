# Inference-Perf CLI Flags

These command line flags are automatically generated from the internal `Config` schema. You can override any configuration directly from the CLI without using a yaml configuration file.

| Flag | Type | Description |
| --- | --- | --- |
| `--api.type` | Enum (completion, chat) | API surface to target (completion or chat). |
| `--api.streaming` | boolean | Stream the response. Required to measure TTFT, ITL, and TPOT. |
| `--api.headers` | JSON | Custom HTTP headers sent with every request. |
| `--api.slo_unit` | str | Unit for SLO header values (e.g. ms, s). |
| `--api.slo_tpot_header` | str | Name of the header carrying the per-request TPOT SLO. |
| `--api.slo_ttft_header` | str | Name of the header carrying the per-request TTFT SLO. |
| `--api.response_format.type` | Enum (json_schema, json_object) | json_schema (constrain output to a schema) or json_object (any valid JSON object). |
| `--api.response_format.name` | str | Name for the JSON schema (used when type is json_schema). |
| `--api.response_format.json_schema` | JSON | The JSON schema the output must conform to (used when type is json_schema). |
| `--data.type` | Enum (mock, shareGPT, synthetic, random, shared_prefix, cnn_dailymail, infinity_instruct, billsum_conversations, otel_trace_replay, conversation_replay) | Data generation type (see Data types). |
| `--data.path` | str | On-disk dataset path. Required for shareGPT, cnn_dailymail, infinity_instruct, and billsum_conversations. |
| `--data.input_distribution.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.input_distribution.max` | int | Upper clamp on sampled length. |
| `--data.input_distribution.mean` | float | Distribution mean. |
| `--data.input_distribution.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.input_distribution.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.input_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.input_distribution.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.input_distribution.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.output_distribution.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.output_distribution.max` | int | Upper clamp on sampled length. |
| `--data.output_distribution.mean` | float | Distribution mean. |
| `--data.output_distribution.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.output_distribution.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.output_distribution.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.output_distribution.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.num_groups` | int | Number of shared-prefix groups (alias num_unique_system_prompts). |
| `--data.shared_prefix.num_prompts_per_group` | int | Distinct questions per group (alias num_users_per_system_prompt). |
| `--data.shared_prefix.system_prompt_len` | string | Shared prefix length in tokens. |
| `--data.shared_prefix.question_len` | string | Question length in tokens. |
| `--data.shared_prefix.output_len` | string | Output length in tokens. |
| `--data.shared_prefix.seed` | int | Random seed for deterministic generation. |
| `--data.shared_prefix.question_distribution.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.shared_prefix.question_distribution.max` | int | Upper clamp on sampled length. |
| `--data.shared_prefix.question_distribution.mean` | float | Distribution mean. |
| `--data.shared_prefix.question_distribution.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.shared_prefix.question_distribution.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.shared_prefix.question_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.shared_prefix.question_distribution.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.shared_prefix.question_distribution.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.output_distribution.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.shared_prefix.output_distribution.max` | int | Upper clamp on sampled length. |
| `--data.shared_prefix.output_distribution.mean` | float | Distribution mean. |
| `--data.shared_prefix.output_distribution.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.shared_prefix.output_distribution.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.shared_prefix.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.shared_prefix.output_distribution.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.shared_prefix.output_distribution.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.enable_multi_turn_chat` | boolean | Generate multi-turn chats instead of single-turn prompts. |
| `--data.shared_prefix.multimodal.image.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.shared_prefix.multimodal.image.count.max` | int | Upper clamp on sampled length. |
| `--data.shared_prefix.multimodal.image.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.image.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.shared_prefix.multimodal.image.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.shared_prefix.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.shared_prefix.multimodal.image.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.shared_prefix.multimodal.image.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.shared_prefix.multimodal.image.representation` | Enum (png, jpeg) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.shared_prefix.multimodal.video.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.shared_prefix.multimodal.video.count.max` | int | Upper clamp on sampled length. |
| `--data.shared_prefix.multimodal.video.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.video.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.shared_prefix.multimodal.video.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.shared_prefix.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.shared_prefix.multimodal.video.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.shared_prefix.multimodal.video.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.shared_prefix.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.shared_prefix.multimodal.audio.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.shared_prefix.multimodal.audio.count.max` | int | Upper clamp on sampled length. |
| `--data.shared_prefix.multimodal.audio.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.audio.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.shared_prefix.multimodal.audio.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.shared_prefix.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.shared_prefix.multimodal.audio.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.shared_prefix.multimodal.audio.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.shared_prefix.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.multimodal.image.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.multimodal.image.count.max` | int | Upper clamp on sampled length. |
| `--data.multimodal.image.count.mean` | float | Distribution mean. |
| `--data.multimodal.image.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.multimodal.image.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.multimodal.image.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.multimodal.image.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.multimodal.image.representation` | Enum (png, jpeg) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.multimodal.video.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.multimodal.video.count.max` | int | Upper clamp on sampled length. |
| `--data.multimodal.video.count.mean` | float | Distribution mean. |
| `--data.multimodal.video.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.multimodal.video.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.multimodal.video.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.multimodal.video.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.multimodal.audio.count.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.multimodal.audio.count.max` | int | Upper clamp on sampled length. |
| `--data.multimodal.audio.count.mean` | float | Distribution mean. |
| `--data.multimodal.audio.count.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.multimodal.audio.count.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.multimodal.audio.count.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.multimodal.audio.count.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.trace.file` | str | Path to the trace file. |
| `--data.trace.format` | Enum (AzurePublicDataset) | Trace format. Currently only AzurePublicDataset. |
| `--data.otel_trace_replay.use_static_model` | boolean | Use a single static model for all requests |
| `--data.otel_trace_replay.static_model_name` | str | Static model name (required if use_static_model=True) |
| `--data.otel_trace_replay.model_mapping` | JSON | Map recorded model names to target models |
| `--data.otel_trace_replay.default_max_tokens` | int | Default max_tokens if not specified in trace |
| `--data.otel_trace_replay.inject_random_session_id` | boolean | Inject random string into unique segments to invalidate KV-cache between sessions |
| `--data.otel_trace_replay.duplicate_sessions_target` | int | Target number of sessions to reach by duplicating existing sessions. If None, no duplication occurs. |
| `--data.otel_trace_replay.max_wait_ms` | int | Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace. |
| `--data.otel_trace_replay.include_errors` | boolean | Include spans with error status |
| `--data.otel_trace_replay.skip_invalid_files` | boolean | Skip invalid trace files instead of failing |
| `--data.otel_trace_replay.trace_directory` | str | Directory containing OTel JSON trace files |
| `--data.otel_trace_replay.trace_files` | JSON | List of paths to specific OTel JSON trace files |
| `--data.otel_trace_replay.hf_dataset_path` | JSON | HuggingFace dataset path. Can be:
  - String: 'username/dataset-name'
  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'} |
| `--data.conversation_replay.seed` | int | Random seed for deterministic generation |
| `--data.conversation_replay.num_conversations` | int | Number of conversation blueprints to generate |
| `--data.conversation_replay.shared_system_prompt_len` | int | Fixed shared system prompt length in tokens |
| `--data.conversation_replay.dynamic_system_prompt_len.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.conversation_replay.dynamic_system_prompt_len.max` | int | Upper clamp on sampled length. |
| `--data.conversation_replay.dynamic_system_prompt_len.mean` | float | Distribution mean. |
| `--data.conversation_replay.dynamic_system_prompt_len.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.conversation_replay.dynamic_system_prompt_len.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.conversation_replay.dynamic_system_prompt_len.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.conversation_replay.dynamic_system_prompt_len.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.conversation_replay.dynamic_system_prompt_len.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.conversation_replay.turns_per_conversation.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.conversation_replay.turns_per_conversation.max` | int | Upper clamp on sampled length. |
| `--data.conversation_replay.turns_per_conversation.mean` | float | Distribution mean. |
| `--data.conversation_replay.turns_per_conversation.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.conversation_replay.turns_per_conversation.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.conversation_replay.turns_per_conversation.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.conversation_replay.turns_per_conversation.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.conversation_replay.turns_per_conversation.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.conversation_replay.input_tokens_per_turn.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.conversation_replay.input_tokens_per_turn.max` | int | Upper clamp on sampled length. |
| `--data.conversation_replay.input_tokens_per_turn.mean` | float | Distribution mean. |
| `--data.conversation_replay.input_tokens_per_turn.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.conversation_replay.input_tokens_per_turn.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.conversation_replay.input_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.conversation_replay.input_tokens_per_turn.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.conversation_replay.input_tokens_per_turn.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.conversation_replay.output_tokens_per_turn.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.conversation_replay.output_tokens_per_turn.max` | int | Upper clamp on sampled length. |
| `--data.conversation_replay.output_tokens_per_turn.mean` | float | Distribution mean. |
| `--data.conversation_replay.output_tokens_per_turn.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.conversation_replay.output_tokens_per_turn.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.conversation_replay.output_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.conversation_replay.output_tokens_per_turn.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.conversation_replay.output_tokens_per_turn.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.conversation_replay.tool_call_latency_sec.min` | int | Lower clamp on sampled length. Must be <= max. |
| `--data.conversation_replay.tool_call_latency_sec.max` | int | Upper clamp on sampled length. |
| `--data.conversation_replay.tool_call_latency_sec.mean` | float | Distribution mean. |
| `--data.conversation_replay.tool_call_latency_sec.std_dev` | float | Standard deviation. Mutually exclusive with variance. |
| `--data.conversation_replay.tool_call_latency_sec.total_count` | int | Total number of prompts to generate from this distribution. |
| `--data.conversation_replay.tool_call_latency_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed. |
| `--data.conversation_replay.tool_call_latency_sec.variance` | float | Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error. |
| `--data.conversation_replay.tool_call_latency_sec.skew` | float | Shape parameter, only used when type is skew_normal. |
| `--data.conversation_replay.max_model_len` | int | Maximum model context length in tokens |
| `--load.type` | Enum (constant, poisson, trace_replay, concurrent, trace_session_replay) | Load pattern (see Load types). |
| `--load.interval` | float | Seconds between request batches within a stage. |
| `--load.stages` | JSON | Ordered load stages; shape depends on type. |
| `--load.sweep.type` | Enum (geometric, linear) | linear (evenly spaced 1 to saturation) or geometric (clustered near saturation). |
| `--load.sweep.num_requests` | int | Requests used to probe saturation. |
| `--load.sweep.timeout` | float | Seconds to run the saturation probe. |
| `--load.sweep.num_stages` | int | Number of stages to generate. |
| `--load.sweep.stage_duration` | int | Duration of each generated stage. |
| `--load.sweep.saturation_percentile` | float | Percentile of sampled rates taken as the saturation point. |
| `--load.num_workers` | int | Number of worker processes (see Worker model). |
| `--load.worker_max_concurrency` | int | Max in-flight requests per worker. |
| `--load.worker_max_tcp_connections` | int | Max TCP connections per worker. |
| `--load.trace.file` | str | Path to the trace file. |
| `--load.trace.format` | Enum (AzurePublicDataset) | Trace format. Currently only AzurePublicDataset. |
| `--load.circuit_breakers` | JSON | Named circuit breakers to abort a run on breach (see docs/goodput.md). |
| `--load.request_timeout` | float | Per-request timeout in seconds. |
| `--load.lora_traffic_split` | JSON | MultiLoRA traffic weights (see LoRA traffic split). |
| `--load.base_seed` | int | Base random seed; set explicitly for reproducible runs. |
| `--metrics.type` | Enum (prometheus, default) | Metrics client backend: 'default' or 'prometheus'. |
| `--metrics.prometheus.scrape_interval` | int | Metrics scrape interval in seconds; should match the server's scrape interval. |
| `--metrics.prometheus.url` | string | Prometheus server URL to query (for example http://localhost:9090). Mutually exclusive with google_managed. |
| `--metrics.prometheus.filters` | JSON | Metric names to collect; empty collects the default set. |
| `--metrics.prometheus.google_managed` | boolean | Query Google Managed Prometheus (GMP) via its API instead of a URL. Mutually exclusive with url. |
| `--report.request_lifecycle.summary` | boolean | Generate the high-level aggregate summary across the whole run. |
| `--report.request_lifecycle.per_stage` | boolean | Include a breakdown per load stage. |
| `--report.request_lifecycle.per_request` | boolean | Emit detailed per-request records (verbose). |
| `--report.request_lifecycle.per_adapter` | boolean | Group metrics by LoRA adapter. |
| `--report.request_lifecycle.per_adapter_stage` | boolean | Group metrics by adapter and stage. |
| `--report.request_lifecycle.percentiles` | JSON | Percentiles to compute for latency/throughput distributions. |
| `--report.prometheus.summary` | boolean | Include the aggregate Prometheus metrics summary. |
| `--report.prometheus.per_stage` | boolean | Include a Prometheus breakdown per load stage. |
| `--report.session_lifecycle.summary` | boolean | Generate the aggregate session summary. |
| `--report.session_lifecycle.per_stage` | boolean | Include a breakdown per load stage. |
| `--report.session_lifecycle.per_session` | boolean | Emit detailed per-session records (verbose). |
| `--report.goodput.constraints` | JSON | Map of metric name to threshold value; a request counts as "good" when it satisfies all constraints. |
| `--storage.local_storage.path` | str | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `--storage.local_storage.report_file_prefix` | str | Optional prefix prepended to each report filename. |
| `--storage.google_cloud_storage.path` | str | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `--storage.google_cloud_storage.report_file_prefix` | str | Optional prefix prepended to each report filename. |
| `--storage.google_cloud_storage.bucket_name` | str | Target GCS bucket. |
| `--storage.simple_storage_service.path` | str | Destination directory (local) or key prefix (GCS / S3). The default is generated once per run from the start time, formatted reports-YYYYMMDD-HHMMSS. |
| `--storage.simple_storage_service.report_file_prefix` | str | Optional prefix prepended to each report filename. |
| `--storage.simple_storage_service.bucket_name` | str | Target S3 bucket. |
| `--storage.simple_storage_service.endpoint_url` | str | Custom endpoint URL for S3-compatible stores. |
| `--storage.simple_storage_service.region_name` | str | AWS region name. |
| `--storage.simple_storage_service.addressing_style` | string | Bucket addressing: auto, virtual, or path. |
| `--server.type` | Enum (vllm, sglang, tgi, mock) | Backend flavor (vllm, sglang, tgi, or mock). |
| `--server.model_name` | str | Model identifier the server should serve (for example, a Hugging Face repo ID). |
| `--server.base_url` | str | Server endpoint, including scheme, host, and port. |
| `--server.ignore_eos` | boolean | Whether to ignore End-of-Sequence tokens so generation runs to the requested length. |
| `--server.api_key` | str | API key for authenticated endpoints. |
| `--server.cert_path` | str | Path to a client TLS certificate for mutual-TLS endpoints. |
| `--server.key_path` | str | Path to the private key paired with cert_path. |
| `--tokenizer.pretrained_model_name_or_path` | str | HuggingFace model ID or local path the tokenizer is loaded from. Required to activate an explicit tokenizer; if unset, the tokenizer is inferred from the model server. |
| `--tokenizer.trust_remote_code` | boolean | Allow loading custom tokenizer code shipped with the model repo. Leave unset (treated as off) unless the model requires it. |
| `--tokenizer.token` | str | HuggingFace access token for private or gated models. |
| `--circuit_breakers` | JSON | Circuit breakers that abort a run when health constraints are breached (see docs/goodput.md). |
