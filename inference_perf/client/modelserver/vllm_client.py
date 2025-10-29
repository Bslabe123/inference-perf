# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inference_perf.client.metricsclient.prometheus_client.base import (
    PrometheusCounterMetric,
    PrometheusGaugeMetric,
    PrometheusHistogramMetric,
    PrometheusScalarMetric,
)
from inference_perf.client.modelserver.base import ModelServerMetricsMetadata
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class vLLMModelServerClient(openAIModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        uri: str,
        model_name: Optional[str],
        tokenizer_config: Optional[CustomTokenizerConfig],
        max_tcp_connections: int,
        additional_filters: List[str],
        ignore_eos: bool = True,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            metrics_collector,
            api_config,
            uri,
            model_name,
            tokenizer_config,
            max_tcp_connections,
            additional_filters,
            ignore_eos,
            api_key,
            timeout,
        )
        self.metric_filters = [f"model_name='{model_name}'", *additional_filters]

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> ModelServerMetricsMetadata:
        return ModelServerMetricsMetadata(
            # Required metrics
            count=PrometheusScalarMetric(
                "increase", PrometheusCounterMetric("vllm:e2e_request_latency_seconds_count", self.metric_filters)
            ),
            rate=PrometheusScalarMetric(
                "rate", PrometheusCounterMetric("vllm:e2e_request_latency_seconds_count", self.metric_filters)
            ),
            prompt_len=PrometheusCounterMetric("vllm:prompt_tokens_total", self.metric_filters),
            output_len=PrometheusCounterMetric("vllm:generation_tokens_total", self.metric_filters),
            queue_len=PrometheusGaugeMetric("vllm:num_requests_waiting", self.metric_filters),
            request_latency=PrometheusHistogramMetric("vllm:e2e_request_latency_seconds", self.metric_filters),
            time_to_first_token=PrometheusHistogramMetric("vllm:time_to_first_token_seconds", self.metric_filters),
            kv_cache_usage_percentage=PrometheusGaugeMetric("vllm:gpu_cache_usage_perc", self.metric_filters),
            # Optional metrics
            time_per_output_token=PrometheusHistogramMetric(
                "vllm:time_per_output_token_seconds", self.metric_filters
            ),
            inter_token_latency=None,
            num_requests_swapped=PrometheusGaugeMetric("vllm:num_requests_swapped", self.metric_filters),
            num_preemptions_total=PrometheusGaugeMetric("vllm:num_preemptions_total", self.metric_filters),
            prefix_cache_hits=PrometheusCounterMetric("vllm:gpu_prefix_cache_hits_total", self.metric_filters),
            prefix_cache_queries=PrometheusCounterMetric("vllm:gpu_prefix_cache_queries_total", self.metric_filters)
        )
