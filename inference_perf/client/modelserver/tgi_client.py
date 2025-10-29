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

from inference_perf.client.metricsclient.prometheus_client.base import PrometheusCounterMetric, PrometheusGaugeMetric, PrometheusHistogramMetric, PrometheusScalarMetric
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from .base import ModelServerMetricsMetadata
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TGImodelServerClient(openAIModelServerClient):
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
        self.metric_filters = additional_filters

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> ModelServerMetricsMetadata:
        return ModelServerMetricsMetadata(
            count=PrometheusScalarMetric("increase", PrometheusCounterMetric("tgi_request_success", self.metric_filters)),
            rate=PrometheusScalarMetric("rate", PrometheusCounterMetric("tgi_request_success", self.metric_filters)),
            queue_len=PrometheusGaugeMetric("tgi_queue_size", self.metric_filters),
            time_per_output_token=PrometheusHistogramMetric("tgi_request_mean_time_per_token_duration", self.metrics_filters),
            prompt_len=PrometheusHistogramMetric("tgi_request_input_length", self.metric_filters),
            output_len=PrometheusHistogramMetric("tgi_request_generated_tokens", self.metric_filters),
            request_latency=PrometheusHistogramMetric("tgi_request_duration", self.metrics_filters),
        )

