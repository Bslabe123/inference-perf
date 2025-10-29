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
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from inference_perf.client.metricsclient.base import MetricsMetadata
from inference_perf.client.metricsclient.prometheus_client.base import PrometheusScalarMetric, PrometheusVectorMetric
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData


# PrometheusVectorMetricMetadata stores the mapping of metrics to their model server names and types
# and the filters to be applied to them.
# This is used to generate Prometheus query for the metrics.
class ModelServerMetricsMetadata(MetricsMetadata):
    count: PrometheusScalarMetric
    rate: PrometheusScalarMetric

    prompt_len: PrometheusVectorMetric
    output_len: PrometheusVectorMetric
    queue_len: PrometheusVectorMetric
    request_latency: PrometheusVectorMetric
    time_to_first_token: PrometheusVectorMetric
    kv_cache_usage_percentage: PrometheusVectorMetric

    time_per_output_token: Optional[PrometheusVectorMetric]
    inter_token_latency: Optional[PrometheusVectorMetric]
    num_requests_swapped: Optional[PrometheusVectorMetric]
    num_preemptions_total: Optional[PrometheusVectorMetric]
    prefix_cache_hits: Optional[PrometheusVectorMetric]
    prefix_cache_queries: Optional[PrometheusVectorMetric]


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, api_config: APIConfig, timeout: Optional[float] = None, *args: Tuple[int, ...]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        self.api_config = api_config
        self.timeout = timeout

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> ModelServerMetricsMetadata:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError
