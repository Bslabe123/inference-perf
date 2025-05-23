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
from inference_perf.client.modelserver.base import ModelServerClient, ModelServerMetrics


class PerfRuntimeParameters:
    def __init__(self, start_time: float, duration: float, model_server_client: ModelServerClient) -> None:
        self.start_time = start_time
        self.duration = duration
        self.model_server_client = model_server_client


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_model_server_metrics(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics | None:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
