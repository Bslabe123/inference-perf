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

import json
import os
from typing import List
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.circuit_breaker import feed_breakers


class JSONLRequestDataCollector(RequestDataCollector):
    """Responsible for writing client request metrics to a JSONL file"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        # Convert metric to dict and use Pydantic json encoder if needed, but dict() usually works if it's a Pydantic model
        # RequestLifecycleMetric is a pydantic model in apis/base.py
        try:
            # Pydantic v2 use model_dump_json, v1 use json()
            if hasattr(metric, "model_dump_json"):
                json_str = metric.model_dump_json()
            else:
                json_str = metric.json()
            
            with open(self.file_path, "a") as f:
                f.write(json_str + "\n")
        except Exception as e:
            # Log error or ignore? For now let's print, but in prod we might need proper logging
            print(f"Error recording metric to JSONL: {e}")

        feed_breakers(metric)

    def get_metrics(self) -> List[RequestLifecycleMetric]:
        metrics: List[RequestLifecycleMetric] = []
        if not os.path.exists(self.file_path):
            return metrics

        with open(self.file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    # Parse back into RequestLifecycleMetric
                    # Pydantic v2 use model_validate_json, v1 use parse_raw
                    if hasattr(RequestLifecycleMetric, "model_validate_json"):
                        metric = RequestLifecycleMetric.model_validate_json(line)
                    else:
                        metric = RequestLifecycleMetric.parse_raw(line)
                    metrics.append(metric)
                except Exception as e:
                    print(f"Error parsing metric from JSONL: {e}")
        return metrics
