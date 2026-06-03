# Copyright 2026 The Kubernetes Authors.
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
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(
        default=True, description="Generate the high-level aggregate summary across the whole run."
    )
    per_stage: Optional[bool] = Field(default=True, description="Include a breakdown per load stage.")
    per_request: Optional[bool] = Field(default=False, description="Emit detailed per-request records (verbose).")
    per_adapter: Optional[bool] = Field(default=True, description="Group metrics by LoRA adapter.")
    per_adapter_stage: Optional[bool] = Field(default=False, description="Group metrics by adapter and stage.")
    percentiles: List[float] = Field(
        default=[0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9],
        description="Percentiles to compute for latency/throughput distributions.",
    )


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(default=True, description="Include the aggregate Prometheus metrics summary.")
    per_stage: Optional[bool] = Field(default=False, description="Include a Prometheus breakdown per load stage.")


class SessionLifecycleReportConfig(BaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate the aggregate session summary.")
    per_stage: Optional[bool] = Field(default=True, description="Include a breakdown per load stage.")
    per_session: Optional[bool] = Field(default=False, description="Emit detailed per-session records (verbose).")


class GoodputConfig(BaseModel):
    constraints: Dict[str, float] = Field(
        default={},
        description='Map of metric name to threshold value; a request counts as "good" when it satisfies all constraints.',
    )


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = Field(
        default=RequestLifecycleMetricsReportConfig(),
        description="Latency/throughput metrics derived from request lifecycle timings.",
    )
    prometheus: Optional[PrometheusMetricsReportConfig] = Field(
        default=PrometheusMetricsReportConfig(),
        description="Server-side metrics scraped from the model server's Prometheus endpoint. Set to null to disable.",
    )
    session_lifecycle: SessionLifecycleReportConfig = Field(
        default=SessionLifecycleReportConfig(),
        description="Multi-turn session metrics for session-based load.",
    )
    goodput: Optional[GoodputConfig] = Field(
        default=None,
        description="Goodput constraints; when set, requests are scored against per-metric thresholds.",
    )
