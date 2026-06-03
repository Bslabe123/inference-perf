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
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class PrometheusClientConfig(BaseModel):
    scrape_interval: int = Field(
        default=15,
        description="Metrics scrape interval in seconds; should match the server's scrape interval.",
    )
    url: Optional[HttpUrl] = Field(
        default=None,
        description="Prometheus server URL to query (for example http://localhost:9090). Mutually exclusive with google_managed.",
    )
    filters: List[str] = Field(
        default=[],
        description="Metric names to collect; empty collects the default set.",
    )
    google_managed: bool = Field(
        default=False,
        description="Query Google Managed Prometheus (GMP) via its API instead of a URL. Mutually exclusive with url.",
    )

    @model_validator(mode="after")
    def check_exclusive_fields(self) -> "PrometheusClientConfig":
        if bool(self.url) == bool(self.google_managed):
            raise ValueError("Exactly one of 'url' or 'google_managed' must be set.")
        return self
