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
"""Arrival-schedule delivery.

``ArrivalSchedule`` describes *when* requests are dispatched, distinct from
*what* they contain. Today the equivalent setting (``data.trace:
TraceConfig``) is misfiled under the source surface even though it modifies
only timing. This module reframes it as a delivery concern; the existing
:class:`TraceConfig` is reused as the inner schema.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from inference_perf.config.datagen.replay import TraceConfig


class TraceFileArrival(BaseModel):
    """Replay request arrivals from a recorded trace file."""

    type: Literal["trace_file"] = "trace_file"
    trace: TraceConfig = Field(
        description="Path and format of the recorded arrival trace.",
    )


ArrivalSchedule = Annotated[
    Union[TraceFileArrival,],
    Field(discriminator="type"),
]
# Other arrival shapes (constant, poisson, custom-distribution) can be added
# here once their drivers are surfaced as delivery-level concerns. Today
# they live inside LoadConfig.type / LoadConfig.interval.
