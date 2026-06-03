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
from pydantic import BaseModel, Field
from typing import Literal, Union


class TriggerConsecutive(BaseModel):
    type: Literal["consecutive"] = Field(..., description="Discriminator selecting the consecutive-hits trigger.")
    threshold: int = Field(..., ge=1, description="Number of consecutive hits required to open the circuit breaker.")


class TriggerRateOverWindow(BaseModel):
    type: Literal["rate_over_window"] = Field(..., description="Discriminator selecting the rate-over-window trigger.")
    window_sec: float = Field(
        ..., gt=0.0, description="Length of the sliding time window, in seconds, over which the hit rate is computed."
    )
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Hit-rate threshold (0.0 to 1.0) that opens the circuit breaker when reached."
    )
    min_samples: int = Field(0, ge=0, description="Minimum number of samples in the window before the hit rate is evaluated.")


TriggerSpec = Union[
    TriggerConsecutive,
    TriggerRateOverWindow,
]
