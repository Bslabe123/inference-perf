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
from enum import Enum
from math import sqrt
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DistributionType(str, Enum):
    NORMAL = "normal"
    SKEW_NORMAL = "skew_normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    POISSON = "poisson"
    FIXED = "fixed"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = Field(default=10, description="Lower clamp on sampled length. Must be <= max.")
    max: int = Field(default=1024, description="Upper clamp on sampled length.")
    mean: float = Field(default=512, description="Distribution mean.")
    std_dev: float = Field(default=200, description="Standard deviation. Mutually exclusive with variance.")
    total_count: Optional[int] = Field(default=None, description="Total number of prompts to generate from this distribution.")
    # New fields for configurable distribution types (default to normal for backward compat)
    type: DistributionType = Field(
        default=DistributionType.NORMAL,
        description="Sampling distribution: normal, skew_normal, lognormal, uniform, poisson, or fixed.",
    )
    variance: Optional[float] = Field(
        default=None, description="Alternative to std_dev (std_dev = sqrt(variance)). Setting both is an error."
    )
    skew: float = Field(
        default=0.0, description="Shape parameter, only used when type is skew_normal."
    )  # Only used for skew_normal

    @model_validator(mode="after")
    def validate_distribution(self) -> "Distribution":
        if self.variance is not None and self.std_dev > 0:
            raise ValueError("Specify either 'std_dev' or 'variance', not both.")
        if self.variance is not None:
            if self.variance < 0:
                raise ValueError("Variance cannot be negative.")
            self.std_dev = sqrt(self.variance)
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max}).")
        if self.std_dev < 0:
            raise ValueError("std_dev cannot be negative.")
        return self
