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
from typing import Annotated, Any, Optional, Set

from pydantic import BaseModel, BeforeValidator, model_validator


class DistributionType(str, Enum):
    NORMAL = "normal"
    SKEW_NORMAL = "skew_normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    POISSON = "poisson"
    FIXED = "fixed"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200
    total_count: Optional[int] = None
    # New fields for configurable distribution types (default to normal for backward compat)
    type: DistributionType = DistributionType.NORMAL
    variance: Optional[float] = None
    skew: float = 0.0  # Only used for skew_normal

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


def make_expression_validator(allowed_symbols: Set[str], allow_random: bool) -> BeforeValidator:
    """Build a pydantic validator for SymPy expression strings.

    Args:
        allowed_symbols: The deterministic free symbols permitted in the
            expression (e.g. ``{"t"}`` for a time-varying rate). An expression
            referencing any other free symbol is rejected.
        allow_random: Whether the expression may contain random variables (named
            distributions such as ``Normal(...)``).

    The returned validator passes non-string inputs through untouched so it can
    be used as one arm of a ``Union`` (e.g. ``Union[Distribution, LengthExpression]``).
    """

    def validator(v: Any) -> Any:
        if not isinstance(v, str):
            return v

        from inference_perf.utils.numeric.expression import (
            free_symbol_names,
            has_random_variables,
            parse_expression,
        )

        try:
            parse_expression(v)
        except Exception as e:
            raise ValueError(f"Invalid math expression '{v}': {e}") from e

        if not allow_random and has_random_variables(v):
            raise ValueError(f"Expression must be deterministic (no random variables): '{v}'")

        unauthorized = free_symbol_names(v) - allowed_symbols
        if unauthorized:
            allowed = sorted(allowed_symbols) or "none"
            raise ValueError(f"Expression '{v}' uses unauthorized symbol(s) {sorted(unauthorized)}; allowed: {allowed}")

        return v

    return BeforeValidator(validator)


# A length expression: may draw from random distributions, but must not depend
# on any free variable (lengths are not time-varying). Bounds, if needed, are
# expressed inline, e.g. "Min(Max(Normal(512, 200), 10), 1024)".
LengthExpression = Annotated[str, make_expression_validator(allowed_symbols=set(), allow_random=True)]

# A deterministic expression in time ``t`` (seconds since the stage started),
# e.g. "10 + t/60" for a ramping request rate.
DeterministicTimeExpression = Annotated[str, make_expression_validator(allowed_symbols={"t"}, allow_random=False)]
