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
"""Distribution sampling as a thin wrapper over the numeric expression core.

Each distribution type is translated into a SymPy expression string and handed
to :mod:`inference_perf.utils.numeric.expression`, which does the actual
sampling. Results are clamped to ``[min, max]`` and rounded to integers, since
these distributions produce token / item counts.

Reproducibility note: single-random-variable distributions (normal, lognormal,
uniform, poisson) are sampled with an integer seed derived from ``rng`` and are
fully reproducible. ``skew_normal`` is built from two *independent* standard
normals (the Azzalini construction), and SymPy's seeding would force those two
variables to be perfectly correlated; it is therefore sampled without a seed and
is not reproducible across runs.
"""

from __future__ import annotations

from math import log, sqrt
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression import sample_expression

if TYPE_CHECKING:
    from inference_perf.config import Distribution

# Upper bound (exclusive) for seeds derived from a numpy Generator.
_SEED_RANGE = 2**32


def _clamp_expr(core: str, min: float, max: float) -> str:
    """Wrap a core expression so its samples are clamped to ``[min, max]``."""
    return f"Min(Max({core}, {min}), {max})"


def _normal_expr(mean: float, std_dev: float) -> str:
    return f"Normal({mean}, {std_dev})"


def _lognormal_expr(mean: float, std_dev: float) -> str:
    # Moment-match the requested mean/std_dev to the underlying normal's mu/sigma.
    sigma_sq = log(1.0 + (std_dev / mean) ** 2)
    mu = log(mean) - sigma_sq / 2.0
    sigma = sqrt(sigma_sq)
    return f"LogNormal({mu}, {sigma})"


def _uniform_expr(min: float, max: float) -> str:
    return f"Uniform({min}, {max})"


def _poisson_expr(mean: float) -> str:
    lam = mean if mean > 0 else 1.0
    return f"Poisson({lam})"


def _skew_normal_expr(mean: float, std_dev: float, skew: float) -> str:
    # Azzalini construction: with N1, N2 iid standard normal and
    # delta = skew / sqrt(1 + skew^2), the variable
    #   mean + std_dev * (delta * |N1| + sqrt(1 - delta^2) * N2)
    # is skew-normal. The two normals must stay independent, so this expression
    # is sampled without a seed (see module docstring).
    delta = skew / sqrt(1.0 + skew**2)
    root = sqrt(1.0 - delta**2)
    return f"{mean} + {std_dev} * ({delta} * Abs(Normal(0, 1)) + {root} * Normal(0, 1))"


def distribution_to_expression(config: "Distribution") -> str:
    """Translate a Distribution config into a clamped SymPy expression string.

    The returned string is self-contained: it embeds the ``[min, max]`` clamp so
    it can be evaluated directly by the expression core (or reused anywhere an
    expression is accepted).
    """
    from inference_perf.config import DistributionType

    min_val, max_val = float(config.min), float(config.max)

    # Degenerate cases collapse to a constant: an explicit fixed type, zero
    # spread, or a collapsed range (min == max, where the clamp forces a single
    # value and a distribution core like Uniform(x, x) would be ill-defined).
    if config.type == DistributionType.FIXED or config.std_dev == 0 or config.min == config.max:
        return _clamp_expr(str(config.mean), min_val, max_val)

    if config.type == DistributionType.NORMAL:
        core = _normal_expr(config.mean, config.std_dev)
    elif config.type == DistributionType.SKEW_NORMAL:
        core = _skew_normal_expr(config.mean, config.std_dev, config.skew)
    elif config.type == DistributionType.LOGNORMAL:
        if config.mean <= 0:
            raise ValueError("Lognormal distribution requires mean > 0.")
        core = _lognormal_expr(config.mean, config.std_dev)
    elif config.type == DistributionType.UNIFORM:
        core = _uniform_expr(min_val, max_val)
    elif config.type == DistributionType.POISSON:
        core = _poisson_expr(config.mean)
    else:
        raise ValueError(f"Unsupported distribution type: {config.type}")

    return _clamp_expr(core, min_val, max_val)


def _derive_seed(rng: Optional[np.random.Generator]) -> Optional[int]:
    if rng is None:
        return None
    return int(rng.integers(0, _SEED_RANGE))


def _finalize(samples: NDArray[np.float64], min: int, max: int) -> NDArray[np.int_]:
    """Round to integers and clamp to ``[min, max]``."""
    result = np.round(np.clip(samples, min, max)).astype(int)
    result = np.clip(result, min, max)
    return cast(NDArray[np.int_], result)


def sample_from_distribution(
    config: "Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.int_]:
    """Sample integer values from a Distribution config.

    Translates ``config`` into an expression and samples it. Dispatches on
    ``config.type`` to support normal, skew_normal, lognormal, uniform, poisson,
    and fixed distributions. Results are clamped to ``[config.min, config.max]``.

    Args:
        config: A Distribution specifying the distribution type and parameters.
        count: Number of samples to generate.
        rng: Optional numpy Generator for deterministic seeding. If None, a
            non-deterministic seed is used.

    Returns:
        A numpy array of integers clamped to ``[config.min, config.max]``.
    """
    from inference_perf.config import DistributionType

    if count <= 0:
        raise ValueError("Count must be a positive integer.")
    if config.min > config.max:
        raise ValueError(f"min ({config.min}) cannot be greater than max ({config.max}).")

    expr = distribution_to_expression(config)

    # skew_normal relies on two independent normals; seeding would correlate them.
    seed = None if config.type == DistributionType.SKEW_NORMAL else _derive_seed(rng)

    samples = sample_expression(expr, size=count, seed=seed)
    return _finalize(samples, config.min, config.max)


def sample_lengths(
    spec: "int | str | Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.int_]:
    """Sample ``count`` non-negative integer lengths from a length spec.

    Unifies the three ways a length can be configured:

    * ``int``  -> a fixed length, broadcast to ``count`` values.
    * ``Distribution`` -> delegated to :func:`sample_from_distribution`.
    * ``str`` -> a length expression (e.g. ``"Min(Max(Normal(512, 200), 10), 1024)"``)
      sampled via the expression core. The expression is expected to encode its
      own bounds; results are rounded to integers and floored at 0 so lengths
      are never negative.

    Args:
        spec: The length specification.
        count: Number of lengths to generate.
        rng: Optional numpy Generator for deterministic seeding.

    Returns:
        A numpy array of non-negative integers of length ``count``.
    """
    from inference_perf.config import Distribution

    if count <= 0:
        raise ValueError("Count must be a positive integer.")

    if isinstance(spec, str):
        samples = sample_expression(spec, size=count, seed=_derive_seed(rng))
        result = np.round(samples).astype(int)
        return cast(NDArray[np.int_], np.clip(result, 0, None))

    if isinstance(spec, Distribution):
        return sample_from_distribution(spec, count, rng)

    return cast(NDArray[np.int_], np.full(count, int(spec), dtype=int))


def generate_distribution(
    min: int,
    max: int,
    mean: float,
    std_dev: float,
    total_count: int,
    dist_type: str = "normal",  # one of: "normal", "lognormal", "uniform", "fixed"
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """Generate an integer array adhering to the specified distribution.

    Lower-level counterpart to :func:`sample_from_distribution` that takes loose
    parameters instead of a Distribution config. Like that function, it
    translates the request into an expression and samples it.

    Args:
        min: The minimum allowed value (inclusive).
        max: The maximum allowed value (inclusive).
        mean: The target mean of the distribution.
        std_dev: The target standard deviation of the distribution.
        total_count: The total number of values to generate.
        dist_type: Distribution type — "normal", "lognormal", "uniform", or "fixed".
        rng: Optional numpy Generator for deterministic output.

    Returns:
        A numpy array of integers clamped to ``[min, max]``.

    Raises:
        ValueError: If constraints are impossible (e.g., min > max).
    """
    if min > max:
        raise ValueError("Minimum value cannot be greater than maximum value.")
    if total_count <= 0:
        raise ValueError("Total count must be a positive integer.")
    if std_dev < 0:
        raise ValueError("Standard deviation cannot be negative.")

    min_val, max_val = float(min), float(max)

    if dist_type == "fixed" or std_dev == 0 or min == max:
        expr = _clamp_expr(str(mean), min_val, max_val)
    elif dist_type == "uniform":
        expr = _clamp_expr(_uniform_expr(min_val, max_val), min_val, max_val)
    elif dist_type == "lognormal":
        if mean <= 0:
            raise ValueError("Lognormal distribution requires mean > 0.")
        expr = _clamp_expr(_lognormal_expr(mean, std_dev), min_val, max_val)
    elif dist_type == "normal":
        if mean < min or mean > max:
            raise ValueError("Mean cannot be outside min and max range.")
        expr = _clamp_expr(_normal_expr(mean, std_dev), min_val, max_val)
    else:
        raise ValueError(f"Unknown dist_type {dist_type!r}. Supported types: 'normal', 'lognormal', 'uniform', 'fixed'.")

    samples = sample_expression(expr, size=total_count, seed=_derive_seed(rng))
    return _finalize(samples, min, max)
