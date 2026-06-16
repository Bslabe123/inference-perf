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
"""Core logic for evaluating and sampling numeric expressions.

Expressions are parsed with SymPy so a single string can describe either a
deterministic, possibly time-varying quantity (e.g. ``"10 + t/60"`` for a
request rate) or a random variable drawn from a named distribution
(e.g. ``"Normal(512, 200)"``). Every probability distribution exposed by
``sympy.stats`` whose constructor takes a leading ``name`` argument is made
available by its class name, with the name supplied automatically.

This module is the shared numeric core: :mod:`inference_perf.utils.numeric`
sub-packages (such as ``distribution``) translate their typed configs into
expression strings and delegate the actual evaluation here.
"""

from __future__ import annotations

import inspect
import itertools
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import sympy  # type: ignore[import-untyped]
import sympy.stats  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sympy.parsing.sympy_parser import parse_expr  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Symbol reserved for time (seconds since a stage started) in rate expressions.
TIME_SYMBOL = sympy.Symbol("t")

# Each occurrence of a distribution in an expression must become an *independent*
# random variable, so every constructor call gets a fresh, unique name. Without
# this, ``Normal(0, 1) - Normal(0, 1)`` would collapse to a single symbol and
# evaluate to a constant zero instead of a variance-2 random variable.
_rv_counter = itertools.count()


def _make_rv_factory(constructor: Callable[..., Any]) -> Callable[..., Any]:
    def factory(*args: Any) -> Any:
        return constructor(f"rv_{next(_rv_counter)}", *args)

    return factory


def _build_distribution_mapping() -> Dict[str, Callable[..., Any]]:
    mapping: Dict[str, Callable[..., Any]] = {}
    for name in dir(sympy.stats):
        if name.startswith("_") or not name[0].isupper():
            continue
        obj = getattr(sympy.stats, name)
        try:
            params = list(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        if params and params[0] == "name":
            mapping[name] = _make_rv_factory(obj)
    return mapping


# Maps distribution class names (Normal, LogNormal, Poisson, Uniform, ...) to a
# zero-config constructor that auto-names the random variable.
_DISTRIBUTION_MAPPING = _build_distribution_mapping()


def parse_expression(expr_str: str) -> Any:
    """Parse a string into a SymPy expression.

    Named distributions resolve to auto-named random variables; ``t`` is the
    reserved time symbol. Raises on malformed input.
    """
    try:
        return parse_expr(expr_str, local_dict=_DISTRIBUTION_MAPPING)
    except Exception as e:
        logger.error(f"Failed to parse expression '{expr_str}': {e}")
        raise


def has_random_variables(expr_str: str) -> bool:
    """Return True if the expression contains any random variable."""
    try:
        expr = parse_expression(expr_str)
    except Exception:
        return False
    if isinstance(expr, (int, float)):
        return False
    return bool(sympy.stats.random_symbols(expr))


def free_symbol_names(expr_str: str) -> set[str]:
    """Return the names of the deterministic free symbols in the expression.

    Random variables are excluded: their presence is governed separately by
    :func:`has_random_variables`. The remaining free symbols are the ones a
    caller may want to restrict (e.g. allowing only ``t`` in a rate expression).
    """
    expr = parse_expression(expr_str)
    if isinstance(expr, (int, float)):
        return set()
    random_syms = set(sympy.stats.random_symbols(expr))
    return {sym.name for sym in expr.free_symbols if sym not in random_syms}


def sample_expression(
    expr_str: str | float | int,
    size: int = 1,
    seed: Optional[int] = None,
    t: Optional[float] = None,
) -> NDArray[np.float64]:
    """Draw ``size`` samples from an expression as a float array.

    Args:
        expr_str: Expression string, or a bare number.
        size: Number of samples to draw.
        seed: Optional integer seed for reproducible sampling. NOTE: SymPy seeds
            every random variable in the expression with the same value, so an
            expression containing two or more identically distributed variables
            (e.g. a skew-normal built from two standard normals) will have them
            perfectly correlated when a seed is supplied. Pass ``None`` for such
            expressions so the variables stay independent.
        t: Optional value to substitute for the reserved time symbol ``t``.

    Returns:
        A 1-D ``float64`` array of length ``size``.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    if isinstance(expr_str, (int, float)):
        return np.full(size, float(expr_str), dtype=np.float64)

    expr = parse_expression(expr_str)

    if t is not None and not isinstance(expr, (int, float)) and TIME_SYMBOL in expr.free_symbols:
        expr = expr.subs(TIME_SYMBOL, t)

    # No randomness left: evaluate once and broadcast.
    if isinstance(expr, (int, float)) or not sympy.stats.random_symbols(expr):
        value = float(sympy.sympify(expr).evalf())
        return np.full(size, value, dtype=np.float64)

    try:
        samples = sympy.stats.sample(expr, size=(size,), library="numpy", seed=seed)
    except Exception as e:
        logger.error(f"Failed to sample from expression '{expr_str}': {e}")
        raise
    return np.asarray(samples, dtype=np.float64)


def evaluate_rate(expr_str: str | float | int, t: float) -> float:
    """Evaluate a deterministic, possibly time-varying expression at time ``t``.

    Intended for rate / interval / concurrency expressions in ``t`` such as
    ``"10 + t/60"``. Raises if the expression contains random variables.
    """
    if isinstance(expr_str, (int, float)):
        return float(expr_str)

    expr = parse_expression(expr_str)
    if isinstance(expr, (int, float)):
        return float(expr)

    if sympy.stats.random_symbols(expr):
        raise ValueError(f"Rate expression must be deterministic, got random variables: '{expr_str}'")

    result = expr.subs(TIME_SYMBOL, t) if TIME_SYMBOL in expr.free_symbols else expr
    return float(result.evalf())
