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
import numpy as np
import pytest

from inference_perf.utils.numeric.expression import (
    evaluate_rate,
    has_random_variables,
    sample_expression,
)


class TestSampleExpression:
    def test_constant_broadcasts(self) -> None:
        result = sample_expression("42", size=5)
        assert result.shape == (5,)
        assert np.all(result == 42.0)

    def test_numeric_input_broadcasts(self) -> None:
        result = sample_expression(3.5, size=4)
        assert np.all(result == 3.5)

    def test_normal_distribution_shape_and_spread(self) -> None:
        result = sample_expression("Normal(100, 15)", size=20000, seed=0)
        assert result.shape == (20000,)
        assert abs(float(result.mean()) - 100.0) < 1.0
        assert abs(float(result.std()) - 15.0) < 1.0

    def test_seeded_is_reproducible(self) -> None:
        a = sample_expression("Normal(0, 1)", size=50, seed=7)
        b = sample_expression("Normal(0, 1)", size=50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = sample_expression("Normal(0, 1)", size=50, seed=1)
        b = sample_expression("Normal(0, 1)", size=50, seed=2)
        assert not np.array_equal(a, b)

    def test_repeated_distributions_are_independent_when_unseeded(self) -> None:
        # Var(N1 - N2) for independent standard normals is ~2; if the two
        # variables collapsed to one symbol it would be 0.
        result = sample_expression("Normal(0, 1) - Normal(0, 1)", size=20000)
        assert abs(float(result.var()) - 2.0) < 0.2

    def test_clamped_expression_respects_bounds(self) -> None:
        result = sample_expression("Min(Max(Normal(50, 30), 10), 90)", size=5000, seed=3)
        assert result.min() >= 10
        assert result.max() <= 90

    def test_time_substitution(self) -> None:
        # Deterministic expression in t, sampled at a fixed time.
        result = sample_expression("10 + t", size=3, t=5.0)
        assert np.all(result == 15.0)

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sample_expression("Normal(0, 1)", size=0)


class TestHasRandomVariables:
    def test_distribution_is_random(self) -> None:
        assert has_random_variables("Normal(0, 1)") is True

    def test_deterministic_is_not_random(self) -> None:
        assert has_random_variables("10 + t/60") is False

    def test_constant_is_not_random(self) -> None:
        assert has_random_variables("512") is False


class TestEvaluateRate:
    def test_constant(self) -> None:
        assert evaluate_rate("10", t=99.0) == 10.0

    def test_numeric_input(self) -> None:
        assert evaluate_rate(7.5, t=1.0) == 7.5

    def test_time_varying(self) -> None:
        assert evaluate_rate("10 + t/60", t=120.0) == pytest.approx(12.0)

    def test_random_expression_rejected(self) -> None:
        with pytest.raises(ValueError, match="deterministic"):
            evaluate_rate("Normal(10, 1)", t=0.0)
