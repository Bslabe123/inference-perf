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
from pydantic import BaseModel, ValidationError

from inference_perf.config import Distribution, DistributionType
from inference_perf.config.common import DeterministicTimeExpression, LengthExpression
from inference_perf.utils.numeric.distribution import sample_lengths


class _LengthModel(BaseModel):
    value: LengthExpression


class _RateModel(BaseModel):
    value: DeterministicTimeExpression


class TestLengthExpressionValidator:
    def test_accepts_random_distribution(self) -> None:
        m = _LengthModel(value="Min(Max(Normal(512, 200), 10), 1024)")
        assert isinstance(m.value, str)

    def test_accepts_plain_number(self) -> None:
        assert _LengthModel(value="512").value == "512"

    def test_rejects_free_symbol(self) -> None:
        # Lengths may not depend on time (or any free symbol).
        with pytest.raises(ValidationError, match="unauthorized symbol"):
            _LengthModel(value="10 + t")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError, match="Invalid math expression"):
            _LengthModel(value="Normal(")


class TestDeterministicTimeExpressionValidator:
    def test_accepts_time_expression(self) -> None:
        assert _RateModel(value="10 + t/60").value == "10 + t/60"

    def test_accepts_constant(self) -> None:
        assert _RateModel(value="42").value == "42"

    def test_rejects_random(self) -> None:
        with pytest.raises(ValidationError, match="deterministic"):
            _RateModel(value="Normal(10, 1)")

    def test_rejects_other_symbol(self) -> None:
        with pytest.raises(ValidationError, match="unauthorized symbol"):
            _RateModel(value="10 + x")


class TestSampleLengths:
    def test_int_spec_is_fixed(self) -> None:
        result = sample_lengths(7, 100)
        assert result.shape == (100,)
        assert np.all(result == 7)

    def test_distribution_spec(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=200.0, min=10, max=400, std_dev=30.0)
        result = sample_lengths(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 10
        assert result.max() <= 400

    def test_expression_spec_bounds_and_nonnegative(self) -> None:
        result = sample_lengths("Min(Max(Normal(100, 50), 5), 300)", 2000, rng=np.random.default_rng(0))
        assert result.min() >= 5
        assert result.max() <= 300
        assert np.all(result >= 0)

    def test_expression_spec_reproducible(self) -> None:
        a = sample_lengths("Normal(100, 20)", 50, rng=np.random.default_rng(1))
        b = sample_lengths("Normal(100, 20)", 50, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(a, b)

    def test_invalid_count(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sample_lengths(5, 0)
