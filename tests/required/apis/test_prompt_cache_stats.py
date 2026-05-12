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
"""Unit coverage for :meth:`PromptCacheStats.from_usage`.

The helper extracts vLLM's ``usage.prompt_tokens_details.cached_tokens`` from
a server-reported ``usage`` dict and returns ``None`` for any shape that
doesn't carry the field — that "absent vs zero" distinction is what lets the
summary code omit the prompt_cache block entirely when no request reported
it. These tests pin the helper's behavior across all the shapes we expect
to see in the wild.
"""

from typing import Any

import pytest

from inference_perf.apis import PromptCacheStats


def test_from_usage_returns_none_when_usage_is_none() -> None:
    assert PromptCacheStats.from_usage(None) is None


def test_from_usage_returns_none_when_usage_is_empty() -> None:
    assert PromptCacheStats.from_usage({}) is None


def test_from_usage_returns_none_when_prompt_tokens_details_missing() -> None:
    # vLLM started without --enable-prompt-tokens-details, or another server.
    assert PromptCacheStats.from_usage({"prompt_tokens": 100, "completion_tokens": 50}) is None


def test_from_usage_returns_none_when_prompt_tokens_details_not_dict() -> None:
    # Defensive against unexpected payload shapes.
    assert PromptCacheStats.from_usage({"prompt_tokens": 100, "prompt_tokens_details": "garbage"}) is None


def test_from_usage_returns_none_when_cached_tokens_missing() -> None:
    assert (
        PromptCacheStats.from_usage({"prompt_tokens": 100, "prompt_tokens_details": {}})
        is None
    )


def test_from_usage_returns_none_when_cached_tokens_not_int() -> None:
    assert (
        PromptCacheStats.from_usage(
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": "30"}}
        )
        is None
    )


def test_from_usage_returns_none_when_prompt_tokens_zero() -> None:
    # Avoid division by zero on the hit_rate.
    assert (
        PromptCacheStats.from_usage(
            {"prompt_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}
        )
        is None
    )


def test_from_usage_returns_none_when_prompt_tokens_negative() -> None:
    assert (
        PromptCacheStats.from_usage(
            {"prompt_tokens": -1, "prompt_tokens_details": {"cached_tokens": 0}}
        )
        is None
    )


def test_from_usage_zero_hit() -> None:
    stats = PromptCacheStats.from_usage(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}
    )
    assert stats is not None
    assert stats.cached_tokens == 0
    assert stats.total_tokens == 100
    assert stats.hit_rate == 0.0


def test_from_usage_partial_hit() -> None:
    stats = PromptCacheStats.from_usage(
        {"prompt_tokens": 200, "prompt_tokens_details": {"cached_tokens": 50}}
    )
    assert stats is not None
    assert stats.cached_tokens == 50
    assert stats.total_tokens == 200
    assert stats.hit_rate == 0.25


def test_from_usage_full_hit() -> None:
    stats = PromptCacheStats.from_usage(
        {"prompt_tokens": 128, "prompt_tokens_details": {"cached_tokens": 128}}
    )
    assert stats is not None
    assert stats.cached_tokens == 128
    assert stats.total_tokens == 128
    assert stats.hit_rate == 1.0


def test_from_usage_ignores_unrelated_usage_fields() -> None:
    """Extra fields in the usage dict (completion_tokens, total_tokens) don't
    affect prompt_cache derivation."""
    stats = PromptCacheStats.from_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": 5},
        }
    )
    assert stats is not None
    assert stats.cached_tokens == 40
    assert stats.total_tokens == 100
    assert stats.hit_rate == 0.4


@pytest.mark.parametrize(
    "usage",
    [
        # vLLM with --enable-prompt-tokens-details, normal stream chunk shape.
        {
            "prompt_tokens": 1024,
            "completion_tokens": 256,
            "total_tokens": 1280,
            "prompt_tokens_details": {"cached_tokens": 768},
        },
        # vLLM without the flag (no prompt_tokens_details key at all).
        {
            "prompt_tokens": 1024,
            "completion_tokens": 256,
            "total_tokens": 1280,
        },
    ],
)
def test_from_usage_handles_realistic_vllm_shapes(usage: dict[str, Any]) -> None:
    """Round-trip on shapes vLLM actually emits."""
    stats = PromptCacheStats.from_usage(usage)
    if "prompt_tokens_details" in usage:
        assert stats is not None
        assert stats.cached_tokens == 768
        assert stats.total_tokens == 1024
        assert stats.hit_rate == 0.75
    else:
        assert stats is None
