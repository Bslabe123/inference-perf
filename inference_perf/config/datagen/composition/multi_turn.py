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
from typing import Literal, Optional

from pydantic import BaseModel, Field

from inference_perf.config.common import Distribution


class MultiTurnComposition(BaseModel):
    """Multi-turn / session-based composition.

    Carries only composition concerns: how many concurrent conversation
    blueprints, the shared / dynamic system prompt structure, turns per
    conversation, and tool-call timing. Prompt content per turn comes from
    the configured source; today's :class:`ConversationReplayConfig` will be
    split so its per-turn token-length distributions move to the source side
    once this surface is wired into ``DataConfig``.
    """

    type: Literal["multi_turn"] = "multi_turn"

    num_conversations: int = Field(
        200,
        gt=0,
        description="Number of conversation blueprints to generate.",
    )
    shared_system_prompt_len: int = Field(
        8359,
        ge=0,
        description="Fixed shared system prompt length in tokens.",
    )
    dynamic_system_prompt_len: Optional[Distribution] = Field(
        default=None,
        description="Per-conversation dynamic system prompt length distribution.",
    )
    turns_per_conversation: Optional[Distribution] = Field(
        default=None,
        description="Number of turns per conversation distribution.",
    )
    tool_call_latency_sec: Optional[Distribution] = Field(
        default=None,
        description=(
            "Per-turn tool execution latency distribution in seconds. When "
            "set, each turn sleeps for the sampled duration after model "
            "inference completes and before the next turn begins. The sleep "
            "is async; the GPU remains free to serve other concurrent "
            "conversations during the wait, so this correctly models offline "
            "agentic workloads without artificially lowering throughput. "
            "May migrate to the agent surface once that exists."
        ),
    )
    max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum model context length in tokens; safety reset above this.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Composition-level seed for deterministic conversation generation.",
    )
