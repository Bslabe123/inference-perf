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
from typing import Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from inference_perf.config.common import Distribution
from inference_perf.config.datagen.multimodal import SyntheticMultimodalDatagenConfig
from inference_perf.config.datagen.replay import (
    ConversationReplayConfig,
    OTelTraceReplayConfig,
    TraceConfig,
)


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"
    Random = "random"
    SharedPrefix = "shared_prefix"
    CNNDailyMail = "cnn_dailymail"
    InfinityInstruct = "infinity_instruct"
    BillsumConversations = "billsum_conversations"
    OTelTraceReplay = "otel_trace_replay"
    ConversationReplay = "conversation_replay"


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
        description="Number of shared-prefix groups (alias num_unique_system_prompts).",
    )

    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
        description="Distinct questions per group (alias num_users_per_system_prompt).",
    )

    system_prompt_len: Union[int, Distribution] = Field(default=100, description="Shared prefix length in tokens.")
    question_len: Union[int, Distribution] = Field(default=50, description="Question length in tokens.")
    output_len: Union[int, Distribution] = Field(default=50, description="Output length in tokens.")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic generation.")

    # Legacy distribution fields — kept for backward compatibility.
    # Prefer using inline distribution syntax on question_len/output_len instead.
    question_distribution: Optional[Distribution] = Field(
        default=None,
        description="Legacy: distribution for question lengths. Prefer an inline distribution on question_len.",
    )
    output_distribution: Optional[Distribution] = Field(
        default=None,
        description="Legacy: distribution for output lengths. Prefer an inline distribution on output_len.",
    )

    enable_multi_turn_chat: bool = Field(
        default=False, description="Generate multi-turn chats instead of single-turn prompts."
    )
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(default=None, description="Media to attach per request.")

    @model_validator(mode="after")
    def validate_no_ambiguous_distributions(self) -> "SharedPrefix":
        if isinstance(self.question_len, Distribution) and self.question_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'question_len' and legacy 'question_distribution'."
                " Use one or the other."
            )
        if isinstance(self.output_len, Distribution) and self.output_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'output_len' and legacy 'output_distribution'."
                " Use one or the other."
            )
        return self


class DataConfig(BaseModel):
    type: DataGenType = Field(default=DataGenType.Mock, description="Data generation type (see Data types).")

    # Valid only for shareGPT type at this moment
    path: Optional[str] = Field(
        default=None,
        description="On-disk dataset path. Required for shareGPT, cnn_dailymail, infinity_instruct, and billsum_conversations.",
    )  # path to the downloaded shareGPT dataset

    # Distributions are only supported for synthetic/random dataset at this moment
    input_distribution: Optional[Distribution] = Field(
        default=None, description="Prompt-length distribution. Used by synthetic / random."
    )
    output_distribution: Optional[Distribution] = Field(
        default=None, description="Generation-length distribution. Used by synthetic / random."
    )
    shared_prefix: Optional[SharedPrefix] = Field(default=None, description="Settings for the shared_prefix type.")
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        default=None, description="Media to attach per request. Valid for synthetic and shared_prefix."
    )

    # Trace file is only supported for random dataset at this moment
    trace: Optional[TraceConfig] = Field(default=None, description="Trace file source. Supported for the random type.")

    # OTel trace replay configuration
    otel_trace_replay: Optional[OTelTraceReplayConfig] = Field(default=None, description="OTel trace replay settings.")

    # Conversation replay configuration
    conversation_replay: Optional[ConversationReplayConfig] = Field(default=None, description="Conversation replay settings.")
