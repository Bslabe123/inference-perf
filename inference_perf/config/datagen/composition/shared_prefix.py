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
from typing import Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from inference_perf.config.common import Distribution


class SharedPrefixComposition(BaseModel):
    """Shared-prefix composition: groups of prompts share a system prefix.

    Carries only composition concerns (how prompts are organised into groups
    with a shared prefix). Prompt content itself comes from the configured
    source, which lets teams use this composition with any compatible source
    (synthetic, ShareGPT, Billsum, etc.) instead of being locked to the
    synthetic-only ``SharedPrefix`` datagen of today.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    type: Literal["shared_prefix"] = "shared_prefix"

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
        description="Number of distinct shared prefixes (system prompts).",
    )
    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
        description="Number of requests dispatched per shared-prefix group.",
    )
    system_prompt_len: Union[int, Distribution] = Field(
        100,
        description="Shared-prefix token length; fixed int or a Distribution.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Composition-level seed for group assignment and prefix layout.",
    )
