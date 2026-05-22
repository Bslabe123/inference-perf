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
"""Composition configuration.

A ``Composition`` describes *how prompts are shaped into requests*
(single-turn, shared-prefix groups, multi-turn sessions, future agentic
loops), independent of where the prompt content originates. The union below
covers the shapes expressible today; new compositions (agent loops, branching
flows) are additions to this union.

Composition is many-to-one with API choice: a multi-turn composition only
works with the Chat API, while single-turn and shared-prefix work with either
the Chat or Completion API. ``DataConfig`` validation enforces this against
``config.api`` once this surface is wired in.
"""

from typing import Annotated, Union

from pydantic import Field

from inference_perf.config.datagen.composition.multi_turn import MultiTurnComposition
from inference_perf.config.datagen.composition.shared_prefix import SharedPrefixComposition
from inference_perf.config.datagen.composition.single_turn import SingleTurnComposition

Composition = Annotated[
    Union[
        SingleTurnComposition,
        SharedPrefixComposition,
        MultiTurnComposition,
    ],
    Field(discriminator="type"),
]
