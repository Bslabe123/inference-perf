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

from pydantic import Field

from inference_perf.config.common import Distribution
from inference_perf.config.datagen.source.markers import TextSource


class SyntheticSource(TextSource):
    """Synthetic prompts sampled from configurable token-length distributions."""

    type: Literal["synthetic"] = "synthetic"
    input_distribution: Optional[Distribution] = Field(
        default=None,
        description="Distribution sampled for prompt token length.",
    )
    output_distribution: Optional[Distribution] = Field(
        default=None,
        description="Distribution sampled for target output token length.",
    )
