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
from typing import Optional

from pydantic import BaseModel, Field


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path the tokenizer is loaded from. Required to activate an explicit tokenizer; if unset, the tokenizer is inferred from the model server.",
    )
    trust_remote_code: Optional[bool] = Field(
        default=None,
        description="Allow loading custom tokenizer code shipped with the model repo. Leave unset (treated as off) unless the model requires it.",
    )
    token: Optional[str] = Field(
        default=None,
        description="HuggingFace access token for private or gated models.",
    )
