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
from typing import Optional

from pydantic import BaseModel, Field


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"


class ModelServerClientConfig(BaseModel):
    type: ModelServerType = Field(default=ModelServerType.VLLM, description="Backend flavor (vllm, sglang, tgi, or mock).")
    model_name: Optional[str] = Field(
        default=None, description="Model identifier the server should serve (for example, a Hugging Face repo ID)."
    )
    base_url: str = Field(..., description="Server endpoint, including scheme, host, and port.")
    ignore_eos: bool = Field(
        default=True,
        description="Whether to ignore End-of-Sequence tokens so generation runs to the requested length.",
    )
    api_key: Optional[str] = Field(default=None, description="API key for authenticated endpoints.")
    cert_path: Optional[str] = Field(default=None, description="Path to a client TLS certificate for mutual-TLS endpoints.")
    key_path: Optional[str] = Field(default=None, description="Path to the private key paired with cert_path.")
