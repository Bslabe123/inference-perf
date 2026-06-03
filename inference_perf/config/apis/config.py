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
from typing import Any, Optional

from pydantic import BaseModel, Field


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class ResponseFormatType(Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class ResponseFormat(BaseModel):
    """Configuration for structured output via response_format parameter.

    See vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    type: ResponseFormatType = Field(
        default=ResponseFormatType.JSON_SCHEMA,
        description="json_schema (constrain output to a schema) or json_object (any valid JSON object).",
    )
    name: str = Field(
        default="structured_output",
        description="Name for the JSON schema (used when type is json_schema).",
    )
    json_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="The JSON schema the output must conform to (used when type is json_schema).",
    )

    def to_api_format(self) -> dict[str, Any]:
        """Convert to the format expected by vLLM/OpenAI API."""
        if self.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        # json_schema type
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.json_schema,
            },
        }


class APIConfig(BaseModel):
    type: APIType = Field(
        default=APIType.Completion,
        description="API surface to target (completion or chat).",
    )
    streaming: bool = Field(
        default=False,
        description="Stream the response. Required to measure TTFT, ITL, and TPOT.",
    )
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Custom HTTP headers sent with every request.",
    )
    slo_unit: Optional[str] = Field(
        default=None,
        description="Unit for SLO header values (e.g. ms, s).",
    )
    slo_tpot_header: Optional[str] = Field(
        default=None,
        description="Name of the header carrying the per-request TPOT SLO.",
    )
    slo_ttft_header: Optional[str] = Field(
        default=None,
        description="Name of the header carrying the per-request TTFT SLO.",
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Structured-output spec.",
    )
