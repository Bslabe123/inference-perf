# Copyright 2025 The Kubernetes Authors.
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

from typing import Any, List
import aiohttp
from pydantic import BaseModel
from inference_perf.prompts import InferenceData, PromptLifecycleMetric, ResponseData, ResponsesSummary
from inference_perf.prompts.base import safe_float, summarize
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class LlmChatCompletionInferenceData(InferenceData):
    messages: List[ChatMessage]

    def get_route(self) -> str:
        return "/v1/chat/completions"

    def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
        }

    async def process_response(self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        output_text = choices[0].get("message", {}).get("content", "")
        output_len = tokenizer.count_tokens(output_text)
        return ResponseData(
            info={
                "output_text": output_text,
                "output_len": output_len,
            },
            error=None,
        )

    def summarize_requests(self, metrics: List[PromptLifecycleMetric]) -> ResponsesSummary:
        all_successful: List[PromptLifecycleMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[PromptLifecycleMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
            },
            successes={
                "count": len(all_successful),
                "request_latency": summarize([(successful.end_time - successful.start_time) for successful in all_successful]),
                "output_len": summarize(
                    [v for success in all_successful if (v := safe_float(success.response.info.get("output_len"))) is not None]
                ),
                "normalized_time_per_output_token": summarize(
                    [
                        ((metric.end_time - metric.start_time) / output_len) if output_len and output_len != 0 else 0
                        for metric in all_successful
                        for output_len in [safe_float(metric.response.info.get("output_len"))]
                    ]
                ),
            },
            failures={
                "count": len(all_failed),
                "request_latency": summarize([(failed.end_time - failed.start_time) for failed in all_failed]),
            },
        )
