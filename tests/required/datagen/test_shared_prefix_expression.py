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
from typing import Any

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, SharedPrefix
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class _MockHFTokenizer:
    vocab_size = 50000

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join([str(tid) for tid in token_ids])

    def batch_decode(self, token_ids_list: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids) for ids in token_ids_list]


class _MockCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _MockHFTokenizer()

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_shared_prefix_accepts_length_expression() -> None:
    """A question_len expression is sampled per-prompt and bounds the prompt size."""
    api_config = APIConfig(type=APIType.Completion)

    shared_prefix_cfg = SharedPrefix(
        num_groups=2,
        num_prompts_per_group=10,
        system_prompt_len=64,
        # Clamped normal: every question length lands in [30, 50].
        question_len="Min(Max(Normal(40, 8), 30), 50)",
        output_len=16,
        enable_multi_turn_chat=False,
        seed=123,
    )
    config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)
    tokenizer = _MockCustomTokenizer()

    generator = SharedPrefixDataGenerator(api_config, config, tokenizer)

    assert len(generator.prompts) == 20
    for prompt in generator.prompts:
        # prompt tokens = system_prompt_len (64) + question_len in [30, 50].
        assert 94 <= tokenizer.count_tokens(prompt) <= 114
