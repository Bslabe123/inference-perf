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
"""Data source configuration.

A ``DataSource`` describes *where prompts come from* (a real dataset, a
synthetic generator, a mocked input), independent of how those prompts are
shaped into requests (single-turn, shared-prefix, multi-turn).

Layout (one subpackage per origin category, no leaf source files at this
level):
  - :mod:`.synthetic`: in-process generators (mock, synthetic, random).
  - :mod:`.dataset`: real-world corpora, sub-categorised by loader
    mechanism (today only ``.hf`` for the HuggingFace datasets API, and
    ``.hf.gated`` for HF datasets that require auth).
  - :mod:`.recorded`: trace replays whose recording bundles composition
    constraints with the source itself.

Composition-bound generators that are not sources (conversation replay,
shared-prefix synthetic) remain modelled separately; they consume request
prompts rather than just supplying them.
"""

from typing import Annotated, Union

from pydantic import Field

from inference_perf.config.datagen.source.dataset.hf.billsum_conversations import BillsumConversationsSource
from inference_perf.config.datagen.source.dataset.hf.cnn_dailymail import CNNDailyMailSource
from inference_perf.config.datagen.source.dataset.hf.infinity_instruct import InfinityInstructSource
from inference_perf.config.datagen.source.dataset.hf.sharegpt import ShareGPTSource
from inference_perf.config.datagen.source.recorded.otel import OTelTraceReplaySource
from inference_perf.config.datagen.source.synthetic.mock import MockSource
from inference_perf.config.datagen.source.synthetic.random import RandomSource
from inference_perf.config.datagen.source.synthetic.synthetic import SyntheticSource

DataSource = Annotated[
    Union[
        MockSource,
        ShareGPTSource,
        SyntheticSource,
        RandomSource,
        CNNDailyMailSource,
        InfinityInstructSource,
        BillsumConversationsSource,
        OTelTraceReplaySource,
    ],
    Field(discriminator="type"),
]
# ShareGPT4VideoSource will be added in a follow-up once
# sharegpt4video-dataset (PR #494) merges; it will extend GatedHFDatasetSource
# and MultimodalSource, and live at ``dataset/hf/gated/sharegpt4video.py``.
