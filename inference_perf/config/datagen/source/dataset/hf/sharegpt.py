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
from typing import Literal

from inference_perf.config.datagen.source.dataset.hf.config import HFDatasetSource
from inference_perf.config.datagen.source.markers import TextSource


class ShareGPTSource(HFDatasetSource, TextSource):
    """ShareGPT conversations from the ``anon8231489123/ShareGPT_Vicuna_unfiltered`` dataset.

    The repo hosts the corpus as a single JSON file rather than a structured
    HF dataset, so :attr:`data_files` is required alongside :attr:`path`.
    Defaults populate both so the schema works without user configuration.
    """

    type: Literal["sharegpt"] = "sharegpt"
    path: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    data_files: str = "ShareGPT_V3_unfiltered_cleaned_split.json"
