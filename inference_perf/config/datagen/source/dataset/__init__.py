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
"""Real-corpus sources.

Sources that pull prompt content from real-world datasets. Sub-categorised
by loader mechanism:

  - :mod:`.hf`: HuggingFace datasets-API loaders (load_dataset by dataset
    identifier). The ``path`` field holds the canonical HF identifier and
    the HF library handles caching and revision resolution. Gated datasets
    that require auth live under :mod:`.hf.gated`.

Non-HF loader mechanisms (raw file paths, S3 URIs, etc.) will land as
sibling subpackages here when there's an actual source that needs them.
"""

from inference_perf.config.datagen.source.dataset.hf import (
    BillsumConversationsSource,
    CNNDailyMailSource,
    GatedHFDatasetSource,
    HFDatasetSource,
    InfinityInstructSource,
    ShareGPTSource,
)

__all__ = [
    "BillsumConversationsSource",
    "CNNDailyMailSource",
    "GatedHFDatasetSource",
    "HFDatasetSource",
    "InfinityInstructSource",
    "ShareGPTSource",
]
