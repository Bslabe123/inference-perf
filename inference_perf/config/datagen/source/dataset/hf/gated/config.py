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
"""Gated HuggingFace dataset source base.

Adds token resolution on top of :class:`HFDatasetSource`. Subclasses are
sources that require HuggingFace authentication; the loader resolves the
token from the explicit ``token`` field first, then falls back to the
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars.
"""

from typing import Optional

from pydantic import Field

from inference_perf.config.datagen.source.dataset.hf.config import HFDatasetSource


class GatedHFDatasetSource(HFDatasetSource):
    """HuggingFace dataset source that requires authentication."""

    token: Optional[str] = Field(
        default=None,
        description=(
            "HuggingFace access token used to download the gated dataset. "
            "Falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars "
            "when omitted."
        ),
    )
