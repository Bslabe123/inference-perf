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
"""Delivery configuration.

Delivery describes *how data is dispatched* at runtime once source content
has been generated or loaded: arrival timing, per-modality media pooling,
and similar runtime mechanics. It is the bottom of the canonical
source -> composition -> delivery stack, and validates against the source's
marker bases (a synthetic pool config is only valid against ``TextSource``
or a generative-multimodal source; not against a ``RecordedSource`` or a
dataset-multimodal source whose media is fixed by the recording / dataset).
"""

from typing import Optional

from pydantic import BaseModel, Field

from inference_perf.config.datagen.delivery.arrival import ArrivalSchedule


class DeliveryConfig(BaseModel):
    """Delivery aspects, all optional and independently composable."""

    arrival: Optional[ArrivalSchedule] = Field(
        default=None,
        description="Optional arrival schedule. When set, request dispatch follows the named schedule rather than the load-generator's default interval / rate.",
    )
    # ``pool`` (per-modality distinct-media pool) lands here once the
    # corresponding runtime work merges. See branch 8651713.
