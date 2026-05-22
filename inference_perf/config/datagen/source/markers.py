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
"""Source capability markers.

Empty pydantic base classes used as type-level markers so delivery
validators (and any other consumer that needs to reason about a source's
output) can use ``isinstance`` checks rather than string fiddling. They
carry no fields; concrete sources mix them in via multiple inheritance
alongside their loader hierarchy (HFDatasetSource, GatedHFDatasetSource,
RecordedSource).
"""

from pydantic import BaseModel


class TextSource(BaseModel):
    """Marker base for sources that emit text-only prompts."""

    pass


class MultimodalSource(BaseModel):
    """Marker base for sources that emit prompts with non-text media (image, video, audio)."""

    pass
