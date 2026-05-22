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
"""HuggingFace dataset source base.

:class:`HFDatasetSource` is the intermediate base for any source loaded via
the HuggingFace ``datasets`` library. It holds the loader knobs every HF
source shares (path, split, revision, data_files) and is meant to be
inherited rather than instantiated directly. Concrete public datasets live
in sibling modules (``cnn_dailymail.py`` etc.); gated datasets live under
:mod:`.gated`. Subclasses set their own canonical default for ``path`` so
the schema works without user configuration.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class HFDatasetSource(BaseModel):
    """Common HuggingFace dataset loader configuration.

    Intermediate type; not directly addressable as a ``DataSource`` variant.
    Subclasses add a ``type`` discriminator and provide a default value for
    ``path`` (the HuggingFace dataset identifier).
    """

    path: str = Field(
        description="HuggingFace dataset identifier, e.g. ``username/dataset-name``. Subclasses populate the canonical default.",
    )
    split: str = Field(
        default="train",
        description="HuggingFace split to stream.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Optional revision (branch / tag / commit) to pin.",
    )
    data_files: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Optional ``data_files`` glob(s) forwarded to ``load_dataset``.",
    )
