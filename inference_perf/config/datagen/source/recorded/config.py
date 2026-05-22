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
"""Recorded sources.

:class:`RecordedSource` is the intermediate base for sources whose underlying
recording carries composition constraints. Unlike text or HF dataset sources,
which can be paired with any compatible composition, a recorded source's
shape (multi-turn structure, dependency edges, timing) is dictated by the
recording itself. Subclasses declare ``valid_compositions`` so config
validation can reject incoherent pairings at load time.
"""

from typing import TYPE_CHECKING, ClassVar, FrozenSet, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    # Forward reference to avoid circular import; the Composition union lives
    # in a sibling subpackage and need not be loaded just to define this base.
    from inference_perf.config.datagen.composition.config import Composition  # noqa: F401


class RecordedSource(BaseModel):
    """Intermediate base for sources whose recording bundles composition.

    Subclasses set :attr:`valid_compositions` to the set of composition
    classes that can legally wrap this source. ``DataConfig`` validation
    consults this set when both a recorded source and a composition are
    configured.
    """

    valid_compositions: ClassVar[FrozenSet[Type["BaseModel"]]] = frozenset()
