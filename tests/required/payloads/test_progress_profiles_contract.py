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
"""Contract: every concrete provenance variant declares a progress profile.

A reporter looks up ``cls.progress_profiles`` to decide how to format
progress for that variant. A variant that forgets to override falls back to
the empty default on :class:`MediaSpec` and disappears from reports
silently. This test prevents that drift.
"""

from typing import Any, Type, get_args

import pytest

from inference_perf.payloads import AudioSpecUnion, ImageSpecUnion, VideoSpecUnion
from inference_perf.payloads.media_base import MediaSpec


def _concrete_variants(union_type: Any) -> list[Type[MediaSpec[Any]]]:
    annotated_args = get_args(union_type)
    union = annotated_args[0]
    return list(get_args(union))


@pytest.mark.parametrize(
    "union_alias, label",
    [
        (ImageSpecUnion, "ImageSpecUnion"),
        (AudioSpecUnion, "AudioSpecUnion"),
        (VideoSpecUnion, "VideoSpecUnion"),
    ],
)
def test_every_variant_declares_progress_profiles(union_alias: Any, label: str) -> None:
    variants = _concrete_variants(union_alias)
    assert variants, f"{label} is empty: did the union annotation change?"

    bad = [cls.__name__ for cls in variants if not getattr(cls, "progress_profiles", ())]
    assert not bad, (
        f"{label} member(s) do not declare progress_profiles. Reporters can't "
        f"format progress for these variants. Add a ClassVar that references "
        f"the appropriate profile from inference_perf.observability. Offenders: {bad}"
    )


def test_media_base_default_is_empty_tuple() -> None:
    # Sanity: the default exists so abstract bases don't accidentally pass the
    # contract test by inheritance. Concrete variants must override.
    assert MediaSpec.progress_profiles == ()
