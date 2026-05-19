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
"""Profile schema invariants.

Profiles are referenced by identity across spec classes, so accidental
duplication of "remote download" with slightly different counters would
silently fragment reporter output. The shape tests pin down the contract
spec classes are coding against.
"""

from inference_perf.observability import (
    LOCAL_INDEX_PREP,
    PRE_ENCODED_PER_REQ,
    REMOTE_DOWNLOAD_PREP,
    SYNTHETIC_AUDIO_PER_REQ,
    SYNTHETIC_IMAGE_PER_REQ,
    SYNTHETIC_VIDEO_FRAMES_PER_REQ,
    SYNTHETIC_VIDEO_MP4_PER_REQ,
    BoundBy,
    Phase,
    ProgressProfile,
)


_ALL_PROFILES = [
    REMOTE_DOWNLOAD_PREP,
    LOCAL_INDEX_PREP,
    SYNTHETIC_IMAGE_PER_REQ,
    SYNTHETIC_VIDEO_MP4_PER_REQ,
    SYNTHETIC_VIDEO_FRAMES_PER_REQ,
    SYNTHETIC_AUDIO_PER_REQ,
    PRE_ENCODED_PER_REQ,
]


def test_profiles_have_unique_names() -> None:
    names = [p.name for p in _ALL_PROFILES]
    assert len(names) == len(set(names)), f"Duplicate profile name(s): {names}"


def test_profiles_have_non_empty_counters() -> None:
    for p in _ALL_PROFILES:
        assert p.counters, f"{p.name} has no counters declared"


def test_rate_counter_references_declared_counter() -> None:
    for p in _ALL_PROFILES:
        if p.rate_counter is None:
            continue
        names = {c.name for c in p.counters}
        assert p.rate_counter in names, f"{p.name}.rate_counter={p.rate_counter!r} not in declared counters {names}"


def test_remote_download_is_network_bound_prep() -> None:
    assert REMOTE_DOWNLOAD_PREP.phase is Phase.PREP
    assert REMOTE_DOWNLOAD_PREP.bound_by is BoundBy.NETWORK


def test_local_index_is_disk_bound_prep() -> None:
    assert LOCAL_INDEX_PREP.phase is Phase.PREP
    assert LOCAL_INDEX_PREP.bound_by is BoundBy.DISK


def test_synthetic_profiles_are_cpu_per_request() -> None:
    for p in (SYNTHETIC_IMAGE_PER_REQ, SYNTHETIC_VIDEO_MP4_PER_REQ, SYNTHETIC_VIDEO_FRAMES_PER_REQ, SYNTHETIC_AUDIO_PER_REQ):
        assert p.phase is Phase.PER_REQUEST
        assert p.bound_by is BoundBy.CPU


def test_profiles_are_frozen_dataclasses() -> None:
    # Mutation would let one consumer corrupt every other consumer's view.
    import dataclasses

    p = REMOTE_DOWNLOAD_PREP
    assert dataclasses.is_dataclass(p)
    try:
        p.name = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ProgressProfile is mutable; should be frozen")


def test_profile_identity_is_shared() -> None:
    # Spec classes import the singletons; identity (not equality) is what
    # lets a reporter de-dupe progress surfaces across spec instances.
    from inference_perf.observability import REMOTE_DOWNLOAD_PREP as a
    from inference_perf.observability import REMOTE_DOWNLOAD_PREP as b

    assert a is b
    assert isinstance(a, ProgressProfile)
