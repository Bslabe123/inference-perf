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
"""Static schema describing what progress means for a (media, provenance) pair.

Two pieces:

- :class:`ProgressProfile` is the *schema* (which counters exist, in what
  units, what limits throughput). It is data-free and frozen, shared by
  reference across spec classes that produce the same kind of work (every
  remote loader shares :data:`REMOTE_DOWNLOAD_PREP`).
- Each :class:`MediaSpec` subclass exposes its applicable profiles via
  :attr:`MediaSpec.progress_profiles`. Reporters dispatch on the spec
  instance, look up profiles, and emit progress accordingly.

The split keeps spec classes pure-data and lets one definition of "remote
download progress" drive image, video, and audio reporters identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Phase(str, Enum):
    """When progress happens relative to load."""

    PREP = "prep"
    PER_REQUEST = "per_request"


class BoundBy(str, Enum):
    """What limits throughput. Drives reporter formatting (e.g. show MB/s for NETWORK)."""

    CPU = "cpu"
    NETWORK = "network"
    DISK = "disk"
    TRIVIAL = "trivial"


class Unit(str, Enum):
    ITEMS = "items"
    BYTES = "bytes"
    SECONDS = "seconds"
    MS = "ms"
    TOKENS = "tokens"
    FRAMES = "frames"
    ROWS = "rows"
    FILES = "files"


class Kind(str, Enum):
    """How a counter accumulates."""

    CUMULATIVE = "cumulative"
    GAUGE = "gauge"


@dataclass(frozen=True)
class CounterDef:
    """One counter slot in a profile."""

    name: str
    unit: Unit
    kind: Kind
    label: str


@dataclass(frozen=True)
class ProgressProfile:
    """Schema for a single progress surface.

    A spec class may declare multiple profiles when its work spans phases
    (e.g. local-file loaders index at prep then read per-request).
    """

    name: str
    phase: Phase
    bound_by: BoundBy
    counters: Tuple[CounterDef, ...]
    rate_counter: Optional[str] = None


# ---- Module-level singletons. Spec classes reference these by identity. ----

REMOTE_DOWNLOAD_PREP = ProgressProfile(
    name="remote_download",
    phase=Phase.PREP,
    bound_by=BoundBy.NETWORK,
    counters=(
        CounterDef("bytes_downloaded", Unit.BYTES, Kind.CUMULATIVE, "downloaded"),
        CounterDef("bytes_total", Unit.BYTES, Kind.GAUGE, "total"),
        CounterDef("files_completed", Unit.FILES, Kind.CUMULATIVE, "files"),
        CounterDef("retries", Unit.ITEMS, Kind.CUMULATIVE, "retries"),
    ),
    rate_counter="bytes_downloaded",
)

LOCAL_INDEX_PREP = ProgressProfile(
    name="local_index",
    phase=Phase.PREP,
    bound_by=BoundBy.DISK,
    counters=(
        CounterDef("files_indexed", Unit.FILES, Kind.CUMULATIVE, "indexed"),
        CounterDef("missing_paths", Unit.ITEMS, Kind.CUMULATIVE, "missing"),
    ),
    rate_counter="files_indexed",
)

SYNTHETIC_IMAGE_PER_REQ = ProgressProfile(
    name="synthetic_image",
    phase=Phase.PER_REQUEST,
    bound_by=BoundBy.CPU,
    counters=(
        CounterDef("items", Unit.ITEMS, Kind.CUMULATIVE, "images"),
        CounterDef("bytes", Unit.BYTES, Kind.CUMULATIVE, "bytes"),
        CounterDef("encode_ms", Unit.MS, Kind.CUMULATIVE, "encode time"),
    ),
    rate_counter="items",
)

SYNTHETIC_VIDEO_MP4_PER_REQ = ProgressProfile(
    name="synthetic_video_mp4",
    phase=Phase.PER_REQUEST,
    bound_by=BoundBy.CPU,
    counters=(
        CounterDef("videos", Unit.ITEMS, Kind.CUMULATIVE, "videos"),
        CounterDef("frames", Unit.FRAMES, Kind.CUMULATIVE, "frames"),
        CounterDef("bytes", Unit.BYTES, Kind.CUMULATIVE, "bytes"),
        CounterDef("encode_ms", Unit.MS, Kind.CUMULATIVE, "encode time"),
    ),
    rate_counter="videos",
)

SYNTHETIC_VIDEO_FRAMES_PER_REQ = ProgressProfile(
    name="synthetic_video_frames",
    phase=Phase.PER_REQUEST,
    bound_by=BoundBy.CPU,
    counters=(
        CounterDef("videos", Unit.ITEMS, Kind.CUMULATIVE, "videos"),
        CounterDef("frames", Unit.FRAMES, Kind.CUMULATIVE, "frames"),
        CounterDef("bytes", Unit.BYTES, Kind.CUMULATIVE, "bytes"),
        CounterDef("frame_encode_ms", Unit.MS, Kind.CUMULATIVE, "encode time"),
    ),
    rate_counter="frames",
)

SYNTHETIC_AUDIO_PER_REQ = ProgressProfile(
    name="synthetic_audio",
    phase=Phase.PER_REQUEST,
    bound_by=BoundBy.CPU,
    counters=(
        CounterDef("items", Unit.ITEMS, Kind.CUMULATIVE, "clips"),
        CounterDef("duration_s", Unit.SECONDS, Kind.CUMULATIVE, "duration"),
        CounterDef("bytes", Unit.BYTES, Kind.CUMULATIVE, "bytes"),
        CounterDef("encode_ms", Unit.MS, Kind.CUMULATIVE, "encode time"),
    ),
    rate_counter="items",
)

# Pre-encoded variants do no meaningful work on either side; one shared profile
# is enough to keep contract tests passing without inviting per-item reporting.
PRE_ENCODED_PER_REQ = ProgressProfile(
    name="pre_encoded_passthrough",
    phase=Phase.PER_REQUEST,
    bound_by=BoundBy.TRIVIAL,
    counters=(
        CounterDef("items", Unit.ITEMS, Kind.CUMULATIVE, "items"),
        CounterDef("bytes", Unit.BYTES, Kind.CUMULATIVE, "bytes"),
    ),
    rate_counter="items",
)


__all__ = [
    "BoundBy",
    "CounterDef",
    "Kind",
    "LOCAL_INDEX_PREP",
    "PRE_ENCODED_PER_REQ",
    "Phase",
    "ProgressProfile",
    "REMOTE_DOWNLOAD_PREP",
    "SYNTHETIC_AUDIO_PER_REQ",
    "SYNTHETIC_IMAGE_PER_REQ",
    "SYNTHETIC_VIDEO_FRAMES_PER_REQ",
    "SYNTHETIC_VIDEO_MP4_PER_REQ",
    "Unit",
]
