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
"""Datagen-side progress observability.

The user-facing contract is ``BaseGenerator.prepare(reporter)``: any pre-load
work a datagen needs to do (download dataset shards, build an index, scan a
local directory) is announced through a :class:`ProgressReporter`. The
reporter implementation decides how progress surfaces (INFO log lines for
cluster job logs, ``rich`` progress bars for interactive terminals, no-op for
tests).

The *shape* of what each datagen reports is determined by the (media,
provenance) pair on the request it produces: a synthetic image generator
reports items encoded, a remote-video loader reports bytes downloaded. That
mapping is statically declared on each spec class via
:attr:`MediaSpec.progress_profiles` and the singletons in
:mod:`.progress_profile`; reporters consume the declared profiles rather than
hard-coding any one shape.
"""

from .progress_profile import (
    BoundBy,
    CounterDef,
    Kind,
    Phase,
    ProgressProfile,
    Unit,
    LOCAL_INDEX_PREP,
    PRE_ENCODED_PER_REQ,
    REMOTE_DOWNLOAD_PREP,
    SYNTHETIC_AUDIO_PER_REQ,
    SYNTHETIC_IMAGE_PER_REQ,
    SYNTHETIC_VIDEO_FRAMES_PER_REQ,
    SYNTHETIC_VIDEO_MP4_PER_REQ,
)
from .reporter import (
    LogReporter,
    NullReporter,
    ProgressReporter,
    RichReporter,
    Task,
    make_reporter,
)

__all__ = [
    "BoundBy",
    "CounterDef",
    "Kind",
    "LOCAL_INDEX_PREP",
    "LogReporter",
    "NullReporter",
    "PRE_ENCODED_PER_REQ",
    "Phase",
    "ProgressProfile",
    "ProgressReporter",
    "REMOTE_DOWNLOAD_PREP",
    "RichReporter",
    "SYNTHETIC_AUDIO_PER_REQ",
    "SYNTHETIC_IMAGE_PER_REQ",
    "SYNTHETIC_VIDEO_FRAMES_PER_REQ",
    "SYNTHETIC_VIDEO_MP4_PER_REQ",
    "Task",
    "Unit",
    "make_reporter",
]
