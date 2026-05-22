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
from typing import ClassVar, FrozenSet, Literal, Type

from pydantic import BaseModel

from inference_perf.config.datagen.replay import OTelTraceReplayConfig
from inference_perf.config.datagen.source.markers import TextSource
from inference_perf.config.datagen.source.recorded.config import RecordedSource


class OTelTraceReplaySource(OTelTraceReplayConfig, RecordedSource, TextSource):
    """OTel trace replay source.

    The recording dictates the multi-turn dependency structure of the
    benchmark, so this source is coupled to multi-turn composition. Today
    the existing :class:`OTelTraceReplayConfig` carries both source-shaped
    fields (trace_directory / trace_files / hf_dataset_path,
    include_errors, skip_invalid_files) and composition-shaped fields
    (model_mapping, use_static_model, default_max_tokens). The split
    between source and composition concerns is left for a follow-up; for
    now this class relabels the existing config as a recorded source so
    the categorisation is expressible.
    """

    type: Literal["otel_trace_replay"] = "otel_trace_replay"

    valid_compositions: ClassVar[FrozenSet[Type[BaseModel]]] = frozenset()
    # Populated after MultiTurnComposition is wired in to avoid a circular import.
