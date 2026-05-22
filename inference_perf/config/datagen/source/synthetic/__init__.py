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
"""In-process synthetic sources.

Sources that generate prompt content in-process, without loading any
external dataset. Useful for capability tests, reproducible benchmarks
under fabricated load, and pipeline validation.
"""

from inference_perf.config.datagen.source.synthetic.mock import MockSource
from inference_perf.config.datagen.source.synthetic.random import RandomSource
from inference_perf.config.datagen.source.synthetic.synthetic import SyntheticSource

__all__ = [
    "MockSource",
    "RandomSource",
    "SyntheticSource",
]
