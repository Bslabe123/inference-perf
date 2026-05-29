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
"""Validity rules for the VisionArena ``data:`` block (``Config.data.visionarena``)."""

import pytest
from pydantic import ValidationError

from inference_perf.config import Config, DataGenType, VisionArenaConfig


def test_visionarena_defaults() -> None:
    config = Config.model_validate({"data": {"type": DataGenType.VisionArena, "visionarena": {}}})
    assert config.data.type == DataGenType.VisionArena
    va = config.data.visionarena
    assert va is not None
    assert va.hf_dataset_name == "lmarena-ai/VisionArena-Chat"
    assert va.hf_split == "train"
    assert va.num_rows == 1000
    assert va.max_images_per_request == 1
    assert va.max_image_dim is None
    assert va.insertion_point == 0.0


def test_visionarena_custom_values_preserved() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.VisionArena,
                "visionarena": {
                    "hf_dataset_name": "my-org/mirror",
                    "hf_split": "validation",
                    "num_rows": 50,
                    "max_images_per_request": 3,
                    "max_image_dim": 512,
                    "insertion_point": 1.0,
                },
            }
        }
    )
    va = config.data.visionarena
    assert va is not None
    assert va.hf_dataset_name == "my-org/mirror"
    assert va.hf_split == "validation"
    assert va.num_rows == 50
    assert va.max_images_per_request == 3
    assert va.max_image_dim == 512
    assert va.insertion_point == 1.0


def test_visionarena_absent_for_other_types() -> None:
    config = Config.model_validate({"data": {"type": DataGenType.Mock}})
    assert config.data.visionarena is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_rows", 0),
        ("num_rows", -1),
        ("max_images_per_request", 0),
        ("max_image_dim", 0),
        ("insertion_point", -0.1),
        ("insertion_point", 1.1),
    ],
)
def test_visionarena_rejects_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        VisionArenaConfig(**{field: value})
