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
"""Loader behavior for the VisionArena-Chat data generator.

``load_dataset`` is monkeypatched with an in-memory iterable mirroring the
VisionArena schema (``conversation`` turns + undecoded ``images``), so these
tests never touch the network.
"""

import io
from typing import Any, List

import pytest
from PIL import Image as PILImage

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.config import Config, DataGenType
from inference_perf.datagen import visionarena_datagen
from inference_perf.datagen.visionarena_datagen import VisionArenaDataGenerator
from inference_perf.payloads import ImageRepresentation, PreEncodedImageSpec


def _png_bytes(w: int = 16, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(w: int = 16, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (30, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _image_field(raw: bytes) -> dict:
    """Mirror the undecoded HuggingFace Image form (decode=false)."""
    return {"bytes": raw, "path": None}


def _row(prompt: str, image_blobs: List[bytes]) -> dict:
    return {
        "conversation": [{"role": "user", "content": prompt}],
        "images": [_image_field(b) for b in image_blobs],
    }


def _make_generator(monkeypatch: pytest.MonkeyPatch, rows: List[Any], **va_overrides: Any) -> VisionArenaDataGenerator:
    monkeypatch.setattr(visionarena_datagen, "load_dataset", lambda *a, **k: list(rows))
    config = Config.model_validate(
        {"api": {"type": "chat"}, "data": {"type": DataGenType.VisionArena, "visionarena": va_overrides}}
    )
    return VisionArenaDataGenerator(config.api, config.data, None)


def test_pool_built_and_request_carries_pre_encoded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _png_bytes()
    gen = _make_generator(monkeypatch, [_row("describe", [raw]), _row("what is this", [_png_bytes(8, 8)])])
    assert len(gen._pool) == 2

    first = next(gen.get_data())
    assert isinstance(first, LazyLoadInferenceAPIData)

    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
    assert isinstance(data, ChatCompletionAPIData)
    assert data.messages[0].content == "describe"
    assert data.multimodal_spec is not None
    images = data.multimodal_spec.images
    assert len(images) == 1
    spec = images[0]
    assert isinstance(spec, PreEncodedImageSpec)
    assert spec.image_bytes == raw  # PNG passed through verbatim, not re-encoded
    assert spec.representation == ImageRepresentation.PNG
    assert (spec.width, spec.height) == (16, 8)


def test_cycles_through_pool_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(monkeypatch, [_row("a", [_png_bytes()]), _row("b", [_png_bytes()])])
    assert gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).messages[0].content == "a"
    assert gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1)).messages[0].content == "b"
    # Wraps around.
    assert gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=2)).messages[0].content == "a"


def test_skips_rows_without_prompt_or_images(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"conversation": [{"role": "assistant", "content": "no user turn"}], "images": [_image_field(_png_bytes())]},
        _row("has prompt but no images", []),
        _row("good row", [_png_bytes()]),
    ]
    gen = _make_generator(monkeypatch, rows)
    assert len(gen._pool) == 1
    assert gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).messages[0].content == "good row"


def test_num_rows_caps_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(f"prompt {i}", [_png_bytes()]) for i in range(5)]
    gen = _make_generator(monkeypatch, rows, num_rows=2)
    assert len(gen._pool) == 2


def test_max_images_per_request_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(
        monkeypatch,
        [_row("multi", [_png_bytes(), _png_bytes(), _png_bytes()])],
        max_images_per_request=2,
    )
    images = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).multimodal_spec.images
    assert len(images) == 2


def test_jpeg_passes_through_as_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _jpeg_bytes()
    gen = _make_generator(monkeypatch, [_row("jpeg", [raw])])
    spec = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).multimodal_spec.images[0]
    assert spec.representation == ImageRepresentation.JPEG
    assert spec.image_bytes == raw


def test_bare_pil_image_is_reencoded_to_png(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some HF configs hand back decoded PIL images instead of undecoded bytes;
    # the loader must re-encode those (no original encoded bytes to pass through).
    row = {"conversation": [{"role": "user", "content": "decoded"}], "images": [PILImage.new("RGB", (12, 9), (0, 128, 0))]}
    gen = _make_generator(monkeypatch, [row])
    spec = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).multimodal_spec.images[0]
    assert spec.representation == ImageRepresentation.PNG
    assert (spec.width, spec.height) == (12, 9)
    # Re-encoded bytes are a valid PNG.
    assert PILImage.open(io.BytesIO(spec.image_bytes)).format == "PNG"


def test_max_image_dim_downscales(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(monkeypatch, [_row("big", [_png_bytes(100, 40)])], max_image_dim=50)
    spec = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0)).multimodal_spec.images[0]
    assert max(spec.width, spec.height) == 50
    assert (spec.width, spec.height) == (50, 20)


def test_empty_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError):
        _make_generator(monkeypatch, [])
