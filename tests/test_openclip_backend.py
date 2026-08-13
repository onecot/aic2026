"""Network-free tests for the optional OpenCLIP backend."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aic2026.config import OpenClipConfig, load_config
from aic2026.embeddings import Embedding, validate_query_compatibility
from aic2026.model_backends import openclip
from aic2026.model_backends.openclip import OpenClipEncoder


class _Tensor:
    """Minimal Torch-tensor stand-in used to test the backend boundary."""

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.device: str | None = None

    def to(self, device: str) -> _Tensor:
        self.device = device
        return self

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


@dataclass
class _Availability:
    available: bool

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _Availability(cuda_available)
        self.backends = type("Backends", (), {"mps": _Availability(mps_available)})()

    @staticmethod
    def stack(tensors: list[_Tensor]) -> _Tensor:
        return _Tensor(np.stack([tensor.values for tensor in tensors]))

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _FakeModel:
    def __init__(self) -> None:
        self.visual = type("Visual", (), {"output_dim": 3})()
        self.evaluated = False

    def eval(self) -> None:
        self.evaluated = True

    @staticmethod
    def encode_text(tokens: _Tensor) -> _Tensor:
        return _Tensor(tokens.values + np.array([0.0, 1.0, 2.0], dtype=np.float32))

    @staticmethod
    def encode_image(batch: _Tensor) -> _Tensor:
        return _Tensor(batch.values + np.array([2.0, 1.0, 0.0], dtype=np.float32))


class _FakeOpenClip:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.create_args: dict[str, str] | None = None

    def create_model_and_transforms(self, **kwargs: str) -> tuple[_FakeModel, object, object]:
        self.create_args = kwargs
        return self.model, object(), lambda image: _Tensor([float(len(str(image))), 1.0, 1.0])

    @staticmethod
    def get_tokenizer(model_name: str) -> Any:
        def tokenize(texts: list[str]) -> _Tensor:
            return _Tensor([[float(len(text)), 1.0, 1.0] for text in texts])

        return tokenize


def _config(**overrides: str) -> OpenClipConfig:
    values = {
        "backend": "openclip",
        "model_name": "test-model",
        "pretrained": "test-checkpoint",
        "device": "auto",
        "precision": "fp32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
        "tokenizer_id": "test-tokenizer-v1",
    }
    values.update(overrides)
    return OpenClipConfig(**values)


def _encoder(monkeypatch: pytest.MonkeyPatch, torch: _FakeTorch | None = None) -> tuple[OpenClipEncoder, _FakeOpenClip]:
    fake_openclip = _FakeOpenClip()
    monkeypatch.setattr(openclip, "_load_dependencies", lambda: (torch or _FakeTorch(), fake_openclip))
    return OpenClipEncoder(_config()), fake_openclip


def test_default_config_exposes_openclip_settings() -> None:
    """The file-driven default configuration includes OpenCLIP settings."""
    config = load_config()
    assert config.openclip.backend == "openclip"
    assert config.openclip.device == "auto"


def test_backend_initialization_and_embedding_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialization loads the configured runtime and declares its full space."""
    encoder, fake_openclip = _encoder(monkeypatch)
    assert fake_openclip.model.evaluated
    assert fake_openclip.create_args == {
        "model_name": "test-model",
        "pretrained": "test-checkpoint",
        "device": "cpu",
        "precision": "fp32",
    }
    assert encoder.embedding_space.backend == "openclip"
    assert encoder.embedding_space.model_id == "test-model"
    assert encoder.embedding_space.checkpoint_id == "test-checkpoint"
    assert encoder.embedding_space.embedding_dim == 3
    assert encoder.embedding_space.output_dtype == "float32"
    assert encoder.embedding_space.normalization == "l2"
    assert "test-preprocess-v1" in encoder.embedding_space.preprocessing_id
    assert "test-tokenizer-v1" in encoder.embedding_space.preprocessing_id


def test_single_text_and_image_encoding_share_compatible_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-modality protocol methods return normalized contract values."""
    encoder, _ = _encoder(monkeypatch)
    text = encoder.encode_text("tiny text")
    image = encoder.encode_image("tiny image")
    assert isinstance(text, Embedding)
    assert text.kind == "text"
    assert image.kind == "image"
    assert text.space == image.space == encoder.embedding_space
    assert text.vector.dtype == np.float32
    assert image.shape == (3,)
    assert np.linalg.norm(text.vector) == pytest.approx(1.0, rel=1e-6)
    assert np.linalg.norm(image.vector) == pytest.approx(1.0, rel=1e-6)
    validate_query_compatibility(text, image.space)


def test_batched_text_and_image_encoding_preserve_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch helpers encode multiple values without changing their shared space."""
    encoder, _ = _encoder(monkeypatch)
    texts = encoder.encode_text_batch(["a", "longer"])
    images = encoder.encode_image_batch(["x", "longer-image"])
    assert [item.kind for item in texts] == ["text", "text"]
    assert [item.kind for item in images] == ["image", "image"]
    assert [item.space for item in texts + images] == [encoder.embedding_space] * 4
    assert not np.array_equal(texts[0].vector, texts[1].vector)
    assert not np.array_equal(images[0].vector, images[1].vector)


def test_precision_controls_output_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured fp16 output is represented consistently in metadata and vectors."""
    fake_openclip = _FakeOpenClip()
    monkeypatch.setattr(openclip, "_load_dependencies", lambda: (_FakeTorch(), fake_openclip))
    encoder = OpenClipEncoder(_config(precision="fp16"))
    assert encoder.embedding_space.output_dtype == "float16"
    assert encoder.encode_text("text").vector.dtype == np.float16


@pytest.mark.parametrize(
    "overrides",
    [{"device": "invalid"}, {"precision": "bf16"}, {"normalization": "unknown"}],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, str]) -> None:
    """Unsupported configuration values fail before model loading."""
    with pytest.raises(ValueError):
        _config(**overrides)


def test_unavailable_requested_device_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable accelerator is reported instead of silently falling back."""
    fake_openclip = _FakeOpenClip()
    monkeypatch.setattr(openclip, "_load_dependencies", lambda: (_FakeTorch(), fake_openclip))
    with pytest.raises(ValueError, match="not available"):
        OpenClipEncoder(_config(device="cuda"))


def test_auto_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto resolution supports MPS when it is the available accelerator."""
    fake_openclip = _FakeOpenClip()
    monkeypatch.setattr(openclip, "_load_dependencies", lambda: (_FakeTorch(mps_available=True), fake_openclip))
    assert OpenClipEncoder(_config()).device == "mps"


def test_batch_input_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty and malformed batch inputs fail clearly."""
    encoder, _ = _encoder(monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_text_batch([])
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_image_batch([])
    with pytest.raises(ValueError, match="non-empty strings"):
        encoder.encode_text_batch([""])
