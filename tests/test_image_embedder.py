"""Tests for the reusable image-embedder interface, device handling, and registry."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aic2026.embeddings import EmbeddingSpace
from aic2026.image_embedder import (
    DEFAULT_BACKEND,
    DEFAULT_PRETRAINED,
    ImageEmbedder,
    _parse_model_spec,
    available_devices,
    create_image_embedder,
    create_text_encoder,
    register_embedder,
    resolve_device,
)
from aic2026.model_backends.openclip import OpenClipImageEmbedder


def _parse(model: str, *, default_pretrained: str = DEFAULT_PRETRAINED) -> tuple[str, str, str]:
    return _parse_model_spec(model, default_pretrained)


def _space(dim: int = 3, **overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "test-model",
        "checkpoint_id": "test-checkpoint",
        "space_id": "test-image-space-v1",
        "embedding_dim": dim,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


# ------------------------------------------------------------------ device resolution


@pytest.mark.parametrize(
    "requested,cuda,mps,expected",
    [
        ("auto", True, True, "cuda"),
        ("auto", True, False, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cuda", True, False, "cuda"),
        ("mps", False, True, "mps"),
        ("cpu", True, True, "cpu"),
    ],
)
def test_resolve_device_priority(requested: str, cuda: bool, mps: bool, expected: str) -> None:
    assert resolve_device(requested, cuda_available=cuda, mps_available=mps) == expected


def test_resolve_device_unavailable_accelerator_raises() -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda", cuda_available=False, mps_available=False)
    with pytest.raises(ValueError, match="not available"):
        resolve_device("mps", cuda_available=False, mps_available=False)


def test_resolve_device_unknown_choice_raises() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu", cuda_available=False, mps_available=False)


def test_resolve_device_detects_when_not_given() -> None:
    # Flags omitted -> auto-detect; must resolve to a concrete device.
    assert resolve_device("auto") in ("cuda", "mps", "cpu")


def test_available_devices_returns_bool_pair() -> None:
    cuda, mps = available_devices()
    assert isinstance(cuda, bool)
    assert isinstance(mps, bool)


# ------------------------------------------------------------------ model spec / registry


def test_parse_model_spec_forms() -> None:
    assert _parse("openclip:ViT-L-14:laion2b_s32b_b82k") == ("openclip", "ViT-L-14", "laion2b_s32b_b82k")
    assert _parse("openclip:ViT-L-14") == ("openclip", "ViT-L-14", DEFAULT_PRETRAINED)
    assert _parse("ViT-L-14") == (DEFAULT_BACKEND, "ViT-L-14", DEFAULT_PRETRAINED)
    assert _parse("ViT-L-14", default_pretrained="my-checkpoint") == (
        DEFAULT_BACKEND, "ViT-L-14", "my-checkpoint",
    )


def test_parse_model_spec_invalid() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _parse("")
    with pytest.raises(ValueError, match="invalid model spec"):
        _parse("a:b:c:d")


class _StubEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._space = _space(2, space_id="stub-space")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._space.embedding_dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_images(self, batch_images: list[object]) -> np.ndarray:
        return np.zeros((len(batch_images), self._space.embedding_dim), dtype=np.float32)


@register_embedder("stub")
def _stub_factory(model_name, pretrained, device, precision, config):
    return _StubEmbedder(model_name=f"stub:{model_name}", device=device)


def test_registered_backend_is_constructible() -> None:
    embedder = create_image_embedder("stub:tiny", device="cpu")
    assert embedder.model_name == "stub:tiny"
    assert embedder.device == "cpu"
    assert embedder.embedding_dim == 2


def test_stub_satisfies_image_embedder_protocol() -> None:
    assert isinstance(_StubEmbedder("x", "cpu"), ImageEmbedder)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown model backend"):
        create_image_embedder("nosuch:model")


def test_jina_clip_backend_is_registered() -> None:
    """The shipped Jina CLIP backend name resolves through the registry."""
    from aic2026.image_embedder import EMBEDDER_FACTORIES

    assert "jina-clip" in EMBEDDER_FACTORIES


def test_jina_clip_backend_is_constructible_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``jina-clip`` model spec constructs through the registry with fake deps."""
    from aic2026.image_embedder import create_image_embedder
    from aic2026.model_backends import jina

    class _FakeRegistryTensor:
        def __init__(self, values: object) -> None:
            self.values = np.asarray(values, dtype=np.float32)
            self.device: str | None = None

        def to(self, device: str) -> _FakeRegistryTensor:
            self.device = device
            return self

        def detach(self) -> _FakeRegistryTensor:
            return self

        def cpu(self) -> _FakeRegistryTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    class _FakeRegistryProcessor:
        def __call__(self, **kwargs: object) -> object:
            return type("Encoded", (), {"to": lambda self, device: {"num_items": 2}})()

    class _FakeRegistryModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"projection_dim": 3})()
            self.placement: list[str] = []

        def eval(self) -> None:
            pass

        def to(self, device: str) -> _FakeRegistryModel:
            self.placement.append(device)
            return self

        def get_image_features(self, **kwargs: object) -> _FakeRegistryTensor:
            return _FakeRegistryTensor(np.zeros((2, 3), dtype=np.float32))

    class _FakeRegistryTransformers:
        def __init__(self) -> None:
            self.model = _FakeRegistryModel()
            self.processor = _FakeRegistryProcessor()
            self.AutoModel = type(
                "AutoModel",
                (),
                {"from_pretrained": staticmethod(lambda *a, **k: self.model)},
            )
            self.AutoProcessor = type(
                "AutoProcessor",
                (),
                {"from_pretrained": staticmethod(lambda *a, **k: self.processor)},
            )

    class _FakeRegistryTorch:
        def __init__(self) -> None:
            self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
            self.backends = type(
                "Backends",
                (),
                {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()},
            )

        @staticmethod
        def inference_mode() -> object:
            return nullcontext()

    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeRegistryTorch(), _FakeRegistryTransformers()),
    )
    embedder = create_image_embedder("jina-clip:jina-clip-v2:jinaai/jina-clip-v2", device="cpu")
    assert isinstance(embedder, ImageEmbedder)
    assert embedder.embedding_dim == 3
    assert embedder.embedding_space.backend == "jina-clip"


# ------------------------------------------------------------------ OpenCLIP adapter (fakes)


class _Tensor:
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
        return _Tensor(np.stack([t.values for t in tensors]))

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _FakeModel:
    def __init__(self) -> None:
        self.visual = type("Visual", (), {"output_dim": 3})()

    def eval(self) -> None:
        pass

    @staticmethod
    def encode_image(batch: _Tensor) -> _Tensor:
        return _Tensor(batch.values + np.array([2.0, 1.0, 0.0], dtype=np.float32))


class _FakeOpenClip:
    def __init__(self) -> None:
        self.model = _FakeModel()

    def create_model_and_transforms(self, **kwargs: str) -> tuple[_FakeModel, object, object]:
        return self.model, object(), lambda image: _Tensor([float(len(str(image))), 1.0, 1.0])

    @staticmethod
    def get_tokenizer(model_name: str) -> Any:
        return lambda texts: None


def _adapter(monkeypatch: pytest.MonkeyPatch) -> OpenClipImageEmbedder:
    monkeypatch.setattr(
        "aic2026.model_backends.openclip._load_dependencies",
        lambda: (_FakeTorch(), _FakeOpenClip()),
    )
    return OpenClipImageEmbedder(model_name="ViT-test", pretrained="checkpoint-test")


def test_openclip_adapter_exposes_identity_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)
    assert isinstance(adapter, ImageEmbedder)
    assert adapter.model_name == "openclip:ViT-test:checkpoint-test"
    assert adapter.embedding_dim == 3
    assert adapter.device == "cpu"
    assert adapter.embedding_space.backend == "openclip"


def test_openclip_adapter_encode_images_stacks_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)
    vectors = adapter.encode_images(["a", "longer-image"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32


def test_openclip_adapter_reuses_wrapped_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)
    assert adapter.encode_images(["one"])[0].shape == (3,)


# ------------------------------------------------------------------ text encoder selection


def _make_config(*, with_jina: bool) -> Any:
    from aic2026.config import Config, JinaClipConfig, OpenClipConfig

    from pathlib import Path

    openclip = OpenClipConfig(
        backend="openclip",
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device="cpu",
        precision="fp32",
        normalization="l2",
        preprocessing_id="openclip-default-v1",
        tokenizer_id="openclip-default-v1",
    )
    jina = None
    if with_jina:
        jina = JinaClipConfig(
            backend="jina-clip",
            model_name="jina-clip-v2",
            pretrained="jinaai/jina-clip-v2",
            device="cpu",
            precision="fp32",
            normalization="l2",
            preprocessing_id="jina-clip-v2-preprocess-v1",
            tokenizer_id="jina-clip-v2-tokenizer-v1",
        )
    return Config(
        data_raw_dir=Path("data/raw/aic2026_batch1"),
        data_indexes_dir=Path("data/interim/indexes"),
        data_keyframes_dir=Path("data/keyframes/adaptive_sbd_v1"),
        openclip=openclip,
        jina=jina,
    )


def test_create_text_encoder_selects_openclip_without_jina_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class _FakeOpenClipEncoder:
        def __init__(self, config: Any) -> None:
            captured.append(config.backend)

    monkeypatch.setattr(
        "aic2026.model_backends.openclip.OpenClipEncoder", _FakeOpenClipEncoder
    )
    encoder = create_text_encoder(_make_config(with_jina=False))
    assert captured == ["openclip"]
    assert isinstance(encoder, _FakeOpenClipEncoder)


def test_create_text_encoder_selects_jina_when_jina_section_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class _FakeJinaClipEncoder:
        def __init__(self, config: Any) -> None:
            captured.append(config.backend)

    monkeypatch.setattr(
        "aic2026.model_backends.jina.JinaClipEncoder", _FakeJinaClipEncoder
    )
    encoder = create_text_encoder(_make_config(with_jina=True))
    assert captured == ["jina-clip"]
    assert isinstance(encoder, _FakeJinaClipEncoder)