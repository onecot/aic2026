"""Network-free tests for the optional Jina CLIP backend."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from aic2026.embeddings import Embedding, validate_query_compatibility
from aic2026.image_embedder import DEFAULT_PRETRAINED
from aic2026.model_backends import jina
from aic2026.model_backends.jina import (
    DEFAULT_CHECKPOINT,
    ModelConfig,
    JinaClipEncoder,
    JinaClipImageEmbedder,
    JinaClipRuntimeError,
)


class _Tensor:
    """Minimal Torch-tensor stand-in used to test the backend boundary.

    ``float32`` tensors behave like real Torch CPU tensors. ``bfloat16``
    tensors simulate Jina CLIP v2 on CUDA: ``numpy()`` without an explicit
    ``float()`` cast raises exactly like NumPy rejecting the dtype, so the
    regression tests prove the fix casts before conversion.
    """

    def __init__(self, values: Any, *, dtype: str = "float32") -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.dtype = dtype
        self.device: str | None = None
        self._cast_to_float: bool = False

    def to(self, device: str) -> _Tensor:
        self.device = device
        return self

    def detach(self) -> _Tensor:
        return self

    def float(self) -> _Tensor:
        self._cast_to_float = True
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        if self.dtype == "bfloat16" and not self._cast_to_float:
            raise TypeError("NumPy does not support the bfloat16 dtype")
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
    def inference_mode() -> Any:
        return nullcontext()


class _Encoded:
    """Stand-in for a processor output carrying the batch size."""

    def __init__(self, num_items: int) -> None:
        self._num_items = num_items

    def to(self, device: str) -> dict[str, object]:
        return {"num_items": self._num_items, "device": device}


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _Encoded:
        self.calls.append(kwargs)
        if "text" in kwargs:
            return _Encoded(len(kwargs["text"]))  # type: ignore[arg-type]
        return _Encoded(len(kwargs["images"]))  # type: ignore[arg-type]


class _FakeModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"projection_dim": 3})()
        self.evaluated = False
        self.placement: list[str] = []
        self.last_input_device: str | None = None

    def eval(self) -> None:
        self.evaluated = True

    def to(self, device: str) -> _FakeModel:
        self.placement.append(device)
        return self

    def _out(self, kwargs: dict[str, object]) -> _Tensor:
        self.last_input_device = kwargs.get("device")
        num_items = int(kwargs["num_items"])  # type: ignore[arg-type]
        rows = np.array([[float(i), float(i) + 1.0, float(i) + 2.0] for i in range(num_items)])
        return _Tensor(rows)

    def get_text_features(self, **kwargs: object) -> _Tensor:
        return self._out(kwargs)

    def get_image_features(self, **kwargs: object) -> _Tensor:
        return self._out(kwargs)


class _FakeTransformers:
    """Stand-in for the ``transformers`` module with ``AutoModel``/``AutoProcessor``."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.processor = _FakeProcessor()
        self.loaded: list[tuple[str, str, dict[str, object]]] = []
        self.AutoModel = type(
            "AutoModel",
            (),
            {"from_pretrained": staticmethod(self._load_model)},
        )
        self.AutoProcessor = type(
            "AutoProcessor",
            (),
            {"from_pretrained": staticmethod(self._load_processor)},
        )

    def _load_model(self, pretrained: str, **kwargs: object) -> _FakeModel:
        self.loaded.append(("model", pretrained, kwargs))
        return self.model

    def _load_processor(self, pretrained: str, **kwargs: object) -> _FakeProcessor:
        self.loaded.append(("processor", pretrained, kwargs))
        return self.processor


def _config(**overrides: str) -> ModelConfig:
    values = {
        "backend": "jina-clip",
        "model_name": "test-model",
        "pretrained": "test-checkpoint",
        "device": "auto",
        "precision": "fp32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
        "tokenizer_id": "test-tokenizer-v1",
    }
    values.update(overrides)
    return ModelConfig(**values)


def _encoder(
    monkeypatch: pytest.MonkeyPatch,
    transformers: _FakeTransformers | None = None,
) -> tuple[JinaClipEncoder, _FakeTransformers]:
    fake_transformers = transformers or _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    return JinaClipEncoder(_config()), fake_transformers


def test_missing_optional_runtime_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Jina runtime dependency fails with an actionable message."""
    with pytest.raises(JinaClipRuntimeError, match=r"\.\[jina\]"):
        monkeypatch.setattr(
            jina,
            "import_module",
            lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        )
        jina._load_dependencies()


def test_missing_runtime_reports_each_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-flight check names exactly which required packages are absent."""
    monkeypatch.setattr(jina, "REQUIRED_RUNTIME_PACKAGES", ("alpha", "beta", "gamma"))

    def selective_import(name: str) -> object:
        if name == "beta":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(jina, "import_module", selective_import)
    with pytest.raises(JinaClipRuntimeError, match=r"missing packages: beta"):
        jina._load_dependencies()


def test_required_runtime_packages_declared_in_pyproject() -> None:
    """Every package the remote modeling code imports is in the jina extra."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
    text = pyproject.read_text(encoding="utf-8")
    section = text.split("jina = [", 1)[1].split("]", 1)[0]
    declared = {line.strip().strip('",') for line in section.splitlines() if line.strip()}
    for package in jina.REQUIRED_RUNTIME_PACKAGES:
        dist_name = {"PIL": "Pillow"}.get(package, package)
        assert any(line.startswith(f"{dist_name}") for line in declared), (
            f"{package} missing from the jina extra in pyproject.toml"
        )


def test_jina_extra_allows_current_transformers_stack() -> None:
    """The jina extra must not force a pre-5.x Transformers downgrade.

    The backend ships a compatibility shim for ``clip_loss`` (removed in
    Transformers 5.6 and replaced by the identical
    ``image_text_contrastive_loss``), so the extra may install a recent
    Transformers. A modern ``tokenizers`` is required to parse the
    ``jinaai/jina-clip-v2`` tokenizer (Metaspace + precompiled normalizer).
    """
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
    text = pyproject.read_text(encoding="utf-8")
    section = text.split("jina = [", 1)[1].split("]", 1)[0]
    declared = {line.strip().strip('",') for line in section.splitlines() if line.strip()}
    transformers_specs = sorted(
        line for line in declared if line.startswith("transformers") or line.startswith("transformers ")
    )
    assert transformers_specs, "transformers missing from the jina extra in pyproject.toml"
    for spec in transformers_specs:
        upper_bounds = [part for part in spec.split(",") if part.startswith("<")]
        for bound in upper_bounds:
            assert not bound.lstrip("<").startswith("5"), (
                f"jina extra must not cap transformers below 5.x; got {spec}"
            )
    tokenizers_specs = sorted(
        line for line in declared if line.startswith("tokenizers") or line.startswith("tokenizers ")
    )
    assert tokenizers_specs, "tokenizers missing from the jina extra in pyproject.toml"
    assert any(">=" in part for part in tokenizers_specs[0].split(",")), (
        "jina extra should pin a minimum tokenizers version to parse the Jina tokenizer"
    )


def _fake_clip_modeling(**attributes: Any) -> ModuleType:
    """Build a stand-in for ``transformers.models.clip.modeling_clip``."""
    module = ModuleType("transformers.models.clip.modeling_clip")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _install_fake_clip_modeling(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    """Route the shim's import to a fake module for a network-free test."""
    monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", module)


def test_clip_loss_shim_is_noop_when_legacy_symbol_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Transformers that still ship ``clip_loss``, the shim leaves it as-is."""
    original = lambda similarity: similarity  # noqa: E731
    module = _fake_clip_modeling(clip_loss=original)
    _install_fake_clip_modeling(monkeypatch, module)
    jina._ensure_clip_loss_compat()
    assert module.clip_loss is original


def test_clip_loss_shim_aliases_current_loss_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """On modern Transformers, the shim aliases ``image_text_contrastive_loss``."""
    def current_loss(similarity: Any) -> Any:
        return similarity

    module = _fake_clip_modeling(image_text_contrastive_loss=current_loss)
    _install_fake_clip_modeling(monkeypatch, module)
    jina._ensure_clip_loss_compat()
    assert module.clip_loss is current_loss


def test_clip_loss_shim_fails_without_any_loss_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected Transformers without either symbol fails with a clear error."""
    module = _fake_clip_modeling()
    _install_fake_clip_modeling(monkeypatch, module)
    with pytest.raises(JinaClipRuntimeError, match="neither 'clip_loss'"):
        jina._ensure_clip_loss_compat()


def test_backend_initialization_and_embedding_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialization loads the runtime with trust_remote_code and declares its space."""
    encoder, fake_transformers = _encoder(monkeypatch)
    assert fake_transformers.model.evaluated
    assert fake_transformers.loaded[0] == ("model", "test-checkpoint", {"trust_remote_code": True})
    assert fake_transformers.loaded[1] == ("processor", "test-checkpoint", {"trust_remote_code": True})
    assert encoder.embedding_space.backend == "jina-clip"
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


class _Bfloat16Model:
    """Model stand-in whose outputs mimic Jina CLIP v2 bfloat16 on CUDA."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"projection_dim": 3})()

    def eval(self) -> None:
        pass

    def to(self, device: str) -> _Bfloat16Model:
        return self

    @staticmethod
    def _out(kwargs: dict[str, object]) -> _Tensor:
        num_items = int(kwargs["num_items"])  # type: ignore[arg-type]
        rows = np.array([[float(i), float(i) + 1.0, float(i) + 2.0] for i in range(num_items)])
        return _Tensor(rows, dtype="bfloat16")

    def get_text_features(self, **kwargs: object) -> _Tensor:
        return self._out(kwargs)

    def get_image_features(self, **kwargs: object) -> _Tensor:
        return self._out(kwargs)


@pytest.mark.parametrize(
    "method",
    ["encode_text", "encode_image", "encode_text_batch", "encode_image_batch"],
)
def test_bfloat16_model_output_is_cast_before_numpy_conversion(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """A bfloat16 model output (CUDA) converts cleanly and keeps the contract.

    ``torch.Tensor.numpy()`` rejects bfloat16, so the backend must cast to
    float32 first. The emitted embeddings remain float32 with the declared
    L2 normalization regardless of the model's internal dtype.
    """
    fake_transformers = _FakeTransformers()
    fake_transformers.model = _Bfloat16Model()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    encoder = JinaClipEncoder(_config())

    if method == "encode_text":
        embeddings = [encoder.encode_text("a query")]
    elif method == "encode_image":
        embeddings = [encoder.encode_image("an image")]
    elif method == "encode_text_batch":
        embeddings = encoder.encode_text_batch(["a query", "another query"])
    else:
        embeddings = encoder.encode_image_batch(["an image", "another image"])

    assert len(embeddings) == 2 if "batch" in method else len(embeddings) == 1
    for embedding in embeddings:
        assert embedding.vector.dtype == np.float32
        assert embedding.space == encoder.embedding_space
        assert np.linalg.norm(embedding.vector) == pytest.approx(1.0, rel=1e-6)
        assert embedding.vector.shape == (3,)


def test_precision_controls_output_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured fp16 output is represented consistently in metadata and vectors."""
    encoder, _ = _encoder(monkeypatch)
    fp16_encoder = JinaClipEncoder(_config(precision="fp16"))
    assert fp16_encoder.embedding_space.output_dtype == "float16"
    assert fp16_encoder.encode_text("text").vector.dtype == np.float16


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "invalid"},
        {"precision": "bf16"},
        {"normalization": "unknown"},
        {"model_name": ""},
        {"pretrained": ""},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, str]) -> None:
    """Unsupported configuration values fail before model loading."""
    with pytest.raises(ValueError):
        _config(**overrides)


def test_unavailable_requested_device_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable accelerator is reported instead of silently falling back."""
    fake_transformers = _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    with pytest.raises(ValueError, match="not available"):
        JinaClipEncoder(_config(device="cuda"))


def test_auto_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto resolution supports MPS when it is the available accelerator."""
    fake_transformers = _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(mps_available=True), fake_transformers),
    )
    assert JinaClipEncoder(_config()).device == "mps"


@pytest.mark.parametrize(
    ("torch_factory", "expected_device"),
    [
        (lambda: _FakeTorch(), "cpu"),
        (lambda: _FakeTorch(mps_available=True), "mps"),
        (lambda: _FakeTorch(cuda_available=True), "cuda"),
    ],
)
def test_model_is_moved_to_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
    torch_factory: Any,
    expected_device: str,
) -> None:
    """Model weights are placed on the resolved device before inference."""
    fake_transformers = _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (torch_factory(), fake_transformers),
    )
    encoder = JinaClipEncoder(_config())
    assert encoder.device == expected_device
    assert fake_transformers.model.placement == [expected_device]
    assert fake_transformers.model.evaluated


@pytest.mark.parametrize(
    ("torch_factory", "expected_device"),
    [
        (lambda: _FakeTorch(), "cpu"),
        (lambda: _FakeTorch(mps_available=True), "mps"),
        (lambda: _FakeTorch(cuda_available=True), "cuda"),
    ],
)
def test_inference_does_not_mix_input_and_model_devices(
    monkeypatch: pytest.MonkeyPatch,
    torch_factory: Any,
    expected_device: str,
) -> None:
    """Encoded inputs are moved to the same device the model was placed on."""
    fake_transformers = _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (torch_factory(), fake_transformers),
    )
    encoder = JinaClipEncoder(_config())
    encoder.encode_image_batch(["a", "b"])
    encoder.encode_text_batch(["a query"])
    assert fake_transformers.model.placement == [expected_device]
    assert fake_transformers.model.last_input_device == expected_device


def test_batch_input_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty and malformed batch inputs fail clearly."""
    encoder, _ = _encoder(monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_text_batch([])
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_image_batch([])
    with pytest.raises(ValueError, match="non-empty strings"):
        encoder.encode_text_batch([""])


def test_shape_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model output that violates the declared dimension fails clearly."""
    fake_transformers = _FakeTransformers()
    fake_transformers.model.get_image_features = lambda **kwargs: _Tensor(
        np.zeros((int(kwargs["num_items"]), 4), dtype=np.float32)  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    encoder = JinaClipEncoder(_config())
    with pytest.raises(ValueError, match="output shape"):
        encoder.encode_image_batch(["a", "b"])


def test_zero_norm_embedding_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-norm model output cannot be L2-normalized and fails clearly."""
    fake_transformers = _FakeTransformers()
    fake_transformers.model.get_image_features = lambda **kwargs: _Tensor(
        np.zeros((int(kwargs["num_items"]), 3), dtype=np.float32)  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    encoder = JinaClipEncoder(_config())
    with pytest.raises(ValueError, match="zero-norm"):
        encoder.encode_image_batch(["a"])


def test_embedding_dim_falls_back_to_text_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """When projection_dim is absent, the text config dimension is used."""
    fake_transformers = _FakeTransformers()
    fake_transformers.model.config = type(
        "Config",
        (),
        {"text_config": type("TextConfig", (), {"hidden_size": 5})()},
    )()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    assert JinaClipEncoder(_config()).embedding_space.embedding_dim == 5


def test_embedding_dim_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model without any declared dimension fails clearly during init."""
    fake_transformers = _FakeTransformers()
    fake_transformers.model.config = type("Config", (), {})()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    with pytest.raises(ValueError, match="embedding dimension"):
        JinaClipEncoder(_config())


# ------------------------------------------------------------------ image embedder adapter


def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: str,
) -> tuple[JinaClipImageEmbedder, _FakeTransformers]:
    fake_transformers = _FakeTransformers()
    monkeypatch.setattr(
        jina,
        "_load_dependencies",
        lambda: (_FakeTorch(), fake_transformers),
    )
    return JinaClipImageEmbedder(**kwargs), fake_transformers


def test_adapter_exposes_identity_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter presents a stable model identity and the resolved device."""
    adapter, _ = _adapter(monkeypatch, model_name="jina-clip-v2", pretrained="jinaai/jina-clip-v2")
    assert adapter.model_name == "jina-clip:jina-clip-v2:jinaai/jina-clip-v2"
    assert adapter.embedding_dim == 3
    assert adapter.device == "cpu"
    assert adapter.embedding_space.backend == "jina-clip"


def test_adapter_encode_images_stacks_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter batch encoding returns a stacked float array in input order."""
    adapter, _ = _adapter(monkeypatch, model_name="jina-clip-v2", pretrained="jinaai/jina-clip-v2")
    vectors = adapter.encode_images(["a", "longer-image"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert not np.array_equal(vectors[0], vectors[1])


def test_adapter_uses_jina_default_checkpoint_for_openclip_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spec resolved against the OpenCLIP default checkpoint loads Jina's default."""
    _, fake_transformers = _adapter(monkeypatch, model_name="jina-clip-v2", pretrained=DEFAULT_PRETRAINED)
    assert fake_transformers.loaded[0] == ("model", DEFAULT_CHECKPOINT, {"trust_remote_code": True})


def test_adapter_uses_jina_default_model_and_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting model/checkpoint falls back to Jina CLIP v2 defaults."""
    adapter, fake_transformers = _adapter(monkeypatch)
    assert fake_transformers.loaded[0] == ("model", DEFAULT_CHECKPOINT, {"trust_remote_code": True})
    assert adapter.model_name == f"jina-clip:{jina.DEFAULT_MODEL}:{DEFAULT_CHECKPOINT}"
