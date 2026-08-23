"""Reusable, hardware-aware image-embedding interface and backend registry.

This module is the boundary between the shard embedding pipeline
(:mod:`aic2026.assets.embedding_shard`) and concrete model backends. It defines:

- the :class:`ImageEmbedder` protocol (``encode_images`` + model identity);
- portable device selection (``resolve_device`` / ``available_devices``);
- a backend registry (``register_embedder`` / ``create_image_embedder``) that
  the shard CLI and tests use to construct an embedder from a model spec.

It intentionally imports no model SDK at module level: the OpenCLIP and Jina
CLIP adapters are imported lazily when their embedder is constructed, so the
core package stays independent of optional model runtimes. Extensible for
other backends (SigLIP, MobileCLIP, ...) via :func:`register_embedder`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aic2026.config import Config
from aic2026.embeddings import EmbeddingEncoder, EmbeddingSpace

#: Backend used when a bare model name (no ``backend:`` prefix) is given.
DEFAULT_BACKEND = "jina-clip"

#: Pretrained/checkpoint used when a model spec omits it.
DEFAULT_PRETRAINED = "jinaai/jina-clip-v2"

#: Valid device selection values for the CLI / resolver.
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


@runtime_checkable
class ImageEmbedder(Protocol):
    """Reusable interface for image-embedding model backends.

    Attributes:
        - model_name: stable identity of the loaded model (backend + arch +
          checkpoint), e.g. ``jina-clip:jina-clip-v2:jinaai/jina-clip-v2``.
        - embedding_dim: positive vector length emitted per image.
        - device: the concrete resolved runtime device (``cuda``, ``mps``,
          or ``cpu``).

    Methods:
        - ``encode_images(batch_images)``: encode a sequence of backend-native
          images into an ``(B, D)`` floating numpy array in the same order.
        - ``embedding_space``: full embedding-space identity (dimension, dtype,
          normalization, preprocessing) used for artifact metadata.

    Expected input for ``encode_images``:
        - batch_images: a non-empty sequence of images accepted by the backend
          preprocess pipeline.

    Returns:
        - An ``(B, D)`` numpy array whose rows are normalized embeddings in the
          model's ``output_dtype``.
    """

    model_name: str
    embedding_dim: int
    device: str

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return metadata for embeddings emitted by this embedder."""
        ...

    def encode_images(self, batch_images: Sequence[object]) -> np.ndarray:
        """Encode a batch of images into an ``(B, D)`` embedding array."""
        ...


# ------------------------------------------------------------------ device handling


def available_devices() -> tuple[bool, bool]:
    """Detect available accelerator backends without requiring an accelerator.

    Expected input:
        - None.

    Returns:
        - ``(cuda_available, mps_available)``. When Torch is unavailable both
          flags are ``False`` so CPU is always the safe fallback.

    Raises:
        - No exceptions.
    """
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return False, False
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return cuda_available, mps_available


def resolve_device(
    requested: str,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    """Resolve a portable device request to a concrete runtime device.

    Auto-selection priority is CUDA, then MPS, then CPU.

    Expected input:
        - requested: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - cuda_available / mps_available: optional availability flags; when
          omitted they are detected via :func:`available_devices`.

    Returns:
        - A concrete device string: ``cuda``, ``mps``, or ``cpu``.

    Raises:
        - ValueError: if the requested device is unsupported or unavailable.
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unsupported device {requested!r}; expected auto, cpu, mps, or cuda")
    if cuda_available is None or mps_available is None:
        detected_cuda, detected_mps = available_devices()
        if cuda_available is None:
            cuda_available = detected_cuda
        if mps_available is None:
            mps_available = detected_mps
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and cuda_available:
        return "cuda"
    if requested == "mps" and mps_available:
        return "mps"
    raise ValueError(f"requested device {requested!r} is not available")


# ------------------------------------------------------------------ model spec parsing


def _parse_model_spec(model: str, default_pretrained: str) -> tuple[str, str, str]:
    """Split a model spec into ``(backend, model_name, pretrained)``.

    Accepted forms:
        - ``jina-clip:jina-clip-v2:jinaai/jina-clip-v2`` (backend + model + pretrained)
        - ``jina-clip:jina-clip-v2`` (backend + model, default pretrained)
        - ``ViT-L-14`` (bare model, default backend + default pretrained)

    Expected input:
        - model: a non-empty model spec string.
        - default_pretrained: checkpoint used when the spec omits one.

    Returns:
        - ``(backend, model_name, pretrained)``.

    Raises:
        - ValueError: if the spec is empty or has too many parts.
    """
    if not isinstance(model, str) or model == "":
        raise ValueError("model must be a non-empty string")
    parts = model.split(":")
    if len(parts) == 1:
        return DEFAULT_BACKEND, parts[0], default_pretrained
    if len(parts) == 2:
        return parts[0], parts[1], default_pretrained
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f"invalid model spec {model!r}; expected [backend:]model[:pretrained]"
    )


# ------------------------------------------------------------------ backend registry

#: Registered backend factories: backend name -> factory. A factory receives
#: ``model_name``, ``pretrained``, ``device``, ``precision``, and ``config``
#: keyword arguments and returns an :class:`ImageEmbedder`.
EMBEDDER_FACTORIES: dict[str, Callable[..., ImageEmbedder]] = {}


def register_embedder(
    backend: str,
) -> Callable[[Callable[..., ImageEmbedder]], Callable[..., ImageEmbedder]]:
    """Register a backend factory so :func:`create_image_embedder` can use it.

    Expected input:
        - backend: the backend name used as the ``backend:`` prefix of a model
          spec (e.g. ``jina-clip``; register others only if a real backend is added).

    Returns:
        - A decorator that registers the given factory.
    """

    def decorator(
        factory: Callable[..., ImageEmbedder],
    ) -> Callable[..., ImageEmbedder]:
        EMBEDDER_FACTORIES[backend] = factory
        return factory

    return decorator


def _jina_factory(
    model_name: str,
    pretrained: str | None,
    device: str,
    precision: str,
    config: ModelConfig | None,
) -> ImageEmbedder:
    """Construct the Jina CLIP embedder (lazy import avoids a circular import)."""
    from aic2026.model_backends.jina import JinaClipImageEmbedder

    return JinaClipImageEmbedder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        config=config,
    )


EMBEDDER_FACTORIES[DEFAULT_BACKEND] = _jina_factory


#: Callable ``path -> backend-native image`` used by pipelines that decode
#: keyframe images from disk.
ImageLoader = Callable[[Path], object]


def default_image_loader() -> ImageLoader:
    """Return a Pillow-based image loader (lazy optional dependency).

    Returns:
        - A callable ``path -> fully-loaded RGB PIL image``.

    Raises:
        - ImportError: if Pillow is unavailable.
    """
    def load(path: Path) -> object:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "image loading requires Pillow; install pillow or pass image_loader=..."
            ) from exc
        with Image.open(path) as img:
            return img.convert("RGB")

    return load


def create_image_embedder(
    model: str,
    device: str = "auto",
    precision: str = "fp32",
    config: ModelConfig | None = None,
) -> ImageEmbedder:
    """Create an :class:`ImageEmbedder` from a model spec and device.

    Expected input:
        - model: a model spec ``[backend:]model[:pretrained]``; the default
          backend is ``jina-clip``.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda`` (resolved by the
          backend to a concrete runtime device).
        - precision: output precision ``fp32`` or ``fp16`` (backend-dependent).
        - config: optional base :class:`aic2026.config.ModelConfig`
          supplying default pretrained/normalization/preprocessing when not
          overridden.

    Returns:
        - A ready-to-use :class:`ImageEmbedder` for the requested model.

    Raises:
        - ValueError: if the model spec is invalid or the backend is unknown.
        - RuntimeError: if the backend's optional model dependencies are
          unavailable or the model/checkpoint cannot be loaded.
    """
    backend, model_name, pretrained = _parse_model_spec(
        model, default_pretrained=(config.pretrained if config else DEFAULT_PRETRAINED)
    )
    factory = EMBEDDER_FACTORIES.get(backend)
    if factory is None:
        raise ValueError(
            f"unknown model backend {backend!r}; registered: {sorted(EMBEDDER_FACTORIES)}"
        )
    return factory(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        config=config,
    )


def create_text_encoder(config: Config) -> EmbeddingEncoder:
    """Create the text encoder for the active model configuration.

    Expected input:
        - config: effective configuration whose ``model`` section selects
          the backend (production default: ``jina-clip`` / jina-clip-v2).

    Returns:
        - A ready-to-use text encoder matching the active embedding space.

    Raises:
        - ValueError: if the configured backend has no registered encoder.
        - RuntimeError: if the backend optional dependencies or model
          checkpoint cannot be loaded.
    """
    if config.model.backend == "jina-clip":
        from aic2026.model_backends.jina import JinaClipEncoder

        return JinaClipEncoder(config.model)
    raise ValueError(f"unknown embedding backend {config.model.backend!r}")
