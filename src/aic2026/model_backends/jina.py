"""Jina CLIP implementation of the model-independent embedding contract.

Jina CLIP v2 is a bilingual (English/Chinese) multimodal model whose image and
text embeddings share one embedding space. It is loaded through HuggingFace
Transformers (``trust_remote_code``). Transformers and Torch are imported only
when :class:`JinaClipEncoder` is constructed, keeping the core package and its
unit tests independent of optional model runtime dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import numpy as np

from aic2026.embeddings import Embedding, EmbeddingKind, EmbeddingSpace, Normalization
from aic2026.image_embedder import DEFAULT_PRETRAINED, resolve_device

#: Backend name used as the ``backend:`` prefix of a model spec.
BACKEND = "jina-clip"

#: Runtime packages the Jina CLIP remote modeling code (loaded through
#: ``trust_remote_code=True``) imports unconditionally. Kept in sync with the
#: ``jina`` optional-dependency group in ``pyproject.toml``.
REQUIRED_RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "PIL",
    "timm",
    "einops",
    "requests",
)

#: Default architecture name used when a model spec omits ``model``.
DEFAULT_MODEL = "jina-clip-v2"

#: Default HuggingFace repository holding the checkpoint.
DEFAULT_CHECKPOINT = "jinaai/jina-clip-v2"

#: Preprocessing / tokenizer revision identifiers (explicit run metadata).
PREPROCESSING_ID = "jina-clip-v2-preprocess-v1"
TOKENIZER_ID = "jina-clip-v2-tokenizer-v1"


class JinaClipRuntimeError(RuntimeError):
    """Raised when the optional Jina CLIP runtime cannot be loaded or used."""


def _load_dependencies() -> tuple[Any, Any]:
    """Import the Jina CLIP runtime packages only when the backend is used.

    Pre-flight checks every package the remote modeling code imports
    unconditionally so that a missing optional dependency fails with an
    actionable message instead of an obscure error deep inside
    ``trust_remote_code`` model loading.

    Returns:
        - The imported ``torch`` and ``transformers`` modules.

    Raises:
        - JinaClipRuntimeError: if any required optional dependency is missing.
    """
    missing = []
    for name in REQUIRED_RUNTIME_PACKAGES:
        try:
            import_module(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise JinaClipRuntimeError(
            "Jina CLIP requires the optional 'jina' dependency group; "
            f"missing packages: {', '.join(missing)}. "
            "Install with pip install -e '.[jina]'"
        )
    return import_module("torch"), import_module("transformers")


def _resolve_device(requested: str, torch: Any) -> str:
    """Resolve a portable configured device to an available Torch device.

    Expected input:
        - requested: one of ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - torch: imported Torch module.

    Returns:
        - A concrete device string accepted by Torch/Transformers.

    Raises:
        - ValueError: if the requested device is unsupported or unavailable.
    """
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return resolve_device(
        requested,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


def _output_dtype(precision: str) -> np.dtype[Any]:
    """Map a configured output precision to its NumPy dtype.

    Expected input:
        - precision: ``fp32`` or ``fp16``.

    Returns:
        - The corresponding floating NumPy dtype.

    Raises:
        - ValueError: if precision is unsupported.
    """
    if precision == "fp32":
        return np.dtype("float32")
    if precision == "fp16":
        return np.dtype("float16")
    raise ValueError(f"unsupported precision {precision!r}; expected fp32 or fp16")


def _model_version() -> str:
    """Return the installed Transformers package version for artifact metadata.

    Returns:
        - Installed ``transformers`` version, or ``unknown`` if package
          metadata is unavailable.
    """
    try:
        return version("transformers")
    except PackageNotFoundError:
        return "unknown"


def _embedding_dim(model: Any) -> int:
    """Read the model's declared shared output dimension.

    Expected input:
        - model: a loaded Jina CLIP model exposing ``config``.

    Returns:
        - Positive embedding dimension.

    Raises:
        - ValueError: if the loaded model does not expose an output dim.
    """
    model_config = getattr(model, "config", None)
    candidates: list[Any] = []
    if model_config is not None:
        candidates.append(getattr(model_config, "projection_dim", None))
        candidates.append(getattr(model_config, "hidden_size", None))
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            candidates.append(getattr(text_config, "projection_dim", None))
            candidates.append(getattr(text_config, "hidden_size", None))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    raise ValueError("Jina CLIP model does not expose a positive embedding dimension")


@dataclass(frozen=True)
class JinaClipConfig:
    """Configuration for the optional Jina CLIP embedding backend.

    Fields:
        - backend: must be ``"jina-clip"``.
        - model_name: model architecture name (e.g. ``jina-clip-v2``).
        - pretrained: HuggingFace repository id or checkpoint path.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - precision: output precision, ``fp32`` or ``fp16``.
        - normalization: common image/text vector convention, ``l2`` or
          ``none``.
        - preprocessing_id: preprocessing pipeline revision identifier.
        - tokenizer_id: tokenizer revision identifier.
    """

    backend: str
    model_name: str
    pretrained: str
    device: str
    precision: str
    normalization: str
    preprocessing_id: str
    tokenizer_id: str

    def __post_init__(self) -> None:
        if self.backend != BACKEND:
            raise ValueError(f"jina-clip backend must be {BACKEND!r}, got {self.backend!r}")
        for name, value in (
            ("model_name", self.model_name),
            ("pretrained", self.pretrained),
            ("preprocessing_id", self.preprocessing_id),
            ("tokenizer_id", self.tokenizer_id),
        ):
            if not isinstance(value, str) or value == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be one of: fp32, fp16")
        if self.normalization not in ("l2", "none"):
            raise ValueError("normalization must be one of: l2, none")


class JinaClipEncoder:
    """Encode text and images using a configured Jina CLIP model.

    The single-item ``encode_text`` and ``encode_image`` methods satisfy the
    backend-independent :class:`aic2026.embeddings.EmbeddingEncoder` protocol.
    The explicit ``*_batch`` methods provide efficient multi-input encoding
    without changing that core interface. Both modalities share one embedding
    space and use the same configured normalization convention.
    """

    def __init__(self, config: JinaClipConfig) -> None:
        """Load the configured Jina CLIP model and processor.

        Expected input:
            - config: validated Jina CLIP backend configuration.

        Returns:
            - A ready-to-use encoder.

        Raises:
            - JinaClipRuntimeError: if optional model dependencies are missing.
            - ValueError: if the configured device or model output is invalid.
            - RuntimeError: if Transformers cannot load the requested model.
        """
        self._config = config
        self._torch, transformers = _load_dependencies()
        self._device = _resolve_device(config.device, self._torch)
        self._model = transformers.AutoModel.from_pretrained(
            config.pretrained,
            trust_remote_code=True,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            config.pretrained,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()
        embedding_dim = _embedding_dim(self._model)
        dtype = _output_dtype(config.precision)
        self._embedding_space = EmbeddingSpace(
            backend=BACKEND,
            model_id=config.model_name,
            checkpoint_id=config.pretrained,
            space_id=(
                f"{BACKEND}:{config.model_name}:{config.pretrained}:"
                f"{config.precision}:{config.normalization}:"
                f"{config.preprocessing_id}:{config.tokenizer_id}"
            ),
            embedding_dim=embedding_dim,
            output_dtype=dtype.name,
            normalization=cast(Normalization, config.normalization),
            preprocessing_id=f"{config.preprocessing_id};tokenizer={config.tokenizer_id}",
            model_version=_model_version(),
        )

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return the shared image/text embedding-space metadata."""
        return self._embedding_space

    @property
    def device(self) -> str:
        """Return the resolved concrete runtime device."""
        return self._device

    def encode_text(self, text: str) -> Embedding:
        """Encode one text query.

        Expected input:
            - text: one non-empty text query.

        Returns:
            - A text :class:`Embedding` in this encoder's space.

        Raises:
            - ValueError: if text is invalid.
        """
        return self.encode_text_batch([text])[0]

    def encode_text_batch(self, texts: Sequence[str]) -> list[Embedding]:
        """Encode text queries in one model invocation.

        Expected input:
            - texts: a non-empty sequence of non-empty strings.

        Returns:
            - Text embeddings in the same order as ``texts``.

        Raises:
            - ValueError: if no text is supplied or an item is invalid.
        """
        values = self._validate_texts(texts)
        encoded = self._processor(
            text=values,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)
        with self._torch.inference_mode():
            vectors = self._model.get_text_features(**encoded)
        return self._make_embeddings("text", vectors, len(values))

    def encode_image(self, image: object) -> Embedding:
        """Encode one backend-native image input.

        Expected input:
            - image: an image accepted by the configured processor.

        Returns:
            - An image :class:`Embedding` in this encoder's space.

        Raises:
            - ValueError: if image input is invalid.
        """
        return self.encode_image_batch([image])[0]

    def encode_image_batch(self, images: Sequence[object]) -> list[Embedding]:
        """Encode images in one model invocation.

        Expected input:
            - images: a non-empty sequence accepted by the configured
              processor.

        Returns:
            - Image embeddings in the same order as ``images``.

        Raises:
            - ValueError: if no image is supplied or model output is invalid.
        """
        if len(images) == 0:
            raise ValueError("images must be a non-empty sequence")
        encoded = self._processor(images=list(images), return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            vectors = self._model.get_image_features(**encoded)
        return self._make_embeddings("image", vectors, len(images))

    def _make_embeddings(self, kind: EmbeddingKind, vectors: Any, expected_rows: int) -> list[Embedding]:
        """Convert one batched model output into validated contract records.

        Expected input:
            - kind: ``image`` or ``text``.
            - vectors: Torch tensor returned by the corresponding model method.
            - expected_rows: number of input items.

        Returns:
            - One embedding per input item.

        Raises:
            - ValueError: if output shape does not match the input contract.
        """
        matrix = np.asarray(vectors.detach().cpu().numpy(), dtype=np.float32)
        expected_shape = (expected_rows, self.embedding_space.embedding_dim)
        if matrix.shape != expected_shape:
            raise ValueError(f"Jina CLIP output shape {matrix.shape} != expected {expected_shape}")
        if self.embedding_space.normalization == "l2":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0.0):
                raise ValueError("Jina CLIP returned a zero-norm embedding")
            matrix = matrix / norms
        dtype = np.dtype(self.embedding_space.output_dtype)
        return [
            Embedding(kind=kind, vector=row.astype(dtype, copy=True), space=self.embedding_space)
            for row in matrix
        ]

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        """Validate and materialize a text batch.

        Expected input:
            - texts: candidate text strings.

        Returns:
            - The validated values as a list.

        Raises:
            - ValueError: if the sequence is empty or contains invalid text.
        """
        values = list(texts)
        if not values:
            raise ValueError("texts must be a non-empty sequence")
        if any(not isinstance(text, str) or text == "" for text in values):
            raise ValueError("texts must contain only non-empty strings")
        return values


class JinaClipImageEmbedder:
    """Jina CLIP adapter for the reusable :class:`ImageEmbedder` interface.

    Wraps :class:`JinaClipEncoder` and exposes ``encode_images`` plus model
    identity and the resolved runtime device, so the shard embedding pipeline
    can drive Jina CLIP through the same interface as the OpenCLIP backend.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str = "auto",
        precision: str = "fp32",
        config: JinaClipConfig | None = None,
    ) -> None:
        """Load the configured Jina CLIP model as an image embedder.

        Expected input:
            - model_name: Jina architecture name; defaults to
              ``jina-clip-v2``.
            - pretrained: HuggingFace repository id; defaults to
              ``jinaai/jina-clip-v2``. When the spec carries the default
              OpenCLIP checkpoint (e.g. a two-part spec ``jina-clip:<model>``
              resolved against the OpenCLIP config), the Jina default is used.
            - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
            - precision: ``fp32`` or ``fp16``.
            - config: optional base :class:`JinaClipConfig` supplying defaults
              for normalization/preprocessing/tokenizer identity.

        Returns:
            - A ready-to-use image embedder.

        Raises:
            - JinaClipRuntimeError: if optional model dependencies are missing.
            - ValueError: if the configured device or model output is invalid.
            - RuntimeError: if Transformers cannot load the requested model.
        """
        base = config if isinstance(config, JinaClipConfig) else None
        effective_model = model_name or DEFAULT_MODEL
        effective_pretrained = pretrained or DEFAULT_CHECKPOINT
        if effective_pretrained == DEFAULT_PRETRAINED:
            effective_pretrained = DEFAULT_CHECKPOINT
        cfg = JinaClipConfig(
            backend=BACKEND,
            model_name=effective_model,
            pretrained=effective_pretrained,
            device=device,
            precision=precision or (base.precision if base else "fp32"),
            normalization=base.normalization if base else "l2",
            preprocessing_id=base.preprocessing_id if base else PREPROCESSING_ID,
            tokenizer_id=base.tokenizer_id if base else TOKENIZER_ID,
        )
        self._encoder = JinaClipEncoder(cfg)
        self._model_name = f"{BACKEND}:{cfg.model_name}:{cfg.pretrained}"

    @property
    def model_name(self) -> str:
        """Return the stable model identity (backend + arch + checkpoint)."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the model's embedding dimension."""
        return self._encoder.embedding_space.embedding_dim

    @property
    def device(self) -> str:
        """Return the resolved concrete runtime device."""
        return self._encoder.device

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return the embedding-space metadata of the wrapped encoder."""
        return self._encoder.embedding_space

    def encode_images(self, batch_images: Sequence[object]) -> np.ndarray:
        """Encode a batch of images into an ``(B, D)`` embedding array.

        Expected input:
            - batch_images: a non-empty sequence of images accepted by the
              configured processor.

        Returns:
            - An ``(B, D)`` float array of normalized embeddings in the model's
              ``output_dtype``, in the same order as ``batch_images``.

        Raises:
            - ValueError: if the batch is empty or the model output is invalid.
        """
        embeddings = self._encoder.encode_image_batch(batch_images)
        return np.stack([embedding.vector for embedding in embeddings])
