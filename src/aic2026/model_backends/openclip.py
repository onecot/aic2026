"""OpenCLIP implementation of the model-independent embedding contract.

OpenCLIP and Torch are imported only when :class:`OpenClipEncoder` is
constructed. This keeps the core package and its unit tests independent of
optional model runtime dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import numpy as np

from aic2026.config import OpenClipConfig
from aic2026.embeddings import Embedding, EmbeddingKind, EmbeddingSpace, Normalization


class OpenClipRuntimeError(RuntimeError):
    """Raised when the optional OpenCLIP runtime cannot be loaded or used."""


def _load_dependencies() -> tuple[Any, Any]:
    """Import Torch and OpenCLIP only when the backend is instantiated.

    Returns:
        - The imported ``torch`` and ``open_clip`` modules.

    Raises:
        - OpenClipRuntimeError: if the optional dependencies are unavailable.
    """
    try:
        return import_module("torch"), import_module("open_clip")
    except ModuleNotFoundError as exc:
        raise OpenClipRuntimeError(
            "OpenCLIP requires the optional 'openclip' dependency group; "
            "install with pip install -e '.[openclip]'"
        ) from exc


def _resolve_device(requested: str, torch: Any) -> str:
    """Resolve a portable configured device to an available Torch device.

    Expected input:
        - requested: one of ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - torch: imported Torch module.

    Returns:
        - A concrete device string accepted by Torch/OpenCLIP.

    Raises:
        - ValueError: if the requested device is unsupported or unavailable.
    """
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
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
    if requested in ("cuda", "mps"):
        raise ValueError(f"requested device {requested!r} is not available")
    raise ValueError(f"unsupported device {requested!r}; expected auto, cpu, mps, or cuda")


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
    """Return the installed OpenCLIP package version for artifact metadata.

    Returns:
        - Installed ``open-clip-torch`` version, or ``unknown`` if package
          metadata is unavailable.
    """
    try:
        return version("open-clip-torch")
    except PackageNotFoundError:
        return "unknown"


class OpenClipEncoder:
    """Encode text and images using a configured OpenCLIP model.

    The single-item ``encode_text`` and ``encode_image`` methods satisfy the
    backend-independent :class:`aic2026.embeddings.EmbeddingEncoder` protocol.
    The explicit ``*_batch`` methods provide efficient multi-input encoding
    without changing that core interface. Both modalities always use the same
    configured normalization convention and embedding space.
    """

    def __init__(self, config: OpenClipConfig) -> None:
        """Load the configured OpenCLIP model, tokenizer, and preprocess pipeline.

        Expected input:
            - config: validated OpenCLIP backend configuration.

        Returns:
            - A ready-to-use encoder.

        Raises:
            - OpenClipRuntimeError: if optional model dependencies are missing.
            - ValueError: if the configured device or model output is invalid.
            - RuntimeError: if OpenCLIP cannot load the requested model/checkpoint.
        """
        self._config = config
        self._torch, open_clip = _load_dependencies()
        self._device = _resolve_device(config.device, self._torch)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name=config.model_name,
            pretrained=config.pretrained,
            device=self._device,
            precision=config.precision,
        )
        self._tokenizer = open_clip.get_tokenizer(config.model_name)
        self._model.eval()
        embedding_dim = self._embedding_dim()
        dtype = _output_dtype(config.precision)
        self._embedding_space = EmbeddingSpace(
            backend="openclip",
            model_id=config.model_name,
            checkpoint_id=config.pretrained,
            space_id=(
                f"openclip:{config.model_name}:{config.pretrained}:"
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
        tokens = self._tokenizer(values).to(self._device)
        with self._torch.inference_mode():
            vectors = self._model.encode_text(tokens)
        return self._make_embeddings("text", vectors, len(values))

    def encode_image(self, image: object) -> Embedding:
        """Encode one backend-native image input.

        Expected input:
            - image: an image accepted by OpenCLIP's configured preprocess.

        Returns:
            - An image :class:`Embedding` in this encoder's space.

        Raises:
            - ValueError: if image input is invalid.
        """
        return self.encode_image_batch([image])[0]

    def encode_image_batch(self, images: Sequence[object]) -> list[Embedding]:
        """Encode images in one model invocation.

        Expected input:
            - images: a non-empty sequence accepted by OpenCLIP preprocess.

        Returns:
            - Image embeddings in the same order as ``images``.

        Raises:
            - ValueError: if no image is supplied or model output is invalid.
        """
        if len(images) == 0:
            raise ValueError("images must be a non-empty sequence")
        batch = self._torch.stack([self._preprocess(image) for image in images]).to(self._device)
        with self._torch.inference_mode():
            vectors = self._model.encode_image(batch)
        return self._make_embeddings("image", vectors, len(images))

    def _embedding_dim(self) -> int:
        """Read the model's declared shared output dimension.

        Returns:
            - Positive embedding dimension.

        Raises:
            - ValueError: if the loaded model does not expose an output dim.
        """
        visual = getattr(self._model, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError("OpenCLIP model does not expose a positive visual.output_dim")
        return output_dim

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
            raise ValueError(f"OpenCLIP output shape {matrix.shape} != expected {expected_shape}")
        if self.embedding_space.normalization == "l2":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0.0):
                raise ValueError("OpenCLIP returned a zero-norm embedding")
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
