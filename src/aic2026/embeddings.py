"""Model-independent embedding and encoder contracts.

This module is the boundary between model backends and index/retrieval core
logic.  It intentionally imports no model SDK: a backend is responsible for
loading a model and constructing :class:`Embedding` values, while the core
only uses the vector and declared embedding-space metadata.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

EmbeddingKind = Literal["image", "text"]
Normalization = Literal["l2", "none", "unknown"]


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity and comparison-relevant metadata for one vector space.

    Expected input:
        - backend: implementation family that produced the embeddings.
        - model_id: backend-specific model identifier or name.
        - checkpoint_id: pretrained/checkpoint identifier, or ``None`` when
          the backend has no separate checkpoint concept.
        - space_id: stable identifier for the exact jointly comparable space.
        - embedding_dim: required 1-D vector length.
        - output_dtype: dtype emitted by the encoder before index storage.
        - normalization: vector normalization convention.
        - preprocessing_id: relevant preprocessing/tokenization revision.
        - model_version: optional backend/model implementation revision.

    The complete value is compared for compatibility.  Matching vector
    dimensions alone never establishes interchangeability.
    """

    backend: str
    model_id: str
    checkpoint_id: str | None
    space_id: str
    embedding_dim: int
    output_dtype: str
    normalization: Normalization
    preprocessing_id: str
    model_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("backend", self.backend),
            ("model_id", self.model_id),
            ("space_id", self.space_id),
            ("output_dtype", self.output_dtype),
            ("preprocessing_id", self.preprocessing_id),
        ):
            _require(isinstance(value, str) and value != "", f"{name} must be a non-empty string")
        _require(
            self.checkpoint_id is None or (isinstance(self.checkpoint_id, str) and self.checkpoint_id != ""),
            "checkpoint_id must be a non-empty string or None",
        )
        _require(
            self.model_version is None or (isinstance(self.model_version, str) and self.model_version != ""),
            "model_version must be a non-empty string or None",
        )
        _require(
            isinstance(self.embedding_dim, int) and not isinstance(self.embedding_dim, bool)
            and self.embedding_dim >= 1,
            "embedding_dim must be a positive integer",
        )
        _require(self.normalization in ("l2", "none", "unknown"), "normalization must be 'l2', 'none', or 'unknown'")
        try:
            dtype = np.dtype(self.output_dtype)
        except TypeError as exc:
            raise ValueError(f"invalid output_dtype {self.output_dtype!r}") from exc
        _require(np.issubdtype(dtype, np.floating), "output_dtype must be a floating numpy dtype")

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-safe dictionary.

        Returns:
            - All fields required to identify and compare the embedding space.
        """
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "space_id": self.space_id,
            "embedding_dim": self.embedding_dim,
            "output_dtype": self.output_dtype,
            "normalization": self.normalization,
            "preprocessing_id": self.preprocessing_id,
            "model_version": self.model_version,
        }

    def is_compatible_with(self, other: EmbeddingSpace) -> bool:
        """Return whether two spaces are mutually comparable.

        ``model_version`` is build-environment metadata (the installed model
        runtime revision) and is intentionally excluded: vectors produced by
        the same model/checkpoint/preprocessing convention live in the same
        space regardless of the runtime patch that generated them.

        Expected input:
            - other: another embedding-space description.

        Returns:
            - True when every comparison-relevant field matches.
        """
        return (
            self.backend == other.backend
            and self.model_id == other.model_id
            and self.checkpoint_id == other.checkpoint_id
            and self.space_id == other.space_id
            and self.embedding_dim == other.embedding_dim
            and self.output_dtype == other.output_dtype
            and self.normalization == other.normalization
            and self.preprocessing_id == other.preprocessing_id
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSpace:
        """Deserialize validated embedding-space metadata.

        Expected input:
            - data: dictionary produced by :meth:`to_dict`.

        Returns:
            - A validated embedding-space description.

        Raises:
            - KeyError: if a required field is missing.
            - ValueError: if a field violates the contract.
        """
        return cls(
            backend=data["backend"],
            model_id=data["model_id"],
            checkpoint_id=data["checkpoint_id"],
            space_id=data["space_id"],
            embedding_dim=int(data["embedding_dim"]),
            output_dtype=data["output_dtype"],
            normalization=data["normalization"],
            preprocessing_id=data["preprocessing_id"],
            model_version=data.get("model_version"),
        )


@dataclass(frozen=True)
class Embedding:
    """One image or text vector together with its embedding-space identity.

    Expected input:
        - kind: ``"image"`` or ``"text"``.
        - vector: owned 1-D floating numpy array matching ``space``.
        - space: metadata identifying the vector's comparable space.

    The stored dtype and shape are explicit through ``vector`` and retained
    losslessly by serialization.
    """

    kind: EmbeddingKind
    vector: np.ndarray
    space: EmbeddingSpace

    def __post_init__(self) -> None:
        _require(self.kind in ("image", "text"), "kind must be 'image' or 'text'")
        _require(
            isinstance(self.vector, np.ndarray) and self.vector.ndim == 1,
            "embedding vector must be a 1-D numpy array",
        )
        _require(np.issubdtype(self.vector.dtype, np.floating), "embedding vector dtype must be floating")
        _require(
            int(self.vector.shape[0]) == self.space.embedding_dim,
            f"embedding vector dim {self.vector.shape[0]} != space dim {self.space.embedding_dim}",
        )
        _require(
            self.vector.dtype == np.dtype(self.space.output_dtype),
            f"embedding vector dtype {self.vector.dtype} != space output_dtype {self.space.output_dtype}",
        )
        object.__setattr__(self, "vector", np.array(self.vector, copy=True))

    @property
    def embedding_dim(self) -> int:
        """Return the one-dimensional vector length."""
        return int(self.vector.shape[0])

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the vector dtype."""
        return self.vector.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the vector shape."""
        return tuple(self.vector.shape)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the embedding losslessly to a JSON-safe dictionary.

        Returns:
            - Kind, space metadata, dtype/shape, and base64-encoded bytes.
        """
        return {
            "kind": self.kind,
            "vector": base64.b64encode(self.vector.tobytes()).decode("ascii"),
            "shape": list(self.vector.shape),
            "dtype": self.vector.dtype.str,
            "space": self.space.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Embedding:
        """Deserialize a losslessly serialized embedding.

        Expected input:
            - data: dictionary produced by :meth:`to_dict`.

        Returns:
            - A validated embedding.

        Raises:
            - KeyError: if a required field is missing.
            - ValueError: if encoded bytes or metadata are inconsistent.
        """
        try:
            vector = np.frombuffer(
                base64.b64decode(data["vector"], validate=True), dtype=np.dtype(data["dtype"])
            ).reshape(tuple(data["shape"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid serialized embedding vector") from exc
        return cls(kind=data["kind"], vector=np.array(vector, copy=True), space=EmbeddingSpace.from_dict(data["space"]))


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Minimal backend-agnostic interface implemented by future encoders."""

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return metadata for embeddings emitted by this encoder."""
        ...

    def encode_image(self, image: object) -> Embedding:
        """Encode one backend-native image input into an image embedding."""
        ...

    def encode_text(self, text: str) -> Embedding:
        """Encode one text query into a text embedding."""
        ...


def legacy_embedding_space(embedding_dim: int, output_dtype: str) -> EmbeddingSpace:
    """Describe an existing artifact whose source model metadata is unavailable.

    Expected input:
        - embedding_dim: positive vector length from the legacy artifact.
        - output_dtype: floating dtype stored by the artifact.

    Returns:
        - An explicit unknown/legacy embedding-space identity.

    Raises:
        - ValueError: if dimension or dtype is invalid.
    """
    return EmbeddingSpace(
        backend="legacy",
        model_id="unknown",
        checkpoint_id=None,
        space_id=f"legacy-unknown-d{embedding_dim}-{np.dtype(output_dtype).name}",
        embedding_dim=embedding_dim,
        output_dtype=np.dtype(output_dtype).name,
        normalization="unknown",
        preprocessing_id="unknown",
    )


def validate_query_compatibility(query: Embedding, index_space: EmbeddingSpace) -> None:
    """Require a query embedding to be comparable with an index space.

    Expected input:
        - query: a text or image embedding to search with.
        - index_space: embedding-space metadata recorded by the index.

    Returns:
        - None when every comparison-relevant metadata field matches.

    Raises:
        - ValueError: if the query and index have different embedding spaces.
    """
    if not query.space.is_compatible_with(index_space):
        raise ValueError(
            "incompatible embedding spaces: query "
            f"{query.space.space_id!r} ({query.space.backend}/{query.space.model_id}) != index "
            f"{index_space.space_id!r} ({index_space.backend}/{index_space.model_id})"
        )
