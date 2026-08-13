"""Contract for the image-embedding-stage artifact.

This module defines the on-disk contract for the artifact produced by the
image embedding stage (``aic2026.embedding_runner``). It is deliberately
independent of both the raw BTC directory layout and any model backend
(OpenCLIP or otherwise): it only declares the files, schema, and
serialization helpers for one embedding artifact.

Scope note: this module only defines the contract (schema, file layout,
serialization helpers, and validation). It does not encode images and does
not run a model.

Data flow::

    Dataset (canonical)  --embedding runner-->  embedding artifact files
        ----------------------------------------------------------->
                          index builder reads the artifact (via rows)

Artifact layout (one directory per ``artifact_id``)::

    <embeddings_dir>/<artifact_id>/
        manifest.json       versioned, self-describing metadata
        embeddings.npy      dense (N, D) matrix; row i <-> frames[i]
        frames.ndjson       one canonical KeyframeTiming per line (row i)

Row ordering contract: every ``i`` in ``[0, N)`` refers to the same
keyframe across ``embeddings.npy[i]`` and ``frames.ndjson[i]``. This is
the stable lookup key the index stage relies on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.canonical import KeyframeTiming
from aic2026.embeddings import EmbeddingSpace

#: Schema version of the embedding artifact.
EMBEDDING_ARTIFACT_SCHEMA_VERSION = "1.0"
SUPPORTED_EMBEDDING_ARTIFACT_SCHEMA_VERSIONS = (EMBEDDING_ARTIFACT_SCHEMA_VERSION,)

#: Relative filenames within an artifact directory.
MANIFEST_FILENAME = "manifest.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
FRAMES_FILENAME = "frames.ndjson"


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EmbeddingManifest:
    """Versioned metadata describing one embedding artifact.

    Fields:
        - schema_version: the artifact schema version this manifest
          conforms to (``EMBEDDING_ARTIFACT_SCHEMA_VERSION``).
        - artifact_id: unique identifier for this artifact (run identity).
        - embedding_space: full model/backend identity of the embeddings
          (dimension, output dtype, normalization, preprocessing, ...).
        - embedding_dim: dimensionality ``D`` of each embedding vector.
        - embedding_dtype: numpy dtype name of the stored matrix, e.g.
          ``float32``.
        - num_rows: number ``N`` of encoded keyframes (rows).
        - num_videos: number of distinct videos covered.
        - source_dataset_id: identifier of the source dataset, e.g.
          ``aic2026_batch1`` (storage-layout independent).
        - canonical_schema_version: canonical schema version the
          ``frames.ndjson`` records conform to.
        - created_at: ISO-8601 timestamp of artifact creation.
    """

    schema_version: str
    artifact_id: str
    embedding_space: EmbeddingSpace
    embedding_dim: int
    embedding_dtype: str
    num_rows: int
    num_videos: int
    source_dataset_id: str
    canonical_schema_version: str
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version in SUPPORTED_EMBEDDING_ARTIFACT_SCHEMA_VERSIONS,
            f"unsupported artifact schema version {self.schema_version!r}; "
            f"supported: {SUPPORTED_EMBEDDING_ARTIFACT_SCHEMA_VERSIONS}",
        )
        _require(isinstance(self.artifact_id, str) and self.artifact_id != "", "artifact_id required")
        _require(isinstance(self.embedding_space, EmbeddingSpace), "embedding_space required")
        _require(
            isinstance(self.embedding_dim, int)
            and not isinstance(self.embedding_dim, bool)
            and self.embedding_dim >= 1,
            "embedding_dim must be >= 1",
        )
        _require(
            self.embedding_dim == self.embedding_space.embedding_dim,
            "embedding_space.embedding_dim must match embedding_dim",
        )
        _require(
            isinstance(self.embedding_dtype, str) and self.embedding_dtype != "",
            "embedding_dtype required",
        )
        try:
            embedding_dtype = np.dtype(self.embedding_dtype)
        except TypeError as exc:
            raise ValueError(f"invalid embedding_dtype {self.embedding_dtype!r}") from exc
        _require(np.issubdtype(embedding_dtype, np.floating), "embedding_dtype must be a floating numpy dtype")
        _require(
            embedding_dtype == np.dtype(self.embedding_space.output_dtype),
            "embedding_space.output_dtype must match embedding_dtype",
        )
        _require(isinstance(self.num_rows, int) and self.num_rows >= 0, "num_rows must be >= 0")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.source_dataset_id, str) and self.source_dataset_id != "", "source_dataset_id required")
        _require(
            isinstance(self.canonical_schema_version, str) and self.canonical_schema_version != "",
            "canonical_schema_version required",
        )
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            - Dict representation including the schema version.
        """
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "embedding_space": self.embedding_space.to_dict(),
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": self.embedding_dtype,
            "num_rows": self.num_rows,
            "num_videos": self.num_videos,
            "source_dataset_id": self.source_dataset_id,
            "canonical_schema_version": self.canonical_schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingManifest:
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid or the schema version is
              unsupported.
        """
        return cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            embedding_space=EmbeddingSpace.from_dict(data["embedding_space"]),
            embedding_dim=int(data["embedding_dim"]),
            embedding_dtype=data["embedding_dtype"],
            num_rows=int(data["num_rows"]),
            num_videos=int(data["num_videos"]),
            source_dataset_id=data["source_dataset_id"],
            canonical_schema_version=data["canonical_schema_version"],
            created_at=data["created_at"],
        )


# ------------------------------------------------------------------ serialization


def write_manifest(path: Path, manifest: EmbeddingManifest) -> None:
    """Write a manifest to ``manifest.json``.

    Expected input:
        - path: a file path (usually ``<artifact_dir>/manifest.json``).
        - manifest: the manifest to persist.

    Returns:
        - None.

    Raises:
        - OSError: if the file cannot be written.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")


def read_manifest(path: Path) -> EmbeddingManifest:
    """Read and validate a manifest from disk.

    Expected input:
        - path: a file path to ``manifest.json``.

    Returns:
        - A validated :class:`EmbeddingManifest`.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if the JSON is malformed.
        - ValueError: if the manifest is invalid or has an unsupported
          schema version.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return EmbeddingManifest.from_dict(payload)


def write_frames(path: Path, frames: Iterable[KeyframeTiming]) -> int:
    """Write frame records as NDJSON, one ``KeyframeTiming`` per line.

    Expected input:
        - path: a file path (usually ``<artifact_dir>/frames.ndjson``).
        - frames: iterable of canonical :class:`KeyframeTiming` in row
          order (index ``i`` becomes line ``i``).

    Returns:
        - Number of lines written.

    Raises:
        - OSError: if the file cannot be written.
    """
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_dict()) + "\n")
            count += 1
    return count


def iter_frames(path: Path) -> Iterator[KeyframeTiming]:
    """Yield canonical :class:`KeyframeTiming` records from an NDJSON file.

    Expected input:
        - path: a file path to ``frames.ndjson``.

    Returns:
        - Iterator of :class:`KeyframeTiming` in file (row) order.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            yield KeyframeTiming.from_dict(json.loads(line))


# ------------------------------------------------------------------ validation


def validate_embedding_artifact(manifest: EmbeddingManifest, artifact_dir: Path) -> list[str]:
    """Validate that an artifact directory matches its manifest.

    Expected input:
        - manifest: the artifact's manifest.
        - artifact_dir: the directory containing the artifact files.

    Returns:
        - List of human-readable problems; empty when the artifact is
          consistent.

    Raises:
        - No exceptions (problems are accumulated and returned).
    """
    problems: list[str] = []

    frames_path = artifact_dir / FRAMES_FILENAME
    embeddings_path = artifact_dir / EMBEDDINGS_FILENAME

    if not frames_path.is_file():
        problems.append(f"missing {FRAMES_FILENAME}")
    else:
        try:
            n_frames = sum(1 for _ in iter_frames(frames_path))
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            problems.append(f"{FRAMES_FILENAME} unreadable: {exc}")
            n_frames = -1
        if n_frames != manifest.num_rows:
            problems.append(
                f"{FRAMES_FILENAME} rows ({n_frames}) != manifest.num_rows ({manifest.num_rows})"
            )

    if not embeddings_path.is_file():
        problems.append(f"missing {EMBEDDINGS_FILENAME}")
    else:
        try:
            matrix = np.load(embeddings_path, mmap_mode="r")
            if matrix.ndim != 2:
                problems.append(f"{EMBEDDINGS_FILENAME} must be 2-D, got ndim={matrix.ndim}")
            else:
                if matrix.shape[0] != manifest.num_rows:
                    problems.append(
                        f"{EMBEDDINGS_FILENAME} rows ({matrix.shape[0]}) != "
                        f"manifest.num_rows ({manifest.num_rows})"
                    )
                if matrix.shape[1] != manifest.embedding_dim:
                    problems.append(
                        f"{EMBEDDINGS_FILENAME} dim ({matrix.shape[1]}) != "
                        f"manifest.embedding_dim ({manifest.embedding_dim})"
                    )
                if matrix.dtype != np.dtype(manifest.embedding_dtype):
                    problems.append(
                        f"{EMBEDDINGS_FILENAME} dtype ({matrix.dtype}) != "
                        f"manifest.embedding_dtype ({manifest.embedding_dtype})"
                    )
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            problems.append(f"{EMBEDDINGS_FILENAME} unreadable: {exc}")

    return problems
