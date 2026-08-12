"""Contract for the indexing-stage artifact.

This module defines the on-disk contract for the artifact produced by the
indexing stage. It is deliberately independent of both the raw BTC
directory layout and any particular ANN/indexing algorithm (FAISS, etc.).

Retrieval and evaluation stages consume this artifact instead of the raw
dataset. They depend only on the files and metadata declared here.

Scope note: this module only defines the contract (schema, file layout,
serialization helpers, and validation). It does not build an index and
does not perform retrieval.

Data flow::

    Dataset (canonical)  --index builder (future)-->  index artifact files
        ----------------------------------------------------------->
                     retrieval / evaluation read artifact only

Artifact layout (one directory per ``artifact_id``)::

    <indexes_dir>/<artifact_id>/
        manifest.json       versioned, self-describing metadata
        frames.ndjson       one canonical KeyframeTiming per row (row index p)
        embeddings.npy      dense (N, D) float16 matrix; row p <-> frames[p]
        norms.npy           (N,) float32 L2 norms; row p <-> frames[p]
        index.bin           RESERVED slot for algorithm payload (not produced)

Row ordering contract: every ``p`` in ``[0, N)`` refers to the same indexed
keyframe across ``frames.ndjson[p]``, ``embeddings.npy[p]`` and
``norms.npy[p]``. This gives retrieval a single, stable lookup key.

Score computation: for cosine similarity, a retrieval stage computes
``dot(query, embeddings[p]) / (norm(query) * norms[p])`` using the stored
norms, so it never needs to read the raw dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.canonical import KeyframeTiming
from aic2026.embeddings import EmbeddingSpace, legacy_embedding_space

#: Schema version of the index artifact. Version 1.1 adds explicit
#: embedding-space metadata. Version 1.0 manifests remain readable as
#: explicitly marked legacy/unknown spaces.
INDEX_SCHEMA_VERSION = "1.1"
SUPPORTED_INDEX_SCHEMA_VERSIONS = ("1.0", INDEX_SCHEMA_VERSION)

#: Relative filenames within an artifact directory.
MANIFEST_FILENAME = "manifest.json"
FRAMES_FILENAME = "frames.ndjson"
EMBEDDINGS_FILENAME = "embeddings.npy"
NORMS_FILENAME = "norms.npy"
PAYLOAD_FILENAME = "index.bin"

#: Score metric the artifact supports out of the box.
SUPPORTED_SCORE_METRICS = ("cosine",)


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class IndexManifest:
    """Versioned metadata describing one index artifact.

    Fields:
        - schema_version: the artifact schema version this manifest
          conforms to (``INDEX_SCHEMA_VERSION``).
        - artifact_id: unique identifier for this artifact (run identity).
        - score_metric: the similarity metric supported, e.g. ``cosine``.
        - embedding_dim: dimensionality ``D`` of each embedding vector.
        - embedding_dtype: numpy dtype name of the stored matrix, e.g.
          ``float16``.
        - num_rows: number ``N`` of indexed keyframes (rows).
        - num_videos: number of distinct videos covered.
        - created_at: ISO-8601 timestamp of artifact creation.
        - source_dataset_id: identifier of the source dataset, e.g.
          ``aic2026_batch1`` (storage-layout independent).
        - canonical_schema_version: canonical schema version the
          ``frames.ndjson`` records conform to.
        - embedding_space: model/backend identity and all metadata required
          to determine whether a query can be compared with this index.
    """

    schema_version: str
    artifact_id: str
    score_metric: str
    embedding_dim: int
    embedding_dtype: str
    num_rows: int
    num_videos: int
    created_at: str
    source_dataset_id: str
    canonical_schema_version: str
    embedding_space: EmbeddingSpace | None = None

    def __post_init__(self) -> None:
        _require(
            self.schema_version in SUPPORTED_INDEX_SCHEMA_VERSIONS,
            f"unsupported artifact schema version {self.schema_version!r}; "
            f"supported: {SUPPORTED_INDEX_SCHEMA_VERSIONS}",
        )
        _require(isinstance(self.artifact_id, str) and self.artifact_id != "", "artifact_id required")
        _require(
            self.score_metric in SUPPORTED_SCORE_METRICS,
            f"unsupported score_metric {self.score_metric!r}; "
            f"supported: {SUPPORTED_SCORE_METRICS}",
        )
        _require(isinstance(self.embedding_dim, int) and self.embedding_dim >= 1, "embedding_dim must be >= 1")
        _require(isinstance(self.embedding_dtype, str) and self.embedding_dtype != "", "embedding_dtype required")
        try:
            embedding_dtype = np.dtype(self.embedding_dtype)
        except TypeError as exc:
            raise ValueError(f"invalid embedding_dtype {self.embedding_dtype!r}") from exc
        _require(np.issubdtype(embedding_dtype, np.floating), "embedding_dtype must be a floating numpy dtype")
        embedding_space = self.embedding_space
        if embedding_space is None:
            embedding_space = legacy_embedding_space(self.embedding_dim, embedding_dtype.name)
            object.__setattr__(self, "embedding_space", embedding_space)
        _require(
            embedding_space.embedding_dim == self.embedding_dim,
            "embedding_space.embedding_dim must match embedding_dim",
        )
        _require(
            np.dtype(embedding_space.output_dtype) == embedding_dtype,
            "embedding_space.output_dtype must match embedding_dtype",
        )
        _require(isinstance(self.num_rows, int) and self.num_rows >= 0, "num_rows must be >= 0")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")
        _require(isinstance(self.source_dataset_id, str) and self.source_dataset_id != "", "source_dataset_id required")
        _require(
            isinstance(self.canonical_schema_version, str) and self.canonical_schema_version != "",
            "canonical_schema_version required",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            - Dict representation including the schema version.
        """
        embedding_space = self.embedding_space
        if embedding_space is None:
            raise ValueError("embedding_space must be resolved before serialization")
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "score_metric": self.score_metric,
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": self.embedding_dtype,
            "num_rows": self.num_rows,
            "num_videos": self.num_videos,
            "created_at": self.created_at,
            "source_dataset_id": self.source_dataset_id,
            "canonical_schema_version": self.canonical_schema_version,
            "embedding_space": embedding_space.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexManifest:
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid or the schema version is
              unsupported.
        """
        return cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            score_metric=data["score_metric"],
            embedding_dim=int(data["embedding_dim"]),
            embedding_dtype=data["embedding_dtype"],
            num_rows=int(data["num_rows"]),
            num_videos=int(data["num_videos"]),
            created_at=data["created_at"],
            source_dataset_id=data["source_dataset_id"],
            canonical_schema_version=data["canonical_schema_version"],
            embedding_space=(
                EmbeddingSpace.from_dict(data["embedding_space"])
                if "embedding_space" in data
                else None
            ),
        )


# ------------------------------------------------------------------ serialization


def write_manifest(path: Path, manifest: IndexManifest) -> None:
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


def read_manifest(path: Path) -> IndexManifest:
    """Read and validate a manifest from disk.

    Expected input:
        - path: a file path to ``manifest.json``.

    Returns:
        - A validated :class:`IndexManifest`.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if the JSON is malformed.
        - ValueError: if the manifest is invalid or has an unsupported
          schema version.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return IndexManifest.from_dict(payload)


def write_frames(path: Path, frames: Iterable[KeyframeTiming]) -> int:
    """Write frame records as NDJSON, one ``KeyframeTiming`` per line.

    Expected input:
        - path: a file path (usually ``<artifact_dir>/frames.ndjson``).
        - frames: iterable of canonical :class:`KeyframeTiming` in row
          order (index ``p`` becomes line ``p``).

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


def validate_index_files(manifest: IndexManifest, artifact_dir: Path) -> list[str]:
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
    norms_path = artifact_dir / NORMS_FILENAME

    if not frames_path.is_file():
        problems.append(f"missing {FRAMES_FILENAME}")
    else:
        n_frames = -1
        try:
            n_frames = sum(1 for _ in iter_frames(frames_path))
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            problems.append(f"{FRAMES_FILENAME} unreadable: {exc}")
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

    if not norms_path.is_file():
        problems.append(f"missing {NORMS_FILENAME}")
    else:
        try:
            norms = np.load(norms_path)
            if norms.ndim != 1 or norms.shape[0] != manifest.num_rows:
                problems.append(
                    f"{NORMS_FILENAME} must be 1-D of length {manifest.num_rows}, "
                    f"got shape={norms.shape}"
                )
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            problems.append(f"{NORMS_FILENAME} unreadable: {exc}")

    return problems