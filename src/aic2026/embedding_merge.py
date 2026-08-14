"""Deterministic merge of completed embedding shards into one artifact.

Combines multiple completed shard directories (each produced by
``aic2026.embedding_shard.embed_shard``) into a single embedding artifact::

    merged/
        embeddings.npy    concatenated (N, D) matrix; row i <-> frames[i]
        frames.ndjson     concatenated keyframe records (row order)
        manifest.json     versioned merge metadata

Merge guarantees:

- every shard must be completed (a ``manifest.json`` exists) and internally
  consistent;
- all shards must share the same ``model``, ``embedding_dim``, and
  ``embedding_space`` (mixing is rejected);
- no keyframe may appear more than once across shards (duplicates are
  rejected);
- the merge order is deterministic: shards are ordered by ``worker_id`` (with
  a directory-name tie-break), and within a shard rows are kept in file order;
- ``embeddings[i]`` always corresponds exactly to ``frames.ndjson[i]``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.embedding_shard import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    ShardManifest,
)
from aic2026.embeddings import EmbeddingSpace

#: Schema version of the merge manifest.
MERGE_MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename of the merge manifest inside the output directory.
MERGE_MANIFEST_FILENAME = "manifest.json"

logger = logging.getLogger(__name__)


class MergeError(RuntimeError):
    """Raised when shards cannot be merged (incomplete, inconsistent, or duplicated)."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class MergeManifest:
    """Versioned metadata describing one merged embedding artifact.

    Fields:
        - schema_version: merge schema version
          (``MERGE_MANIFEST_SCHEMA_VERSION``).
        - model: shared model identity across all merged shards.
        - embedding_dim: shared dimensionality ``D`` of each embedding vector.
        - total_embeddings: total number ``N`` of merged keyframes (rows).
        - source_shards: the merged shard directory paths, in merge order.
        - embedding_space: shared embedding-space identity of the vectors.
        - created_at: ISO-8601 timestamp of merge completion.
    """

    schema_version: str
    model: str
    embedding_dim: int
    total_embeddings: int
    source_shards: tuple[str, ...]
    embedding_space: EmbeddingSpace
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == MERGE_MANIFEST_SCHEMA_VERSION,
            f"unsupported merge schema version {self.schema_version!r}",
        )
        _require(isinstance(self.model, str) and self.model != "", "model required")
        _require(isinstance(self.embedding_dim, int) and self.embedding_dim >= 1, "embedding_dim must be >= 1")
        _require(isinstance(self.total_embeddings, int) and self.total_embeddings >= 0, "total_embeddings must be >= 0")
        _require(len(self.source_shards) >= 1, "source_shards must not be empty")
        _require(isinstance(self.embedding_space, EmbeddingSpace), "embedding_space required")
        _require(
            self.embedding_dim == self.embedding_space.embedding_dim,
            "embedding_dim must match embedding_space.embedding_dim",
        )
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "total_embeddings": self.total_embeddings,
            "source_shards": list(self.source_shards),
            "embedding_space": self.embedding_space.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeManifest":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            model=data["model"],
            embedding_dim=int(data["embedding_dim"]),
            total_embeddings=int(data["total_embeddings"]),
            source_shards=tuple(data["source_shards"]),
            embedding_space=EmbeddingSpace.from_dict(data["embedding_space"]),
            created_at=data["created_at"],
        )


def order_shards(shard_dirs: list[Path]) -> list[Path]:
    """Order shard directories deterministically for merging.

    Shards are ordered by the ``worker_id`` recorded in each shard manifest,
    with a directory-path tie-break for shards sharing a worker id.

    Expected input:
        - shard_dirs: completed shard directories.

    Returns:
        - The same directories sorted into merge order.

    Raises:
        - MergeError: if a shard is not complete (its ``manifest.json`` is
          missing) so its worker id cannot be read.
    """

    def order_key(directory: Path) -> tuple[int, str]:
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise MergeError(
                f"shard is not complete (missing {MANIFEST_FILENAME}): {directory}"
            )
        worker_id = ShardManifest.from_dict(_read_json(manifest_path)).worker_id
        return worker_id, str(directory)

    return sorted(shard_dirs, key=order_key)


def _load_shard(
    shard_dir: Path,
    reference: ShardManifest,
) -> tuple[ShardManifest, np.ndarray, list[dict[str, Any]]]:
    """Load and validate one shard against a reference manifest.

    Raises:
        - MergeError: if the shard is incomplete or inconsistent with the
          reference (model, embedding_dim, embedding_space, or dtype).
    """
    manifest_path = shard_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MergeError(f"shard is not complete (missing {MANIFEST_FILENAME}): {shard_dir}")
    manifest = ShardManifest.from_dict(_read_json(manifest_path))

    if manifest.model != reference.model:
        raise MergeError(
            f"shard model {manifest.model!r} != merged model {reference.model!r}: {shard_dir}"
        )
    if manifest.embedding_dim != reference.embedding_dim:
        raise MergeError(
            f"shard embedding_dim {manifest.embedding_dim} != merged {reference.embedding_dim}: {shard_dir}"
        )
    if manifest.embedding_space != reference.embedding_space:
        raise MergeError(f"shard embedding_space does not match the merged space: {shard_dir}")
    if manifest.dtype != reference.dtype:
        raise MergeError(f"shard dtype {manifest.dtype!r} != merged {reference.dtype!r}: {shard_dir}")

    embeddings_path = shard_dir / EMBEDDINGS_FILENAME
    frames_path = shard_dir / FRAMES_FILENAME
    if not embeddings_path.is_file() or not frames_path.is_file():
        raise MergeError(f"shard is missing embeddings.npy or frames.ndjson: {shard_dir}")

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != manifest.num_embeddings:
        raise MergeError(
            f"shard embeddings rows {embeddings.shape[0]} != manifest {manifest.num_embeddings}: {shard_dir}"
        )
    with frames_path.open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle if line.strip()]
    if len(frames) != manifest.num_embeddings:
        raise MergeError(
            f"shard frames rows {len(frames)} != manifest {manifest.num_embeddings}: {shard_dir}"
        )
    return manifest, embeddings, frames


def _collect_keys(frames: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return ``(video_id, keyframe_ordinal)`` keys for frame records."""
    keys: list[tuple[str, int]] = []
    for row in frames:
        kf = row["keyframe"]
        keys.append((kf["video_id"]["value"], int(kf["keyframe_ordinal"])))
    return keys


def merge_shards(
    shard_dirs: list[Path | str],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> MergeManifest:
    """Merge completed embedding shards into a single embedding artifact.

    Expected input:
        - shard_dirs: completed shard directories (each with ``manifest.json``,
          ``embeddings.npy``, ``frames.ndjson``).
        - output_dir: target directory for the merged artifact (created if
          absent).
        - overwrite: if True, replace an existing merged artifact; otherwise
          raise ``MergeError`` when the output already exists.

    Returns:
        - A :class:`MergeManifest` describing the merged artifact.

    Raises:
        - MergeError: on incomplete/inconsistent shards, mixed models or
          dimensions, overlapping (duplicate) frames, or a pre-existing
          non-overwritten output.
        - FileNotFoundError: if a shard directory is missing.
        - OSError: if outputs cannot be written.
    """
    shard_dirs = [Path(d) for d in shard_dirs]
    if not shard_dirs:
        raise MergeError("at least one shard directory is required")
    for directory in shard_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"shard directory does not exist: {directory}")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise MergeError(
            f"merge output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )

    ordered = order_shards(shard_dirs)

    # Reference identity comes from the first shard in merge order.
    reference = ShardManifest.from_dict(_read_json(ordered[0] / MANIFEST_FILENAME))

    all_embeddings: list[np.ndarray] = []
    all_frames: list[list[dict[str, Any]]] = []
    seen_keys: set[tuple[str, int]] = set()
    for directory in ordered:
        manifest, embeddings, frames = _load_shard(directory, reference)
        keys = _collect_keys(frames)
        for key in keys:
            if key in seen_keys:
                raise MergeError(f"duplicate keyframe across shards: {key}")
            seen_keys.add(key)
        all_embeddings.append(embeddings)
        all_frames.append(frames)

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, reference.embedding_dim), dtype=np.dtype(reference.dtype))
    total = int(embeddings.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    # Write the embeddings matrix (mmap write is atomic-ish; a final manifest
    # write marks the artifact complete).
    np.save(output_dir / EMBEDDINGS_FILENAME, embeddings, allow_pickle=False)
    frames_out = output_dir / FRAMES_FILENAME
    tmp_frames = frames_out.with_name(frames_out.name + ".tmp")
    with tmp_frames.open("w", encoding="utf-8") as handle:
        for frames in all_frames:
            for row in frames:
                handle.write(json.dumps(row) + "\n")
    tmp_frames.replace(frames_out)

    manifest = MergeManifest(
        schema_version=MERGE_MANIFEST_SCHEMA_VERSION,
        model=reference.model,
        embedding_dim=reference.embedding_dim,
        total_embeddings=total,
        source_shards=tuple(str(d) for d in ordered),
        embedding_space=reference.embedding_space,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_atomic(output_dir / MERGE_MANIFEST_FILENAME, manifest.to_dict())
    logger.info("merged %d shards -> %d embeddings (model=%s)", len(ordered), total, reference.model)
    return manifest