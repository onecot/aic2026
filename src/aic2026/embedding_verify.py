"""Verification of a completed embedding artifact.

The embedding stage produces an artifact directory containing:

.. code-block:: text

    <output-dir>/
        embeddings.npy    dense (N, D) matrix; row i <-> frames[i]
        frames.ndjson     one canonical KeyframeTiming per line (row i)
        manifest.json     versioned metadata describing the artifact

This module verifies the artifact for internal consistency, independent of
any model backend:

- every required file exists (``embeddings.npy``, ``frames.ndjson``,
  ``manifest.json``);
- the manifest parses and records model metadata (embedding-space identity);
- ``embeddings.npy`` is a 2-D floating matrix whose row count matches
  ``frames.ndjson``;
- the matrix dimension matches the manifest's ``embedding_dim``;
- the matrix dtype matches the manifest's dtype when the manifest records one;
- no embedding vector is empty (all-zero) or contains NaN/Inf.

Both artifact manifest schemas are supported: the per-shard/merged
``EmbeddingManifest`` (``aic2026.embedding_artifact``) and the merge output
``MergeManifest`` (``aic2026.embedding_merge``). It never runs a model and
never requires media files or images.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.canonical import KeyframeTiming
from aic2026.embedding_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    EmbeddingManifest,
)
from aic2026.embedding_merge import MergeManifest
from aic2026.embeddings import EmbeddingSpace

logger = logging.getLogger(__name__)

#: Model identity when ``manifest.json`` is absent or cannot be parsed.
MODEL_ID_UNKNOWN = "unknown"


@dataclass(frozen=True)
class EmbeddingVerificationIssue:
    """One problem found while verifying an embedding artifact.

    Fields:
        - section: which artifact component the issue belongs to
          (``artifact``, ``manifest``, ``frames``, ``embeddings``).
        - message: human-readable description of the problem.
    """

    section: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"section": self.section, "message": self.message}


@dataclass(frozen=True)
class EmbeddingVerificationReport:
    """Result of verifying one embedding artifact.

    Fields:
        - output_dir: the artifact directory that was verified.
        - valid: True when no issues were found.
        - issues: every problem found (empty when ``valid``).
        - rows: number of rows in ``embeddings.npy`` (``-1`` when unreadable).
        - frames: number of records in ``frames.ndjson`` (``-1`` when
          unreadable).
        - dim: embedding vector dimension ``D`` (``None`` when unavailable).
        - dtype: numpy dtype name stored by the matrix (``None`` when
          unreadable).
        - empty_vectors: number of all-zero embedding rows.
        - nan_vectors: number of rows containing NaN or Inf.
        - model_id: model identifier recorded in the manifest (or
          :data:`MODEL_ID_UNKNOWN`).
    """

    output_dir: str
    valid: bool
    issues: tuple[EmbeddingVerificationIssue, ...]
    rows: int
    frames: int
    dim: int | None
    dtype: str | None
    empty_vectors: int
    nan_vectors: int
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "output_dir": self.output_dir,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "rows": self.rows,
            "frames": self.frames,
            "dim": self.dim,
            "dtype": self.dtype,
            "empty_vectors": self.empty_vectors,
            "nan_vectors": self.nan_vectors,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class _ManifestInfo:
    """Normalized metadata shared by both embedding manifest schemas."""

    schema_version: str
    artifact_id: str
    model_id: str
    embedding_dim: int
    embedding_dtype: str | None
    num_rows: int


def _model_id_from_space(space: EmbeddingSpace) -> str:
    """Return the model identifier of an embedding space, or UNKNOWN."""
    if space is None or not space.model_id:
        return MODEL_ID_UNKNOWN
    return space.model_id


def _read_manifest(
    output_dir: Path,
    issues: list[EmbeddingVerificationIssue],
) -> _ManifestInfo | None:
    """Parse ``manifest.json`` or record the failure.

    Supports both the runner/merge ``EmbeddingManifest`` and the merge-output
    ``MergeManifest`` schemas.

    Returns:
        - Normalized manifest metadata, or ``None`` when the file is
          missing/invalid.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(
            EmbeddingVerificationIssue(
                "artifact", f"manifest.json missing: {manifest_path}"
            )
        )
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        issues.append(
            EmbeddingVerificationIssue("manifest", f"invalid manifest.json: {exc}")
        )
        return None

    if not isinstance(data, dict):
        issues.append(
            EmbeddingVerificationIssue("manifest", "manifest.json is not a JSON object")
        )
        return None

    manifest: EmbeddingManifest | MergeManifest
    try:
        if "total_embeddings" in data and "source_shards" in data:
            manifest = MergeManifest.from_dict(data)
        else:
            manifest = EmbeddingManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(
            EmbeddingVerificationIssue("manifest", f"invalid manifest.json: {exc}")
        )
        return None

    if isinstance(manifest, MergeManifest):
        return _ManifestInfo(
            schema_version=manifest.schema_version,
            artifact_id=MODEL_ID_UNKNOWN,
            model_id=manifest.model,
            embedding_dim=manifest.embedding_dim,
            embedding_dtype=None,
            num_rows=manifest.total_embeddings,
        )

    return _ManifestInfo(
        schema_version=manifest.schema_version,
        artifact_id=manifest.artifact_id,
        model_id=_model_id_from_space(manifest.embedding_space),
        embedding_dim=manifest.embedding_dim,
        embedding_dtype=manifest.embedding_dtype,
        num_rows=manifest.num_rows,
    )


def _read_frames(
    output_dir: Path,
    issues: list[EmbeddingVerificationIssue],
) -> int:
    """Count canonical records in ``frames.ndjson``.

    Returns:
        - Number of valid canonical records, or ``-1`` when the file is
          missing/unreadable.
    """
    frames_path = output_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        issues.append(
            EmbeddingVerificationIssue("artifact", f"frames.ndjson missing: {frames_path}")
        )
        return -1
    count = 0
    try:
        with frames_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    KeyframeTiming.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    issues.append(
                        EmbeddingVerificationIssue(
                            "frames",
                            f"invalid frame record on line {line_no}: {exc}",
                        )
                    )
                    continue
                count += 1
    except OSError as exc:
        issues.append(
            EmbeddingVerificationIssue("frames", f"frames.ndjson unreadable: {exc}")
        )
        return -1
    if count == 0:
        issues.append(
            EmbeddingVerificationIssue(
                "frames", f"{FRAMES_FILENAME} contains no frame records"
            )
        )
    return count


def _check_matrix(
    output_dir: Path,
    issues: list[EmbeddingVerificationIssue],
    manifest: _ManifestInfo | None,
) -> tuple[int, int | None, str | None, int, int]:
    """Validate ``embeddings.npy`` shape, dtype, and vector health.

    Returns:
        - ``(rows, dim, dtype, empty_vectors, nan_vectors)``; ``rows`` is
          ``-1`` and ``dim``/``dtype`` are ``None`` when the file is
          missing/unreadable.
    """
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    if not embeddings_path.is_file():
        issues.append(
            EmbeddingVerificationIssue(
                "embeddings", f"embeddings.npy missing: {embeddings_path}"
            )
        )
        return -1, None, None, 0, 0

    try:
        matrix = np.load(embeddings_path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - surface any load failure
        issues.append(
            EmbeddingVerificationIssue("embeddings", f"embeddings.npy unreadable: {exc}")
        )
        return -1, None, None, 0, 0

    if matrix.ndim != 2:
        issues.append(
            EmbeddingVerificationIssue(
                "embeddings", f"embeddings.npy must be 2-D, got ndim={matrix.ndim}"
            )
        )
        return -1, None, None, 0, 0

    rows = int(matrix.shape[0])
    dim = int(matrix.shape[1])
    dtype_name = str(matrix.dtype)

    if not np.issubdtype(matrix.dtype, np.floating):
        issues.append(
            EmbeddingVerificationIssue(
                "embeddings",
                f"embeddings.npy dtype {matrix.dtype} is not floating",
            )
        )

    if manifest is not None:
        if manifest.embedding_dim != dim:
            issues.append(
                EmbeddingVerificationIssue(
                    "embeddings",
                    f"embeddings.npy dim ({dim}) != manifest embedding_dim "
                    f"({manifest.embedding_dim})",
                )
            )
        if manifest.embedding_dtype is not None and np.dtype(manifest.embedding_dtype) != matrix.dtype:
            issues.append(
                EmbeddingVerificationIssue(
                    "embeddings",
                    f"embeddings.npy dtype ({matrix.dtype}) != manifest dtype "
                    f"({manifest.embedding_dtype})",
                )
            )
        if manifest.num_rows != rows:
            issues.append(
                EmbeddingVerificationIssue(
                    "embeddings",
                    f"embeddings.npy rows ({rows}) != manifest num_rows "
                    f"({manifest.num_rows})",
                )
            )

    empty_vectors = 0
    nan_vectors = 0
    if rows:
        nan_rows = np.isnan(matrix).any(axis=1) | np.isinf(matrix).any(axis=1)
        nan_vectors = int(nan_rows.sum())
        zero_rows = np.all(matrix == 0, axis=1)
        empty_vectors = int(zero_rows.sum())
        if nan_vectors:
            issues.append(
                EmbeddingVerificationIssue(
                    "embeddings",
                    f"{nan_vectors} embedding rows contain NaN/Inf",
                )
            )
        if empty_vectors:
            issues.append(
                EmbeddingVerificationIssue(
                    "embeddings",
                    f"{empty_vectors} embedding rows are all-zero (empty vectors)",
                )
            )

    return rows, dim, dtype_name, empty_vectors, nan_vectors


def verify_embedding(output_dir: Path | str) -> EmbeddingVerificationReport:
    """Verify a completed embedding artifact.

    Checks artifact existence, manifest parsing and model metadata, frame
    record validity and count, and the embedding matrix (2-D floating, row
    count matches frames and manifest, dimension and dtype match the manifest
    when recorded, no empty or NaN/Inf vectors).

    Expected input:
        - output_dir: directory of an embedding artifact (``embeddings.npy`` +
          ``frames.ndjson`` + ``manifest.json``).

    Returns:
        - An :class:`EmbeddingVerificationReport`; the CLI maps ``valid`` to
          its exit code.

    Raises:
        - FileNotFoundError: if the artifact directory is absent.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"embedding output directory not found: {output_dir}")

    issues: list[EmbeddingVerificationIssue] = []
    manifest = _read_manifest(output_dir, issues)
    frames = _read_frames(output_dir, issues)
    rows, dim, dtype, empty_vectors, nan_vectors = _check_matrix(
        output_dir, issues, manifest
    )

    if rows >= 0 and frames >= 0 and rows != frames:
        issues.append(
            EmbeddingVerificationIssue(
                "embeddings",
                f"embeddings.npy rows ({rows}) != frames.ndjson records ({frames})",
            )
        )

    if manifest is not None:
        model_id = manifest.model_id
        if not model_id or model_id == MODEL_ID_UNKNOWN:
            issues.append(
                EmbeddingVerificationIssue(
                    "manifest", "model metadata missing: no model identity recorded"
                )
            )
    else:
        model_id = MODEL_ID_UNKNOWN

    return EmbeddingVerificationReport(
        output_dir=str(output_dir),
        valid=not issues,
        issues=tuple(issues),
        rows=rows,
        frames=frames,
        dim=dim,
        dtype=dtype,
        empty_vectors=empty_vectors,
        nan_vectors=nan_vectors,
        model_id=model_id,
    )
