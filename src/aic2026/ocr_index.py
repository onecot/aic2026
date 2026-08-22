"""OCR-only dense retrieval index over temporal OCR chunks.

Encodes the ``text`` field of one completed OCR chunk artifact (see
``aic2026.ocr_chunks``) with an existing text encoder (e.g. Jina CLIP v2)
and stores a plain numpy exact-cosine index::

    index/
        manifest.json   versioned metadata incl. embedding_space
        embeddings.npy  float32 (num_rows, embedding_dim), L2-normalized
        chunks.ndjson   per-row mapping metadata (chunk identity + text)

Row ``i`` of ``embeddings.npy`` corresponds to row ``i`` of
``chunks.ndjson``, which in turn corresponds to chunk row ``i`` of the
source artifact (source order is preserved). Retrieval is exact cosine
similarity with deterministic tie-breaking by row index; no FAISS or other
vector database is involved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aic2026.embeddings import Embedding, EmbeddingSpace, validate_query_compatibility
from aic2026.ocr_chunks import read_ocr_chunks

logger = logging.getLogger(__name__)

#: OCR index artifact schema version.
OCR_INDEX_SCHEMA_VERSION = "1.0"

#: Numpy file holding the L2-normalized chunk embedding matrix.
EMBEDDINGS_FILENAME = "embeddings.npy"

#: NDJSON file holding per-row chunk mapping metadata.
CHUNKS_FILENAME = "chunks.ndjson"

#: Manifest file describing the index artifact.
MANIFEST_FILENAME = "manifest.json"


class OcrIndexError(RuntimeError):
    """Raised when an OCR index artifact cannot be built or is inconsistent."""


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


@dataclass(frozen=True)
class OcrIndexRow:
    """Per-row mapping metadata for one indexed OCR chunk.

    Fields:
        - chunk_id / video_id: source chunk identity.
        - representative_frame_index: retrievable original-video frame.
        - frame_start / frame_end: inclusive original-video frame span of
          the source chunk.
        - text: the exact OCR chunk text that was encoded.
    """

    chunk_id: str
    video_id: str
    representative_frame_index: int
    frame_start: int
    frame_end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "representative_frame_index": self.representative_frame_index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrIndexRow":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
        """
        return cls(
            chunk_id=str(data["chunk_id"]),
            video_id=str(data["video_id"]),
            representative_frame_index=int(data["representative_frame_index"]),
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class OcrIndexManifest:
    """Versioned metadata describing one OCR retrieval index.

    Fields:
        - schema_version: index schema version (``OCR_INDEX_SCHEMA_VERSION``).
        - artifact_id: identifier for this artifact.
        - created_at: UTC creation timestamp (ISO-8601).
        - source_chunks_artifact: path of the OCR chunk artifact.
        - embedding_space: resolved space of the encoding model (records
          backend, model id, checkpoint id, normalization, dim).
        - embedding_dim: 1-D vector length (equal to the space's dim).
        - num_rows: number of indexed chunks.
        - normalization: stored-vector normalization convention.
    """

    schema_version: str
    artifact_id: str
    created_at: str
    source_chunks_artifact: str
    embedding_space: EmbeddingSpace
    embedding_dim: int
    num_rows: int
    normalization: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "source_chunks_artifact": self.source_chunks_artifact,
            "embedding_space": self.embedding_space.to_dict(),
            "embedding_dim": self.embedding_dim,
            "num_rows": self.num_rows,
            "normalization": self.normalization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrIndexManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value violates the contract.
        """
        manifest = cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            created_at=str(data["created_at"]),
            source_chunks_artifact=str(data["source_chunks_artifact"]),
            embedding_space=EmbeddingSpace.from_dict(data["embedding_space"]),
            embedding_dim=int(data["embedding_dim"]),
            num_rows=int(data["num_rows"]),
            normalization=str(data["normalization"]),
        )
        _require(
            manifest.schema_version == OCR_INDEX_SCHEMA_VERSION,
            f"unsupported OCR index schema version: {manifest.schema_version!r}",
        )
        _require(manifest.embedding_dim == manifest.embedding_space.embedding_dim,
                 "embedding_dim must match the embedding space")
        _require(manifest.num_rows >= 0, "num_rows must be >= 0")
        return manifest


@dataclass(frozen=True)
class OcrSearchResult:
    """One ranked OCR chunk hit.

    Fields:
        - rank: 1-based rank in the result list.
        - chunk_id / video_id: source chunk identity.
        - representative_frame_index: retrievable original-video frame.
        - frame_start / frame_end: inclusive chunk frame span.
        - score: cosine similarity in ``[-1, 1]``.
        - text: the chunk's OCR text.
    """

    rank: int
    chunk_id: str
    video_id: str
    representative_frame_index: int
    frame_start: int
    frame_end: int
    score: float
    text: str


@dataclass(frozen=True)
class OcrIndexHandle:
    """A loaded OCR index ready for exact cosine search.

    Fields:
        - manifest: validated artifact metadata.
        - embeddings: float32 ``(num_rows, embedding_dim)`` matrix.
        - rows: per-row chunk metadata in matrix row order.
        - norms: precomputed float32 row L2 norms.

    Raises:
        - ValueError: if constructed with inconsistent contents.
    """

    manifest: OcrIndexManifest
    embeddings: np.ndarray
    rows: tuple[OcrIndexRow, ...]
    norms: np.ndarray

    def __post_init__(self) -> None:
        _require(self.embeddings.ndim == 2, "embeddings must be a 2-D matrix")
        n, d = self.embeddings.shape
        _require(n == self.manifest.num_rows, "matrix rows != manifest.num_rows")
        _require(d == self.manifest.embedding_dim, "matrix dim != manifest.embedding_dim")
        _require(len(self.rows) == n, "row metadata count != matrix rows")
        _require(self.norms.shape == (n,), "norms shape != matrix rows")

    def search(self, query: Embedding, top_k: int = 100) -> list[OcrSearchResult]:
        """Return top_k chunks ranked by exact cosine similarity.

        Expected input:
            - query: a text :class:`Embedding` comparable with the index
              space (validated before scoring).
            - top_k: positive integer number of results.

        Returns:
            - Ranked results (fewer than ``top_k`` when the index is smaller).

        Raises:
            - ValueError: on incompatible spaces, invalid shapes/dim, zero
              query norm, or invalid top_k.
        """
        _require(isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0,
                 f"top_k must be a positive integer, got {top_k!r}")
        validate_query_compatibility(query, self.manifest.embedding_space)
        vector = np.asarray(query.vector, dtype=np.float32)
        _require(vector.ndim == 1 and int(vector.shape[0]) == self.manifest.embedding_dim,
                 "query vector shape mismatch")
        query_norm = float(np.linalg.norm(vector))
        _require(query_norm > 0.0, "query vector must have non-zero L2 norm")
        n = self.manifest.num_rows
        if n == 0:
            return []
        scores = (self.embeddings @ vector) / (self.norms * np.float32(query_norm))
        order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
        results: list[OcrSearchResult] = []
        for rank, row_idx in enumerate(order[: min(top_k, n)], start=1):
            row = self.rows[int(row_idx)]
            results.append(
                OcrSearchResult(
                    rank=rank,
                    chunk_id=row.chunk_id,
                    video_id=row.video_id,
                    representative_frame_index=row.representative_frame_index,
                    frame_start=row.frame_start,
                    frame_end=row.frame_end,
                    score=float(scores[int(row_idx)]),
                    text=row.text,
                )
            )
        return results


def build_ocr_index(
    chunks_artifact: Path | str,
    encoder: Any,
    *,
    artifact_id: str = "ocr-index",
    batch_size: int = 32,
    created_at: str | None = None,
) -> tuple[OcrIndexManifest, np.ndarray, tuple[OcrIndexRow, ...]]:
    """Encode an OCR chunk artifact into an exact-search index.

    Expected input:
        - chunks_artifact: completed OCR chunk artifact directory.
        - encoder: a text encoder exposing ``encode_text_batch`` and
          ``embedding_space`` (e.g. ``JinaClipEncoder``).
        - artifact_id: identifier recorded in the manifest.
        - batch_size: texts encoded per encoder invocation (>= 1).
        - created_at: optional fixed UTC timestamp (mainly for tests).

    Returns:
        - ``(manifest, embeddings, rows)`` where row ``i`` of ``embeddings``
          encodes ``rows[i].text`` and ``rows[i]`` mirrors source chunk row
          ``i``.

    Raises:
        - ValueError: if the chunk artifact is invalid or ``batch_size < 1``.
        - RuntimeError: if the encoder fails.
    """
    _require(isinstance(batch_size, int) and batch_size >= 1, "batch_size must be >= 1")
    _, chunks = read_ocr_chunks(chunks_artifact)
    rows = tuple(
        OcrIndexRow(
            chunk_id=chunk.chunk_id,
            video_id=chunk.video_id,
            representative_frame_index=chunk.representative_frame_index,
            frame_start=chunk.frame_start,
            frame_end=chunk.frame_end,
            text=chunk.text,
        )
        for chunk in chunks
    )

    vectors: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        embeddings = encoder.encode_text_batch([row.text for row in batch])
        vectors.extend(embedding.vector.astype(np.float32, copy=False) for embedding in embeddings)
        logger.info("encoded %d/%d OCR chunks", min(start + batch_size, len(rows)), len(rows))

    if rows:
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
    else:
        matrix = np.zeros((0, encoder.embedding_space.embedding_dim), dtype=np.float32)

    space: EmbeddingSpace = encoder.embedding_space
    manifest = OcrIndexManifest(
        schema_version=OCR_INDEX_SCHEMA_VERSION,
        artifact_id=artifact_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        source_chunks_artifact=str(Path(chunks_artifact).resolve()),
        embedding_space=space,
        embedding_dim=int(matrix.shape[1]),
        num_rows=int(matrix.shape[0]),
        normalization=space.normalization,
    )
    return manifest, matrix, rows


def write_ocr_index(
    output_dir: Path | str,
    manifest: OcrIndexManifest,
    embeddings: np.ndarray,
    rows: Sequence[OcrIndexRow],
) -> Path:
    """Write an OCR index artifact atomically.

    Expected input:
        - output_dir: target directory (created if absent).
        - manifest / embeddings / rows: as returned by
          :func:`build_ocr_index`.

    Returns:
        - The output directory.

    Raises:
        - OSError: if outputs cannot be written.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings_path = output / EMBEDDINGS_FILENAME
    tmp_path = embeddings_path.with_name(embeddings_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(embeddings, dtype=np.float32), allow_pickle=False)
    tmp_path.replace(embeddings_path)
    chunks_path = output / CHUNKS_FILENAME
    tmp_chunks = chunks_path.with_name(chunks_path.name + ".tmp")
    with tmp_chunks.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    tmp_chunks.replace(chunks_path)
    _write_json_atomic(output / MANIFEST_FILENAME, manifest.to_dict())
    return output


def read_ocr_index(artifact_dir: Path | str) -> OcrIndexHandle:
    """Load and validate a completed OCR index artifact.

    Expected input:
        - artifact_dir: directory containing ``manifest.json``,
          ``embeddings.npy``, and ``chunks.ndjson``.

    Returns:
        - A ready-to-search :class:`OcrIndexHandle`.

    Raises:
        - FileNotFoundError: if a required file is missing.
        - ValueError: if the manifest or files are inconsistent.
    """
    directory = Path(artifact_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing OCR index manifest: {manifest_path}")
    manifest = OcrIndexManifest.from_dict(_read_json(manifest_path))
    embeddings_path = directory / EMBEDDINGS_FILENAME
    chunks_path = directory / CHUNKS_FILENAME
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"missing OCR index embeddings: {embeddings_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"missing OCR index rows: {chunks_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r").astype(np.float32)
    rows: list[OcrIndexRow] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            rows.append(OcrIndexRow.from_dict(json.loads(line)))
    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
    return OcrIndexHandle(manifest=manifest, embeddings=embeddings, rows=tuple(rows), norms=norms)


def search_ocr_chunks(
    index: Path | str | OcrIndexHandle,
    query_text: str,
    *,
    encoder: Any,
    top_k: int = 100,
) -> list[OcrSearchResult]:
    """Encode one query text and return ranked OCR chunks.

    Expected input:
        - index: an :class:`OcrIndexHandle` or an artifact directory.
        - query_text: non-empty query text.
        - encoder: text encoder whose space must match the index.
        - top_k: positive integer number of results.

    Returns:
        - Ranked :class:`OcrSearchResult` list.

    Raises:
        - ValueError: on invalid input or incompatible embedding spaces.
    """
    _require(isinstance(query_text, str) and query_text.strip() != "", "query_text must be non-empty")
    handle = index if isinstance(index, OcrIndexHandle) else read_ocr_index(index)
    query = encoder.encode_text(query_text)
    return handle.search(query, top_k=top_k)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
