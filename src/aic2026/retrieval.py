"""Core vector retrieval over a built index artifact.

This module is intentionally independent of raw dataset layout and
implements exact cosine retrieval over the artifact contract.

The frame-level :class:`RetrievalCandidate` values returned here are the
candidate-generation output consumed by the task-independent foundation
(:mod:`aic2026.retrieval_candidates`) for video and window-level grouping,
aggregation, and ranking in future QA/TRAKE layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic2026.embeddings import Embedding, validate_query_compatibility
from aic2026.index_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    NORMS_FILENAME,
    IndexManifest,
    iter_frames,
    read_manifest,
    validate_index_files,
)


@dataclass(frozen=True)
class RetrievalCandidate:
    """One ranked retrieval hit."""

    rank: int
    video_id: str
    keyframe_ordinal: int
    frame_index: int
    pts_time: float
    score: float


class IndexRetriever:
    """Load an index artifact once and run exact cosine search."""

    def __init__(self, artifact_dir: Path | str) -> None:
        artifact_dir = Path(artifact_dir)
        manifest = read_manifest(artifact_dir / "manifest.json")
        problems = validate_index_files(manifest, artifact_dir)
        if problems:
            raise ValueError(f"invalid index artifact: {problems}")
        self._artifact_dir = artifact_dir
        self._manifest = manifest
        self._frames = tuple(iter_frames(artifact_dir / FRAMES_FILENAME))
        self._embeddings = np.load(artifact_dir / EMBEDDINGS_FILENAME, mmap_mode="r")
        self._norms = np.load(artifact_dir / NORMS_FILENAME, mmap_mode="r")

    @property
    def manifest(self) -> IndexManifest:
        """The validated index artifact manifest."""
        return self._manifest

    @property
    def artifact_dir(self) -> Path:
        """The directory of the loaded index artifact."""
        return self._artifact_dir


    def search(
        self, query_vector: Sequence[float] | np.ndarray | Embedding, top_k: int = 10
    ) -> list[RetrievalCandidate]:
        """Return top_k cosine-ranked candidates.

        Expected input:
            - query_vector: an :class:`Embedding` whose declared space
              matches the index, or a 1-D raw vector only for an explicitly
              legacy/unknown index artifact.
            - top_k: positive integer number of results to return.

        Returns:
            - Ranked list of retrieval candidates.

        Raises:
            - ValueError: for incompatible embedding spaces, invalid query
              shape/dim, zero query norm, or invalid top_k.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        embedding_space = self._manifest.embedding_space
        if embedding_space is None:
            raise ValueError("index artifact is missing resolved embedding-space metadata")
        if isinstance(query_vector, Embedding):
            validate_query_compatibility(query_vector, embedding_space)
            query = query_vector.vector
        else:
            if embedding_space.backend != "legacy":
                raise ValueError(
                    "model-identified indexes require an Embedding query so embedding-space "
                    "compatibility can be validated"
                )
            query = np.asarray(query_vector)

        if query.ndim != 1:
            raise ValueError(f"query_vector must be 1-D, got ndim={query.ndim}")
        if int(query.shape[0]) != self._manifest.embedding_dim:
            raise ValueError(
                f"query_vector dim {query.shape[0]} != index dim {self._manifest.embedding_dim}"
            )
        if self._manifest.num_rows == 0:
            return []
        query32 = query.astype(np.float32, copy=False)
        query_norm = float(np.linalg.norm(query32))
        if query_norm == 0.0:
            raise ValueError("query_vector must have non-zero L2 norm")

        dots = self._embeddings.astype(np.float32) @ query32
        denom = self._norms.astype(np.float32) * np.float32(query_norm)
        scores = np.divide(
            dots,
            denom,
            out=np.full_like(dots, fill_value=-np.inf, dtype=np.float32),
            where=denom > 0.0,
        )

        n = self._manifest.num_rows
        order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
        k = min(top_k, n)
        out: list[RetrievalCandidate] = []
        for rank, row_idx in enumerate(order[:k], start=1):
            frame = self._frames[int(row_idx)]
            out.append(
                RetrievalCandidate(
                    rank=rank,
                    video_id=frame.keyframe.video_id.value,
                    keyframe_ordinal=frame.keyframe.keyframe_ordinal,
                    frame_index=frame.frame_index,
                    pts_time=frame.pts_time,
                    score=float(scores[int(row_idx)]),
                )
            )
        return out
