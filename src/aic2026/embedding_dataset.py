"""Dataset-like read-only source backed by one embedding artifact.

This module lets the index stage consume an artifact produced by
``aic2026.embedding_runner.build_embeddings`` (``manifest.json``,
``embeddings.npy``, ``frames.ndjson``) without re-encoding images or touching
the raw BTC dataset. It exposes the same small accessor API that
``build_index`` requires so an index can be built directly from stored
embeddings.

Data flow::

    EmbeddingArtifactDataset (reads embedding artifact)
        --build_index-->  index artifact files

Row identity is taken exclusively from ``frames.ndjson``: each line is one
canonical :class:`~aic2026.canonical.KeyframeTiming`, and line ``i`` maps to
row ``i`` of ``embeddings.npy``. The adapter therefore preserves the artifact's
video ids, keyframe ordinals, ``frame_index``/``pts_time`` values, and
embedding dtype/dimension exactly as stored.

Differences from :class:`~aic2026.dataset.Dataset`:
    - ``raw_root`` returns the artifact directory (recorded as the index
      ``source_dataset_id``).
    - ``metadata()`` is unsupported: the embedding artifact contract does not
      carry per-video media metadata (author/title/...).
    - video ordering follows the artifact's ``frames.ndjson`` first-appearance
      row order, not the raw collection layout.

Typical usage::

    source = EmbeddingArtifactDataset("data/interim/embeddings/<artifact_id>")
    build_index(source, out_dir, artifact_id="...", embedding_space=source.embedding_space)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from aic2026.canonical import ClipEmbedding, KeyframeTiming, VideoId, VideoMetadata
from aic2026.dataset import CorruptRecordError, MissingRecordError
from aic2026.embedding_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    EmbeddingManifest,
    iter_frames,
    read_manifest,
)
from aic2026.embeddings import EmbeddingSpace


class EmbeddingArtifactDataset:
    """Dataset-like read-only source backed by one embedding artifact.

    Expected input:
        - artifact_dir: directory containing a valid embedding artifact
          (``manifest.json``, ``embeddings.npy``, ``frames.ndjson``).

    Raises:
        - FileNotFoundError: if ``artifact_dir`` does not exist.
        - CorruptRecordError: if the artifact is missing required files or its
          content contradicts its manifest.
    """

    def __init__(self, artifact_dir: Path | str) -> None:
        artifact_dir = Path(artifact_dir)
        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                f"embedding artifact directory not found: {artifact_dir}"
            )

        manifest_path = artifact_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise CorruptRecordError(
                f"embedding artifact missing {MANIFEST_FILENAME}: {manifest_path}"
            )
        try:
            manifest = read_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            raise CorruptRecordError(
                f"embedding artifact manifest is unreadable: {exc}"
            ) from exc

        embeddings_path = artifact_dir / EMBEDDINGS_FILENAME
        if not embeddings_path.is_file():
            raise CorruptRecordError(
                f"embedding artifact missing {EMBEDDINGS_FILENAME}: {embeddings_path}"
            )
        try:
            matrix = np.load(embeddings_path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise CorruptRecordError(
                f"embedding matrix is unreadable: {exc}"
            ) from exc
        if matrix.ndim != 2:
            raise CorruptRecordError(
                f"embedding matrix must be 2-D, got ndim={matrix.ndim}"
            )
        if (
            matrix.shape[0] != manifest.num_rows
            or matrix.shape[1] != manifest.embedding_dim
        ):
            raise CorruptRecordError(
                f"embedding matrix shape {matrix.shape} != manifest "
                f"(rows={manifest.num_rows}, dim={manifest.embedding_dim})"
            )
        if matrix.dtype != np.dtype(manifest.embedding_dtype):
            raise CorruptRecordError(
                f"embedding matrix dtype {matrix.dtype} != manifest "
                f"{manifest.embedding_dtype}"
            )

        frames_path = artifact_dir / FRAMES_FILENAME
        if not frames_path.is_file():
            raise CorruptRecordError(
                f"embedding artifact missing {FRAMES_FILENAME}: {frames_path}"
            )

        # Build the per-video row map in a single pass over frames.ndjson.
        # Row order (and video first-appearance order) is the artifact order.
        rows_by_video: dict[str, list[tuple[int, KeyframeTiming]]] = {}
        video_order: list[str] = []
        row_count = 0
        try:
            for row_index, timing in enumerate(iter_frames(frames_path)):
                row_count += 1
                video_id = timing.keyframe.video_id.value
                if video_id not in rows_by_video:
                    rows_by_video[video_id] = []
                    video_order.append(video_id)
                rows_by_video[video_id].append((row_index, timing))
        except ValueError as exc:
            raise CorruptRecordError(
                f"embedding {FRAMES_FILENAME} is malformed: {exc}"
            ) from exc
        if row_count != manifest.num_rows:
            raise CorruptRecordError(
                f"frames.ndjson rows ({row_count}) != manifest.num_rows "
                f"({manifest.num_rows})"
            )

        self._artifact_dir = artifact_dir
        self._manifest = manifest
        self._matrix = matrix
        self._rows_by_video = rows_by_video
        self._video_order = video_order
        self._timings_by_video = {
            video_id: sorted(
                (timing for _, timing in rows),
                key=lambda timing: timing.keyframe.keyframe_ordinal,
            )
            for video_id, rows in rows_by_video.items()
        }

    # ------------------------------------------------------------------ source identity

    @property
    def raw_root(self) -> Path:
        """Return the artifact directory (recorded as the index source id).

        Returns:
            - The artifact directory path.
        """
        return self._artifact_dir

    @property
    def manifest(self) -> EmbeddingManifest:
        """Return the validated embedding-artifact manifest."""
        return self._manifest

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return the embedding-space identity recorded by the manifest.

        Pass this to ``build_index(..., embedding_space=...)`` so the index
        manifest preserves the artifact's space instead of a legacy/unknown id.

        Returns:
            - The artifact's :class:`~aic2026.embeddings.EmbeddingSpace`.
        """
        return self._manifest.embedding_space

    @property
    def artifact_id(self) -> str:
        """Return the artifact's unique run identity."""
        return self._manifest.artifact_id

    @property
    def num_rows(self) -> int:
        """Return the total number of embedding rows in the artifact."""
        return self._manifest.num_rows

    @property
    def num_videos(self) -> int:
        """Return the number of distinct videos covered by the artifact."""
        return self._manifest.num_videos

    @property
    def embedding_dim(self) -> int:
        """Return the embedding vector dimension ``D``."""
        return self._manifest.embedding_dim

    @property
    def embedding_dtype(self) -> str:
        """Return the stored matrix dtype name, e.g. ``float32``."""
        return self._manifest.embedding_dtype

    # ------------------------------------------------------------------ discovery

    def video_ids(self) -> list[str]:
        """Return all video ids present, in artifact first-appearance order.

        Returns:
            - Ordered list of ``L*_V*`` ids exactly as grouped in
              ``frames.ndjson``.
        """
        return list(self._video_order)

    def iter_video_ids(self) -> Iterator[str]:
        """Yield video ids in artifact first-appearance order.

        Returns:
            - Iterator of ``L*_V*`` video id strings.
        """
        yield from self._video_order

    def __iter__(self) -> Iterator[VideoId]:
        """Yield one canonical :class:`VideoId` per video, in artifact order.

        Returns:
            - Iterator of canonical :class:`VideoId` records.
        """
        for video_id in self._video_order:
            yield VideoId(video_id)

    def __len__(self) -> int:
        """Return the number of distinct videos in the artifact.

        Returns:
            - Count of unique video ids.
        """
        return len(self._video_order)

    def __contains__(self, video_id: str) -> bool:
        """Return whether ``video_id`` is part of the artifact.

        Expected input:
            - video_id: the ``L*_V*`` id to look for.

        Returns:
            - True if the video id was found in the artifact.
        """
        return self.has_video(video_id)

    def has_video(self, video_id: str) -> bool:
        """Return whether ``video_id`` is part of the artifact.

        Expected input:
            - video_id: the ``L*_V*`` id to look for.

        Returns:
            - True if the video id was found in the artifact.
        """
        return video_id in self._rows_by_video

    # ------------------------------------------------------------------ video-level

    def timings(self, video_id: str) -> list[KeyframeTiming]:
        """Return the canonical per-keyframe timings for one video.

        Timings are returned in keyframe-ordinal order, matching
        ``Dataset.timings`` semantics that ``build_index`` relies on (it
        indexes ``timings[keyframe_ordinal - 1]``). All temporal fields
        (``pts_time``, ``fps``, ``frame_index``) are preserved as stored.

        Expected input:
            - video_id: the ``L*_V*`` id present in the artifact.

        Returns:
            - List of :class:`KeyframeTiming` in keyframe-ordinal order.

        Raises:
            - MissingRecordError: if ``video_id`` is not part of the artifact.
        """
        self._require_video(video_id)
        return list(self._timings_by_video[video_id])

    def iter_embeddings(self, video_id: str) -> Iterator[ClipEmbedding]:
        """Stream canonical embeddings for one video in artifact row order.

        Rows are read from the memory-mapped matrix slice for the video, so
        only per-row vectors are materialized. The artifact's dtype, dimension,
        and per-row keyframe identity are preserved exactly.

        Expected input:
            - video_id: the ``L*_V*`` id present in the artifact.

        Returns:
            - Iterator of :class:`ClipEmbedding`, one per artifact row of the
              video, in artifact row order.

        Raises:
            - MissingRecordError: if ``video_id`` is not part of the artifact.
        """
        self._require_video(video_id)
        matrix = self._matrix
        for row_index, timing in self._rows_by_video[video_id]:
            yield ClipEmbedding(keyframe=timing.keyframe, vector=matrix[row_index])

    def metadata(self, video_id: str) -> VideoMetadata:
        """Raise ``NotImplementedError``: the artifact carries no media metadata.

        Per-video media metadata (author/title/...) is not part of the
        embedding artifact contract; use ``aic2026.dataset.Dataset`` against the
        raw dataset to read media-info records.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Raises:
            - NotImplementedError: always.
        """
        raise NotImplementedError(
            "media metadata is not part of the embedding artifact contract; "
            "use aic2026.dataset.Dataset to read raw media-info records"
        )

    # ------------------------------------------------------------------- helpers

    def _require_video(self, video_id: str) -> None:
        """Raise unless ``video_id`` is present in the artifact.

        Raises:
            - MissingRecordError: if ``video_id`` is not part of the artifact.
        """
        if video_id not in self._rows_by_video:
            raise MissingRecordError(
                f"video {video_id!r} not found in embedding artifact "
                f"{self._artifact_dir}"
            )

    def close(self) -> None:
        """Close the underlying memory-mapped embedding matrix.

        Returns:
            - None.
        """
        mmap = getattr(self._matrix, "_mmap", None)
        if mmap is not None and hasattr(mmap, "close"):
            mmap.close()
