"""Read-only accessor over the raw BTC dataset yielding canonical records.

This is the loading layer for the pipeline. It hides the raw BTC
directory/file layout (collection folder names, filenames, zero-padding,
CSV/JSON structure, and how the CLIP embeddings are stored) behind a
small, stable API that returns :mod:`aic2026.canonical` entities.

Responsibilities and separation:
    - raw parsing            -> ``aic2026.entities``
    - raw -> canonical       -> ``aic2026.converters``
    - canonical definitions  -> ``aic2026.canonical``
    - dataset access/routing -> this module
    - CLI                    -> ``aic2026.cli`` (interface only)

Data flow::

    raw BTC layout  --entities.parsers-->  raw records
        --converters-->  canonical records  <-Dataset.xxx

The loader is strictly read-only with respect to ``data/raw/``: it never
writes, renames, copies, or generates files under the raw root. It also
does not maintain any cache or materialized copy of the dataset.

Lazy access: embeddings are streamed row-by-row from a memory-mapped
``.npy`` matrix, and detections are parsed one keyframe JSON at a time,
so a full dataset (all videos) is never loaded into memory at once.

Failure policy: records are never silently skipped. A missing expected
raw file raises :class:`MissingRecordError`; a present-but-malformed or
inconsistent raw file raises :class:`CorruptRecordError`. Both wrap the
underlying cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from aic2026.canonical import (
    ClipEmbedding,
    KeyframeDetections,
    KeyframeKey,
    KeyframeTiming,
    VideoId,
    VideoMetadata,
)
from aic2026.contracts import COLLECTION_NAMES, video_id_from_collection_name
from aic2026.converters import (
    keyframe_key_from_parts,
    keyframe_map_rows_to_timings,
    media_dict_to_metadata,
    object_frame_to_detections,
)
from aic2026.entities import (
    parse_keyframe_map,
    parse_media_info,
    parse_object_detections,
)
from aic2026.frame_images import find_keyframe_image
from aic2026.inspection import discover_video_ids


class DatasetError(Exception):
    """Base error for failures in the dataset loading layer."""


class MissingRecordError(DatasetError):
    """An expected raw record (file or directory) is absent."""


class CorruptRecordError(DatasetError):
    """A raw record exists but is malformed or inconsistent."""


class Dataset:
    """Read-only loading layer over the raw BTC dataset.

    Expected input:
        - raw_root: path to the raw dataset root (string or pathlib.Path).

    Returns:
        - A ``Dataset`` instance.

    Raises:
        - FileNotFoundError: if ``raw_root`` does not exist or is not a
          directory.
    """

    def __init__(self, raw_root: Path | str) -> None:
        self._raw_root = Path(raw_root)
        if not self._raw_root.is_dir():
            raise FileNotFoundError(f"raw dataset root not found: {self._raw_root}")

    @property
    def raw_root(self) -> Path:
        """Return the raw dataset root path.

        Returns:
            - The configured raw root as a :class:`pathlib.Path`.
        """
        return self._raw_root

    # ------------------------------------------------------------------ discovery

    def video_ids(self) -> list[str]:
        """Return all discovered video ids, sorted.

        Returns:
            - Sorted list of ``L*_V*`` video id strings present across
              all raw collections.

        Raises:
            - No exceptions when the root exists (checked at init).
        """
        return discover_video_ids(self._raw_root)

    def iter_video_ids(self) -> Iterator[str]:
        """Yield discovered video ids in sorted order.

        Returns:
            - Iterator of ``L*_V*`` video id strings.
        """
        yield from self.video_ids()

    def __iter__(self) -> Iterator[VideoId]:
        """Yield one :class:`VideoId` per discovered video.

        Returns:
            - Iterator of canonical :class:`VideoId` records.
        """
        for video_id in self.video_ids():
            yield VideoId(video_id)

    def __len__(self) -> int:
        """Return the number of discovered videos.

        Returns:
            - Count of unique video ids.
        """
        return len(self.video_ids())

    def __contains__(self, video_id: str) -> bool:
        """Return whether ``video_id`` is part of the dataset.

        Expected input:
            - video_id: the ``L*_V*`` id to look for.

        Returns:
            - True if the video id was found in any raw collection.
        """
        return self.has_video(video_id)

    def has_video(self, video_id: str) -> bool:
        """Return whether ``video_id`` appears in any raw collection.

        Expected input:
            - video_id: the ``L*_V*`` id to look for.

        Returns:
            - True if the video id was found in any raw collection.

        Raises:
            - No exceptions (non-matching ids simply return False).
        """
        if video_id_from_collection_name(video_id) is None:
            return False
        for collection in COLLECTION_NAMES:
            collection_dir = self._raw_root / collection
            if not collection_dir.is_dir():
                continue
            for entry in collection_dir.iterdir():
                if video_id_from_collection_name(entry.name) == video_id:
                    return True
        return False

    # ---------------------------------------------------------------- video-level

    def metadata(self, video_id: str) -> VideoMetadata:
        """Load the canonical metadata record for one video.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - Canonical :class:`VideoMetadata`.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the media-info file is absent.
            - CorruptRecordError: if the file is malformed or the payload
              violates the media-info contract.
        """
        self._require_valid_video_id(video_id)
        path = self._raw_root / "media-info" / f"{video_id}.json"
        if not path.is_file():
            raise MissingRecordError(f"media-info record missing for {video_id}: {path}")
        try:
            payload = parse_media_info(path)
        except Exception as exc:  # noqa: BLE001 - wrap with context
            raise CorruptRecordError(
                f"media-info record is corrupt for {video_id}: {exc}"
            ) from exc
        try:
            return media_dict_to_metadata(video_id, payload)
        except ValueError as exc:
            raise CorruptRecordError(
                f"media-info record is corrupt for {video_id}: {exc}"
            ) from exc

    def timings(self, video_id: str) -> list[KeyframeTiming]:
        """Load the canonical per-keyframe timings for one video.

        The timing list defines the canonical, ordered set of keyframes
        for the video (ordinals ``1..N``).

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - Ordered list of :class:`KeyframeTiming`, one per keyframe.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the map-keyframes CSV is absent.
            - CorruptRecordError: if the CSV is malformed or a row
              violates the timing contract.
        """
        self._require_valid_video_id(video_id)
        path = self._raw_root / "map-keyframes" / f"{video_id}.csv"
        if not path.is_file():
            raise MissingRecordError(f"map-keyframes record missing for {video_id}: {path}")
        try:
            rows = parse_keyframe_map(path)
        except Exception as exc:  # noqa: BLE001 - wrap with context
            raise CorruptRecordError(
                f"map-keyframes record is corrupt for {video_id}: {exc}"
            ) from exc
        try:
            return keyframe_map_rows_to_timings(video_id, rows)
        except ValueError as exc:
            raise CorruptRecordError(
                f"map-keyframes record is corrupt for {video_id}: {exc}"
            ) from exc

    def keyframe_ordinals(self, video_id: str) -> list[int]:
        """Return the ordered keyframe ordinals for one video.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - List of ordinals ``[1, 2, ..., N]`` as per the map.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the map-keyframes CSV is absent.
            - CorruptRecordError: if the CSV is malformed.
        """
        return [timing.keyframe.keyframe_ordinal for timing in self.timings(video_id)]

    def keyframe_image_path(self, video_id: str, keyframe_ordinal: int) -> Path:
        """Return the raw image path for one keyframe.

        Delegates to the canonical resolver
        (:func:`aic2026.frame_images.find_keyframe_image`); the raw filename
        extension and zero-padding stay hidden behind this accessor.

        Expected input:
            - video_id: the ``L*_V*`` id.
            - keyframe_ordinal: the 1-based keyframe ordinal to look up.

        Returns:
            - The image file path for that keyframe.

        Raises:
            - ValueError: if ``video_id`` is malformed or the ordinal is
              not a positive integer.
            - MissingRecordError: if no keyframe image exists for the
              ordinal.
        """
        self._require_valid_video_id(video_id)
        try:
            return find_keyframe_image(self._raw_root, video_id, keyframe_ordinal)
        except FileNotFoundError as exc:
            raise MissingRecordError(str(exc)) from exc

    def iter_keyframes(self, video_id: str) -> Iterator[KeyframeKey]:
        """Yield one canonical :class:`KeyframeKey` per keyframe.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - Iterator of :class:`KeyframeKey` in map order.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the map-keyframes CSV is absent.
            - CorruptRecordError: if the CSV is malformed.
        """
        for timing in self.timings(video_id):
            yield timing.keyframe

    # ---------------------------------------------------------------- lazy stream

    def iter_embeddings(self, video_id: str) -> Iterator[ClipEmbedding]:
        """Stream canonical embeddings for one video, one row at a time.

        The CLIP feature matrix is opened memory-mapped and rows are
        yielded one by one, so neither the full matrix nor the full
        dataset is ever copied into memory at once. Row ``i`` maps to
        keyframe ordinal ``i + 1``.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - Iterator of :class:`ClipEmbedding`, one per matrix row.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the matrix file is absent.
            - CorruptRecordError: if the file is unreadable or is not a
              2-D matrix.
        """
        self._require_valid_video_id(video_id)
        path = self._raw_root / "clip-features-32" / f"{video_id}.npy"
        if not path.is_file():
            raise MissingRecordError(f"embedding matrix missing for {video_id}: {path}")
        try:
            matrix = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise CorruptRecordError(
                f"embedding matrix is corrupt for {video_id}: {exc}"
            ) from exc
        try:
            if matrix.ndim != 2:
                raise CorruptRecordError(
                    f"embedding matrix must be 2-D for {video_id}: got ndim={matrix.ndim}"
                )
            for ordinal, vector in enumerate(matrix, start=1):
                yield ClipEmbedding(
                    keyframe=keyframe_key_from_parts(video_id, ordinal),
                    vector=vector,
                )
        finally:
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None and hasattr(mmap, "close"):
                mmap.close()

    def iter_detections(self, video_id: str) -> Iterator[KeyframeDetections]:
        """Stream canonical detections for one video, one keyframe at a time.

        A separate on-disk JSON is parsed per keyframe, so memory use
        stays proportional to a single keyframe's detections.

        Expected input:
            - video_id: the ``L*_V*`` id.

        Returns:
            - Iterator of :class:`KeyframeDetections` in map order.

        Raises:
            - ValueError: if ``video_id`` is malformed.
            - MissingRecordError: if the map-keyframes CSV or an object
              detections file is absent.
            - CorruptRecordError: if the CSV or a detection file is
              malformed.
        """
        self._require_valid_video_id(video_id)
        for timing in self.timings(video_id):
            ordinal = timing.keyframe.keyframe_ordinal
            path = self._raw_root / "objects" / video_id / f"{ordinal:03d}.json"
            if not path.is_file():
                raise MissingRecordError(
                    f"object detections missing for {video_id} keyframe {ordinal}: {path}"
                )
            try:
                frame = parse_object_detections(path, video_id=video_id)
                yield object_frame_to_detections(frame)
            except Exception as exc:  # noqa: BLE001 - wrap with context
                raise CorruptRecordError(
                    f"object detections corrupt for {video_id} keyframe {ordinal}: {exc}"
                ) from exc

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _require_valid_video_id(video_id: str) -> None:
        """Validate that ``video_id`` matches the ``L*_V*`` pattern.

        Raises:
            - ValueError: if the id is malformed.
        """
        VideoId(video_id)