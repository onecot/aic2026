"""Deterministic, workload-balanced split of a video manifest.

Given a ``videos.json`` video manifest, this module partitions its entries into
``N`` worker manifests such that:

- each video is kept intact (never split across workers);
- each worker receives an approximately balanced number of frames (using the
  optional ``num_frames`` weight; videos without it weigh 1);
- the result is deterministic (same input + same ``num_workers`` -> identical
  splits);
- no video is duplicated or dropped (the union of the splits equals the input).

Split directory layout::

    <output-dir>/
        split_manifest.json   versioned split metadata
        worker_000/
            videos.json
        worker_001/
            videos.json
        ...

Each ``worker_<id>/videos.json`` is a valid :class:`VideoManifest` suitable as
direct input to ``aic2026.keyframe_extraction.extract_keyframes``.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.keyframe_extraction import (
    VIDEOS_MANIFEST_FILENAME,
    VIDEOS_MANIFEST_SCHEMA_VERSION,
    VideoEntry,
    VideoManifest,
    load_video_manifest,
    save_video_manifest,
)

#: Schema version of the split manifest.
SPLIT_MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename of the split manifest inside the output directory.
SPLIT_MANIFEST_FILENAME = "split_manifest.json"

logger = logging.getLogger(__name__)


class SplitError(RuntimeError):
    """Raised when a split is invalid or a split target already exists."""


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


def worker_dir_name(worker_id: int) -> str:
    """Return the deterministic ``worker_<id>`` directory name.

    Expected input:
        - worker_id: a non-negative worker machine identifier.

    Returns:
        - ``worker_<id:03d>`` (e.g. ``worker_000``).
    """
    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0:
        raise ValueError(f"worker_id must be a non-negative integer, got {worker_id!r}")
    return f"worker_{worker_id:03d}"


def assign_videos(videos: list[VideoEntry], num_workers: int) -> list[list[VideoEntry]]:
    """Assign whole videos to workers, balancing estimated frame counts.

    Uses a greedy longest-processing-time assignment: videos are considered in
    descending weight (ties broken by video id), and each video is placed into
    the worker with the smallest accumulated weight (ties broken by the lowest
    worker id). Within each worker, videos are re-ordered by their original
    appearance order, so the split is fully deterministic.

    Expected input:
        - videos: the video entries in file order.
        - num_workers: number of worker splits (>= 1).

    Returns:
        - A list of ``num_workers`` video lists, index ``i`` being worker ``i``.

    Raises:
        - ValueError: if ``num_workers`` is not a positive integer.
    """
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
        raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")
    if not videos:
        return [[] for _ in range(num_workers)]

    def weight(entry: VideoEntry) -> int:
        return entry.num_frames if entry.num_frames is not None else 1

    assignment_order = sorted(videos, key=lambda v: (-weight(v), v.video_id))
    loads = [0] * num_workers
    assigned: list[list[str]] = [[] for _ in range(num_workers)]
    for entry in assignment_order:
        target = min(range(num_workers), key=lambda w: (loads[w], w))
        assigned[target].append(entry.video_id)
        loads[target] += weight(entry)

    by_id = {entry.video_id: entry for entry in videos}
    per_worker: list[list[VideoEntry]] = []
    for worker_id in range(num_workers):
        per_worker.append([by_id[vid] for vid in assigned[worker_id]])
    return per_worker


@dataclass(frozen=True)
class SplitWorkerInfo:
    """Metadata for one worker split.

    Fields:
        - worker_id: worker machine identifier.
        - num_videos: number of videos assigned to the worker.
        - estimated_frames: sum of ``num_frames`` weights (videos without a
          weight contribute 1).
        - video_ids: sorted video ids assigned to the worker.
    """

    worker_id: int
    num_videos: int
    estimated_frames: int
    video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.worker_id, int) and self.worker_id >= 0, "worker_id must be >= 0")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.estimated_frames, int) and self.estimated_frames >= 0, "estimated_frames must be >= 0")
        _require(all(isinstance(vid, str) and vid != "" for vid in self.video_ids), "video_ids must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "worker_id": self.worker_id,
            "num_videos": self.num_videos,
            "estimated_frames": self.estimated_frames,
            "video_ids": list(self.video_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitWorkerInfo":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            worker_id=int(data["worker_id"]),
            num_videos=int(data["num_videos"]),
            estimated_frames=int(data["estimated_frames"]),
            video_ids=tuple(data["video_ids"]),
        )


@dataclass(frozen=True)
class SplitManifest:
    """Versioned metadata describing one split of a video manifest.

    Fields:
        - schema_version: split schema version
          (``SPLIT_MANIFEST_SCHEMA_VERSION``).
        - source_videos: the input ``videos.json`` path that was split.
        - num_workers: total number of worker splits.
        - workers: per-worker metadata in worker order.
    """

    schema_version: str
    source_videos: str
    num_workers: int
    workers: tuple[SplitWorkerInfo, ...]

    def __post_init__(self) -> None:
        _require(
            self.schema_version == SPLIT_MANIFEST_SCHEMA_VERSION,
            f"unsupported split schema version {self.schema_version!r}",
        )
        _require(isinstance(self.source_videos, str) and self.source_videos != "", "source_videos required")
        _require(isinstance(self.num_workers, int) and self.num_workers >= 1, "num_workers must be >= 1")
        _require(len(self.workers) == self.num_workers, "workers must match num_workers")
        _require(
            [w.worker_id for w in self.workers] == list(range(self.num_workers)),
            "workers must be listed in ascending worker_id order",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "source_videos": self.source_videos,
            "num_workers": self.num_workers,
            "workers": [w.to_dict() for w in self.workers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            source_videos=data["source_videos"],
            num_workers=int(data["num_workers"]),
            workers=tuple(SplitWorkerInfo.from_dict(w) for w in data["workers"]),
        )


def _validate_split(per_worker: list[list[VideoEntry]], original: list[VideoEntry]) -> None:
    """Check that the splits reproduce the original manifest exactly.

    Asserts every video appears exactly once across workers (no duplicates, no
    missing videos).

    Raises:
        - SplitError: if a video is duplicated, missing, or the counts differ.
    """
    total = sum(len(w) for w in per_worker)
    if total != len(original):
        raise SplitError(f"split total videos {total} != original {len(original)}")
    seen: set[str] = set()
    for worker_videos in per_worker:
        for entry in worker_videos:
            if entry.video_id in seen:
                raise SplitError(f"duplicate video {entry.video_id} across splits")
            seen.add(entry.video_id)
    original_ids = {entry.video_id for entry in original}
    if seen != original_ids:
        raise SplitError("split videos do not match the original manifest exactly")


def split_video_manifest(
    videos_path: Path | str,
    num_workers: int,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> SplitManifest:
    """Split a ``videos.json`` manifest into ``num_workers`` worker manifests.

    Expected input:
        - videos_path: path to a ``videos.json`` video manifest.
        - num_workers: number of worker splits to produce (>= 1).
        - output_dir: target directory for the split (created if absent).
        - overwrite: if True, replace an existing split; otherwise raise
          ``SplitError`` when a split already exists there.

    Returns:
        - A :class:`SplitManifest` describing the produced splits.

    Raises:
        - FileNotFoundError: if ``videos_path`` does not exist.
        - ValueError: if ``num_workers`` is invalid or the manifest is invalid.
        - SplitError: if the split already exists and ``overwrite`` is False.
    """
    videos_path = Path(videos_path)
    output_dir = Path(output_dir)
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
        raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SplitError(
            f"split output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )

    manifest = load_video_manifest(videos_path)
    videos = list(manifest.videos)
    per_worker = assign_videos(videos, num_workers)
    _validate_split(per_worker, videos)

    output_dir.mkdir(parents=True, exist_ok=True)
    for worker_id, worker_videos in enumerate(per_worker):
        worker_dir = output_dir / worker_dir_name(worker_id)
        worker_dir.mkdir(parents=True, exist_ok=True)
        save_video_manifest(
            worker_dir / VIDEOS_MANIFEST_FILENAME,
            VideoManifest(schema_version=VIDEOS_MANIFEST_SCHEMA_VERSION, videos=tuple(worker_videos)),
        )

    worker_infos: list[SplitWorkerInfo] = []
    for worker_id, worker_videos in enumerate(per_worker):
        worker_infos.append(
            SplitWorkerInfo(
                worker_id=worker_id,
                num_videos=len(worker_videos),
                estimated_frames=sum(
                    (entry.num_frames if entry.num_frames is not None else 1) for entry in worker_videos
                ),
                video_ids=tuple(sorted(entry.video_id for entry in worker_videos)),
            )
        )
    split_manifest = SplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        source_videos=str(videos_path.resolve()),
        num_workers=num_workers,
        workers=tuple(worker_infos),
    )
    _write_json_atomic(output_dir / SPLIT_MANIFEST_FILENAME, split_manifest.to_dict())
    logger.info(
        "split %d videos across %d workers (source=%s)",
        len(videos),
        num_workers,
        videos_path,
    )
    return split_manifest