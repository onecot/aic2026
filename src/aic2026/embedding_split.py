"""Deterministic, workload-balanced split of a canonical keyframe set.

Given a canonical artifact ``frames.ndjson``, this module partitions its
records into ``N`` worker input files such that:

- each video is kept intact (never split across workers);
- each worker receives an approximately balanced number of frames;
- the result is deterministic (same input + same ``num_workers`` -> identical
  splits);
- no frames are duplicated or dropped (the union of the splits equals the
  original file);
- the order of a video's frames, and the relative order of videos, is
  preserved inside each split.

Balancing is workload-based (number of frames per video), not a plain row
split, so videos whose keyframe sets would be split are instead assigned
whole to the least-loaded worker.

Split directory layout::

    <output-dir>/
        split_manifest.json   versioned split metadata
        worker_000/
            frames.ndjson
        worker_001/
            frames.ndjson
        ...

Each ``worker_<id>/frames.ndjson`` is a canonical artifact suitable as direct
input to ``aic2026.embedding_shard.embed_shard``.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.canonical import KeyframeTiming
from aic2026.embedding_shard import read_frames

#: Schema version of the split manifest.
SPLIT_MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename of the split manifest inside the output directory.
SPLIT_MANIFEST_FILENAME = "split_manifest.json"

#: Filename of a worker input file inside a ``worker_<id>`` directory.
SPLIT_FRAMES_FILENAME = "frames.ndjson"

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


def group_by_video(frames: list[KeyframeTiming]) -> list[tuple[str, list[KeyframeTiming]]]:
    """Group frames by video, preserving the original video/frame order.

    Expected input:
        - frames: canonical keyframe records in row order.

    Returns:
        - A list of ``(video_id, [frames...])`` pairs ordered by each video's
          first appearance; each video's frames stay in their original order.
    """
    ordered: OrderedDict[str, list[KeyframeTiming]] = OrderedDict()
    for frame in frames:
        video_id = frame.keyframe.video_id.value
        ordered.setdefault(video_id, []).append(frame)
    return [(video_id, group) for video_id, group in ordered.items()]


def duplicate_frame_keys(frames: Sequence[KeyframeTiming]) -> list[tuple[str, int]]:
    """Return duplicate keyframe keys ``(video_id, frame_index)``.

    A keyframe is identified by ``(keyframe.video_id.value, frame_index)``
    (the 0-based position in the original encoded video); two records sharing
    that key refer to the same original frame. Splitting such duplicates into
    different workers would later make the merge reject the artifact.

    Expected input:
        - frames: canonical keyframe records in row order.

    Returns:
        - The duplicated keys, one entry each, in first-seen order.
    """
    counts: dict[tuple[str, int], int] = {}
    for frame in frames:
        key = (frame.keyframe.video_id.value, frame.frame_index)
        counts[key] = counts.get(key, 0) + 1
    return [key for key, count in counts.items() if count > 1]


def validate_no_duplicate_frames(frames: Sequence[KeyframeTiming], source: str) -> None:
    """Fail fast when the input contains duplicate keyframes.

    Duplicates are detected by ``(video_id, frame_index)``. This is a hard
    error (not silent deduplication): a split must preserve the experiment
    input exactly, and duplicates would be distributed across workers and
    rejected by the later merge.

    Expected input:
        - frames: canonical keyframe records in row order.
        - source: human-readable input description for the error message.

    Raises:
        - SplitError: if any duplicate keyframe key is present, reporting the
          total duplicate count and a few example keys.
    """
    duplicate_keys = duplicate_frame_keys(frames)
    if not duplicate_keys:
        return
    unique_keys = {(frame.keyframe.video_id.value, frame.frame_index) for frame in frames}
    duplicate_count = len(frames) - len(unique_keys)
    examples = ", ".join(repr(key) for key in duplicate_keys[:5])
    raise SplitError(
        f"{source} contains {duplicate_count} duplicate keyframes "
        f"(keyed by (video_id, frame_index)); duplicate keys e.g. {examples}. "
        "Splitting would spread duplicates across workers and the merge would "
        "reject them; create a deduplicated frames artifact first instead of "
        "splitting the original."
    )


def assign_frames(frames: list[KeyframeTiming], num_workers: int) -> list[list[KeyframeTiming]]:
    """Assign whole videos to workers, balancing frame counts.

    Uses a greedy longest-processing-time assignment: videos are considered in
    descending frame count (ties broken by video id), and each video is placed
    into the worker with the fewest assigned frames (ties broken by the lowest
    worker id). Within each worker, videos are re-ordered by their original
    appearance order and frames by their original order, so the split is fully
    deterministic and preserves input order as much as possible.

    Expected input:
        - frames: canonical keyframe records in row order.
        - num_workers: number of worker splits (>= 1).

    Returns:
        - A list of ``num_workers`` frame lists, index ``i`` being the frames
          for worker ``i``.

    Raises:
        - ValueError: if ``num_workers`` is not a positive integer.
    """
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
        raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")
    if not frames:
        return [[] for _ in range(num_workers)]

    ordered_videos = group_by_video(frames)

    # Assign videos greedily by workload (descending), tie-breaking on id.
    assignment_order = sorted(
        ordered_videos, key=lambda v: (-len(v[1]), v[0])
    )
    loads = [0] * num_workers
    assigned: list[list[str]] = [[] for _ in range(num_workers)]
    for video_id, group in assignment_order:
        target = min(range(num_workers), key=lambda w: (loads[w], w))
        assigned[target].append(video_id)
        loads[target] += len(group)

    # Rebuild per-worker frame lists in original appearance order.
    video_to_frames = {video_id: group for video_id, group in ordered_videos}
    per_worker: list[list[KeyframeTiming]] = []
    for worker_id in range(num_workers):
        frames_for_worker: list[KeyframeTiming] = []
        for video_id in assigned[worker_id]:
            frames_for_worker.extend(video_to_frames[video_id])
        per_worker.append(frames_for_worker)
    return per_worker


def _write_worker_frames(directory: Path, frames: list[KeyframeTiming]) -> None:
    """Write the canonical frames.ndjson for one worker (atomic)."""
    path = directory / SPLIT_FRAMES_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_dict()) + "\n")
    tmp.replace(path)


@dataclass(frozen=True)
class SplitWorkerInfo:
    """Metadata for one worker split.

    Fields:
        - worker_id: worker machine identifier.
        - num_frames: number of keyframe records assigned to the worker.
        - num_videos: number of videos assigned to the worker.
        - video_ids: sorted video ids assigned to the worker.
    """

    worker_id: int
    num_frames: int
    num_videos: int
    video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.worker_id, int) and self.worker_id >= 0, "worker_id must be >= 0")
        _require(isinstance(self.num_frames, int) and self.num_frames >= 0, "num_frames must be >= 0")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(
            all(isinstance(vid, str) and vid != "" for vid in self.video_ids),
            "video_ids must be non-empty strings",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "worker_id": self.worker_id,
            "num_frames": self.num_frames,
            "num_videos": self.num_videos,
            "video_ids": list(self.video_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitWorkerInfo":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            worker_id=int(data["worker_id"]),
            num_frames=int(data["num_frames"]),
            num_videos=int(data["num_videos"]),
            video_ids=tuple(data["video_ids"]),
        )


@dataclass(frozen=True)
class SplitManifest:
    """Versioned metadata describing one split of a canonical keyframe set.

    Fields:
        - schema_version: split schema version
          (``SPLIT_MANIFEST_SCHEMA_VERSION``).
        - source_frames: the input ``frames.ndjson`` path that was split.
        - num_workers: total number of worker splits.
        - workers: per-worker metadata in worker order.
    """

    schema_version: str
    source_frames: str
    num_workers: int
    workers: tuple[SplitWorkerInfo, ...]

    def __post_init__(self) -> None:
        _require(
            self.schema_version == SPLIT_MANIFEST_SCHEMA_VERSION,
            f"unsupported split schema version {self.schema_version!r}",
        )
        _require(isinstance(self.source_frames, str) and self.source_frames != "", "source_frames required")
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
            "source_frames": self.source_frames,
            "num_workers": self.num_workers,
            "workers": [w.to_dict() for w in self.workers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            source_frames=data["source_frames"],
            num_workers=int(data["num_workers"]),
            workers=tuple(SplitWorkerInfo.from_dict(w) for w in data["workers"]),
        )


def _validate_frames(
    per_worker: list[list[KeyframeTiming]], original: list[KeyframeTiming]
) -> None:
    """Check that the splits reproduce the original set exactly.

    Asserts that total frames across splits equals the original count and that
    every keyframe appears exactly once (no duplicates, no missing frames).

    Raises:
        - SplitError: if a frame is duplicated, missing, or the counts differ.
    """
    total = sum(len(w) for w in per_worker)
    if total != len(original):
        raise SplitError(
            f"split total frames {total} != original {len(original)}"
        )
    seen: set[tuple[str, int]] = set()
    for worker_frames in per_worker:
        for frame in worker_frames:
            key = (frame.keyframe.video_id.value, frame.keyframe.keyframe_ordinal)
            if key in seen:
                raise SplitError(f"duplicate frame {key} across splits")
            seen.add(key)
    original_keys = {
        (frame.keyframe.video_id.value, frame.keyframe.keyframe_ordinal) for frame in original
    }
    if seen != original_keys:
        raise SplitError("split frames do not match the original set exactly")


def split_frames(
    frames_path: Path | str,
    num_workers: int,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> SplitManifest:
    """Split a canonical ``frames.ndjson`` into ``num_workers`` input files.

    Expected input:
        - frames_path: path to a canonical artifact ``frames.ndjson``.
        - num_workers: number of worker splits to produce (>= 1).
        - output_dir: target directory for the split (created if absent).
        - overwrite: if True, replace an existing split; otherwise raise
          ``SplitError`` when a split already exists there.

    Returns:
        - A :class:`SplitManifest` describing the produced splits.

    Raises:
        - FileNotFoundError: if ``frames_path`` does not exist.
        - json.JSONDecodeError / ValueError: if a frame line is malformed.
        - ValueError: if ``num_workers`` is invalid.
        - SplitError: if the split already exists and ``overwrite`` is False,
          or if the input contains duplicate keyframes (keyed by
          ``(video_id, frame_index)``).
    """
    frames_path = Path(frames_path)
    output_dir = Path(output_dir)
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
        raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SplitError(
            f"split output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )

    frames = read_frames(frames_path)
    validate_no_duplicate_frames(frames, f"input frames file {frames_path}")
    per_worker = assign_frames(frames, num_workers)
    _validate_frames(per_worker, frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    for worker_id, worker_frames in enumerate(per_worker):
        worker_dir = output_dir / worker_dir_name(worker_id)
        worker_dir.mkdir(parents=True, exist_ok=True)
        _write_worker_frames(worker_dir, worker_frames)

    worker_infos: list[SplitWorkerInfo] = []
    for worker_id, worker_frames in enumerate(per_worker):
        video_ids = sorted({frame.keyframe.video_id.value for frame in worker_frames})
        worker_infos.append(
            SplitWorkerInfo(
                worker_id=worker_id,
                num_frames=len(worker_frames),
                num_videos=len(video_ids),
                video_ids=tuple(video_ids),
            )
        )
    manifest = SplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        source_frames=str(frames_path.resolve()),
        num_workers=num_workers,
        workers=tuple(worker_infos),
    )
    _write_json_atomic(output_dir / SPLIT_MANIFEST_FILENAME, manifest.to_dict())
    logger.info(
        "split %d frames across %d workers (source=%s)",
        len(frames),
        num_workers,
        frames_path,
    )
    return manifest