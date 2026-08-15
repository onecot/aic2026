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


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


@dataclass(frozen=True)
class SplitVerificationIssue:
    """One problem found while verifying a split.

    Fields:
        - worker_id: worker identifier (``None`` when the issue is split-wide).
        - message: human-readable description of the problem.
    """

    worker_id: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"worker_id": self.worker_id, "message": self.message}


@dataclass(frozen=True)
class SplitWorkerVerification:
    """Per-worker video/frame counts from one verified split.

    Fields:
        - worker_id: worker identifier.
        - num_videos: number of videos in the worker manifest.
        - num_frames: sum of ``num_frames`` (1 per video when unknown).
        - manifest_found: whether ``worker_<id>/videos.json`` was found.
    """

    worker_id: int
    num_videos: int
    num_frames: int
    manifest_found: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "worker_id": self.worker_id,
            "num_videos": self.num_videos,
            "num_frames": self.num_frames,
            "manifest_found": self.manifest_found,
        }


@dataclass(frozen=True)
class SplitVerificationReport:
    """Result of verifying one split directory.

    Fields:
        - split_dir: the split directory that was verified.
        - valid: True when no issues were found.
        - issues: every problem found (empty when ``valid``).
        - workers: per-worker counts in worker order.
        - total_videos: total videos across all worker manifests.
        - unique_videos: distinct video ids across all worker manifests.
        - duplicates: video ids appearing in more than one worker manifest.
        - total_frames: sum of ``num_frames`` (1 per video when unknown).
    """

    split_dir: str
    valid: bool
    issues: tuple[SplitVerificationIssue, ...]
    workers: tuple[SplitWorkerVerification, ...]
    total_videos: int
    unique_videos: int
    duplicates: int
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "split_dir": self.split_dir,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "workers": [worker.to_dict() for worker in self.workers],
            "total_videos": self.total_videos,
            "unique_videos": self.unique_videos,
            "duplicates": self.duplicates,
            "total_frames": self.total_frames,
        }


def _frame_weight(entry: VideoEntry) -> int:
    """Return the split weight of a video entry (``num_frames`` or 1)."""
    return entry.num_frames if entry.num_frames is not None else 1


def verify_video_split(split_dir: Path | str) -> SplitVerificationReport:
    """Verify that a split directory is internally consistent.

    Checks:
        - ``split_manifest.json`` exists and parses;
        - every ``worker_<id>/videos.json`` recorded in the manifest exists
          and parses as a valid :class:`VideoManifest`;
        - the total videos across worker manifests equals the total recorded
          in ``split_manifest.json``;
        - no ``video_id`` is duplicated across workers;
        - every ``video_id`` recorded in ``split_manifest.json`` appears in
          exactly one worker manifest;
        - each worker's recorded frame count matches its manifest (videos
          without ``num_frames`` count as 1 frame, matching the split weight).

    This only verifies internal consistency of the split directory: the source
    ``videos.json`` (recorded as ``source_videos``) is not required to exist.

    Expected input:
        - split_dir: the split directory containing ``split_manifest.json``.

    Returns:
        - A :class:`SplitVerificationReport`; the CLI maps ``valid`` to its
          exit code.

    Raises:
        - FileNotFoundError: if ``split_manifest.json`` is absent.
    """
    split_dir = Path(split_dir)
    issues: list[SplitVerificationIssue] = []

    split_manifest_path = split_dir / SPLIT_MANIFEST_FILENAME
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_path}")

    try:
        data = _read_json(split_manifest_path)
        split_manifest = SplitManifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        issues.append(SplitVerificationIssue(None, f"invalid split_manifest.json: {exc}"))
        return SplitVerificationReport(
            split_dir=str(split_dir),
            valid=False,
            issues=tuple(issues),
            workers=(),
            total_videos=0,
            unique_videos=0,
            duplicates=0,
            total_frames=0,
        )

    seen_videos: dict[str, int] = {}
    total_videos = 0
    total_frames = 0
    manifest_videos = 0
    manifest_frames = 0
    workers: list[SplitWorkerVerification] = []

    for worker in split_manifest.workers:
        worker_dir = split_dir / worker_dir_name(worker.worker_id)
        worker_manifest_path = worker_dir / VIDEOS_MANIFEST_FILENAME
        if not worker_manifest_path.is_file():
            workers.append(
                SplitWorkerVerification(
                    worker_id=worker.worker_id, num_videos=0, num_frames=0, manifest_found=False
                )
            )
            issues.append(
                SplitVerificationIssue(
                    worker.worker_id, f"worker manifest missing: {worker_manifest_path}"
                )
            )
            continue
        try:
            worker_manifest = load_video_manifest(worker_manifest_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            workers.append(
                SplitWorkerVerification(
                    worker_id=worker.worker_id, num_videos=0, num_frames=0, manifest_found=False
                )
            )
            issues.append(
                SplitVerificationIssue(
                    worker.worker_id, f"invalid worker manifest: {worker_manifest_path} ({exc})"
                )
            )
            continue

        for entry in worker_manifest.videos:
            seen_videos[entry.video_id] = seen_videos.get(entry.video_id, 0) + 1
        total_videos += len(worker_manifest.videos)
        total_frames += sum(_frame_weight(entry) for entry in worker_manifest.videos)
        workers.append(
            SplitWorkerVerification(
                worker_id=worker.worker_id,
                num_videos=len(worker_manifest.videos),
                num_frames=sum(_frame_weight(entry) for entry in worker_manifest.videos),
                manifest_found=True,
            )
        )

        actual_ids = {entry.video_id for entry in worker_manifest.videos}
        recorded_ids = set(worker.video_ids)
        if actual_ids != recorded_ids:
            missing = sorted(recorded_ids - actual_ids)
            extra = sorted(actual_ids - recorded_ids)
            detail: list[str] = []
            if missing:
                detail.append(f"missing from manifest: {missing}")
            if extra:
                detail.append(f"unexpected in manifest: {extra}")
            issues.append(
                SplitVerificationIssue(
                    worker.worker_id,
                    f"videos do not match split_manifest.json ({'; '.join(detail)})",
                )
            )
        if len(worker_manifest.videos) != worker.num_videos:
            issues.append(
                SplitVerificationIssue(
                    worker.worker_id,
                    f"worker has {len(worker_manifest.videos)} videos; "
                    f"split_manifest.json records {worker.num_videos}",
                )
            )
        worker_frames = sum(_frame_weight(entry) for entry in worker_manifest.videos)
        if worker_frames != worker.estimated_frames:
            issues.append(
                SplitVerificationIssue(
                    worker.worker_id,
                    f"worker has {worker_frames} frames; "
                    f"split_manifest.json records {worker.estimated_frames}",
                )
            )
        manifest_videos += worker.num_videos
        manifest_frames += worker.estimated_frames

    duplicates = sum(count - 1 for count in seen_videos.values() if count > 1)
    for video_id, count in sorted(seen_videos.items()):
        if count > 1:
            issues.append(
                SplitVerificationIssue(
                    None,
                    f"video {video_id!r} appears in {count} worker manifests",
                )
            )

    if total_videos != manifest_videos:
        issues.append(
            SplitVerificationIssue(
                None,
                f"total videos {total_videos} != split_manifest.json total {manifest_videos}",
            )
        )
    if total_frames != manifest_frames:
        issues.append(
            SplitVerificationIssue(
                None,
                f"total frames {total_frames} != split_manifest.json total {manifest_frames}",
            )
        )

    return SplitVerificationReport(
        split_dir=str(split_dir),
        valid=not issues,
        issues=tuple(issues),
        workers=tuple(workers),
        total_videos=total_videos,
        unique_videos=len(seen_videos),
        duplicates=duplicates,
        total_frames=total_frames,
    )