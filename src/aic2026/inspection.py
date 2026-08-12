"""Read-only dataset inspection and validation.

This module discovers the raw BTC-provided dataset layout, cross-checks
the per-video collections, and produces a concise report. It performs
no writes, so it is safe to run repeatedly.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aic2026.contracts import (
    COLLECTION_NAMES,
    KEYFRAME_FILENAME_PATTERN,
    OBJECT_FILENAME_PATTERN,
    video_id_from_collection_name,
)
from aic2026.entities import (
    parse_keyframe_map,
    parse_media_info,
    validate_object_payload,
)

logger = logging.getLogger(__name__)

_COLLECTION_SUFFIXES = {
    "clip-features-32": ".npy",
    "map-keyframes": ".csv",
    "media-info": ".json",
}


@dataclass
class VideoStatus:
    """Per-video snapshot used for validation.

    Fields:
        - video_id: the ``L*_V*`` id.
        - n_keyframes: number of keyframe files found.
        - n_objects: number of object JSON files found.
        - map_keyframe_indices: sorted keyframe indices from the map CSV.
        - object_indices: set of keyframe indices with object files.
        - keyframe_indices: set of keyframe indices with image files.
        - clip_rows: number of rows in the CLIP feature matrix.
        - clip_dim: embedding dimensionality (columns).
        - clip_dtype: dtype name of the CLIP matrix.
        - fps: frame rate from the map CSV (first row), if any.
        - media_problems: list of schema problems in media-info.
        - object_problems: list of schema problems across object files.
        - missing_files: list of expected-but-absent file descriptions.
    """

    video_id: str
    n_keyframes: int = 0
    n_objects: int = 0
    map_keyframe_indices: list[int] = field(default_factory=list)
    object_indices: set[int] = field(default_factory=set)
    keyframe_indices: set[int] = field(default_factory=set)
    clip_rows: int | None = None
    clip_dim: int | None = None
    clip_dtype: str | None = None
    fps: float | None = None
    media_problems: list[str] = field(default_factory=list)
    object_problems: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)


def discover_video_ids(raw_root: Path) -> list[str]:
    """Return the union of video ids found across all collections.

    Expected input:
        - raw_root: path to the raw dataset root.

    Returns:
        - Sorted list of unique video ids.

    Raises:
        - FileNotFoundError: if ``raw_root`` does not exist.
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw dataset root not found: {raw_root}")
    ids: set[str] = set()
    for name in COLLECTION_NAMES:
        ids |= collection_video_ids(raw_root, name)
    return sorted(ids)


def collection_video_ids(raw_root: Path, collection: str) -> set[str]:
    """Return the video ids present in one collection directory.

    Expected input:
        - raw_root: path to the raw dataset root.
        - collection: one of `COLLECTION_NAMES`.

    Returns:
        - Set of video ids found directly under that collection.

    Raises:
        - FileNotFoundError: if the collection directory does not exist.
    """
    collection_dir = raw_root / collection
    if not collection_dir.is_dir():
        raise FileNotFoundError(f"collection not found: {collection_dir}")
    ids: set[str] = set()
    for entry in collection_dir.iterdir():
        vid = video_id_from_collection_name(entry.name)
        if vid:
            ids.add(vid)
    return ids


def _iter_keyframe_indices(keyframes_dir: Path) -> set[int]:
    """Extract 1-based keyframe indices from keyframe image filenames."""
    indices: set[int] = set()
    for entry in keyframes_dir.iterdir():
        match = KEYFRAME_FILENAME_PATTERN.match(entry.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def _iter_object_indices(objects_dir: Path) -> set[int]:
    """Extract 1-based keyframe indices from object JSON filenames."""
    indices: set[int] = set()
    for entry in objects_dir.iterdir():
        match = OBJECT_FILENAME_PATTERN.match(entry.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def inspect_video(raw_root: Path, video_id: str, sample_objects: int) -> VideoStatus:
    """Inspect one video across all collections (read-only).

    Expected input:
        - raw_root: path to the raw dataset root.
        - video_id: the ``L*_V*`` id to inspect.
        - sample_objects: number of object JSON files to schema-check per
          video (0 to skip sampling).

    Returns:
        - A :class:`VideoStatus` describing the video's records.

    Raises:
        - No exceptions: missing pieces are recorded in the status.
    """
    status = VideoStatus(video_id=video_id)
    # Keyframe images
    kf_dir = raw_root / "keyframes" / video_id
    if kf_dir.is_dir():
        status.keyframe_indices = _iter_keyframe_indices(kf_dir)
        status.n_keyframes = len(status.keyframe_indices)
    else:
        status.missing_files.append(f"keyframes/{video_id}/")
    # Object JSONs
    objects_dir = raw_root / "objects" / video_id
    if objects_dir.is_dir():
        status.object_indices = _iter_object_indices(objects_dir)
        status.n_objects = len(status.object_indices)
        if sample_objects:
            for path in sorted(objects_dir.iterdir()):
                if OBJECT_FILENAME_PATTERN.match(path.name):
                    try:
                        with path.open(encoding="utf-8") as handle:
                            import json

                            payload = json.load(handle)
                        problems = validate_object_payload(payload)
                    except (OSError, ValueError) as exc:
                        problems = [f"unreadable: {exc}"]
                    status.object_problems.extend(
                        f"{path.name}: {p}" for p in problems
                    )
                    sample_objects -= 1
                    if sample_objects <= 0:
                        break
    else:
        status.missing_files.append(f"objects/{video_id}/")
    # Map-keyframes CSV
    map_path = raw_root / "map-keyframes" / f"{video_id}.csv"
    if map_path.is_file():
        try:
            rows = parse_keyframe_map(map_path)
            status.map_keyframe_indices = [row.keyframe_no for row in rows]
            fps_values = {row.fps for row in rows}
            status.fps = (
                sorted(fps_values)[0]
                if len(fps_values) == 1
                else (sorted(fps_values)[0] if fps_values else None)
            )
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            status.missing_files.append(f"map-keyframes/{video_id}.csv parse error: {exc}")
    else:
        status.missing_files.append(f"map-keyframes/{video_id}.csv")
    # CLIP features
    npy_path = raw_root / "clip-features-32" / f"{video_id}.npy"
    if npy_path.is_file():
        try:
            matrix = np.load(npy_path, mmap_mode="r")
            status.clip_rows = int(matrix.shape[0])
            status.clip_dim = int(matrix.shape[1])
            status.clip_dtype = str(matrix.dtype)
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            status.missing_files.append(f"clip-features-32/{video_id}.npy unreadable: {exc}")
    else:
        status.missing_files.append(f"clip-features-32/{video_id}.npy")
    # Media info
    media_path = raw_root / "media-info" / f"{video_id}.json"
    if media_path.is_file():
        try:
            parse_media_info(media_path)
        except Exception as exc:  # noqa: BLE001 - surface as a problem
            status.media_problems.append(f"media-info/{video_id}.json: {exc}")
    else:
        status.missing_files.append(f"media-info/{video_id}.json")
    return status


def validate_video(status: VideoStatus) -> list[str]:
    """Cross-check one video's records for consistency.

    Expected input:
        - status: a :class:`VideoStatus` produced by :func:`inspect_video`.

    Returns:
        - List of human-readable consistency problems; empty when the
          video is internally consistent.

    Raises:
        - No exceptions.
    """
    problems: list[str] = []
    vid = status.video_id
    kf = status.keyframe_indices
    obj = status.object_indices
    if kf != obj:
        problems.append(
            f"keyframe/object index mismatch for {vid}: "
            f"{len(kf)} keyframes vs {len(obj)} objects"
        )
    n_map = len(status.map_keyframe_indices)
    if kf and n_map != len(kf):
        problems.append(
            f"map-keyframes rows ({n_map}) != keyframe files ({len(kf)}) for {vid}"
        )
    if status.clip_rows is not None and kf and status.clip_rows != len(kf):
        problems.append(
            f"CLIP rows ({status.clip_rows}) != keyframe files ({len(kf)}) for {vid}"
        )
    # Map indices should match keyframe indices exactly
    if kf and status.map_keyframe_indices:
        map_set = set(status.map_keyframe_indices)
        if map_set != kf:
            problems.append(
                f"map keyframe indices disagree with keyframe files for {vid}"
            )
    if status.clip_dim is not None and status.clip_dim != 512:
        problems.append(f"unexpected CLIP embedding dim {status.clip_dim} for {vid}")
    # Media info problems already recorded
    if status.media_problems:
        problems.extend(status.media_problems)
    return problems


@dataclass
class DatasetReport:
    """Structured results of a dataset inspection run.

    Fields:
        - raw_root: the raw dataset root that was inspected.
        - video_ids: sorted list of all discovered video ids.
        - collection_counts: mapping collection name -> video id count.
        - total_keyframes: total number of keyframe files.
        - total_objects: total number of object JSON files.
        - object_detection_counts: Counter mapping detection count per
          object frame -> number of frames (sampled).
        - embedding_shapes: Counter mapping `(rows, cols)` -> count.
        - embedding_dtypes: Counter mapping dtype name -> count.
        - fps_histogram: Counter mapping fps -> number of videos.
        - map_rows_total: total number of map-keyframes CSV rows.
        - inconsistent_videos: dict video id -> list of problems.
        - videos_with_media_issues: dict video id -> list of problems.
        - missing_files_total: aggregate count of expected-but-absent
          files across all videos.
        - object_payload_problems: dict video id -> list of sampled
          object payload problems.
        - observations: list of human-readable schema observations.
    """

    raw_root: Path
    video_ids: list[str] = field(default_factory=list)
    collection_counts: dict[str, int] = field(default_factory=dict)
    total_keyframes: int = 0
    total_objects: int = 0
    object_detection_counts: Counter = field(default_factory=Counter)
    embedding_shapes: Counter = field(default_factory=Counter)
    embedding_dtypes: Counter = field(default_factory=Counter)
    fps_histogram: Counter = field(default_factory=Counter)
    map_rows_total: int = 0
    inconsistent_videos: dict[str, list[str]] = field(default_factory=dict)
    videos_with_media_issues: dict[str, list[str]] = field(default_factory=dict)
    missing_files_total: int = 0
    object_payload_problems: dict[str, list[str]] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)


def inspect_dataset(
    raw_root: Path,
    max_videos: int | None = None,
    sample_objects: int = 3,
    video_ids_filter: set[str] | None = None,
) -> DatasetReport:
    """Inspect the raw dataset and return a structured report.

    Expected input:
        - raw_root: path to the raw dataset root.
        - max_videos: optional limit on the number of videos inspected.
        - sample_objects: number of object JSON files to schema-check per
          video (0 to skip).
        - video_ids_filter: optional set of video ids to restrict to.

    Returns:
        - A :class:`DatasetReport` with counts, consistency findings, and
          schema observations.

    Raises:
        - FileNotFoundError: if ``raw_root`` does not exist.
    """
    report = DatasetReport(raw_root=raw_root)

    for name in COLLECTION_NAMES:
        try:
            ids = collection_video_ids(raw_root, name)
        except FileNotFoundError:
            report.observations.append(f"collection missing: {name}")
            ids = set()
        report.collection_counts[name] = len(ids)

    video_ids = discover_video_ids(raw_root)
    if video_ids_filter is not None:
        video_ids = [vid for vid in video_ids if vid in video_ids_filter]
    if max_videos is not None:
        video_ids = video_ids[:max_videos]
    report.video_ids = video_ids

    all_kf_files = 0
    all_obj_files = 0
    all_map_rows = 0
    all_missing = 0
    all_fps = Counter()
    all_shapes = Counter()
    all_dtypes = Counter()

    for video_id in video_ids:
        status = inspect_video(raw_root, video_id, sample_objects)
        all_kf_files += status.n_keyframes
        all_obj_files += status.n_objects
        all_map_rows += len(status.map_keyframe_indices)
        all_missing += len(status.missing_files)
        if status.fps is not None:
            all_fps[status.fps] += 1
        if status.clip_dtype is not None:
            all_dtypes[status.clip_dtype] += 1
        if status.clip_rows is not None and status.clip_dim is not None:
            all_shapes[(status.clip_rows, status.clip_dim)] += 1

        problems = validate_video(status)
        if problems:
            report.inconsistent_videos[video_id] = problems
        if status.media_problems:
            report.videos_with_media_issues[video_id] = status.media_problems
        if status.object_problems:
            report.object_payload_problems[video_id] = status.object_problems

    report.total_keyframes = all_kf_files
    report.total_objects = all_obj_files
    report.map_rows_total = all_map_rows
    report.missing_files_total = all_missing
    report.fps_histogram = all_fps
    report.embedding_shapes = all_shapes
    report.embedding_dtypes = all_dtypes

    if video_ids:
        report.observations.append(
            f"inspected {len(video_ids)} unique video(s)"
        )
    for name, count in report.collection_counts.items():
        report.observations.append(f"{name}: {count} video entries")
    if all_shapes:
        shapes_desc = ", ".join(f"{shape}: {count}" for shape, count in all_shapes.items())
        report.observations.append(f"CLIP matrix shapes: {shapes_desc}")
    if all_dtypes:
        dtypes_desc = ", ".join(f"{dtype}: {count}" for dtype, count in all_dtypes.items())
        report.observations.append(f"CLIP matrix dtypes: {dtypes_desc}")
    if all_fps:
        fps_desc = ", ".join(f"{k}: {v}" for k, v in all_fps.items())
        report.observations.append(f"fps distribution: {fps_desc}")
    if report.inconsistent_videos:
        report.observations.append(
            f"{len(report.inconsistent_videos)} video(s) have consistency problems"
        )
    if report.missing_files_total:
        report.observations.append(
            f"{report.missing_files_total} expected files are missing or unreadable"
        )

    # Embedding dimension consistency
    if all_shapes:
        dim_values = {shape[1] for shape in all_shapes}
        if dim_values == {512}:
            report.observations.append("embedding dimensionality: 512 (uniform)")
        else:
            report.observations.append(f"embedding dimensionality varies: {sorted(dim_values)}")

    # Detect counting parity between map rows and keyframes
    if report.map_rows_total and report.total_keyframes:
        if report.map_rows_total != report.total_keyframes:
            report.observations.append(
                f"map-keyframes total rows ({report.map_rows_total}) != "
                f"keyframe files ({report.total_keyframes})"
            )

    return report