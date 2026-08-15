"""Verification of a completed keyframe extraction shard.

The extractor output artifact (``outputs/keyframes_worker_XXX/``) is verified
for internal consistency before it is uploaded to the coordinator. This module
checks the artifact alone (and optionally against the ``videos.json`` that was
assigned to the worker):

- ``manifest.json`` exists, parses, and records the extraction identity
  (extractor / detector / sampler for the ``omnishotcut_adaptive`` extractor,
  valid schema version, counts);
- ``frames.ndjson`` exists, is non-empty, every line parses as a canonical
  :class:`KeyframeTiming`, contains no duplicated ``(video_id, frame_index)``
  key, and is deterministically ordered;
- the recorded ``num_videos`` / ``num_keyframes`` match the actual records;
- when the input ``videos.json`` is provided, every assigned video appears in
  the output, no unexpected video appears, and the totals match;
- ``map-keyframes/<VIDEO_ID>.csv`` exists for every video, parses, and its
  row count matches the video's records.

This is the post-extraction counterpart of
``aic2026.keyframe_extraction_split.verify_video_split`` (which validates the
*input* split). It never requires media files or the detector model; it only
reads the JSON/CSV artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.canonical import KeyframeTiming
from aic2026.entities import parse_keyframe_map
from aic2026.keyframe_extraction import (
    EXTRACTION_MANIFEST_SCHEMA_VERSION,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    MAP_KEYFRAMES_DIRNAME,
    ExtractionManifest,
    load_video_manifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionVerificationIssue:
    """One problem found while verifying a keyframe extraction artifact.

    Fields:
        - section: which artifact component the issue belongs to
          (``artifact``, ``manifest``, ``frames``, ``coverage``, ``maps``).
        - message: human-readable description of the problem.
    """

    section: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"section": self.section, "message": self.message}


@dataclass(frozen=True)
class KeyframeExtractionVerificationReport:
    """Result of verifying one keyframe extraction artifact.

    Fields:
        - output_dir: the artifact directory that was verified.
        - valid: True when no issues were found.
        - issues: every problem found (empty when ``valid``).
        - expected_videos: expected video count (from the input ``videos.json``
          when provided, otherwise the manifest's ``num_videos``).
        - found_videos: distinct videos represented in ``frames.ndjson``.
        - total_keyframes: number of parsed keyframe records.
        - duplicates: number of duplicated ``(video_id, frame_index)`` keys.
        - extractor / detector / sampler: extraction identity from the
          manifest (``None`` when the manifest is invalid or absent).
    """

    output_dir: str
    valid: bool
    issues: tuple[ExtractionVerificationIssue, ...]
    expected_videos: int
    found_videos: int
    total_keyframes: int
    duplicates: int
    extractor: str | None
    detector: str | None
    sampler: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "output_dir": self.output_dir,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "expected_videos": self.expected_videos,
            "found_videos": self.found_videos,
            "total_keyframes": self.total_keyframes,
            "duplicates": self.duplicates,
            "extractor": self.extractor,
            "detector": self.detector,
            "sampler": self.sampler,
        }


def _read_manifest(
    output_dir: Path, issues: list[ExtractionVerificationIssue]
) -> ExtractionManifest | None:
    """Parse ``manifest.json`` or record the failure.

    Returns:
        - The parsed manifest, or ``None`` when missing/invalid.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(
            ExtractionVerificationIssue("artifact", f"manifest.json missing: {manifest_path}")
        )
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            return ExtractionManifest.from_dict(json.load(handle))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        issues.append(
            ExtractionVerificationIssue("manifest", f"invalid manifest.json: {exc}")
        )
        return None


def _check_manifest(
    manifest: ExtractionManifest | None,
    issues: list[ExtractionVerificationIssue],
) -> None:
    """Check manifest-level consistency that does not depend on the frames."""
    if manifest is None:
        return
    if manifest.schema_version != EXTRACTION_MANIFEST_SCHEMA_VERSION:
        issues.append(
            ExtractionVerificationIssue(
                "manifest",
                f"unsupported schema_version {manifest.schema_version!r}; "
                f"expected {EXTRACTION_MANIFEST_SCHEMA_VERSION!r}",
            )
        )
    if manifest.extractor == "omnishotcut_adaptive":
        for name, value in (
            ("extractor", manifest.extractor),
            ("detector", manifest.detector),
            ("sampler", manifest.sampler),
        ):
            if not value or value == "n/a":
                issues.append(
                    ExtractionVerificationIssue(
                        "manifest",
                        f"adaptive extractor requires field {name!r}, got {value!r}",
                    )
                )


def _read_frames(
    output_dir: Path,
    issues: list[ExtractionVerificationIssue],
) -> list[KeyframeTiming]:
    """Parse ``frames.ndjson`` into canonical records.

    Validates JSON syntax and the canonical record schema (nested
    ``keyframe.video_id.value``, ``keyframe.keyframe_ordinal``, ``pts_time``,
    ``fps``, ``frame_index``) line by line.

    Returns:
        - The parsed records (lines that failed are skipped and reported).
    """
    frames_path = output_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        issues.append(ExtractionVerificationIssue("artifact", f"frames.ndjson missing: {frames_path}"))
        return []
    records: list[KeyframeTiming] = []
    with frames_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(KeyframeTiming.from_dict(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                issues.append(
                    ExtractionVerificationIssue(
                        "frames", f"invalid frame record on line {line_no}: {exc}"
                    )
                )
    return records


def _check_duplicates(
    records: list[KeyframeTiming],
    issues: list[ExtractionVerificationIssue],
) -> int:
    """Record duplicated ``(video_id, frame_index)`` keys and count them."""
    counts: dict[tuple[str, int], int] = {}
    for timing in records:
        key = (timing.keyframe.video_id.value, timing.frame_index)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    for key, count in sorted(counts.items()):
        if count > 1:
            issues.append(
                ExtractionVerificationIssue(
                    "frames",
                    f"duplicated keyframe key (video_id={key[0]!r}, frame_index={key[1]}): {count} records",
                )
            )
    return duplicates


def _check_ordering(
    records: list[KeyframeTiming],
    issues: list[ExtractionVerificationIssue],
) -> None:
    """Verify records are deterministically ordered.

    The artifact contract groups each video's records contiguously, in manifest
    video order, with strictly increasing ``keyframe_ordinal`` within a video.
    """
    last_ordinal: dict[str, int] = {}
    seen: set[str] = set()
    last_video: str | None = None
    for timing in records:
        video_id = timing.keyframe.video_id.value
        ordinal = timing.keyframe.keyframe_ordinal
        if video_id in seen:
            if video_id != last_video:
                issues.append(
                    ExtractionVerificationIssue(
                        "frames",
                        f"video {video_id!r} records are not contiguous in frames.ndjson",
                    )
                )
            if ordinal <= last_ordinal[video_id]:
                issues.append(
                    ExtractionVerificationIssue(
                        "frames",
                        f"records not ordered for video {video_id!r}: "
                        f"keyframe_ordinal {ordinal} after {last_ordinal[video_id]}",
                    )
                )
        seen.add(video_id)
        last_ordinal[video_id] = ordinal
        last_video = video_id


def _check_coverage(
    output_dir: Path,
    records: list[KeyframeTiming],
    manifest: ExtractionManifest | None,
    videos_path: Path | None,
    issues: list[ExtractionVerificationIssue],
) -> tuple[int, int]:
    """Check video coverage against the input manifest when available.

    Returns:
        - ``(expected_videos, found_videos)`` for the report.
    """
    found_ids = {timing.keyframe.video_id.value for timing in records}
    expected_videos = manifest.num_videos if manifest is not None else 0
    if videos_path is not None:
        try:
            videos = load_video_manifest(videos_path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            issues.append(
                ExtractionVerificationIssue(
                    "coverage", f"cannot read input videos.json: {exc}"
                )
            )
        else:
            expected_ids = {entry.video_id for entry in videos.videos}
            expected_videos = len(expected_ids)
            missing = sorted(expected_ids - found_ids)
            unexpected = sorted(found_ids - expected_ids)
            for video_id in missing:
                issues.append(
                    ExtractionVerificationIssue(
                        "coverage", f"video {video_id!r} missing from frames.ndjson"
                    )
                )
            for video_id in unexpected:
                issues.append(
                    ExtractionVerificationIssue(
                        "coverage", f"video {video_id!r} not in the input videos.json"
                    )
                )
            if manifest is not None and manifest.num_videos != len(expected_ids):
                issues.append(
                    ExtractionVerificationIssue(
                        "manifest",
                        f"manifest num_videos {manifest.num_videos} != "
                        f"input videos.json total {len(expected_ids)}",
                    )
                )
    if manifest is not None and manifest.num_videos != len(found_ids):
        issues.append(
            ExtractionVerificationIssue(
                "manifest",
                f"manifest num_videos {manifest.num_videos} != "
                f"videos represented in frames.ndjson ({len(found_ids)})",
            )
        )
    return expected_videos, len(found_ids)


def _check_keyframe_count(
    manifest: ExtractionManifest | None,
    records: list[KeyframeTiming],
    issues: list[ExtractionVerificationIssue],
) -> None:
    """Verify ``manifest.num_keyframes`` matches the actual record count."""
    if manifest is None:
        return
    if manifest.num_keyframes != len(records):
        issues.append(
            ExtractionVerificationIssue(
                "manifest",
                f"manifest num_keyframes {manifest.num_keyframes} != "
                f"records in frames.ndjson ({len(records)})",
            )
        )


def _check_maps(
    output_dir: Path,
    records: list[KeyframeTiming],
    issues: list[ExtractionVerificationIssue],
) -> None:
    """Verify ``map-keyframes/`` presence, parseability, and row counts."""
    maps_dir = output_dir / MAP_KEYFRAMES_DIRNAME
    if not maps_dir.is_dir():
        issues.append(
            ExtractionVerificationIssue("artifact", f"map-keyframes directory missing: {maps_dir}")
        )
        return
    per_video: dict[str, list[KeyframeTiming]] = {}
    for timing in records:
        per_video.setdefault(timing.keyframe.video_id.value, []).append(timing)
    for video_id, timings in sorted(per_video.items()):
        csv_path = maps_dir / f"{video_id}.csv"
        if not csv_path.is_file():
            issues.append(
                ExtractionVerificationIssue(
                    "maps", f"map-keyframes file missing for video {video_id!r}: {csv_path}"
                )
            )
            continue
        try:
            rows = parse_keyframe_map(csv_path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            issues.append(
                ExtractionVerificationIssue(
                    "maps", f"invalid map-keyframes CSV for video {video_id!r}: {exc}"
                )
            )
            continue
        if len(rows) != len(timings):
            issues.append(
                ExtractionVerificationIssue(
                    "maps",
                    f"map-keyframes for video {video_id!r} has {len(rows)} rows; "
                    f"frames.ndjson records {len(timings)}",
                )
            )


def verify_keyframe_extraction(
    output_dir: Path | str,
    *,
    videos_path: Path | str | None = None,
) -> KeyframeExtractionVerificationReport:
    """Verify a completed keyframe extraction artifact.

    Checks artifact existence, manifest consistency, canonical frame records
    (unique ``(video_id, frame_index)`` keys, deterministic ordering), video
    coverage against the assigned ``videos.json`` when provided, and per-video
    ``map-keyframes`` files.

    Expected input:
        - output_dir: directory of a keyframe extraction artifact
          (``frames.ndjson`` + ``map-keyframes/`` + ``manifest.json``).
        - videos_path: optional ``videos.json`` assigned to this worker; when
          provided, coverage is verified against its video ids.

    Returns:
        - A :class:`KeyframeExtractionVerificationReport`; the CLI maps
          ``valid`` to its exit code.

    Raises:
        - FileNotFoundError: if the artifact directory or ``manifest.json`` is
          absent.
    """
    output_dir = Path(output_dir)
    issues: list[ExtractionVerificationIssue] = []

    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"keyframe extraction manifest not found: {manifest_path}")

    manifest = _read_manifest(output_dir, issues)
    _check_manifest(manifest, issues)

    records = _read_frames(output_dir, issues)
    if not records and not any(issue.section == "artifact" for issue in issues):
        issues.append(
            ExtractionVerificationIssue(
                "frames", f"{FRAMES_FILENAME} contains no frame records"
            )
        )

    duplicates = _check_duplicates(records, issues)
    _check_ordering(records, issues)
    _check_keyframe_count(manifest, records, issues)
    expected_videos, found_videos = _check_coverage(
        output_dir, records, manifest, Path(videos_path) if videos_path is not None else None, issues
    )
    _check_maps(output_dir, records, issues)

    return KeyframeExtractionVerificationReport(
        output_dir=str(output_dir),
        valid=not issues,
        issues=tuple(issues),
        expected_videos=expected_videos,
        found_videos=found_videos,
        total_keyframes=len(records),
        duplicates=duplicates,
        extractor=manifest.extractor if manifest is not None else None,
        detector=manifest.detector if manifest is not None else None,
        sampler=manifest.sampler if manifest is not None else None,
    )
