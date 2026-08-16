"""Verification of a completed keyframe-image materialization artifact.

Materialization (``aic2026.keyframe_materialize.materialize_keyframes``)
produces an ``adaptive_sbd_v1`` artifact:

.. code-block:: text

    <output-dir>/
        images/<video_id>/<ordinal:06d>.jpg
        frames.ndjson
        manifest.json

This module verifies the artifact for internal consistency:

- ``manifest.json`` exists, parses, records ``artifact_id == adaptive_sbd_v1``,
  a valid schema version, and a ``num_keyframes`` that matches the records;
- ``frames.ndjson`` exists, is non-empty, every line parses as a canonical
  :class:`KeyframeTiming` (nested ``keyframe.video_id.value`` and
  ``keyframe.keyframe_ordinal`` required), contains no duplicated
  ``(video_id, keyframe_ordinal)`` key, and is deterministically ordered;
- every frame record has exactly one image at
  ``images/<video_id>/<ordinal:06d>.jpg``, no image is missing, no unexpected
  image exists, and the total image count matches the record count.

It never requires media files, ffmpeg, or a model; it only reads the JSON /
image artifact. It is the standalone, delivered-artifact counterpart of the
internal validation ``materialize_keyframes`` runs on the staging directory
before moving it into place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.canonical import KeyframeTiming
from aic2026.keyframe_extraction import FRAMES_FILENAME, MANIFEST_FILENAME
from aic2026.keyframe_materialize import (
    ARTIFACT_ID,
    IMAGE_EXTENSION,
    IMAGES_DIRNAME,
    MATERIALIZE_MANIFEST_SCHEMA_VERSION,
    MaterializeManifest,
    image_filename,
    materialized_image_path,
)

logger = logging.getLogger(__name__)

#: Manifest identity when ``manifest.json`` is absent or cannot be parsed.
ARTIFACT_ID_UNKNOWN = "unknown"


@dataclass(frozen=True)
class MaterializationVerificationIssue:
    """One problem found while verifying a materialized keyframe artifact.

    Fields:
        - section: which artifact component the issue belongs to
          (``artifact``, ``manifest``, ``frames``, ``images``).
        - message: human-readable description of the problem.
    """

    section: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"section": self.section, "message": self.message}


@dataclass(frozen=True)
class KeyframeMaterializationVerificationReport:
    """Result of verifying one materialized keyframe artifact.

    Fields:
        - output_dir: the artifact directory that was verified.
        - artifact_id: ``adaptive_sbd_v1`` (or :data:`ARTIFACT_ID_UNKNOWN`
          when the manifest is invalid or absent).
        - valid: True when no issues were found.
        - issues: every problem found (empty when ``valid``).
        - records: number of parsed ``frames.ndjson`` records.
        - images: number of image files found under ``images/``.
        - missing_images: number of records without a corresponding image.
        - unexpected_images: number of image files not referenced by a record.
        - duplicates: number of duplicated ``(video_id, keyframe_ordinal)``
          keys.
        - num_videos: distinct videos represented in ``frames.ndjson``.
    """

    output_dir: str
    artifact_id: str
    valid: bool
    issues: tuple[MaterializationVerificationIssue, ...]
    records: int
    images: int
    missing_images: int
    unexpected_images: int
    duplicates: int
    num_videos: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "output_dir": self.output_dir,
            "artifact_id": self.artifact_id,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "records": self.records,
            "images": self.images,
            "missing_images": self.missing_images,
            "unexpected_images": self.unexpected_images,
            "duplicates": self.duplicates,
            "num_videos": self.num_videos,
        }


def _read_manifest(
    output_dir: Path,
    issues: list[MaterializationVerificationIssue],
) -> tuple[MaterializeManifest | None, int | None, str]:
    """Parse ``manifest.json`` or record the failure.

    Returns:
        - ``(manifest, num_images, artifact_id)``: the parsed manifest (or
          ``None`` when missing/invalid), the optional recorded ``num_images``
          (or ``None`` when absent/unparsable), and the stored ``artifact_id``
          value (or :data:`ARTIFACT_ID_UNKNOWN` when absent/unparseable).
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(
            MaterializationVerificationIssue(
                "artifact", f"manifest.json missing: {manifest_path}"
            )
        )
        return None, None, ARTIFACT_ID_UNKNOWN
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        issues.append(
            MaterializationVerificationIssue("manifest", f"invalid manifest.json: {exc}")
        )
        return None, None, ARTIFACT_ID_UNKNOWN

    artifact_id = (
        str(data.get("artifact_id") or ARTIFACT_ID_UNKNOWN)
        if isinstance(data, dict)
        else ARTIFACT_ID_UNKNOWN
    )
    num_images: int | None = None
    if isinstance(data, dict) and "num_images" in data:
        raw_num_images = data["num_images"]
        if not isinstance(raw_num_images, int) or isinstance(raw_num_images, bool) or raw_num_images < 0:
            issues.append(
                MaterializationVerificationIssue(
                    "manifest", f"invalid num_images {raw_num_images!r}"
                )
            )
        else:
            num_images = raw_num_images

    if not isinstance(data, dict) or data.get("artifact_id") != ARTIFACT_ID:
        issues.append(
            MaterializationVerificationIssue(
                "manifest",
                f"artifact_id {artifact_id!r} != expected {ARTIFACT_ID!r}",
            )
        )

    if not isinstance(data, dict) or data.get("schema_version") != MATERIALIZE_MANIFEST_SCHEMA_VERSION:
        issues.append(
            MaterializationVerificationIssue(
                "manifest",
                f"unsupported schema_version "
                f"{data.get('schema_version') if isinstance(data, dict) else None!r}; "
                f"expected {MATERIALIZE_MANIFEST_SCHEMA_VERSION!r}",
            )
        )

    if not isinstance(data, dict):
        issues.append(
            MaterializationVerificationIssue("manifest", "manifest.json is not a JSON object")
        )
        return None, num_images, artifact_id

    try:
        manifest = MaterializeManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(
            MaterializationVerificationIssue("manifest", f"invalid manifest.json: {exc}")
        )
        return None, num_images, artifact_id
    return manifest, num_images, artifact_id


def _read_frames(
    output_dir: Path,
    issues: list[MaterializationVerificationIssue],
) -> list[KeyframeTiming]:
    """Parse ``frames.ndjson`` into canonical records.

    Validates JSON syntax and the canonical record schema (nested
    ``keyframe.video_id.value`` and ``keyframe.keyframe_ordinal``) line by
    line.

    Returns:
        - The parsed records (lines that failed are skipped and reported).
    """
    frames_path = output_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        issues.append(
            MaterializationVerificationIssue("artifact", f"frames.ndjson missing: {frames_path}")
        )
        return []
    records: list[KeyframeTiming] = []
    with frames_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(KeyframeTiming.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                issues.append(
                    MaterializationVerificationIssue(
                        "frames", f"invalid frame record on line {line_no}: {exc}"
                    )
                )
    if not records and not any(issue.section == "artifact" for issue in issues):
        issues.append(
            MaterializationVerificationIssue(
                "frames", f"{FRAMES_FILENAME} contains no frame records"
            )
        )
    return records


def _check_duplicates(
    records: list[KeyframeTiming],
    issues: list[MaterializationVerificationIssue],
) -> int:
    """Record duplicated ``(video_id, keyframe_ordinal)`` keys and count them."""
    counts: dict[tuple[str, int], int] = {}
    for timing in records:
        key = (timing.keyframe.video_id.value, timing.keyframe.keyframe_ordinal)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    for key, count in sorted(counts.items()):
        if count > 1:
            issues.append(
                MaterializationVerificationIssue(
                    "frames",
                    f"duplicated keyframe key (video_id={key[0]!r}, "
                    f"keyframe_ordinal={key[1]}): {count} records",
                )
            )
    return duplicates


def _check_ordering(
    records: list[KeyframeTiming],
    issues: list[MaterializationVerificationIssue],
) -> None:
    """Verify records are deterministically ordered.

    The artifact contract groups each video's records contiguously, in
    ``frames.ndjson`` order, with strictly increasing ``keyframe_ordinal``
    within a video.
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
                    MaterializationVerificationIssue(
                        "frames",
                        f"video {video_id!r} records are not contiguous in frames.ndjson",
                    )
                )
            if ordinal <= last_ordinal[video_id]:
                issues.append(
                    MaterializationVerificationIssue(
                        "frames",
                        f"records not ordered for video {video_id!r}: "
                        f"keyframe_ordinal {ordinal} after {last_ordinal[video_id]}",
                    )
                )
        seen.add(video_id)
        last_ordinal[video_id] = ordinal
        last_video = video_id


def _check_images(
    output_dir: Path,
    records: list[KeyframeTiming],
    issues: list[MaterializationVerificationIssue],
) -> tuple[int, int, int]:
    """Verify every record has an image, no image is missing or unexpected.

    Returns:
        - ``(images_found, missing_images, unexpected_images)``.
    """
    images_dir = output_dir / IMAGES_DIRNAME
    if not images_dir.is_dir():
        issues.append(
            MaterializationVerificationIssue(
                "artifact", f"images directory missing: {images_dir}"
            )
        )
        return 0, 0, 0

    expected = {
        (timing.keyframe.video_id.value, timing.keyframe.keyframe_ordinal)
        for timing in records
    }
    found: set[tuple[str, int]] = set()
    misnamed: list[Path] = []
    images_found = 0
    for video_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        video_id = video_dir.name
        for image_path in sorted(video_dir.glob(f"*.{IMAGE_EXTENSION}")):
            images_found += 1
            try:
                ordinal = int(image_path.stem)
            except ValueError:
                misnamed.append(image_path)
                continue
            if image_path.name != image_filename(ordinal):
                misnamed.append(image_path)
                continue
            found.add((video_id, ordinal))

    missing_images = 0
    for video_id, ordinal in sorted(expected):
        expected_path = materialized_image_path(output_dir, video_id, ordinal)
        if not expected_path.is_file():
            missing_images += 1
            issues.append(
                MaterializationVerificationIssue(
                    "images",
                    f"missing image for {video_id} ordinal {ordinal}: {expected_path}",
                )
            )

    unexpected_images = 0
    for video_id, ordinal in sorted(found - expected):
        unexpected_images += 1
        issues.append(
            MaterializationVerificationIssue(
                "images",
                f"unexpected image for {video_id} ordinal {ordinal}: "
                f"{images_dir / video_id / image_filename(ordinal)}",
            )
        )
    for image_path in misnamed:
        unexpected_images += 1
        issues.append(
            MaterializationVerificationIssue(
                "images", f"unexpected image file: {image_path}"
            )
        )
    return images_found, missing_images, unexpected_images


def verify_keyframe_materialization(
    output_dir: Path | str,
) -> KeyframeMaterializationVerificationReport:
    """Verify a completed materialized keyframe artifact.

    Checks artifact existence (``manifest.json``, ``frames.ndjson``,
    ``images/``), manifest consistency (artifact id, schema version, counts),
    canonical frame records (unique ``(video_id, keyframe_ordinal)`` keys,
    deterministic ordering), and image coverage (no missing or unexpected
    images, total count matches records).

    Expected input:
        - output_dir: directory of a materialized artifact (``images/`` +
          ``frames.ndjson`` + ``manifest.json``).

    Returns:
        - A :class:`KeyframeMaterializationVerificationReport`; the CLI maps
          ``valid`` to its exit code.

    Raises:
        - FileNotFoundError: if the artifact directory is absent.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"materialization output directory not found: {output_dir}")

    issues: list[MaterializationVerificationIssue] = []
    manifest, num_images, artifact_id = _read_manifest(output_dir, issues)
    records = _read_frames(output_dir, issues)

    duplicates = _check_duplicates(records, issues)
    _check_ordering(records, issues)

    if manifest is not None and manifest.num_keyframes != len(records):
        issues.append(
            MaterializationVerificationIssue(
                "manifest",
                f"manifest num_keyframes {manifest.num_keyframes} != "
                f"records in frames.ndjson ({len(records)})",
            )
        )

    images_found, missing_images, unexpected_images = _check_images(
        output_dir, records, issues
    )

    if num_images is not None and num_images != images_found:
        issues.append(
            MaterializationVerificationIssue(
                "manifest",
                f"manifest num_images {num_images} != images found ({images_found})",
            )
        )

    if len(records) != images_found:
        issues.append(
            MaterializationVerificationIssue(
                "images",
                f"frames.ndjson records ({len(records)}) != images found ({images_found})",
            )
        )

    return KeyframeMaterializationVerificationReport(
        output_dir=str(output_dir),
        artifact_id=artifact_id,
        valid=not issues,
        issues=tuple(issues),
        records=len(records),
        images=images_found,
        missing_images=missing_images,
        unexpected_images=unexpected_images,
        duplicates=duplicates,
        num_videos=len({timing.keyframe.video_id.value for timing in records}),
    )
