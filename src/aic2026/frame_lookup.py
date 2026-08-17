"""General-purpose frame lookup over canonical keyframe artifacts.

Given a keyframe artifact whose ``frames.ndjson`` follows the canonical
keyframe record schema (:class:`aic2026.canonical.KeyframeTiming`), this
module resolves a ``(video_id, keyframe_ordinal)`` reference to its stored
metadata and optionally derives an annotation GT frame range.

Purpose — this is an *annotation* utility for building:

- KIS ``gt_frame_start`` / ``gt_frame_end`` columns of the evaluation CSV;
- QA evidence frame selection;
- TRAKE event range annotation.

It depends only on the canonical artifact schema:

- ``frames.ndjson`` (one canonical keyframe record per line);
- an optional ``manifest.json`` whose presence/absence is tolerated (its
  ``schema_version`` is reported but no artifact-specific schema, extractor
  name, or artifact id is required);
- an optional canonical ``images/<video_id>/<ordinal:06d>.jpg`` collection.

It deliberately does NOT read or require:

- ``map-keyframes/*.csv`` legacy maps;
- BTC keyframe folders or old baseline mappings;
- any extractor-specific directory layout or artifact id such as
  ``adaptive_sbd_v1``.

Future artifacts work unchanged as long as their ``frames.ndjson`` follows
the canonical keyframe record schema. Stored metadata is never modified; the
``--window-sec`` GT range is derived and clamped, not written back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.canonical import VideoId

#: Filename of the canonical keyframe records inside the artifact.
FRAMES_FILENAME = "frames.ndjson"
#: Optional filename of the artifact manifest inside the artifact.
MANIFEST_FILENAME = "manifest.json"
#: Canonical materialized image directory name (relative to the artifact).
IMAGES_DIRNAME = "images"
#: Zero-padded width of ``keyframe_ordinal`` in canonical image filenames.
IMAGE_ORDINAL_DIGITS = 6
#: Image extension of the canonical materialized images.
IMAGE_EXTENSION = "jpg"


class FrameLookupError(ValueError):
    """Raised for invalid references, malformed artifacts, or bad input."""


@dataclass(frozen=True)
class FrameLookupRecord:
    """Resolved metadata of one keyframe from a canonical artifact.

    Expected input:
        - video_id: canonical video identifier.
        - keyframe_ordinal: 1-based keyframe ordinal within the video.
        - frame_index: 0-based original-video frame index.
        - pts_time: presentation timestamp in seconds, or ``None`` when the
          artifact does not store it.
        - fps: original-video frame rate, or ``None`` when not stored.

    Returns:
        - A validated lookup record.

    Raises:
        - ValueError: if any field violates the contract.
    """

    video_id: str
    keyframe_ordinal: int
    frame_index: int
    pts_time: float | None = None
    fps: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or self.video_id == "":
            raise ValueError("video_id must be a non-empty string")
        if not isinstance(self.keyframe_ordinal, int) or isinstance(self.keyframe_ordinal, bool) or self.keyframe_ordinal < 1:
            raise ValueError("keyframe_ordinal must be a positive integer")
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self.pts_time is not None and not isinstance(self.pts_time, (int, float)):
            raise ValueError("pts_time must be numeric or None")
        if self.fps is not None and not isinstance(self.fps, (int, float)):
            raise ValueError("fps must be numeric or None")


def parse_frame_record(payload: dict[str, Any]) -> FrameLookupRecord:
    """Parse one canonical keyframe record, tolerating optional timing fields.

    The essential canonical identity fields (``keyframe.video_id``,
    ``keyframe.keyframe_ordinal``, ``frame_index``) are required and
    validated; ``pts_time`` and ``fps`` are preserved when present and set
    to ``None`` when absent, so future artifacts that omit them still work.

    Expected input:
        - payload: a parsed JSON object for one ``frames.ndjson`` line.

    Returns:
        - The validated lookup record.

    Raises:
        - FrameLookupError: if the record is not canonical-shaped or an
          essential field is invalid.
    """
    if not isinstance(payload, dict):
        raise FrameLookupError("keyframe record must be a JSON object")
    try:
        keyframe = payload["keyframe"]
        if not isinstance(keyframe, dict):
            raise FrameLookupError("keyframe record is missing the 'keyframe' object")
        video_id = _parse_video_id(keyframe.get("video_id"))
        keyframe_ordinal = int(keyframe["keyframe_ordinal"])
        frame_index = int(payload["frame_index"])
    except KeyError as exc:
        raise FrameLookupError(f"keyframe record is missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise FrameLookupError(f"keyframe record has an invalid essential field: {exc}") from exc
    if keyframe_ordinal < 1:
        raise FrameLookupError("keyframe_ordinal must be a positive integer")
    if frame_index < 0:
        raise FrameLookupError("frame_index must be a non-negative integer")
    pts_time = _optional_float(payload.get("pts_time"), "pts_time", minimum=0.0, strict_minimum=False)
    fps = _optional_float(payload.get("fps"), "fps", minimum=0.0, strict_minimum=True)
    return FrameLookupRecord(
        video_id=video_id,
        keyframe_ordinal=keyframe_ordinal,
        frame_index=frame_index,
        pts_time=pts_time,
        fps=fps,
    )


class KeyframeArtifact:
    """A loaded canonical keyframe artifact with O(1) frame lookup.

    Expected input:
        - artifact_dir: directory containing ``frames.ndjson`` and an
          optional ``manifest.json`` / ``images/`` collection.

    Returns:
        - A validated artifact index; loading fails fast with a clear
          :class:`FrameLookupError` for missing/invalid inputs.

    Raises:
        - FrameLookupError: if the directory is missing, ``frames.ndjson``
          is absent, the manifest is malformed, or a record is invalid.
    """

    def __init__(self, artifact_dir: Path | str) -> None:
        dir_path = Path(artifact_dir)
        if not dir_path.is_dir():
            raise FrameLookupError(f"artifact directory does not exist: {dir_path}")
        frames_path = dir_path / FRAMES_FILENAME
        if not frames_path.is_file():
            raise FrameLookupError(f"artifact has no {FRAMES_FILENAME}: {dir_path}")
        manifest = _read_manifest(dir_path)
        records: dict[tuple[str, int], FrameLookupRecord] = {}
        with frames_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip() == "":
                    continue
                try:
                    record = parse_frame_record(json.loads(line))
                except (json.JSONDecodeError, FrameLookupError) as exc:
                    raise FrameLookupError(f"{frames_path} line {line_number}: {exc}") from exc
                key = (record.video_id, record.keyframe_ordinal)
                if key in records:
                    raise FrameLookupError(
                        f"duplicate keyframe record {record.video_id!r} ordinal {record.keyframe_ordinal} "
                        f"in {frames_path}"
                    )
                records[key] = record
        self._artifact_dir = dir_path
        self._manifest = manifest
        self._frames_path = frames_path
        self._records = records

    @property
    def artifact_dir(self) -> Path:
        """Return the artifact directory path."""
        return self._artifact_dir

    @property
    def manifest(self) -> dict[str, Any] | None:
        """Return the parsed artifact manifest, or ``None`` when absent."""
        return self._manifest

    @property
    def schema_version(self) -> str | None:
        """Return the manifest's ``schema_version``, or ``None``."""
        if self._manifest is None:
            return None
        value = self._manifest.get("schema_version")
        return value if isinstance(value, str) else None

    @property
    def num_keyframes(self) -> int:
        """Return the number of keyframe records in the artifact."""
        return len(self._records)

    def lookup(self, video_id: str, keyframe_ordinal: int) -> FrameLookupRecord:
        """Resolve a ``(video_id, keyframe_ordinal)`` reference.

        Expected input:
            - video_id: canonical video identifier.
            - keyframe_ordinal: 1-based keyframe ordinal.

        Returns:
            - The stored canonical record.

        Raises:
            - FrameLookupError: if the reference is unknown.
        """
        key = (video_id, keyframe_ordinal)
        record = self._records.get(key)
        if record is None:
            raise FrameLookupError(
                f"keyframe not found: video_id={video_id!r} keyframe_ordinal={keyframe_ordinal}"
            )
        return record

    def image_path(self, video_id: str, keyframe_ordinal: int) -> str | None:
        """Return the canonical image path when the image exists.

        Expected input:
            - video_id: canonical video identifier.
            - keyframe_ordinal: 1-based keyframe ordinal.

        Returns:
            - The canonical relative path
              ``images/<video_id>/<ordinal:06d>.jpg`` when the file exists,
              otherwise ``None``.

        Raises:
            - No exceptions.
        """
        candidate = self._artifact_dir / image_relpath(video_id, keyframe_ordinal)
        return image_relpath(video_id, keyframe_ordinal) if candidate.is_file() else None


def parse_keyframe_reference(reference: str) -> tuple[str, int]:
    """Parse a keyframe reference into ``(video_id, keyframe_ordinal)``.

    Accepted formats:
        - ``VIDEO_ID/ORDINAL.ext``, e.g. ``L25_V085/000123.jpg``;
        - ``VIDEO_ID ORDINAL``, e.g. ``L25_V085 123``.

    Expected input:
        - reference: a non-empty keyframe reference string.

    Returns:
        - Tuple ``(video_id, keyframe_ordinal)``.

    Raises:
        - FrameLookupError: if the reference is malformed or the ordinal is
          not a positive integer.
    """
    if not isinstance(reference, str) or reference.strip() == "":
        raise FrameLookupError("keyframe reference must be a non-empty string")
    reference = reference.strip()
    if " " in reference:
        parts = reference.split()
        if len(parts) != 2:
            raise FrameLookupError(
                f"invalid keyframe reference {reference!r}; expected 'VIDEO_ID ORDINAL'"
            )
        video_id_text, ordinal_text = parts
    elif "/" in reference:
        video_id_text, filename = reference.split("/", maxsplit=1)
        if "/" in filename:
            raise FrameLookupError(
                f"invalid keyframe reference {reference!r}; expected 'VIDEO_ID/ORDINAL.ext'"
            )
        ordinal_text = Path(filename).stem
    else:
        raise FrameLookupError(
            f"invalid keyframe reference {reference!r}; "
            "expected 'VIDEO_ID/ORDINAL.ext' or 'VIDEO_ID ORDINAL'"
        )
    try:
        video_id = VideoId(video_id_text).value
    except ValueError as exc:
        raise FrameLookupError(f"invalid video_id {video_id_text!r}: {exc}") from exc
    try:
        keyframe_ordinal = int(ordinal_text)
    except ValueError as exc:
        raise FrameLookupError(f"invalid keyframe_ordinal {ordinal_text!r}") from exc
    if keyframe_ordinal < 1:
        raise FrameLookupError("keyframe_ordinal must be a positive integer")
    return video_id, keyframe_ordinal


def compute_gt_range(record: FrameLookupRecord, window_sec: float) -> tuple[int, int]:
    """Derive an inclusive GT frame range centered on a keyframe.

    Uses the canonical ``frame_index`` and the record's ``fps``: the total
    ``window_sec`` duration is split evenly around the keyframe and converted
    to frames with ``round``. The start is clamped at 0 so negative frame
    indices are never produced. Stored metadata is not modified.

    Expected input:
        - record: a canonical lookup record.
        - window_sec: positive total window duration in seconds.

    Returns:
        - Tuple ``(gt_frame_start, gt_frame_end)``, both non-negative.

    Raises:
        - FrameLookupError: if the record has no ``fps`` (a second-based
          window cannot be converted) or ``window_sec`` is not positive.
    """
    if not isinstance(window_sec, (int, float)) or isinstance(window_sec, bool) or window_sec <= 0:
        raise FrameLookupError(f"window_sec must be a positive number, got {window_sec!r}")
    if record.fps is None:
        raise FrameLookupError("cannot compute gt_frame range: record has no fps")
    half_frames = round(record.fps * float(window_sec) / 2.0)
    start = max(0, record.frame_index - half_frames)
    end = record.frame_index + half_frames
    return start, end


def format_record(
    record: FrameLookupRecord,
    image_path: str | None,
    gt_range: tuple[int, int] | None = None,
) -> str:
    """Format a lookup record as the annotator-facing text output.

    Expected input:
        - record: the resolved lookup record.
        - image_path: canonical image path, or ``None`` when unavailable.
        - gt_range: optional ``(start, end)`` frame range to include.

    Returns:
        - A deterministic, column-aligned multi-line string.

    Raises:
        - No exceptions.
    """
    lines = [
        f"video_id          : {record.video_id}",
        f"keyframe_ordinal  : {record.keyframe_ordinal}",
        f"frame_index       : {record.frame_index}",
        f"pts_time          : {_format_optional_float(record.pts_time)}",
        f"fps               : {_format_optional_float(record.fps)}",
        f"image_path        : {image_path if image_path is not None else 'not available'}",
    ]
    if gt_range is not None:
        lines.append(f"gt_frame_start    : {gt_range[0]}")
        lines.append(f"gt_frame_end      : {gt_range[1]}")
    return "\n".join(lines)


def image_filename(keyframe_ordinal: int) -> str:
    """Return the canonical materialized image filename for an ordinal."""
    if not isinstance(keyframe_ordinal, int) or isinstance(keyframe_ordinal, bool) or keyframe_ordinal < 1:
        raise FrameLookupError("keyframe_ordinal must be a positive integer")
    return f"{keyframe_ordinal:0{IMAGE_ORDINAL_DIGITS}d}.{IMAGE_EXTENSION}"


def image_relpath(video_id: str, keyframe_ordinal: int) -> str:
    """Return the canonical relative image path for a keyframe."""
    return f"{IMAGES_DIRNAME}/{video_id}/{image_filename(keyframe_ordinal)}"


def load_frame_lookup_config(path: Path | str | None) -> dict[str, Any]:
    """Load an optional YAML config; returns the ``frame_lookup`` section.

    Expected input:
        - path: optional YAML config path; ``None`` returns ``{}``.

    Returns:
        - The ``frame_lookup`` mapping from the YAML file (empty when the
          file has no such section).

    Raises:
        - FileNotFoundError: if the path does not exist.
        - yaml.YAMLError: if the YAML cannot be parsed.
        - FrameLookupError: if the ``frame_lookup`` section is not a mapping.
    """
    if path is None:
        return {}
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise FrameLookupError("config file must contain a YAML mapping")
    section = payload.get("frame_lookup", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise FrameLookupError("config 'frame_lookup' section must be a mapping")
    return dict(section)


def _parse_video_id(value: Any) -> str:
    """Normalize and validate a canonical video id from a record field."""
    if isinstance(value, str):
        raw = value
    elif isinstance(value, dict) and isinstance(value.get("value"), str):
        raw = value["value"]
    else:
        raise ValueError("video_id must be a string or {'value': str}")
    return VideoId(raw).value


def _optional_float(value: Any, name: str, *, minimum: float, strict_minimum: bool) -> float | None:
    """Coerce an optional numeric field, validating its bound."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FrameLookupError(f"{name} must be numeric, got {value!r}") from exc
    if strict_minimum and parsed <= minimum:
        raise FrameLookupError(f"{name} must be > {minimum:g}, got {parsed!r}")
    if not strict_minimum and parsed < minimum:
        raise FrameLookupError(f"{name} must be >= {minimum:g}, got {parsed!r}")
    return parsed


def _format_optional_float(value: float | None) -> str:
    """Format an optional float as a short string or ``n/a``."""
    if value is None:
        return "n/a"
    return f"{float(value):g}"


def _read_manifest(dir_path: Path) -> dict[str, Any] | None:
    """Read and parse the optional artifact manifest."""
    manifest_path = dir_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise FrameLookupError(f"invalid manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrameLookupError(f"manifest must be a JSON object: {manifest_path}")
    return payload