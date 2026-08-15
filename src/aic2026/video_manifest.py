"""Canonical video inventory manifest for keyframe extraction.

This module builds the ``videos.json`` artifact consumed by the keyframe
extraction split (``keyframe_extraction_split``) and execution
(``keyframe_extraction``). The serialized format is additive-compatible with
the existing ``aic2026.keyframe_extraction`` video-manifest schema (version
``1.0``): the ``video_id`` / ``path`` / ``num_frames`` fields are preserved
verbatim so existing consumers keep working, and the richer metadata
(``file_size_bytes``, ``duration_seconds``, ``fps``) is added without
changing their meaning.

The generator is deliberately separate from the extractor: video discovery,
metadata probing (``ffprobe``), and manifest serialization are reusable,
testable, and independent of how keyframes are later scored.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.contracts import VIDEO_ID_PATTERN

#: Version of the ``videos.json`` manifest schema (kept at ``1.0`` so the
#: existing split/execution consumers accept the artifact unchanged).
VIDEO_MANIFEST_SCHEMA_VERSION = "1.0"


class VideoManifestError(RuntimeError):
    """Raised when building, probing, or serializing a video manifest fails."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class VideoEntry:
    """One video in the inventory.

    Fields:
        - video_id: the ``L*_V*`` video id (the filename stem).
        - path: path to the video file. When the file is inside the manifest's
          ``source_dir`` the path is stored relative to it (so the manifest is
          relocatable); paths outside ``source_dir`` are stored absolute.
        - file_size_bytes: size of the video file in bytes.
        - duration_seconds: optional duration in seconds (``None`` when
          unknown or metadata probing is disabled).
        - fps: optional video frame rate (``None`` when unknown or metadata
          probing is disabled).
        - num_frames: optional known/estimated frame count (used as a split
          weight); ``None`` when unknown.
    """

    video_id: str
    path: str
    file_size_bytes: int
    duration_seconds: float | None = None
    fps: float | None = None
    num_frames: int | None = None

    def __post_init__(self) -> None:
        _require(
            VIDEO_ID_PATTERN.fullmatch(self.video_id) is not None,
            f"invalid video id: {self.video_id!r} (expected L<lane>_V<number>)",
        )
        _require(isinstance(self.path, str) and self.path != "", "path required")
        _require(
            isinstance(self.file_size_bytes, int) and not isinstance(self.file_size_bytes, bool) and self.file_size_bytes >= 0,
            "file_size_bytes must be a non-negative integer",
        )
        _require(
            self.duration_seconds is None or float(self.duration_seconds) > 0.0,
            "duration_seconds must be > 0 or None",
        )
        _require(self.fps is None or float(self.fps) > 0.0, "fps must be > 0 or None")
        _require(
            self.num_frames is None
            or (isinstance(self.num_frames, int) and not isinstance(self.num_frames, bool) and self.num_frames >= 1),
            "num_frames must be a positive integer or None",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "video_id": self.video_id,
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "num_frames": self.num_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoEntry":
        """Deserialize from a :meth:`to_dict` dict.

        Extra keys (e.g. from richer producers) are ignored, so manifests
        written by ``aic2026.keyframe_extraction`` load unchanged.
        """
        return cls(
            video_id=data["video_id"],
            path=data["path"],
            file_size_bytes=int(data.get("file_size_bytes", 0)),
            duration_seconds=data.get("duration_seconds"),
            fps=data.get("fps"),
            num_frames=data.get("num_frames"),
        )


@dataclass(frozen=True)
class VideoManifest:
    """A versioned, canonical inventory of videos.

    Fields:
        - schema_version: manifest schema version
          (``VIDEO_MANIFEST_SCHEMA_VERSION``).
        - source_dir: the directory the videos were discovered in.
        - videos: the video entries, deterministically ordered by ``video_id``.
        - created_at: ISO-8601 UTC timestamp of manifest creation.
    """

    schema_version: str
    source_dir: str
    videos: tuple[VideoEntry, ...]
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == VIDEO_MANIFEST_SCHEMA_VERSION,
            f"unsupported video manifest schema {self.schema_version!r}",
        )
        _require(isinstance(self.source_dir, str), "source_dir must be a string")
        _require(isinstance(self.created_at, str), "created_at must be a string")
        ids = [entry.video_id for entry in self.videos]
        _require(len(set(ids)) == len(ids), "video ids must be unique within a manifest")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "source_dir": self.source_dir,
            "created_at": self.created_at,
            "videos": [entry.to_dict() for entry in self.videos],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Extra top-level keys are ignored, so manifests written by
        ``aic2026.keyframe_extraction`` (which lack ``source_dir`` /
        ``created_at``) load with empty defaults.
        """
        return cls(
            schema_version=data["schema_version"],
            source_dir=data.get("source_dir", ""),
            created_at=data.get("created_at", ""),
            videos=tuple(VideoEntry.from_dict(entry) for entry in data["videos"]),
        )

    def by_id(self) -> dict[str, VideoEntry]:
        """Return ``video_id -> entry`` mapping."""
        return {entry.video_id: entry for entry in self.videos}


# ---------------------------------------------------------------- ffprobe probing


def _probe_video(path: Path) -> tuple[float, float, int]:
    """Probe ``(fps, duration_seconds, num_frames)`` via ``ffprobe``.

    Expected input:
        - path: path to a video file.

    Returns:
        - ``(fps, duration_seconds, num_frames)`` for the first video stream.
          ``num_frames`` is the ``nb_frames`` value when reported, otherwise
          estimated as ``round(duration * fps)``.

    Raises:
        - VideoManifestError: if ``ffprobe`` is unavailable, fails, or cannot
          report a valid frame rate or duration.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise VideoManifestError(
            "ffprobe is required for video metadata but was not found on PATH"
        ) from exc
    if result.returncode != 0:
        raise VideoManifestError(f"ffprobe failed on {path}: {result.stderr.strip()}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise VideoManifestError(f"no video stream found in {path}")
    stream = streams[0]

    rate = stream.get("avg_frame_rate", "0/1")
    try:
        numerator, denominator = rate.split("/")
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoManifestError(f"cannot parse frame rate {rate!r} for {path}") from exc
    if fps <= 0.0:
        raise VideoManifestError(f"invalid frame rate {rate!r} for {path}")

    duration = stream.get("duration")
    if duration is None:
        duration = (payload.get("format") or {}).get("duration")
    if duration is None:
        raise VideoManifestError(f"cannot determine duration for {path}")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise VideoManifestError(f"cannot parse duration {duration!r} for {path}") from exc
    if duration_seconds <= 0.0:
        raise VideoManifestError(f"invalid duration {duration!r} for {path}")

    raw_frames = stream.get("nb_frames")
    if raw_frames is not None and str(raw_frames).strip() not in ("", "N/A"):
        try:
            num_frames = int(raw_frames)
        except (TypeError, ValueError):
            num_frames = max(1, round(duration_seconds * fps))
    else:
        num_frames = max(1, round(duration_seconds * fps))

    return fps, duration_seconds, num_frames


# -------------------------------------------------------------------- generation


def _relocatable_path(path: Path, source_dir: Path) -> str:
    """Return ``path`` as a path relative to ``source_dir`` when possible.

    Expected input:
        - path: an absolute file path inside ``source_dir`` (e.g. a discovered
          ``*.mp4``).
        - source_dir: the resolved directory the videos were discovered in.

    Returns:
        - ``str(path.relative_to(source_dir))`` when the path is inside
          ``source_dir``; otherwise the absolute path unchanged (so files that
          escape the source tree are never lost).
    """
    try:
        return str(path.resolve().relative_to(source_dir))
    except ValueError:
        return str(path.resolve())


def build_video_manifest(
    video_dir: Path | str,
    include_metadata: bool = True,
) -> VideoManifest:
    """Build a canonical video inventory from a directory of ``.mp4`` files.

    Discovers ``*.mp4`` files (other extensions are ignored), derives
    ``video_id`` from each filename stem, and orders entries deterministically
    by ``video_id``. Empty (zero-byte) files are rejected. When
    ``include_metadata`` is True, ``ffprobe`` supplies ``fps``,
    ``duration_seconds``, and ``num_frames``; a probing failure is raised
    rather than silently producing an invalid manifest.

    Video paths are stored relative to ``source_dir`` so the serialized
    manifest is relocatable across machines; consumers that run on a different
    host resolve the relative paths against a ``video_root`` override (see
    ``aic2026.keyframe_extraction.extract_keyframes``).

    Expected input:
        - video_dir: directory containing ``*.mp4`` files.
        - include_metadata: if True, probe each video with ``ffprobe``.

    Returns:
        - A validated :class:`VideoManifest` whose entry ``path`` fields are
          relative to ``source_dir``.

    Raises:
        - FileNotFoundError: if ``video_dir`` does not exist.
        - ValueError: if ``video_dir`` is not a directory.
        - VideoManifestError: if a video file is empty/corrupt or metadata
          probing fails.
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")
    if not video_dir.is_dir():
        raise ValueError(f"video path is not a directory: {video_dir}")
    source_dir = video_dir.resolve()

    entries: list[VideoEntry] = []
    seen: set[str] = set()
    for path in sorted(video_dir.glob("*.mp4")):
        video_id = path.stem
        if video_id in seen:
            raise VideoManifestError(f"duplicate video id {video_id!r} in {video_dir}")
        seen.add(video_id)

        size = path.stat().st_size
        if size == 0:
            raise VideoManifestError(f"empty video file: {path}")

        duration_seconds: float | None = None
        fps: float | None = None
        num_frames: int | None = None
        if include_metadata:
            fps, duration_seconds, num_frames = _probe_video(path)

        entries.append(
            VideoEntry(
                video_id=video_id,
                path=_relocatable_path(path, source_dir),
                file_size_bytes=int(size),
                duration_seconds=duration_seconds,
                fps=fps,
                num_frames=num_frames,
            )
        )

    entries.sort(key=lambda entry: entry.video_id)
    return VideoManifest(
        schema_version=VIDEO_MANIFEST_SCHEMA_VERSION,
        source_dir=str(source_dir),
        videos=tuple(entries),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------- serialization


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def save_video_manifest(path: Path | str, manifest: VideoManifest) -> None:
    """Persist a ``videos.json`` manifest (atomic, deterministic).

    Expected input:
        - path: target ``videos.json`` path.
        - manifest: the manifest to write.

    Returns:
        - None.

    Raises:
        - OSError: if the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, manifest.to_dict())


def load_video_manifest(path: Path | str) -> VideoManifest:
    """Load and validate a ``videos.json`` manifest.

    Expected input:
        - path: path to a ``videos.json`` file.

    Returns:
        - The parsed :class:`VideoManifest`.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if the file is not valid JSON.
        - ValueError: if the content is invalid.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return VideoManifest.from_dict(json.load(handle))