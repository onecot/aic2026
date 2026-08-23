"""Resumable, deterministic keyframe extraction pipeline.

Given a video manifest (a ``videos.json`` listing ``video_id`` + ``path``
entries), this module runs the DAKE-inspired extractor over every video and
writes a canonical keyframe artifact::

    <output-dir>/
        frames.ndjson             canonical keyframe records (KeyframeTiming)
        map-keyframes/
            L21_V001.csv          per-video ``n,pts_time,fps,frame_idx`` maps
            ...
        manifest.json             versioned extraction metadata
        checkpoint.json           resume state (present during processing)

The ``frames.ndjson`` records exactly match :class:`KeyframeTiming.to_dict()`
and the ``map-keyframes/*.csv`` files match the ``parse_keyframe_map`` column
contract (``n, pts_time, fps, frame_idx``), so the artifact is a drop-in
replacement for the BTC-provided keyframe source.

Pipeline guarantees:

- deterministic: same manifest + same config -> byte-identical records;
- resumable: ``checkpoint.json`` records completed videos; an interrupted run
  skips them on resume; a config change on resume is rejected, never mixed;
- atomic: per-video outputs are written to a staging directory and the final
  ``frames.ndjson``/``manifest.json`` are swapped into place only after all
  videos complete;
- validated: the finished artifact is re-read and checked (unique
  ``(video_id, frame_index)`` keys, every video accounted for) before it is
  marked complete.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aic2026.artifact_io import read_json_object, write_json_atomic
from aic2026.canonical import VIDEO_ID_PATTERN, KeyframeTiming
from aic2026.image_embedder import DEVICE_CHOICES
from aic2026.assets.keyframe_extractor import (
    DakeKeyframeExtractor,
    FRAME_READER_NAMES,
    Keyframe,
    SCORER_NAMES,
    create_frame_reader,
    create_scorer,
)
from aic2026.assets.keyframe_sampling import SAMPLER_NAMES
from aic2026.progress import ProgressReporter
from aic2026.assets.shot_detection import (
    DEFAULT_OMNISHOTCUT_FILENAME,
    SHOT_DETECTOR_NAMES,
    create_shot_detector,
)

#: Schema version of the extraction manifest.
EXTRACTION_MANIFEST_SCHEMA_VERSION = "1.0"

#: Schema version of the extraction checkpoint.
EXTRACTION_CHECKPOINT_SCHEMA_VERSION = "1.0"

#: Schema version of the video manifest.
VIDEOS_MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename of the canonical keyframe output inside the output directory.
FRAMES_FILENAME = "frames.ndjson"

#: Directory holding per-video keyframe maps inside the output directory.
MAP_KEYFRAMES_DIRNAME = "map-keyframes"

#: Filename of the extraction manifest inside the output directory.
MANIFEST_FILENAME = "manifest.json"

#: Filename of the resume checkpoint inside the output directory.
CHECKPOINT_FILENAME = "checkpoint.json"

#: Filename of the video manifest consumed by the pipeline.
VIDEOS_MANIFEST_FILENAME = "videos.json"

#: Name of the staging directory for incomplete per-video outputs.
STAGING_DIRNAME = ".staging"

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when extraction fails or produces an invalid artifact."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _canonicalize(value: Any) -> Any:
    """Recursively sort dict keys so identical content hashes identically."""
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- video manifest


@dataclass(frozen=True)
class VideoEntry:
    """One video to extract keyframes from.

    Fields:
        - video_id: the ``L*_V*`` video id.
        - path: path to the video file.
        - num_frames: optional known frame count (used as a split weight);
          ``None`` when unknown.
    """

    video_id: str
    path: str
    num_frames: int | None = None

    def __post_init__(self) -> None:
        _require(
            VIDEO_ID_PATTERN.fullmatch(self.video_id) is not None,
            f"invalid video id: {self.video_id!r} (expected L<lane>_V<number>)",
        )
        _require(isinstance(self.path, str) and self.path != "", "path required")
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
            "num_frames": self.num_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoEntry":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            video_id=data["video_id"],
            path=data["path"],
            num_frames=data.get("num_frames"),
        )


@dataclass(frozen=True)
class VideoManifest:
    """A versioned list of videos to extract.

    Fields:
        - schema_version: video manifest schema version
          (``VIDEOS_MANIFEST_SCHEMA_VERSION``).
        - source_dir: optional directory the videos were discovered in; used
          to resolve relative ``path`` values that survived a relocatable
          ``videos.json`` (empty when unknown or the paths are absolute).
        - videos: the video entries, in file order.
    """

    schema_version: str
    videos: tuple[VideoEntry, ...]
    source_dir: str = ""

    def __post_init__(self) -> None:
        _require(
            self.schema_version == VIDEOS_MANIFEST_SCHEMA_VERSION,
            f"unsupported video manifest schema {self.schema_version!r}",
        )
        _require(isinstance(self.source_dir, str), "source_dir must be a string")
        ids = [entry.video_id for entry in self.videos]
        _require(len(set(ids)) == len(ids), "video ids must be unique within a manifest")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "source_dir": self.source_dir,
            "videos": [entry.to_dict() for entry in self.videos],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Missing ``source_dir`` (legacy manifests) defaults to ``""``.
        """
        return cls(
            schema_version=data["schema_version"],
            source_dir=data.get("source_dir", ""),
            videos=tuple(VideoEntry.from_dict(entry) for entry in data["videos"]),
        )

    def by_id(self) -> dict[str, VideoEntry]:
        """Return ``video_id -> entry`` mapping."""
        return {entry.video_id: entry for entry in self.videos}


def resolve_video_path(
    path: str,
    *,
    source_dir: str = "",
    video_root: Path | str | None = None,
) -> Path:
    """Resolve a manifest video path to an absolute filesystem path.

    Absolute paths are returned unchanged (backward compatibility with legacy
    manifests). Relative paths resolve against ``video_root`` when provided,
    then against the manifest's ``source_dir`` when present.

    Expected input:
        - path: the ``path`` stored on a :class:`VideoEntry`.
        - source_dir: the manifest's ``source_dir`` (empty when unknown).
        - video_root: optional override root for relative paths.

    Returns:
        - The resolved absolute path.

    Raises:
        - ValueError: if ``path`` is relative and neither ``video_root`` nor
          ``source_dir`` provides a base to resolve it against.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = Path(video_root) if video_root is not None else (Path(source_dir) if source_dir else None)
    if base is None:
        raise ValueError(
            f"cannot resolve relative video path {path!r}: pass a video_root "
            "override or ensure the manifest records a source_dir"
        )
    return base / candidate


def require_media_tools(
    *,
    ffmpeg: bool = True,
    ffprobe: bool = True,
) -> None:
    """Fail before extraction starts if required media tools are unavailable.

    FFmpeg and ffprobe are invoked on ``PATH`` by the frame reader and the
    probe helpers. Detecting their absence up front avoids starting a
    multi-hour job that fails on the first video.

    Expected input:
        - ffmpeg: if True, require ``ffmpeg`` on ``PATH``.
        - ffprobe: if True, require ``ffprobe`` on ``PATH``.

    Returns:
        - None.

    Raises:
        - ExtractionError: listing every required tool that is missing.
    """
    missing = [
        tool
        for tool, required in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        if required and shutil.which(tool) is None
    ]
    if missing:
        raise ExtractionError(
            "required media tools not found on PATH: "
            + ", ".join(missing)
            + ". Install ffmpeg >= 4.4 (with ffprobe) and ensure both are on "
            "PATH before running extraction."
        )


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
    write_json_atomic(path, manifest.to_dict())


# ---------------------------------------------------------------- extraction config


@dataclass(frozen=True)
class ExtractionConfig:
    """Validated parameters for one extraction run.

    Exactly one of ``target_keyframes_per_video`` or ``sampling_ratio`` must
    be provided when ``extractor == "dake"`` (the ``omnishotcut_adaptive``
    extractor requires neither).

    Fields:
        - extractor: extractor implementation name (``dake`` or
          ``omnishotcut_adaptive``).
        - extractor_version: extractor version string.
        - scorer: change-scorer name (``frame_difference``).
        - frame_reader: frame reader name (``ffmpeg`` or ``synthetic``).
        - detector: shot-detector name (``omnishotcut`` or ``synthetic``) for
          the ``omnishotcut_adaptive`` extractor.
        - sampler: sampler name (``adaptive_v1``) for the
          ``omnishotcut_adaptive`` extractor.
        - short_shot_seconds / long_shot_seconds / uniform_interval_seconds:
          adaptive-sampler thresholds (shots shorter than ``short_shot_seconds``
          get only their middle frame; shots up to ``long_shot_seconds`` add a
          uniform sample; longer shots also get high-change frames).
        - omnishotcut_mode / omnishotcut_overlap / device / checkpoint /
          omnishotcut_filename: OmniShotCut model options.
        - min_temporal_distance: minimum frame gap between keyframes.
        - peak_threshold: minimum score for a candidate peak.
        - target_keyframes_per_video: hard total per-video keyframe cap including
          anchors (mutually exclusive with ``sampling_ratio``; ``dake`` only).
        - sampling_ratio: fraction of frames selected as keyframes (mutually
          exclusive with ``target_keyframes_per_video``; ``dake`` only).
        - max_side: longest side (pixels) of downscaled grayscale frames.
        - fps: optional fixed video frame rate override.
        - jpeg_quality: JPEG quality in ``[1, 100]`` for the ``jpeg_size``
          scorer.
        - anchor_interval_seconds: optional uniform anchor spacing in seconds
          (first and last frames are always anchors; anchors are downsampled
          deterministically when they would exceed
          ``target_keyframes_per_video``).
    """

    extractor: str = "dake"
    extractor_version: str = "2.0"
    scorer: str = "frame_difference"
    frame_reader: str = "ffmpeg"
    detector: str = "omnishotcut"
    sampler: str = "adaptive_v1"
    short_shot_seconds: float = 5.0
    long_shot_seconds: float = 60.0
    uniform_interval_seconds: float = 10.0
    omnishotcut_mode: str = "clean_shot"
    omnishotcut_overlap: int = 20
    device: str = "auto"
    checkpoint: str | None = None
    omnishotcut_filename: str = DEFAULT_OMNISHOTCUT_FILENAME
    min_temporal_distance: int = 30
    peak_threshold: float = 8.0
    target_keyframes_per_video: int | None = None
    sampling_ratio: float | None = None
    max_side: int = 64
    fps: float | None = None
    jpeg_quality: int = 70
    anchor_interval_seconds: float | None = None

    def __post_init__(self) -> None:
        _require(
            self.extractor in ("dake", "omnishotcut_adaptive"),
            f"unsupported extractor {self.extractor!r}; expected dake or omnishotcut_adaptive",
        )
        _require(isinstance(self.extractor_version, str) and self.extractor_version != "", "extractor_version required")
        _require(self.scorer in SCORER_NAMES, f"unsupported scorer {self.scorer!r}; expected {SCORER_NAMES}")
        _require(self.frame_reader in FRAME_READER_NAMES, f"unsupported frame reader {self.frame_reader!r}; expected {FRAME_READER_NAMES}")
        _require(
            isinstance(self.min_temporal_distance, int) and not isinstance(self.min_temporal_distance, bool) and self.min_temporal_distance >= 1,
            "min_temporal_distance must be >= 1",
        )
        _require(float(self.peak_threshold) >= 0.0, "peak_threshold must be >= 0")
        _require(
            self.target_keyframes_per_video is None
            or (isinstance(self.target_keyframes_per_video, int) and not isinstance(self.target_keyframes_per_video, bool) and self.target_keyframes_per_video >= 1),
            "target_keyframes_per_video must be a positive integer or None",
        )
        _require(
            self.sampling_ratio is None or 0.0 < float(self.sampling_ratio) <= 1.0,
            "sampling_ratio must be in (0, 1] or None",
        )
        if self.extractor == "omnishotcut_adaptive":
            _require(
                self.target_keyframes_per_video is None and self.sampling_ratio is None,
                "omnishotcut_adaptive does not support target_keyframes_per_video or sampling_ratio",
            )
        else:
            _require(
                (self.target_keyframes_per_video is None) != (self.sampling_ratio is None),
                "exactly one of target_keyframes_per_video or sampling_ratio is required",
            )
        _require(
            isinstance(self.max_side, int) and not isinstance(self.max_side, bool) and self.max_side >= 1,
            "max_side must be a positive integer",
        )
        _require(self.fps is None or float(self.fps) > 0.0, "fps must be > 0 when provided")
        _require(
            isinstance(self.jpeg_quality, int) and not isinstance(self.jpeg_quality, bool) and 1 <= self.jpeg_quality <= 100,
            "jpeg_quality must be an integer in [1, 100]",
        )
        _require(
            self.anchor_interval_seconds is None or float(self.anchor_interval_seconds) > 0.0,
            "anchor_interval_seconds must be > 0 when provided",
        )
        if self.extractor == "omnishotcut_adaptive":
            _require(self.detector in SHOT_DETECTOR_NAMES, f"unsupported shot detector {self.detector!r}; expected {SHOT_DETECTOR_NAMES}")
            _require(self.sampler in SAMPLER_NAMES, f"unsupported sampler {self.sampler!r}; expected {SAMPLER_NAMES}")
            _require(
                0.0 < float(self.short_shot_seconds) < float(self.long_shot_seconds),
                "need 0 < short_shot_seconds < long_shot_seconds",
            )
            _require(float(self.uniform_interval_seconds) > 0.0, "uniform_interval_seconds must be > 0")
            _require(
                self.omnishotcut_mode in ("clean_shot", "default"),
                f"unsupported omnishotcut_mode {self.omnishotcut_mode!r}; expected clean_shot or default",
            )
            _require(
                isinstance(self.omnishotcut_overlap, int) and not isinstance(self.omnishotcut_overlap, bool) and self.omnishotcut_overlap >= 0,
                "omnishotcut_overlap must be a non-negative integer",
            )
            _require(self.device in DEVICE_CHOICES, f"unsupported device {self.device!r}; expected {DEVICE_CHOICES}")
            _require(self.checkpoint is None or (isinstance(self.checkpoint, str) and self.checkpoint != ""), "checkpoint must be a non-empty string or None")
            _require(isinstance(self.omnishotcut_filename, str) and self.omnishotcut_filename != "", "omnishotcut_filename required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        The ``omnishotcut_adaptive`` fields are only included for that
        extractor, so ``dake`` configurations keep a byte-stable identity
        (resume compatibility with existing dake artifacts).
        """
        if self.extractor == "omnishotcut_adaptive":
            return {
                "extractor": self.extractor,
                "extractor_version": self.extractor_version,
                "scorer": self.scorer,
                "frame_reader": self.frame_reader,
                "detector": self.detector,
                "sampler": self.sampler,
                "short_shot_seconds": self.short_shot_seconds,
                "long_shot_seconds": self.long_shot_seconds,
                "uniform_interval_seconds": self.uniform_interval_seconds,
                "omnishotcut_mode": self.omnishotcut_mode,
                "omnishotcut_overlap": self.omnishotcut_overlap,
                "device": self.device,
                "checkpoint": self.checkpoint,
                "omnishotcut_filename": self.omnishotcut_filename,
                "min_temporal_distance": self.min_temporal_distance,
                "peak_threshold": self.peak_threshold,
                "max_side": self.max_side,
                "fps": self.fps,
                "jpeg_quality": self.jpeg_quality,
            }
        return {
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "scorer": self.scorer,
            "frame_reader": self.frame_reader,
            "min_temporal_distance": self.min_temporal_distance,
            "peak_threshold": self.peak_threshold,
            "target_keyframes_per_video": self.target_keyframes_per_video,
            "sampling_ratio": self.sampling_ratio,
            "max_side": self.max_side,
            "fps": self.fps,
            "jpeg_quality": self.jpeg_quality,
            "anchor_interval_seconds": self.anchor_interval_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionConfig":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            extractor=data.get("extractor", "dake"),
            extractor_version=data.get("extractor_version", "2.0"),
            scorer=data.get("scorer", "frame_difference"),
            frame_reader=data.get("frame_reader", "ffmpeg"),
            detector=data.get("detector", "omnishotcut"),
            sampler=data.get("sampler", "adaptive_v1"),
            short_shot_seconds=float(data.get("short_shot_seconds", 5.0)),
            long_shot_seconds=float(data.get("long_shot_seconds", 60.0)),
            uniform_interval_seconds=float(data.get("uniform_interval_seconds", 10.0)),
            omnishotcut_mode=data.get("omnishotcut_mode", "clean_shot"),
            omnishotcut_overlap=int(data.get("omnishotcut_overlap", 20)),
            device=data.get("device", "auto"),
            checkpoint=data.get("checkpoint"),
            omnishotcut_filename=data.get("omnishotcut_filename", DEFAULT_OMNISHOTCUT_FILENAME),
            min_temporal_distance=int(data.get("min_temporal_distance", 30)),
            peak_threshold=float(data.get("peak_threshold", 8.0)),
            target_keyframes_per_video=data.get("target_keyframes_per_video"),
            sampling_ratio=data.get("sampling_ratio"),
            max_side=int(data.get("max_side", 64)),
            fps=data.get("fps"),
            jpeg_quality=int(data.get("jpeg_quality", 70)),
            anchor_interval_seconds=data.get("anchor_interval_seconds"),
        )

    def config_hash(self) -> str:
        """Return a stable sha256 identity of the effective configuration.

        The hash is computed over canonicalized JSON (sorted keys) so that
        re-serializing the same config always yields the same value. It is
        used to reject resume runs that changed experiment parameters.

        Returns:
            - Hex sha256 digest (64 characters).
        """
        payload = json.dumps(_canonicalize(self.to_dict()), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------- extraction manifest


@dataclass(frozen=True)
class ExtractionStatistics:
    """Aggregate statistics for one extraction run.

    Fields:
        - total_videos: number of videos processed.
        - total_frames_scanned: total source frames consumed across videos.
        - total_keyframes: total keyframes selected.
        - avg_keyframes_per_video: mean keyframes per video.
        - min_keyframes: fewest keyframes selected for a video.
        - max_keyframes: most keyframes selected for a video.
    """

    total_videos: int
    total_frames_scanned: int
    total_keyframes: int
    avg_keyframes_per_video: float
    min_keyframes: int
    max_keyframes: int

    def __post_init__(self) -> None:
        _require(isinstance(self.total_videos, int) and self.total_videos >= 0, "total_videos must be >= 0")
        _require(isinstance(self.total_frames_scanned, int) and self.total_frames_scanned >= 0, "total_frames_scanned must be >= 0")
        _require(isinstance(self.total_keyframes, int) and self.total_keyframes >= 0, "total_keyframes must be >= 0")
        _require(isinstance(self.min_keyframes, int) and self.min_keyframes >= 0, "min_keyframes must be >= 0")
        _require(isinstance(self.max_keyframes, int) and self.max_keyframes >= 0, "max_keyframes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "total_videos": self.total_videos,
            "total_frames_scanned": self.total_frames_scanned,
            "total_keyframes": self.total_keyframes,
            "avg_keyframes_per_video": self.avg_keyframes_per_video,
            "min_keyframes": self.min_keyframes,
            "max_keyframes": self.max_keyframes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionStatistics":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            total_videos=int(data["total_videos"]),
            total_frames_scanned=int(data["total_frames_scanned"]),
            total_keyframes=int(data["total_keyframes"]),
            avg_keyframes_per_video=float(data["avg_keyframes_per_video"]),
            min_keyframes=int(data["min_keyframes"]),
            max_keyframes=int(data["max_keyframes"]),
        )


@dataclass(frozen=True)
class ExtractionManifest:
    """Versioned metadata describing one extraction artifact.

    Fields:
        - schema_version: extraction schema version
          (``EXTRACTION_MANIFEST_SCHEMA_VERSION``).
        - extractor: extractor implementation name.
        - extractor_version: extractor version.
        - scorer / frame_reader: scoring and decoding method names.
        - detector / sampler: shot-detector and sampler names (additive;
          ``"n/a"`` for the ``dake`` extractor, which uses neither).
        - parameters: the effective :class:`ExtractionConfig` dict.
        - config_hash: sha256 identity of the effective configuration.
        - source_videos: resolved ``videos.json`` path.
        - num_videos / num_keyframes: artifact counts.
        - statistics: aggregate extraction statistics.
        - created_at: ISO-8601 timestamp of completion.
    """

    schema_version: str
    extractor: str
    extractor_version: str
    scorer: str
    frame_reader: str
    parameters: dict[str, Any]
    config_hash: str
    source_videos: str
    num_videos: int
    num_keyframes: int
    statistics: ExtractionStatistics
    created_at: str
    detector: str = "n/a"
    sampler: str = "n/a"

    def __post_init__(self) -> None:
        _require(
            self.schema_version == EXTRACTION_MANIFEST_SCHEMA_VERSION,
            f"unsupported extraction schema {self.schema_version!r}",
        )
        _require(isinstance(self.extractor, str) and self.extractor != "", "extractor required")
        _require(isinstance(self.extractor_version, str) and self.extractor_version != "", "extractor_version required")
        _require(isinstance(self.scorer, str) and self.scorer != "", "scorer required")
        _require(isinstance(self.frame_reader, str) and self.frame_reader != "", "frame_reader required")
        _require(isinstance(self.detector, str) and self.detector != "", "detector required")
        _require(isinstance(self.sampler, str) and self.sampler != "", "sampler required")
        _require(isinstance(self.parameters, dict), "parameters must be a dict")
        _require(isinstance(self.config_hash, str) and self.config_hash != "", "config_hash required")
        _require(isinstance(self.source_videos, str) and self.source_videos != "", "source_videos required")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.num_keyframes, int) and self.num_keyframes >= 0, "num_keyframes must be >= 0")
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "scorer": self.scorer,
            "frame_reader": self.frame_reader,
            "detector": self.detector,
            "sampler": self.sampler,
            "parameters": self.parameters,
            "config_hash": self.config_hash,
            "source_videos": self.source_videos,
            "num_videos": self.num_videos,
            "num_keyframes": self.num_keyframes,
            "statistics": self.statistics.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionManifest":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            extractor=data["extractor"],
            extractor_version=data["extractor_version"],
            scorer=data["scorer"],
            frame_reader=data["frame_reader"],
            detector=data.get("detector", "n/a"),
            sampler=data.get("sampler", "n/a"),
            parameters=data["parameters"],
            config_hash=data["config_hash"],
            source_videos=data["source_videos"],
            num_videos=int(data["num_videos"]),
            num_keyframes=int(data["num_keyframes"]),
            statistics=ExtractionStatistics.from_dict(data["statistics"]),
            created_at=data["created_at"],
        )


# ---------------------------------------------------------------- checkpoint


@dataclass(frozen=True)
class ExtractionCheckpoint:
    """Resume state persisted after each completed video.

    Fields:
        - schema_version: checkpoint schema version
          (``EXTRACTION_CHECKPOINT_SCHEMA_VERSION``).
        - config_hash: sha256 identity of the effective configuration; a
          resume with a different config is rejected.
        - total_videos: total number of videos in the input manifest.
        - completed_video_ids: video ids whose keyframes are fully staged.
        - frames_scanned_by_video: ``video_id -> source frames scanned`` for
          each completed video (used for aggregate statistics on resume).
        - started_at / timestamp: ISO-8601 creation and last-update stamps.
    """

    schema_version: str
    config_hash: str
    total_videos: int
    completed_video_ids: tuple[str, ...]
    frames_scanned_by_video: dict[str, int]
    started_at: str
    timestamp: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            f"unsupported checkpoint schema {self.schema_version!r}",
        )
        _require(isinstance(self.config_hash, str) and self.config_hash != "", "config_hash required")
        _require(isinstance(self.total_videos, int) and self.total_videos >= 0, "total_videos must be >= 0")
        _require(0 <= len(self.completed_video_ids) <= self.total_videos, "completed_video_ids must fit within total_videos")
        _require(all(isinstance(vid, str) and vid != "" for vid in self.completed_video_ids), "completed_video_ids must be non-empty strings")
        _require(isinstance(self.frames_scanned_by_video, dict), "frames_scanned_by_video must be a dict")
        _require(
            all(isinstance(vid, str) and isinstance(count, int) and count >= 0 for vid, count in self.frames_scanned_by_video.items()),
            "frames_scanned_by_video must map video_id to non-negative ints",
        )
        _require(isinstance(self.started_at, str) and self.started_at != "", "started_at required")
        _require(isinstance(self.timestamp, str) and self.timestamp != "", "timestamp required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "total_videos": self.total_videos,
            "completed_video_ids": list(self.completed_video_ids),
            "frames_scanned_by_video": self.frames_scanned_by_video,
            "started_at": self.started_at,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionCheckpoint":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            config_hash=data["config_hash"],
            total_videos=int(data["total_videos"]),
            completed_video_ids=tuple(data["completed_video_ids"]),
            frames_scanned_by_video=data.get("frames_scanned_by_video", {}),
            started_at=data["started_at"],
            timestamp=data["timestamp"],
        )


# ---------------------------------------------------------------- per-video IO


def write_video_ndjson(path: Path, keyframes: list[Keyframe]) -> None:
    """Write one video's canonical records to a ``.ndjson`` file (atomic).

    Expected input:
        - path: target file path.
        - keyframes: the video's keyframes in ordinal order.

    Returns:
        - None.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for keyframe in keyframes:
            handle.write(json.dumps(keyframe.to_dict()) + "\n")
    tmp.replace(path)


def write_video_map(path: Path, keyframes: list[Keyframe]) -> None:
    """Write one video's ``n,pts_time,fps,frame_idx`` map CSV (atomic).

    The header and column order match ``aic2026.entities.parse_keyframe_map``.

    Expected input:
        - path: target ``<VIDEO_ID>.csv`` path.
        - keyframes: the video's keyframes in ordinal order.

    Returns:
        - None.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("n,pts_time,fps,frame_idx\n")
        for keyframe in keyframes:
            handle.write(
                f"{keyframe.keyframe_ordinal},{keyframe.pts_time},{keyframe.fps},{keyframe.frame_index}\n"
            )
    tmp.replace(path)


def _read_video_ndjson(path: Path) -> list[KeyframeTiming]:
    """Read a per-video staged ``.ndjson`` back into canonical records."""
    timings: list[KeyframeTiming] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                timings.append(KeyframeTiming.from_dict(json.loads(line)))
    return timings


# ---------------------------------------------------------------- validation


def _duplicate_keys(timings: list[KeyframeTiming]) -> list[tuple[str, int]]:
    """Return duplicated ``(video_id, frame_index)`` keys in first-seen order."""
    counts: dict[tuple[str, int], int] = {}
    for timing in timings:
        key = (timing.keyframe.video_id.value, timing.frame_index)
        counts[key] = counts.get(key, 0) + 1
    return [key for key, count in counts.items() if count > 1]


def _validate_staged(output_dir: Path, videos: VideoManifest) -> list[list[KeyframeTiming]]:
    """Validate staged per-video outputs and return them in manifest order.

    Checks every video has staged frames and maps, each staged record parses
    as canonical, no duplicate ``(video_id, frame_index)`` key appears, and
    no keyframe record belongs to an unexpected video.

    Raises:
        - ExtractionError: on any inconsistency.
    """
    staging = output_dir / STAGING_DIRNAME
    if not staging.is_dir():
        raise ExtractionError(f"staging directory missing: {staging}")
    by_id = videos.by_id()
    per_video: list[list[KeyframeTiming]] = []
    seen_keys: set[tuple[str, int]] = set()
    for entry in videos.videos:
        ndjson_path = staging / f"{entry.video_id}.ndjson"
        map_path = staging / f"{entry.video_id}.csv"
        if not ndjson_path.is_file() or not map_path.is_file():
            raise ExtractionError(f"staged output incomplete for video {entry.video_id}: {ndjson_path}, {map_path}")
        timings = _read_video_ndjson(ndjson_path)
        for timing in timings:
            video_id = timing.keyframe.video_id.value
            if video_id != entry.video_id:
                raise ExtractionError(
                    f"staged record for video {video_id!r} found under {entry.video_id}"
                )
            key = (video_id, timing.frame_index)
            if key in seen_keys:
                raise ExtractionError(f"duplicate keyframe key across videos: {key}")
            seen_keys.add(key)
        per_video.append(timings)
    return per_video


def _merge_staged_into_final(
    output_dir: Path,
    per_video: list[list[KeyframeTiming]],
    videos: VideoManifest,
) -> None:
    """Assemble the canonical artifact from validated staged outputs.

    Writes ``frames.ndjson`` and ``map-keyframes/<VIDEO_ID>.csv`` atomically.
    The artifact's record order is the manifest's video order, and each
    video's records keep their ordinal order.

    Raises:
        - ExtractionError: if a video recorded in the manifest has no output.
    """
    frames_path = output_dir / FRAMES_FILENAME
    tmp_frames = frames_path.with_name(frames_path.name + ".tmp")
    with tmp_frames.open("w", encoding="utf-8") as handle:
        for timings in per_video:
            for timing in timings:
                handle.write(json.dumps(timing.to_dict()) + "\n")
    tmp_frames.replace(frames_path)

    maps_dir = output_dir / MAP_KEYFRAMES_DIRNAME
    maps_dir.mkdir(parents=True, exist_ok=True)
    for entry in videos.videos:
        staged_csv = output_dir / STAGING_DIRNAME / f"{entry.video_id}.csv"
        target_csv = maps_dir / f"{entry.video_id}.csv"
        tmp_csv = target_csv.with_name(target_csv.name + ".tmp")
        with staged_csv.open("r", encoding="utf-8") as src, tmp_csv.open("w", encoding="utf-8") as dst:
            dst.write(src.read())
        tmp_csv.replace(target_csv)


# ---------------------------------------------------------------- pipeline


@runtime_checkable
class KeyframeExtractorProtocol(Protocol):
    """Minimal extractor contract the pipeline loop relies on.

    Attributes:
        - extractor_name: extractor implementation name.
        - extractor_version: extractor version string.

    Methods:
        - ``extract(video_path)``: return the video's :class:`Keyframe`
          records.
        - ``last_frames_scanned``: number of source frames scanned by the most
          recent ``extract`` call.
    """

    extractor_name: str
    extractor_version: str

    def extract(self, video_path: Path | str) -> list[Keyframe]:
        """Extract keyframes from a video and return the records."""
        ...

    @property
    def last_frames_scanned(self) -> int:
        """Return the number of source frames scanned by the last extract."""
        ...


def _build_extractor(
    video_id: str,
    config: ExtractionConfig,
    *,
    shot_detector: object | None = None,
) -> KeyframeExtractorProtocol:
    """Create the extractor for one video from the effective configuration.

    The ``omnishotcut_adaptive`` extractor requires a shared ``shot_detector``
    (the OmniShotCut model is loaded once per run and reused across videos).
    """
    frame_reader = create_frame_reader(config.frame_reader, max_side=config.max_side, fps=config.fps)
    scorer = create_scorer(config.scorer, quality=config.jpeg_quality)
    if config.extractor == "omnishotcut_adaptive":
        if shot_detector is None:
            raise ValueError("omnishotcut_adaptive requires a shared shot_detector")
        from aic2026.assets.omnishotcut_extractor import OmniShotCutAdaptiveExtractor

        return OmniShotCutAdaptiveExtractor(
            video_id,
            shot_detector,  # type: ignore[arg-type]
            frame_reader,
            scorer=scorer,
            short_shot_seconds=config.short_shot_seconds,
            long_shot_seconds=config.long_shot_seconds,
            uniform_interval_seconds=config.uniform_interval_seconds,
            min_temporal_distance=config.min_temporal_distance,
            peak_threshold=config.peak_threshold,
            fps=config.fps,
        )
    return DakeKeyframeExtractor(
        video_id,
        frame_reader,
        scorer=scorer,
        min_temporal_distance=config.min_temporal_distance,
        peak_threshold=config.peak_threshold,
        target=config.target_keyframes_per_video,
        sampling_ratio=config.sampling_ratio,
        anchor_interval_seconds=config.anchor_interval_seconds,
    )


def extract_keyframes(
    videos_path: Path | str,
    output_dir: Path | str,
    config: ExtractionConfig,
    *,
    progress: ProgressReporter | None = None,
    resume: bool = True,
    overwrite: bool = False,
    video_root: Path | str | None = None,
) -> ExtractionManifest:
    """Run resumable keyframe extraction over a video manifest.

    Expected input:
        - videos_path: path to a ``videos.json`` manifest.
        - output_dir: target directory for the canonical artifact.
        - config: the validated :class:`ExtractionConfig`.
        - progress: optional :class:`ProgressReporter` (unit should be
          ``videos``) to update per completed video.
        - resume: if True, skip videos already completed in a checkpoint that
          matches ``config``; a config mismatch on resume is rejected.
        - overwrite: if True, replace an existing artifact; otherwise an
          existing non-empty output without a matching complete manifest
          raises :class:`ExtractionError`.
        - video_root: optional root directory that relative video paths in the
          manifest are resolved against (overrides the manifest's
          ``source_dir``). Absolute paths are used unchanged, so legacy
          manifests keep working without this flag.

    Returns:
        - The :class:`ExtractionManifest` describing the produced artifact.

    Raises:
        - FileNotFoundError: if ``videos_path`` does not exist.
        - ValueError: if the manifest or configuration is invalid.
        - ExtractionError: on a config mismatch on resume, an incomplete
          artifact, or invalid staged output.
        - OSError: if outputs cannot be written.
    """
    videos_path = Path(videos_path)
    output_dir = Path(output_dir)
    config_hash = config.config_hash()
    manifest_path = output_dir / MANIFEST_FILENAME
    checkpoint_path = output_dir / CHECKPOINT_FILENAME

    videos = load_video_manifest(videos_path)
    total_videos = len(videos.videos)

    # A complete, matching artifact is idempotently reused.
    if manifest_path.is_file() and not overwrite:
        existing = ExtractionManifest.from_dict(read_json_object(manifest_path))
        if existing.config_hash == config_hash and (output_dir / FRAMES_FILENAME).is_file():
            logger.info("artifact already complete at %s; skipping extraction", output_dir)
            return existing

    # Fail fast on missing media tools before the (potentially long) run.
    if config.frame_reader == "ffmpeg" or (
        config.extractor == "omnishotcut_adaptive" and config.detector == "omnishotcut"
    ):
        require_media_tools(ffmpeg=True, ffprobe=True)

    # A matching checkpoint makes the output a legitimate resume target even
    # though the directory is not empty; anything else non-empty is refused
    # unless overwrite was requested.
    checkpoint: ExtractionCheckpoint | None = None
    if resume and checkpoint_path.is_file():
        checkpoint = ExtractionCheckpoint.from_dict(read_json_object(checkpoint_path))
        if checkpoint.config_hash != config_hash:
            raise ExtractionError(
                f"checkpoint config hash {checkpoint.config_hash} != current {config_hash}; "
                "cannot resume with a different configuration (refusing to mix artifacts)"
            )
        if checkpoint.total_videos != total_videos:
            raise ExtractionError(
                f"checkpoint total_videos {checkpoint.total_videos} != manifest {total_videos}"
            )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite and checkpoint is None:
        raise ExtractionError(
            f"extraction output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )
    if overwrite and output_dir.exists() and checkpoint is None:
        for child in list(output_dir.iterdir()):
            if child.name in (MANIFEST_FILENAME, CHECKPOINT_FILENAME, FRAMES_FILENAME, MAP_KEYFRAMES_DIRNAME, STAGING_DIRNAME):
                if child.is_dir():
                    for sub in list(child.iterdir()):
                        sub.unlink()
                child.unlink() if child.is_file() else child.rmdir()

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    frames_scanned: dict[str, int] = {}
    started_at = datetime.now(timezone.utc).isoformat()
    if checkpoint is not None:
        known = set(checkpoint.completed_video_ids)
        unknown = known - set(videos.by_id())
        if unknown:
            raise ExtractionError(f"checkpoint references unknown videos: {sorted(unknown)}")
        completed_ids = known
        frames_scanned = dict(checkpoint.frames_scanned_by_video)
        started_at = checkpoint.started_at
        logger.info("resuming extraction with %d/%d videos already completed", len(completed_ids), len(videos.videos))

    def write_checkpoint() -> None:
        write_json_atomic(
            checkpoint_path,
            ExtractionCheckpoint(
                schema_version=EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                config_hash=config_hash,
                total_videos=len(videos.videos),
                completed_video_ids=tuple(sorted(completed_ids)),
                frames_scanned_by_video=dict(sorted(frames_scanned.items())),
                started_at=started_at,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict(),
        )

    if progress is not None:
        progress.set_total(len(videos.videos), len(completed_ids))

    # The shot detector is built once per run and shared across videos so the
    # (heavy) OmniShotCut model is loaded a single time. Its construction is
    # lazy: no checkpoint is downloaded until the first video is detected.
    shot_detector = None
    if config.extractor == "omnishotcut_adaptive":
        shot_detector = create_shot_detector(
            config.detector,
            checkpoint=config.checkpoint,
            filename=config.omnishotcut_filename,
            mode=config.omnishotcut_mode,
            overlap=config.omnishotcut_overlap,
            device=config.device,
        )
        logger.info(
            "created shared shot detector %r once for %d videos",
            config.detector,
            total_videos,
        )

    completed_video = 0
    session_keyframes = 0
    for entry in videos.videos:
        if entry.video_id in completed_ids:
            completed_video += 1
            continue
        video_path = resolve_video_path(
            entry.path,
            source_dir=videos.source_dir,
            video_root=video_root,
        )
        extractor = _build_extractor(entry.video_id, config, shot_detector=shot_detector)
        video_started = time.monotonic()
        keyframes = extractor.extract(video_path)
        session_keyframes += len(keyframes)
        frames_scanned[entry.video_id] = extractor.last_frames_scanned
        write_video_ndjson(staging / f"{entry.video_id}.ndjson", keyframes)
        write_video_map(staging / f"{entry.video_id}.csv", keyframes)
        completed_ids.add(entry.video_id)
        completed_video += 1
        write_checkpoint()
        if progress is not None:
            progress.update(
                completed_video,
                note=f"video={entry.video_id} frames={extractor.last_frames_scanned} keyframes={len(keyframes)}",
                keyframes=session_keyframes,
            )
        logger.info(
            "extracted %d keyframes from %d frames for %s (%.1fs)",
            len(keyframes),
            extractor.last_frames_scanned,
            entry.video_id,
            time.monotonic() - video_started,
        )

    if len(completed_ids) != len(videos.videos):
        raise ExtractionError("some videos were not completed")

    per_video = _validate_staged(output_dir, videos)
    _merge_staged_into_final(output_dir, per_video, videos)

    total_frames_scanned = sum(frames_scanned.values())
    counts = [len(timings) for timings in per_video]
    total_keyframes = sum(counts)
    statistics = ExtractionStatistics(
        total_videos=len(videos.videos),
        total_frames_scanned=total_frames_scanned,
        total_keyframes=total_keyframes,
        avg_keyframes_per_video=float(total_keyframes / len(videos.videos)) if videos.videos else 0.0,
        min_keyframes=min(counts) if counts else 0,
        max_keyframes=max(counts) if counts else 0,
    )
    manifest = ExtractionManifest(
        schema_version=EXTRACTION_MANIFEST_SCHEMA_VERSION,
        extractor=config.extractor,
        extractor_version=config.extractor_version,
        scorer=config.scorer,
        frame_reader=config.frame_reader,
        detector=config.detector if config.extractor == "omnishotcut_adaptive" else "n/a",
        sampler=config.sampler if config.extractor == "omnishotcut_adaptive" else "n/a",
        parameters=config.to_dict(),
        config_hash=config_hash,
        source_videos=str(videos_path.resolve()),
        num_videos=len(videos.videos),
        num_keyframes=total_keyframes,
        statistics=statistics,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(manifest_path, manifest.to_dict())
    # Staging and the checkpoint are no longer needed after the artifact is
    # assembled; only the final outputs remain.
    for child in list(staging.iterdir()):
        child.unlink()
    staging.rmdir()
    checkpoint_path.unlink(missing_ok=True)
    if progress is not None:
        progress.finish(note=f"total keyframes={total_keyframes}")
    logger.info(
        "extraction complete: %d keyframes across %d videos -> %s",
        total_keyframes,
        len(videos.videos),
        output_dir,
    )
    return manifest