"""DAKE-inspired lightweight keyframe extraction.

Implements a modular, production-ready keyframe extractor that produces the
same canonical keyframe contract consumed by the embedding/retrieval pipeline
(``frames.ndjson`` records matching :class:`aic2026.canonical.KeyframeTiming`
plus per-video ``map-keyframes/<VIDEO_ID>.csv`` files). It is designed to
replace the BTC-provided keyframe source without touching retrieval, scoring,
evaluation, index building, or embedding code.

Pipeline (V1, DAKE-inspired but independently implemented)::

    video frames
        -> frame-level change score
        -> candidate peaks
        -> temporal suppression
        -> selected keyframes

Design notes:

- The change-scoring method is pluggable through the :class:`ChangeScorer`
  protocol (V1 ships :class:`FrameDifferenceScorer`, a lightweight mean-abs
  difference over downscaled grayscale frames).
- Frame acquisition is pluggable through the :class:`FrameReader` protocol.
  V1 ships :class:`FFmpegFrameReader` (decodes a video to downscaled grayscale
  frames; assumes a constant frame rate so ``frame_index == decoded position``
  and ``pts_time = frame_index / fps``) and :class:`SyntheticFrameReader`
  (reads a lightweight JSON descriptor for deterministic, synthetic tests).
- Selection is deterministic: identical inputs produce identical keyframes.
- A video always yields at least one keyframe (frame 0), and no selected
  keyframe is closer than ``min_temporal_distance`` to another.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId

logger = logging.getLogger(__name__)

#: Schema version of the synthetic video descriptor.
SYNTHETIC_VIDEO_SCHEMA_VERSION = "1.0"

#: Valid ``--frame-reader`` CLI values.
FRAME_READER_NAMES = ("ffmpeg", "synthetic")

#: Valid scoring method names.
SCORER_NAMES = ("frame_difference", "jpeg_size")


class KeyframeExtractionError(RuntimeError):
    """Raised when keyframe extraction fails or produces invalid output."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


# ------------------------------------------------------------------ contracts


@dataclass(frozen=True)
class VideoFrame:
    """One decoded video frame fed to the change scorer.

    Fields:
        - frame_index: 0-based index of the frame in the original video.
        - pts_time: presentation timestamp in seconds.
        - fps: frame rate of the source video.
        - image: downscaled grayscale float array ``(H, W)`` in ``[0, 255]``,
          used by image-based scorers.
    """

    frame_index: int
    pts_time: float
    fps: float
    image: np.ndarray | None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.frame_index, int) and self.frame_index >= 0,
            "frame_index must be a non-negative integer",
        )
        _require(float(self.pts_time) >= 0.0, "pts_time must be >= 0")
        _require(float(self.fps) > 0.0, "fps must be > 0")
        _require(
            self.image is None or (isinstance(self.image, np.ndarray) and self.image.ndim == 2),
            "image must be a 2-D array or None",
        )


@dataclass(frozen=True)
class Keyframe:
    """One selected keyframe in canonical keyframe-record form.

    Fields:
        - video_id: the ``L*_V*`` video id.
        - keyframe_ordinal: 1-based ordinal within the video.
        - pts_time: presentation timestamp in seconds.
        - fps: frame rate of the source video.
        - frame_index: 0-based index in the original video.
    """

    video_id: str
    keyframe_ordinal: int
    pts_time: float
    fps: float
    frame_index: int

    def __post_init__(self) -> None:
        VideoId(self.video_id)  # validates the id pattern
        _require(
            isinstance(self.keyframe_ordinal, int) and not isinstance(self.keyframe_ordinal, bool) and self.keyframe_ordinal >= 1,
            "keyframe_ordinal must be a positive integer",
        )
        _require(float(self.pts_time) >= 0.0, "pts_time must be >= 0")
        _require(float(self.fps) > 0.0, "fps must be > 0")
        _require(
            isinstance(self.frame_index, int) and not isinstance(self.frame_index, bool) and self.frame_index >= 0,
            "frame_index must be a non-negative integer",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical ``frames.ndjson`` record.

        Returns:
            - A dict matching ``KeyframeTiming.to_dict()`` so downstream
              stages consume it without changes.
        """
        return self.to_timing().to_dict()

    def to_timing(self) -> KeyframeTiming:
        """Convert to a canonical :class:`KeyframeTiming`.

        Returns:
            - The canonical timing record.
        """
        return KeyframeTiming(
            keyframe=KeyframeKey(video_id=VideoId(self.video_id), keyframe_ordinal=self.keyframe_ordinal),
            pts_time=self.pts_time,
            fps=self.fps,
            frame_index=self.frame_index,
        )

    @classmethod
    def from_timing(cls, timing: KeyframeTiming) -> "Keyframe":
        """Convert a canonical :class:`KeyframeTiming` back to a :class:`Keyframe`."""
        return cls(
            video_id=timing.keyframe.video_id.value,
            keyframe_ordinal=timing.keyframe.keyframe_ordinal,
            pts_time=timing.pts_time,
            fps=timing.fps,
            frame_index=timing.frame_index,
        )


@runtime_checkable
class FrameReader(Protocol):
    """Protocol for yielding decoded frames from a video path."""

    def read(self, video_path: Path | str) -> Iterator[VideoFrame]:
        """Yield one :class:`VideoFrame` per decoded frame, in order.

        Raises:
            - FileNotFoundError: if the video is absent.
            - KeyframeExtractionError: if the video cannot be decoded.
        """


@runtime_checkable
class ChangeScorer(Protocol):
    """Protocol for computing one change score per frame.

    Scores are expected to be non-negative; a higher score marks a stronger
    visual change. The scorer consumes the frame iterator and returns one
    score per yielded frame (including the first frame, conventionally 0).
    """

    name: str

    def compute(self, frames: Iterable[VideoFrame]) -> np.ndarray:
        """Return ``scores[i]`` for the ``i``-th yielded frame.

        Raises:
            - KeyframeExtractionError: if a frame cannot be scored.
        """


# ------------------------------------------------------------- frame readers


class FFmpegFrameReader:
    """Decode a video to downscaled grayscale frames via the ``ffmpeg`` CLI.

    Frames are decoded at the source frame rate (``-fps_mode passthrough``).
    Because the decoder processes every source frame in order, ``frame_index``
    is the decoded position and ``pts_time = frame_index / fps``; this assumes
    a constant frame rate (true for the current dataset). Requires ``ffmpeg``
    on ``PATH``; a missing binary raises :class:`FileNotFoundError`.

    Expected input:
        - max_side: largest allowed side (pixels) after downscaling.
        - fps: optional override of the video frame rate (otherwise probed).

    Returns:
        - A frame reader instance.
    """

    def __init__(self, *, max_side: int = 64, fps: float | None = None) -> None:
        if not isinstance(max_side, int) or isinstance(max_side, bool) or max_side < 1:
            raise ValueError(f"max_side must be a positive integer, got {max_side!r}")
        if fps is not None and float(fps) <= 0.0:
            raise ValueError("fps must be > 0 when provided")
        self._max_side = max_side
        self._fps_override = float(fps) if fps is not None else None

    def read(self, video_path: Path | str) -> Iterator[VideoFrame]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        fps = self._fps_override
        width, height = 0, 0
        if fps is None or width == 0:
            probed_fps, probed_w, probed_h = self._probe(path)
            if fps is None:
                fps = probed_fps
            width, height = probed_w, probed_h
        scaled_w, scaled_h = self._scaled_size(width, height)

        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-fps_mode",
            "passthrough",
            "-vf",
            f"scale={scaled_w}:{scaled_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ffmpeg is required to decode videos but was not found on PATH"
            ) from exc

        assert process.stdout is not None
        assert process.stderr is not None
        frame_bytes = scaled_w * scaled_h
        frame_index = 0
        try:
            while True:
                buffer = process.stdout.read(frame_bytes)
                if not buffer or len(buffer) < frame_bytes:
                    break
                image = np.frombuffer(buffer, dtype=np.uint8).reshape(scaled_h, scaled_w).astype(np.float32)
                yield VideoFrame(
                    frame_index=frame_index,
                    pts_time=frame_index / fps,
                    fps=fps,
                    image=image,
                )
                frame_index += 1
        finally:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            process.wait()
            if process.returncode != 0 and frame_index == 0:
                raise KeyframeExtractionError(
                    f"ffmpeg failed to decode {path}: {stderr.strip()}"
                )

    @staticmethod
    def _probe(path: Path) -> tuple[float, int, int]:
        """Probe fps, width, and height via ``ffprobe``.

        Returns:
            - ``(fps, width, height)`` for the first video stream.

        Raises:
            - FileNotFoundError: if ``ffprobe`` is not on ``PATH``.
            - KeyframeExtractionError: if probing fails.
        """
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ffprobe is required to decode videos but was not found on PATH"
            ) from exc
        if result.returncode != 0:
            raise KeyframeExtractionError(f"ffprobe failed on {path}: {result.stderr.strip()}")
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise KeyframeExtractionError(f"no video stream found in {path}")
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        rate = stream.get("avg_frame_rate", "0/1")
        try:
            numerator, denominator = rate.split("/")
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError) as exc:
            raise KeyframeExtractionError(f"cannot parse frame rate {rate!r} for {path}") from exc
        if fps <= 0.0:
            raise KeyframeExtractionError(f"invalid frame rate {rate!r} for {path}")
        return fps, width, height

    def _scaled_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the downscaled ``(width, height)`` preserving aspect ratio."""
        if width <= 0 or height <= 0:
            return self._max_side, self._max_side
        scale = min(self._max_side / width, self._max_side / height, 1.0)
        scaled_w = max(1, round(width * scale))
        scaled_h = max(1, round(height * scale))
        return scaled_w, scaled_h


class SyntheticFrameReader:
    """Read a synthetic video descriptor for deterministic tests.

    The descriptor is a JSON file::

        {
          "schema_version": "1.0",
          "fps": 25.0,
          "width": 8,
          "height": 8,
          "frames": [
            {"pts_time": 0.0, "value": 10.0},
            ...
          ]
        }

    Each frame is a constant grayscale image filled with ``value``; the
    frame-difference scorer then measures ``|value_i - value_{i-1}|``, giving
    deterministic, easy-to-assert change peaks. ``frame_index`` is the row
    position. No video codec is required.
    """

    def __init__(self) -> None:
        pass

    def read(self, video_path: Path | str) -> Iterator[VideoFrame]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"synthetic video file does not exist: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise KeyframeExtractionError(f"invalid synthetic video JSON {path}: {exc}") from exc
        if payload.get("schema_version") != SYNTHETIC_VIDEO_SCHEMA_VERSION:
            raise KeyframeExtractionError(
                f"unsupported synthetic video schema {payload.get('schema_version')!r}: {path}"
            )
        fps = float(payload["fps"])
        width = int(payload["width"])
        height = int(payload["height"])
        if fps <= 0.0 or width < 1 or height < 1:
            raise KeyframeExtractionError(f"invalid synthetic video dimensions/fps: {path}")
        for frame_index, row in enumerate(payload["frames"]):
            value = float(row["value"])
            pts_time = float(row.get("pts_time", frame_index / fps))
            image = np.full((height, width), value, dtype=np.float32)
            yield VideoFrame(
                frame_index=frame_index,
                pts_time=pts_time,
                fps=fps,
                image=image,
            )


def write_synthetic_video(
    path: Path | str,
    values: Sequence[float],
    *,
    fps: float = 25.0,
    width: int = 8,
    height: int = 8,
    pts_times: Sequence[float] | None = None,
) -> Path:
    """Write a synthetic video descriptor file.

    Expected input:
        - path: target descriptor path.
        - values: one constant grayscale value per frame.
        - fps / width / height: descriptor metadata.
        - pts_times: optional per-frame presentation times (defaults to
          ``frame_index / fps``).

    Returns:
        - The written path.

    Raises:
        - ValueError: if ``values`` is empty or ``pts_times`` mismatches.
    """
    path = Path(path)
    if not values:
        raise ValueError("values must contain at least one frame")
    if pts_times is not None and len(pts_times) != len(values):
        raise ValueError("pts_times must match the number of frames")
    if width < 1 or height < 1 or fps <= 0.0:
        raise ValueError("width/height must be >= 1 and fps > 0")
    frames = [
        {"pts_time": float(pts_times[i] if pts_times is not None else i / fps), "value": float(values[i])}
        for i in range(len(values))
    ]
    payload = {
        "schema_version": SYNTHETIC_VIDEO_SCHEMA_VERSION,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
    }
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- change scorer


class FrameDifferenceScorer:
    """Lightweight visual-difference scorer (V1).

    ``score[i]`` is the mean absolute difference between frame ``i`` and frame
    ``i - 1`` over the (downscaled grayscale) image; ``score[0] == 0``. This
    is deterministic and cheap, and requires no model.
    """

    name: str = "frame_difference"

    def compute(self, frames: Iterable[VideoFrame]) -> np.ndarray:
        scores: list[float] = []
        previous: np.ndarray | None = None
        for frame in frames:
            if frame.image is None:
                raise KeyframeExtractionError(
                    f"frame {frame.frame_index} has no image for frame-difference scoring"
                )
            image = frame.image
            if previous is None:
                scores.append(0.0)
            else:
                scores.append(float(np.mean(np.abs(image - previous))))
            previous = image
        return np.asarray(scores, dtype=np.float64)


def _jpeg_size(image: np.ndarray, quality: int) -> int:
    """Return the in-memory JPEG byte size of a grayscale image.

    Expected input:
        - image: 2-D float array in ``[0, 255]``.
        - quality: JPEG quality in ``[1, 100]``.

    Returns:
        - The JPEG-encoded byte length.

    Raises:
        - ImportError: if Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "jpeg_size scoring requires Pillow; install pillow"
        ) from exc
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="JPEG", quality=quality)
    return int(buffer.tell())


class JPEGSizeChangeScorer:
    """DAKE-inspired JPEG-compressed-size change scorer (V2).

    For each frame, the downscaled grayscale image is JPEG-encoded in memory
    and its compressed size is measured. ``score[i]`` is the relative change
    in compressed size between frame ``i`` and frame ``i - 1`` (a percentage
    of the larger size), and ``score[0] == 0``. A large change in compressed
    size between consecutive frames marks a strong visual change; this is a
    deterministic, model-free, GPU-free proxy (inspired by DAKE/U-CESE, which
    use JPEG size as a cheap informativeness signal).

    Determinism: for identical input images and the same Pillow version the
    encoder output is deterministic, so repeated runs yield identical scores.

    Expected input:
        - quality: JPEG quality in ``[1, 100]`` used for encoding (default 70).
    """

    name: str = "jpeg_size"

    def __init__(self, quality: int = 70) -> None:
        if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
            raise ValueError(f"quality must be an integer in [1, 100], got {quality!r}")
        self._quality = quality

    def compute(self, frames: Iterable[VideoFrame]) -> np.ndarray:
        scores: list[float] = []
        previous: int | None = None
        for frame in frames:
            if frame.image is None:
                raise KeyframeExtractionError(
                    f"frame {frame.frame_index} has no image for jpeg_size scoring"
                )
            size = _jpeg_size(frame.image, self._quality)
            if previous is None:
                scores.append(0.0)
            else:
                denom = max(previous, size, 1)
                scores.append(100.0 * abs(size - previous) / denom)
            previous = size
        return np.asarray(scores, dtype=np.float64)


# ------------------------------------------------------------ keyframe selection


def candidate_peaks(scores: np.ndarray, peak_threshold: float) -> list[int]:
    """Return indices of strict local maxima with score above the threshold.

    A peak at ``i`` requires ``scores[i] > peak_threshold`` and
    ``scores[i] >= scores[i-1]`` and ``scores[i] > scores[i+1]`` (first and
    last frames are never peaks, matching a strict local-maximum definition).

    Expected input:
        - scores: per-frame change scores.
        - peak_threshold: minimum score for a candidate.

    Returns:
        - Sorted candidate indices.
    """
    n = len(scores)
    peaks: list[int] = []
    for i in range(1, n - 1):
        if scores[i] > peak_threshold and scores[i] >= scores[i - 1] and scores[i] > scores[i + 1]:
            peaks.append(i)
    return peaks


def compute_anchor_frames(
    num_frames: int,
    interval_frames: int | None = None,
) -> list[int]:
    """Compute hybrid temporal anchor frame indices.

    The first and last frames are always anchors. When ``interval_frames``
    is given (and >= 1), every ``interval_frames``-th frame strictly between
    the first and last frame is also an anchor.

    Expected input:
        - num_frames: total number of frames (>= 0).
        - interval_frames: optional uniform anchor spacing, or ``None`` for
          no interval anchors.

    Returns:
        - Sorted anchor indices; ``[0, n-1]`` for ``n >= 1``, ``[]`` for
          ``n == 0``.

    Raises:
        - ValueError: on invalid ``num_frames`` or ``interval_frames``.
    """
    if num_frames < 0:
        raise ValueError(f"num_frames must be >= 0, got {num_frames!r}")
    if interval_frames is not None and (
        not isinstance(interval_frames, int) or isinstance(interval_frames, bool) or interval_frames < 1
    ):
        raise ValueError(f"interval_frames must be a positive integer or None, got {interval_frames!r}")
    if num_frames == 0:
        return []
    anchors = {0, num_frames - 1}
    if interval_frames is not None:
        anchors.update(range(interval_frames, num_frames - 1, interval_frames))
    return sorted(anchors)


def temporal_suppression(
    peaks: Sequence[int],
    scores: np.ndarray,
    *,
    min_temporal_distance: int,
    max_keyframes: int | None = None,
    occupied: Sequence[int] = (),
) -> list[int]:
    """Greedily select peaks while enforcing a minimum temporal distance.

    Peaks are considered in descending score order (ties broken by ascending
    index). A peak is accepted when it is at least ``min_temporal_distance``
    frames from every already-accepted index and from every ``occupied``
    index (anchors). At most ``max_keyframes`` indices are returned
    (excluding anchors).

    Expected input:
        - peaks: candidate peak indices.
        - scores: per-frame change scores (for ordering).
        - min_temporal_distance: minimum allowed frame gap.
        - max_keyframes: optional cap on accepted peaks.
        - occupied: indices that are already selected (e.g. anchors) and that
          peaks must keep their distance from.

    Returns:
        - Accepted peak indices in ascending order.

    Raises:
        - ValueError: if ``min_temporal_distance`` is invalid.
    """
    if min_temporal_distance < 1:
        raise ValueError("min_temporal_distance must be >= 1")
    ordered = sorted(peaks, key=lambda i: (-scores[i], i))
    accepted: list[int] = []
    for index in ordered:
        if max_keyframes is not None and len(accepted) >= max_keyframes:
            break
        if all(abs(index - other) >= min_temporal_distance for other in accepted):
            if all(abs(index - other) >= min_temporal_distance for other in occupied):
                accepted.append(index)
    return sorted(accepted)


def select_keyframes(
    scores: np.ndarray,
    *,
    min_temporal_distance: int,
    peak_threshold: float,
    target: int | None = None,
    anchor_interval_frames: int | None = None,
) -> list[int]:
    """Select keyframe indices from per-frame change scores.

    Anchor frames are always kept: frame 0, the last frame, and (when
    ``anchor_interval_frames`` is given) every ``anchor_interval_frames``-th
    frame between them. Candidate peaks are then suppressed by
    ``min_temporal_distance`` and must also stay ``min_temporal_distance``
    frames away from anchors. When ``target`` is given, the best
    (highest-scoring) peaks are kept so the total (anchors + peaks) does not
    exceed ``target``; if anchors alone reach or exceed ``target``, no peaks
    are added.

    Expected input:
        - scores: per-frame change scores.
        - min_temporal_distance: minimum frame gap between selected keyframes.
        - peak_threshold: minimum score for a candidate peak.
        - target: optional maximum number of keyframes (``None`` = no cap).
        - anchor_interval_frames: optional uniform anchor spacing
          (``None`` = first and last frame only).

    Returns:
        - Sorted keyframe frame indices (anchors included).

    Raises:
        - ValueError: if ``target`` is invalid.
    """
    if target is not None and (not isinstance(target, int) or isinstance(target, bool) or target < 1):
        raise ValueError(f"target must be a positive integer or None, got {target!r}")
    if not len(scores):
        return []
    anchors = compute_anchor_frames(len(scores), anchor_interval_frames)
    max_peaks = None if target is None else max(0, target - len(anchors))
    accepted = temporal_suppression(
        candidate_peaks(scores, peak_threshold),
        scores,
        min_temporal_distance=min_temporal_distance,
        max_keyframes=max_peaks,
        occupied=anchors,
    )
    return sorted(set(anchors) | set(accepted))


# ------------------------------------------------------------- extractor


class KeyframeExtractor(Protocol):
    """Protocol for extracting keyframes from a video path."""

    extractor_name: str
    extractor_version: str

    def extract(self, video_path: Path | str) -> list[Keyframe]:
        """Extract keyframes from a video.

        Expected input:
            - video_path: path to the video to process.

        Returns:
            - The selected :class:`Keyframe` records, ordered by frame index,
              with ordinals ``1..N``.

        Raises:
            - FileNotFoundError: if the video is absent.
            - KeyframeExtractionError: if decoding or scoring fails.
        """


class DakeKeyframeExtractor:
    """V2 DAKE-inspired keyframe extractor.

    Expected input:
        - video_id: the ``L*_V*`` video id stamped on every :class:`Keyframe`.
        - frame_reader: a :class:`FrameReader` producing decoded frames.
        - scorer: a :class:`ChangeScorer` (default frame-difference; the
          ``jpeg_size`` scorer is the DAKE-style JPEG-size-change scorer).
        - min_temporal_distance: minimum frame gap between keyframes.
        - peak_threshold: minimum score for a candidate peak.
        - target: optional maximum keyframes per video (``None`` = no cap).
        - anchor_interval_seconds: optional uniform anchor spacing in seconds
          (first and last frames are always anchors).

    Returns:
        - An extractor instance.

    Raises:
        - ValueError: on invalid parameters.
    """

    extractor_name: str = "dake"
    extractor_version: str = "2.0"

    def __init__(
        self,
        video_id: str,
        frame_reader: FrameReader,
        *,
        scorer: ChangeScorer | None = None,
        min_temporal_distance: int = 30,
        peak_threshold: float = 8.0,
        target: int | None = None,
        sampling_ratio: float | None = None,
        anchor_interval_seconds: float | None = None,
    ) -> None:
        VideoId(video_id)  # validates the id pattern
        if not isinstance(min_temporal_distance, int) or isinstance(min_temporal_distance, bool) or min_temporal_distance < 1:
            raise ValueError(f"min_temporal_distance must be >= 1, got {min_temporal_distance!r}")
        if float(peak_threshold) < 0.0:
            raise ValueError("peak_threshold must be >= 0")
        if target is not None and (not isinstance(target, int) or isinstance(target, bool) or target < 1):
            raise ValueError(f"target must be a positive integer or None, got {target!r}")
        if sampling_ratio is not None and not 0.0 < float(sampling_ratio) <= 1.0:
            raise ValueError(f"sampling_ratio must be in (0, 1], got {sampling_ratio!r}")
        if target is not None and sampling_ratio is not None:
            raise ValueError("target and sampling_ratio are mutually exclusive")
        if anchor_interval_seconds is not None and float(anchor_interval_seconds) <= 0.0:
            raise ValueError(f"anchor_interval_seconds must be > 0, got {anchor_interval_seconds!r}")
        self._video_id = video_id
        self._frame_reader = frame_reader
        self._scorer = scorer if scorer is not None else FrameDifferenceScorer()
        self._min_temporal_distance = min_temporal_distance
        self._peak_threshold = float(peak_threshold)
        self._target = target
        self._sampling_ratio = float(sampling_ratio) if sampling_ratio is not None else None
        self._anchor_interval_seconds = (
            float(anchor_interval_seconds) if anchor_interval_seconds is not None else None
        )
        #: Number of source frames scanned by the most recent :meth:`extract`.
        self._last_frames_scanned = 0

    @property
    def video_id(self) -> str:
        """Return the video id this extractor stamps keyframes with."""
        return self._video_id

    @property
    def last_frames_scanned(self) -> int:
        """Return the number of source frames scanned by the last extract() call.

        Returns:
            - Total frames consumed from the frame reader (>= 0).
        """
        return self._last_frames_scanned

    def extract(self, video_path: Path | str) -> list[Keyframe]:
        """Extract keyframes from a video.

        Expected input:
            - video_path: path to the video to process.

        Returns:
            - Selected :class:`Keyframe` records ordered by frame index.

        Raises:
            - FileNotFoundError: if the video is absent.
            - KeyframeExtractionError: if decoding or scoring fails.
        """
        scores, pts_times, fps_values, frame_indices = self._compute_scores(video_path)
        self._last_frames_scanned = int(len(scores))
        effective_target = self._target
        if effective_target is None and self._sampling_ratio is not None:
            effective_target = max(1, round(len(scores) * self._sampling_ratio))
        anchor_interval_frames = None
        if self._anchor_interval_seconds is not None:
            fps = fps_values[0] if fps_values else None
            if fps is None or fps <= 0.0:
                raise KeyframeExtractionError(
                    "anchor_interval_seconds requires a positive per-frame fps"
                )
            anchor_interval_frames = max(1, round(self._anchor_interval_seconds * fps))
        indices = select_keyframes(
            scores,
            min_temporal_distance=self._min_temporal_distance,
            peak_threshold=self._peak_threshold,
            target=effective_target,
            anchor_interval_frames=anchor_interval_frames,
        )
        keyframes: list[Keyframe] = []
        for ordinal, frame_index in enumerate(indices, start=1):
            keyframes.append(
                Keyframe(
                    video_id=self._video_id,
                    keyframe_ordinal=ordinal,
                    pts_time=pts_times[frame_index],
                    fps=fps_values[frame_index],
                    frame_index=frame_indices[frame_index],
                )
            )
        return keyframes

    def _compute_scores(self, video_path: Path | str) -> tuple[np.ndarray, list[float], list[float], list[int]]:
        """Score all frames, recording per-frame metadata.

        Returns:
            - ``(scores, pts_times, fps_values, frame_indices)`` aligned by
              score position.
        """
        pts_times: list[float] = []
        fps_values: list[float] = []
        frame_indices: list[int] = []

        def metadata_wrapper() -> Iterator[VideoFrame]:
            for frame in self._frame_reader.read(video_path):
                pts_times.append(frame.pts_time)
                fps_values.append(frame.fps)
                frame_indices.append(frame.frame_index)
                yield frame

        scores = self._scorer.compute(metadata_wrapper())
        return scores, pts_times, fps_values, frame_indices


def create_frame_reader(
    name: str,
    *,
    max_side: int = 64,
    fps: float | None = None,
) -> FrameReader:
    """Create a :class:`FrameReader` by name.

    Expected input:
        - name: ``ffmpeg`` (real videos) or ``synthetic`` (test descriptors).
        - max_side / fps: parameters forwarded to the reader.

    Returns:
        - A frame reader instance.

    Raises:
        - ValueError: if ``name`` is not a known reader.
    """
    if name not in FRAME_READER_NAMES:
        raise ValueError(f"unsupported frame reader {name!r}; expected {FRAME_READER_NAMES}")
    if name == "synthetic":
        return SyntheticFrameReader()
    return FFmpegFrameReader(max_side=max_side, fps=fps)


def create_scorer(name: str, *, quality: int = 70) -> ChangeScorer:
    """Create a :class:`ChangeScorer` by name.

    Expected input:
        - name: a known scorer name.
        - quality: JPEG quality in ``[1, 100]`` for the ``jpeg_size`` scorer.

    Returns:
        - A scorer instance.

    Raises:
        - ValueError: if ``name`` is not a known scorer or ``quality`` is
          invalid.
    """
    if name not in SCORER_NAMES:
        raise ValueError(f"unsupported scorer {name!r}; expected {SCORER_NAMES}")
    if name == "jpeg_size":
        return JPEGSizeChangeScorer(quality=quality)
    return FrameDifferenceScorer()


def extract_video_keyframes(
    video_id: str,
    video_path: Path | str,
    *,
    frame_reader: FrameReader,
    scorer: ChangeScorer | None = None,
    min_temporal_distance: int = 30,
    peak_threshold: float = 8.0,
    target: int | None = None,
    sampling_ratio: float | None = None,
    anchor_interval_seconds: float | None = None,
) -> list[Keyframe]:
    """Convenience wrapper: extract keyframes from one video.

    Expected input:
        - video_id: the ``L*_V*`` video id.
        - video_path: path to the video.
        - frame_reader / scorer / selection parameters: see
          :class:`DakeKeyframeExtractor`.
        - anchor_interval_seconds: optional uniform anchor spacing in seconds.

    Returns:
        - Selected :class:`Keyframe` records ordered by frame index.

    Raises:
        - FileNotFoundError: if the video is absent.
        - KeyframeExtractionError: if decoding or scoring fails.
    """
    extractor = DakeKeyframeExtractor(
        video_id,
        frame_reader,
        scorer=scorer,
        min_temporal_distance=min_temporal_distance,
        peak_threshold=peak_threshold,
        target=target,
        sampling_ratio=sampling_ratio,
        anchor_interval_seconds=anchor_interval_seconds,
    )
    return extractor.extract(video_path)