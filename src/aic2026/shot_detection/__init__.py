"""Shot detection for adaptive keyframe extraction.

This package provides a pluggable shot-detection stage that splits a video
into contiguous temporal segments (shots). Shot boundaries are computed
independently of keyframe selection, so the shot detector can be swapped
without touching the sampling or extraction pipeline.

Contract:

- :class:`Shot` is an inclusive ``[start_frame, end_frame]`` segment of a
  source video. Shots are contiguous: the next shot's ``start_frame`` equals
  the previous shot's ``end_frame + 1`` (for full-coverage detectors).
- :class:`ShotDetector.detect`` returns the detected shots and records the
  source video's frame rate / frame count on the instance (``last_fps``,
  ``last_num_frames``), which the extractor uses to build canonical records.
- :func:`create_shot_detector` is the factory used by the pipeline. The
  ``omnishotcut`` implementation runs the OmniShotCut model (device-aware,
  lazy model load); the ``synthetic`` implementation reads deterministic shot
  ranges from a synthetic descriptor for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Valid ``--detector`` CLI values.
SHOT_DETECTOR_NAMES = ("omnishotcut", "synthetic")

#: Default OmniShotCut model checkpoint on the HuggingFace Hub.
DEFAULT_OMNISHOTCUT_REPO = "uva-cv-lab/OmniShotCut"

#: Default OmniShotCut checkpoint filename inside the repository.
DEFAULT_OMNISHOTCUT_FILENAME = "OmniShotCut_ckpt.pth"


class ShotDetectionError(RuntimeError):
    """Raised when shot detection fails or produces invalid output."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Shot:
    """One detected shot as an inclusive source-frame range.

    Fields:
        - start_frame: 0-based index of the shot's first frame.
        - end_frame: 0-based index of the shot's last frame (inclusive).
    """

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        _require(
            isinstance(self.start_frame, int) and not isinstance(self.start_frame, bool) and self.start_frame >= 0,
            "start_frame must be a non-negative integer",
        )
        _require(
            isinstance(self.end_frame, int) and not isinstance(self.end_frame, bool) and self.end_frame >= self.start_frame,
            "end_frame must be an integer >= start_frame",
        )

    @property
    def num_frames(self) -> int:
        """Return the number of source frames in the shot (>= 1)."""
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, int]:
        """Serialize to a JSON-safe dict."""
        return {"start_frame": self.start_frame, "end_frame": self.end_frame}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Shot":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(start_frame=int(data["start_frame"]), end_frame=int(data["end_frame"]))


@runtime_checkable
class ShotDetector(Protocol):
    """Protocol for detecting shots in a video.

    Attributes:
        - name: stable detector identity.
        - last_fps: frame rate of the most recently detected video, or None.
        - last_num_frames: source frame count of the most recently detected
          video, or None.

    Methods:
        - ``detect(video_path)``: return the detected :class:`Shot` list and
          populate ``last_fps`` / ``last_num_frames``.
    """

    name: str

    @property
    def last_fps(self) -> float | None:
        """Frame rate of the most recently detected video (or None)."""
        ...

    @property
    def last_num_frames(self) -> int | None:
        """Source frame count of the most recently detected video (or None)."""
        ...

    def detect(self, video_path: Path | str) -> list[Shot]:
        """Detect shots in a video.

        Expected input:
            - video_path: path to the video to analyze.

        Returns:
            - The detected :class:`Shot` segments in ascending order.

        Raises:
            - FileNotFoundError: if the video is absent.
            - ShotDetectionError: if the video cannot be analyzed.
        """


def normalize_shots(shots: list[Shot], num_frames: int) -> list[Shot]:
    """Normalize detected shots to cover ``[0, num_frames - 1]`` fully.

    Sorts shots, clamps them to the video bounds, merges only overlapping
    shots (shot boundaries are preserved: adjacent shots stay distinct), and
    fills gaps (frames not covered by any shot) with their own shots so the
    union covers the entire video. An empty ``shots`` list yields one
    full-video shot.

    Expected input:
        - shots: detected shot segments (any order).
        - num_frames: total source frames in the video (>= 0).

    Returns:
        - Sorted, non-overlapping shots whose union is exactly
          ``[0, num_frames - 1]`` (``[]`` when ``num_frames == 0``).

    Raises:
        - ValueError: if ``num_frames`` is negative.
    """
    _require(isinstance(num_frames, int) and num_frames >= 0, "num_frames must be >= 0")
    if num_frames == 0:
        return []
    if not shots:
        return [Shot(0, num_frames - 1)]
    ordered = sorted(shots, key=lambda shot: (shot.start_frame, shot.end_frame))
    merged: list[Shot] = []
    for shot in ordered:
        start = max(0, shot.start_frame)
        end = min(num_frames - 1, shot.end_frame)
        if start > end:
            continue
        if merged and start <= merged[-1].end_frame:
            merged[-1] = Shot(merged[-1].start_frame, max(merged[-1].end_frame, end))
        else:
            merged.append(Shot(start, end))
    filled: list[Shot] = []
    cursor = 0
    for shot in merged:
        if shot.start_frame > cursor:
            filled.append(Shot(cursor, shot.start_frame - 1))
        filled.append(shot)
        cursor = shot.end_frame + 1
    if cursor <= num_frames - 1:
        filled.append(Shot(cursor, num_frames - 1))
    return filled


def create_shot_detector(
    name: str,
    *,
    checkpoint: str | None = None,
    filename: str = DEFAULT_OMNISHOTCUT_FILENAME,
    mode: str = "clean_shot",
    overlap: int = 20,
    device: str = "auto",
) -> ShotDetector:
    """Create a :class:`ShotDetector` by name.

    Expected input:
        - name: ``omnishotcut`` (real videos) or ``synthetic`` (test
          descriptors).
        - checkpoint / filename / mode / overlap: OmniShotCut options
          (ignored for the synthetic detector).
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda`` for the OmniShotCut
          model (ignored for the synthetic detector).

    Returns:
        - A shot detector instance.

    Raises:
        - ValueError: if ``name`` is not a known detector.
    """
    if name not in SHOT_DETECTOR_NAMES:
        raise ValueError(f"unsupported shot detector {name!r}; expected {SHOT_DETECTOR_NAMES}")
    if name == "synthetic":
        from aic2026.shot_detection.synthetic import SyntheticShotDetector

        return SyntheticShotDetector()
    from aic2026.shot_detection.omnishotcut import OmniShotCutShotDetector

    return OmniShotCutShotDetector(
        checkpoint=checkpoint,
        filename=filename,
        mode=mode,
        overlap=overlap,
        device=device,
    )
