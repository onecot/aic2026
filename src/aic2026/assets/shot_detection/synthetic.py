"""Synthetic shot detector for deterministic tests.

Reads a lightweight JSON descriptor (no video codec, no model) that declares
the video's frame rate and a list of shot ranges::

    {
      "schema_version": "1.0",
      "fps": 25.0,
      "shots": [
        {"start_frame": 0, "end_frame": 99},
        {"start_frame": 100, "end_frame": 299}
      ]
    }

It also accepts a synthetic *video* descriptor (as written by
:func:`aic2026.assets.keyframe_extractor.write_synthetic_video`) that embeds the same
``shots`` list next to ``frames``, so a single file drives both the frame
reader and the shot detector in adaptive-pipeline tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from aic2026.assets.keyframe_extractor import SYNTHETIC_VIDEO_SCHEMA_VERSION
from aic2026.assets.shot_detection import Shot, ShotDetectionError

#: Schema version of the synthetic shot descriptor.
SYNTHETIC_SHOT_SCHEMA_VERSION = "1.0"

#: Schema version of the synthetic video descriptor (accepted too).
SYNTHETIC_VIDEO_SCHEMA_VERSION = SYNTHETIC_VIDEO_SCHEMA_VERSION


class SyntheticShotDetector:
    """Read deterministic shot ranges from a synthetic JSON descriptor.

    Expected input:
        - None.

    Returns:
        - A shot detector instance.
    """

    name: str = "synthetic"

    def __init__(self) -> None:
        self._last_fps: float | None = None
        self._last_num_frames: int | None = None

    @property
    def last_fps(self) -> float | None:
        """Return the frame rate declared by the most recent descriptor."""
        return self._last_fps

    @property
    def last_num_frames(self) -> int | None:
        """Return the max shot end frame + 1 of the most recent descriptor."""
        return self._last_num_frames

    def detect(self, video_path: Path | str) -> list[Shot]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"synthetic shot file does not exist: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ShotDetectionError(f"invalid synthetic shot JSON {path}: {exc}") from exc
        schema = payload.get("schema_version")
        if schema not in (SYNTHETIC_SHOT_SCHEMA_VERSION, SYNTHETIC_VIDEO_SCHEMA_VERSION):
            raise ShotDetectionError(
                f"unsupported synthetic shot schema {schema!r}: {path}"
            )
        fps = float(payload["fps"])
        if fps <= 0.0:
            raise ShotDetectionError(f"invalid synthetic fps {fps!r}: {path}")
        raw = payload.get("shots")
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ShotDetectionError(f"malformed shots list in {path}")
        shots: list[Shot] = []
        for row in raw:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if start < 0 or end < start:
                raise ShotDetectionError(f"invalid shot [{start}, {end}] in {path}")
            shots.append(Shot(start_frame=start, end_frame=end))
        if not shots:
            raise ShotDetectionError(f"no shots declared in {path}")
        shots.sort(key=lambda shot: shot.start_frame)
        self._last_fps = fps
        self._last_num_frames = max(shot.end_frame for shot in shots) + 1
        return shots


def write_synthetic_shots(
    path: Path | str,
    shots: list[tuple[int, int]] | list[Shot],
    *,
    fps: float = 25.0,
) -> Path:
    """Write a synthetic shot descriptor file for deterministic tests.

    Expected input:
        - path: target descriptor path.
        - shots: ``(start_frame, end_frame)`` tuples (or :class:`Shot`
          instances) in any order.
        - fps: frame rate to declare for the synthetic video.

    Returns:
        - The written path.

    Raises:
        - ValueError: if ``shots`` is empty or contains an invalid range.
    """
    path = Path(path)
    if not shots:
        raise ValueError("shots must contain at least one range")
    rows: list[dict[str, int]] = []
    for item in shots:
        if isinstance(item, Shot):
            start, end = item.start_frame, item.end_frame
        else:
            start, end = int(item[0]), int(item[1])
        if start < 0 or end < start:
            raise ValueError(f"invalid shot [{start}, {end}]")
        rows.append({"start_frame": start, "end_frame": end})
    payload = {
        "schema_version": SYNTHETIC_SHOT_SCHEMA_VERSION,
        "fps": float(fps),
        "shots": rows,
    }
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)
    return path
