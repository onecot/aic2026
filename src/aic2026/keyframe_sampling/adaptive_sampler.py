"""Adaptive per-shot keyframe sampler (``adaptive_v1``).

The sampler picks keyframes inside a detected shot based on the shot's
duration (converted to seconds using the video's frame rate):

- short shot (< ``short_shot_seconds``): the middle frame only;
- medium shot (``short_shot_seconds`` .. ``long_shot_seconds``): the middle
  frame plus a uniform sample every ``uniform_interval_seconds``;
- long shot (> ``long_shot_seconds``): a uniform sample every
  ``uniform_interval_seconds`` plus high-change frames (when per-frame change
  scores are supplied), selected with the shared peak/suppression helpers from
  :mod:`aic2026.keyframe_extractor`.

The class is stateless across calls except for its configuration; a single
instance is reused for all shots of one video.
"""

from __future__ import annotations

from collections.abc import Sequence

from aic2026.keyframe_extractor import candidate_peaks, temporal_suppression
from aic2026.keyframe_sampling import SamplerError
from aic2026.shot_detection import Shot


class AdaptiveSampler:
    """Select keyframes inside a shot using duration-adaptive policies.

    Expected input:
        - fps: frame rate of the video being sampled (positive).
        - short_shot_seconds: shots strictly shorter than this (in seconds)
          produce only their middle frame.
        - long_shot_seconds: shots at least this long (in seconds) use the
          long-shot policy (uniform + high-change) instead of the medium one.
        - uniform_interval_seconds: spacing (in seconds) of uniform samples.
        - min_temporal_distance: minimum number of frames between high-change
          keyframes in long shots.
        - peak_threshold: relative prominence a score peak needs to qualify.

    Returns:
        - A sampler instance.

    Raises:
        - ValueError: if any numeric parameter is invalid.
    """

    name: str = "adaptive_v1"

    def __init__(
        self,
        *,
        fps: float,
        short_shot_seconds: float = 5.0,
        long_shot_seconds: float = 60.0,
        uniform_interval_seconds: float = 10.0,
        min_temporal_distance: int = 30,
        peak_threshold: float = 8.0,
    ) -> None:
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        if short_shot_seconds <= 0.0 or long_shot_seconds <= short_shot_seconds:
            raise ValueError(
                "need 0 < short_shot_seconds < long_shot_seconds, got "
                f"short={short_shot_seconds!r}, long={long_shot_seconds!r}"
            )
        if uniform_interval_seconds <= 0.0:
            raise ValueError(f"uniform_interval_seconds must be positive, got {uniform_interval_seconds!r}")
        if min_temporal_distance < 1 or peak_threshold <= 0.0:
            raise ValueError("min_temporal_distance >= 1 and peak_threshold > 0 are required")
        self._fps = float(fps)
        self._short_shot_seconds = float(short_shot_seconds)
        self._long_shot_seconds = float(long_shot_seconds)
        self._uniform_interval_seconds = float(uniform_interval_seconds)
        self._min_temporal_distance = int(min_temporal_distance)
        self._peak_threshold = float(peak_threshold)
        self._uniform_interval_frames = max(
            1, int(round(self._uniform_interval_seconds * self._fps))
        )

    def sample(
        self,
        shot: Shot,
        frames: Sequence[int],
        *,
        scores: object | None = None,
    ) -> list[int]:
        """Select keyframe frame indices inside ``shot``.

        Expected input:
            - shot: the detected shot segment (inclusive ``start_frame`` /
              ``end_frame``).
            - frames: candidate frame indices inside the shot, ascending.
            - scores: optional per-frame change scores aligned with the
              video's frame indices; required for high-change selection in
              long shots.

        Returns:
            - Selected frame indices, ascending, unique, each in ``frames``.

        Raises:
            - SamplerError: if the shot or frames are inconsistent.
        """
        if shot.start_frame > shot.end_frame:
            raise SamplerError(f"invalid shot {shot} (start after end)")
        candidates = sorted(set(int(f) for f in frames))
        in_shot = [f for f in candidates if shot.start_frame <= f <= shot.end_frame]
        if not in_shot:
            raise SamplerError(f"shot {shot} overlaps no candidate frames")
        if in_shot != candidates:
            raise SamplerError(
                f"shot {shot} does not cover all candidate frames "
                f"(outside: {set(candidates) - set(in_shot)})"
            )

        duration_seconds = (shot.end_frame - shot.start_frame + 1) / self._fps
        middle = (shot.start_frame + shot.end_frame) // 2
        uniform = self._uniform_indices(shot)

        if duration_seconds < self._short_shot_seconds:
            selected = {middle}
        elif duration_seconds <= self._long_shot_seconds:
            selected = set(uniform)
            selected.add(middle)
        else:
            selected = set(uniform)
            if scores is not None:
                selected.update(self._high_change_indices(shot, scores, selected))

        return sorted(f for f in selected if shot.start_frame <= f <= shot.end_frame)

    def _uniform_indices(self, shot: Shot) -> list[int]:
        """Return uniformly spaced frame indices across the shot."""
        indices = [
            shot.start_frame + k * self._uniform_interval_frames
            for k in range((shot.end_frame - shot.start_frame) // self._uniform_interval_frames + 1)
        ]
        return [f for f in indices if f <= shot.end_frame]

    def _high_change_indices(
        self,
        shot: Shot,
        scores: object,
        occupied: set[int],
    ) -> list[int]:
        """Return high-change frame indices inside ``shot`` not already chosen.

        Peaks are detected on the global score array (indexed by frame), then
        restricted to the shot and temporally suppressed relative to each other
        and to the uniform keyframes already selected.
        """
        import numpy as np

        if not hasattr(scores, "__len__") or len(scores) <= shot.end_frame:
            raise SamplerError(
                f"score array too short for shot {shot}: "
                f"len={len(scores)} < end_frame={shot.end_frame}"
            )
        score_arr = np.asarray(scores, dtype=float)
        peaks = candidate_peaks(score_arr, self._peak_threshold)
        peaks = [p for p in peaks if shot.start_frame <= p <= shot.end_frame]
        if not peaks:
            return []
        all_occupied = sorted(set(peaks) | set(occupied))
        surviving = temporal_suppression(
            all_occupied,
            score_arr,
            min_temporal_distance=self._min_temporal_distance,
            occupied=occupied,
        )
        return [p for p in surviving if p not in occupied]
