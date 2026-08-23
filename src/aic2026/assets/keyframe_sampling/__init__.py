"""Keyframe sampling for adaptive keyframe extraction.

This package provides a pluggable sampling stage that, given a detected
:class:`~aic2026.shot_detection.Shot`, selects which source frames become
keyframes. The sampler is independent of how shots are detected and of how
the video is decoded, so either side can be swapped without touching the
other.

Contract:

- :class:`Sampler.sample`` receives a shot and the candidate frame indices
  inside it and returns the selected subset (ascending, no duplicates).
- A sampler may optionally accept per-frame change scores (aligned with the
  video's frame indices) to select high-change frames inside long shots.
- :func:`create_sampler` is the factory used by the pipeline; ``adaptive_v1``
  is the OmniShotCut-adaptive policy:
    - short shots (< ``short_shot_seconds``): the middle frame only;
    - medium shots: the middle frame plus a uniform sample every
      ``uniform_interval_seconds``;
    - long shots (> ``long_shot_seconds``): a uniform sample plus additional
      high-change frames (when scores are available).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from aic2026.assets.shot_detection import Shot

#: Valid ``--sampler`` CLI values.
SAMPLER_NAMES = ("adaptive_v1",)


class SamplerError(RuntimeError):
    """Raised when sampling fails or produces invalid output."""


@runtime_checkable
class Sampler(Protocol):
    """Protocol for selecting keyframes inside a shot.

    Attributes:
        - name: stable sampler identity.

    Methods:
        - ``sample(shot, frames, *, scores=None)``: select a subset of the
          candidate frame indices inside ``shot``.
    """

    name: str

    def sample(
        self,
        shot: Shot,
        frames: Sequence[int],
        *,
        scores: object | None = None,
    ) -> list[int]:
        """Select keyframe frame indices inside ``shot``.

        Expected input:
            - shot: the detected shot segment.
            - frames: candidate frame indices inside the shot (ascending,
              typically ``range(start_frame, end_frame + 1)``).
            - scores: optional per-frame change scores aligned with the
              video's frame indices (``scores[frame_index]``); used to select
              high-change frames in long shots.

        Returns:
            - Selected frame indices as a subset of ``frames``, ascending and
              unique.

        Raises:
            - SamplerError: if the shot or frames are inconsistent.
        """


def create_sampler(
    name: str,
    *,
    fps: float,
    short_shot_seconds: float = 5.0,
    long_shot_seconds: float = 60.0,
    uniform_interval_seconds: float = 10.0,
    min_temporal_distance: int = 30,
    peak_threshold: float = 8.0,
) -> Sampler:
    """Create a :class:`Sampler` by name.

    Expected input:
        - name: a known sampler name (``adaptive_v1``).
        - fps: frame rate of the video being sampled (converts seconds to
          frames).
        - short_shot_seconds / long_shot_seconds: shot-duration thresholds.
        - uniform_interval_seconds: spacing of uniform samples.
        - min_temporal_distance / peak_threshold: high-change selection
          parameters (long shots).

    Returns:
        - A sampler instance.

    Raises:
        - ValueError: if ``name`` is not a known sampler.
    """
    if name not in SAMPLER_NAMES:
        raise ValueError(f"unsupported sampler {name!r}; expected {SAMPLER_NAMES}")
    if name == "adaptive_v1":
        from aic2026.assets.keyframe_sampling.adaptive_sampler import AdaptiveSampler

        return AdaptiveSampler(
            fps=fps,
            short_shot_seconds=short_shot_seconds,
            long_shot_seconds=long_shot_seconds,
            uniform_interval_seconds=uniform_interval_seconds,
            min_temporal_distance=min_temporal_distance,
            peak_threshold=peak_threshold,
        )
    raise ValueError(f"unsupported sampler {name!r}")  # pragma: no cover - defensive
