"""Tests for the adaptive per-shot keyframe sampler.

Deterministic and model-free: samples are computed from shots, frame rate,
and optional change scores (numpy arrays only).
"""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.keyframe_sampling import (
    SAMPLER_NAMES,
    SamplerError,
    create_sampler,
)
from aic2026.keyframe_sampling.adaptive_sampler import AdaptiveSampler
from aic2026.shot_detection import Shot


@pytest.fixture
def fps() -> float:
    return 25.0


def _sampler(**overrides: object) -> AdaptiveSampler:
    kwargs: dict[str, object] = {
        "fps": 25.0,
        "short_shot_seconds": 5.0,
        "long_shot_seconds": 60.0,
        "uniform_interval_seconds": 10.0,
        "min_temporal_distance": 30,
        "peak_threshold": 8.0,
    }
    kwargs.update(overrides)
    return AdaptiveSampler(**kwargs)  # type: ignore[arg-type]


def _shot(start: int, end: int) -> Shot:
    return Shot(start_frame=start, end_frame=end)


class TestAdaptiveSamplerValidation:
    def test_bad_fps(self) -> None:
        with pytest.raises(ValueError):
            _sampler(fps=0.0)

    def test_bad_thresholds(self) -> None:
        with pytest.raises(ValueError):
            _sampler(short_shot_seconds=0.0)
        with pytest.raises(ValueError):
            _sampler(long_shot_seconds=3.0, short_shot_seconds=5.0)
        with pytest.raises(ValueError):
            _sampler(uniform_interval_seconds=0.0)
        with pytest.raises(ValueError):
            _sampler(min_temporal_distance=0)
        with pytest.raises(ValueError):
            _sampler(peak_threshold=0.0)


class TestAdaptiveSamplerPolicy:
    def test_short_shot_middle_only(self) -> None:
        # 25 fps * 4s = 100 frames -> short (4s < 5s).
        sampler = _sampler()
        shot = _shot(0, 99)
        result = sampler.sample(shot, list(range(100)))
        assert result == [49]

    def test_medium_shot_middle_plus_uniform(self) -> None:
        # 25 fps * 20s = 500 frames -> medium.
        sampler = _sampler()
        shot = _shot(0, 499)
        result = sampler.sample(shot, list(range(500)))
        # uniform every 250 frames (10s) + the middle frame (249).
        assert 249 in result
        assert 0 in result
        assert 250 in result
        assert len(result) == 3

    def test_long_shot_without_scores_is_uniform(self) -> None:
        # 25 fps * 100s = 2500 frames -> long (>60s); no scores supplied.
        sampler = _sampler()
        shot = _shot(0, 2499)
        result = sampler.sample(shot, list(range(2500)), scores=None)
        assert result == list(range(0, 2500, 250))

    def test_long_shot_with_scores_adds_high_change(self) -> None:
        # 100s shot with a strong isolated change peak in the middle.
        fps = 25.0
        num_frames = 2500
        scores = np.zeros(num_frames, dtype=np.float64)
        scores[1200] = 100.0  # a big peak far from the uniform anchors
        sampler = _sampler(fps=fps, min_temporal_distance=30, peak_threshold=8.0)
        shot = _shot(0, num_frames - 1)
        result = sampler.sample(shot, list(range(num_frames)), scores=scores)
        assert 1200 in result
        uniform = set(range(0, num_frames, int(round(10.0 * fps))))
        assert set(result) == uniform | {1200}

    def test_result_unique_and_sorted(self) -> None:
        sampler = _sampler()
        shot = _shot(0, 499)
        result = sampler.sample(shot, list(range(500)))
        assert result == sorted(set(result))
        assert len(set(result)) == len(result)


class TestAdaptiveSamplerErrors:
    def test_frames_outside_shot_rejected(self) -> None:
        sampler = _sampler()
        shot = _shot(0, 99)
        with pytest.raises(SamplerError):
            sampler.sample(shot, list(range(50, 150)))

    def test_no_overlap_rejected(self) -> None:
        sampler = _sampler()
        shot = _shot(100, 199)
        with pytest.raises(SamplerError):
            sampler.sample(shot, list(range(0, 10)))

    def test_partial_coverage_rejected(self) -> None:
        sampler = _sampler()
        with pytest.raises(SamplerError):
            sampler.sample(_shot(10, 50), list(range(0, 60)), scores=None)

    def test_scores_too_short_rejected(self) -> None:
        sampler = _sampler()
        shot = _shot(0, 2499)
        scores = np.zeros(10, dtype=np.float64)
        with pytest.raises(SamplerError):
            sampler.sample(shot, list(range(2500)), scores=scores)


class TestCreateSampler:
    def test_adaptive_v1(self) -> None:
        sampler = create_sampler("adaptive_v1", fps=25.0)
        assert isinstance(sampler, AdaptiveSampler)
        assert sampler.name == "adaptive_v1"

    def test_unknown_name(self) -> None:
        with pytest.raises(ValueError):
            create_sampler("nope", fps=25.0)

    def test_names(self) -> None:
        assert "adaptive_v1" in SAMPLER_NAMES
