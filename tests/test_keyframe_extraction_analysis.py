"""Tests for dataset-level OmniShotCut adaptive extraction analysis.

Uses synthetic video descriptors with embedded shot ranges only, so the tests
are fast, deterministic, and never load the OmniShotCut model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.keyframe_extraction import (
    VideoEntry,
    VideoManifest,
    save_video_manifest,
)
from aic2026.keyframe_extraction_analysis import (
    ShotDistributionReport,
    ShotValidationError,
    analyze_shot_distribution,
    estimate_keyframes,
    format_report,
    validate_shots,
    write_report,
)
from aic2026.keyframe_extractor import write_synthetic_video
from aic2026.shot_detection import Shot
from aic2026.shot_detection.synthetic import SyntheticShotDetector


def _write_video(
    tmp_path: Path,
    name: str,
    num_frames: int,
    *,
    fps: float = 25.0,
    shots: list[tuple[int, int]],
    scene_length: int = 40,
) -> Path:
    values = [float((i // scene_length) * 100.0) for i in range(num_frames)]
    return write_synthetic_video(tmp_path / f"{name}.json", values, fps=fps, shots=shots)


def _manifest(tmp_path: Path, videos: list[tuple[str, int, list[tuple[int, int]]]]) -> Path:
    entries = []
    for vid, frames, shots in videos:
        path = _write_video(tmp_path, vid, frames, shots=shots)
        entries.append(VideoEntry(video_id=vid, path=str(path), num_frames=frames))
    path = tmp_path / "videos.json"
    save_video_manifest(path, VideoManifest(schema_version="1.0", videos=tuple(entries)))
    return path


# ------------------------------------------------------------------ validation


class TestValidateShots:
    def test_valid_contiguous_shots(self) -> None:
        validate_shots([Shot(0, 99), Shot(100, 499)], 500, video_id="L21_V001")

    def test_single_full_video_shot(self) -> None:
        validate_shots([Shot(0, 499)], 500, video_id="L21_V001")

    def test_zero_shots_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="0 shots"):
            validate_shots([], 500, video_id="L21_V001")

    def test_invalid_range_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="invalid shot range"):
            validate_shots([_raw_shot(150, 100)], 500, video_id="L21_V001")

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="invalid shot range"):
            validate_shots([_raw_shot(-5, 10)], 500, video_id="L21_V001")

    def test_out_of_bounds_end_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="invalid shot range"):
            validate_shots([Shot(0, 99), Shot(100, 500)], 500, video_id="L21_V001")

    def test_overlap_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="overlapping"):
            validate_shots([Shot(0, 199), Shot(150, 299)], 500, video_id="L21_V001")

    def test_coverage_gap_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="do not cover"):
            validate_shots([Shot(0, 99), Shot(200, 499)], 500, video_id="L21_V001")

    def test_coverage_start_not_zero_rejected(self) -> None:
        with pytest.raises(ShotValidationError, match="do not cover"):
            validate_shots([Shot(10, 499)], 500, video_id="L21_V001")

    def test_invalid_frame_count(self) -> None:
        with pytest.raises(ShotValidationError, match="invalid frame count"):
            validate_shots([Shot(0, 99)], 0, video_id="L21_V001")


def _raw_shot(start: int, end: int) -> Shot:
    """Build a Shot bypassing __post_init__ validation (for defensive checks)."""
    shot = object.__new__(Shot)
    object.__setattr__(shot, "start_frame", start)
    object.__setattr__(shot, "end_frame", end)
    return shot


# ------------------------------------------------------------------ estimation


class TestEstimateKeyframes:
    def test_short_shot_one_keyframe(self) -> None:
        # 100 frames @25fps = 4s < short_shot_seconds(5) -> middle only.
        assert estimate_keyframes([Shot(0, 99)], 25.0, {}) == 1

    def test_medium_shot_uniform_plus_middle(self) -> None:
        # 500 frames @25fps = 20s -> medium: uniform every 250 + middle.
        assert estimate_keyframes([Shot(0, 499)], 25.0, {}) == 3

    def test_long_shot_uniform_lower_bound(self) -> None:
        # 2500 frames @25fps = 100s -> long; no scores -> uniform only.
        assert estimate_keyframes([Shot(0, 2499)], 25.0, {}) == 10

    def test_multiple_shots_summed(self) -> None:
        assert estimate_keyframes([Shot(0, 99), Shot(100, 199)], 25.0, {}) == 2

    def test_short_medium_long_combined(self) -> None:
        shots = [Shot(0, 99), Shot(100, 199), Shot(200, 2499)]
        assert estimate_keyframes(shots, 25.0, {}) == 12  # 1 + 1 + 10

    def test_custom_thresholds(self) -> None:
        # long_shot_seconds=3 with short_shot_seconds=2 makes the 4s shot
        # "long" -> uniform-only at a 2s (50-frame) step: frames 0, 50.
        config = {
            "short_shot_seconds": 2.0,
            "long_shot_seconds": 3.0,
            "uniform_interval_seconds": 2.0,
        }
        assert estimate_keyframes([Shot(0, 99)], 25.0, config) == 2

    def test_invalid_fps(self) -> None:
        with pytest.raises(ValueError):
            estimate_keyframes([Shot(0, 99)], 0.0, {})


# ------------------------------------------------------------------ report


class TestAnalyzeShotDistribution:
    def test_stats_over_known_shots(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path,
            [
                ("L21_V001", 500, [(0, 99), (100, 199), (200, 499)]),
                ("L21_V002", 2500, [(0, 2499)]),
                ("L21_V003", 100, [(0, 99)]),
            ],
        )
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        assert report.total_videos == 3
        assert report.shots.total == 5
        assert report.shots.min == 1
        assert report.shots.max == 3
        assert report.estimated_keyframes.total == 16  # 5 + 10 + 1
        assert report.estimated_keyframes.min == 1
        assert report.estimated_keyframes.max == 10

    def test_shot_durations_in_seconds(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [("L21_V001", 100, [(0, 99)])])
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        assert report.shot_duration_seconds.max == pytest.approx(4.0)  # 100 @25fps

    def test_report_accepts_loaded_manifest(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [("L21_V001", 100, [(0, 99)])])
        manifest = VideoManifest.from_dict(json.loads(path.read_text()))
        report = analyze_shot_distribution(manifest, SyntheticShotDetector(), {})
        assert report.total_videos == 1

    def test_empty_manifest_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        save_video_manifest(path, VideoManifest(schema_version="1.0", videos=()))
        with pytest.raises(ValueError, match="empty"):
            analyze_shot_distribution(path, SyntheticShotDetector(), {})

    def test_missing_video_file(self, tmp_path: Path) -> None:
        path = tmp_path / "videos.json"
        save_video_manifest(
            path,
            VideoManifest(
                schema_version="1.0",
                videos=(VideoEntry(video_id="L21_V001", path=str(tmp_path / "nope.mp4"), num_frames=100),),
            ),
        )
        with pytest.raises(FileNotFoundError):
            analyze_shot_distribution(path, SyntheticShotDetector(), {})

    def test_deterministic_report(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path,
            [
                ("L21_V001", 500, [(0, 99), (100, 199), (200, 499)]),
                ("L21_V002", 2500, [(0, 2499)]),
                ("L21_V003", 100, [(0, 99)]),
            ],
        )
        first = analyze_shot_distribution(path, SyntheticShotDetector(), {}).to_dict()
        second = analyze_shot_distribution(path, SyntheticShotDetector(), {}).to_dict()
        # analyzed_at is a timestamp; drop it before comparing determinism.
        assert first["total_videos"] == second["total_videos"]
        assert first["shots"] == second["shots"]
        assert first["shot_duration_seconds"] == second["shot_duration_seconds"]
        assert first["estimated_keyframes"] == second["estimated_keyframes"]
        assert first["outliers"] == second["outliers"]

    def test_outliers_flagged(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path,
            [
                ("L21_V001", 100, [(0, 99)]),
                ("L21_V002", 100, [(0, 99)]),
                ("L21_V003", 100, [(0, 99)]),
                ("L21_V004", 100, [(0, 99)]),
                ("L21_V005", 100, [(0, 99)]),
                ("L21_V006", 10000, [(0, 2499), (2500, 4999), (5000, 7499), (7500, 9999)]),
            ],
        )
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        assert any(record["video_id"] == "L21_V006" for record in report.outliers)
        for record in report.outliers:
            assert set(record) == {"video_id", "num_shots", "estimated_keyframes"}

    def test_outliers_none_with_no_spread(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path,
            [("L21_V001", 100, [(0, 99)]), ("L21_V002", 100, [(0, 99)])],
        )
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        assert report.outliers == []

    def test_zero_shot_video_rejected(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path,
            [
                ("L21_V001", 100, [(0, 99)]),
                ("L21_V002", 100, [(0, 99)]),
            ],
        )
        # Monkeypatch the synthetic detector to return no shots for one video.
        original = SyntheticShotDetector.detect

        def _empty_detect(self, video_path):
            self._last_fps = 25.0
            self._last_num_frames = 100
            return []

        SyntheticShotDetector.detect = _empty_detect  # type: ignore[method-assign]
        try:
            with pytest.raises(ShotValidationError, match="0 shots"):
                analyze_shot_distribution(path, SyntheticShotDetector(), {})
        finally:
            SyntheticShotDetector.detect = original  # type: ignore[method-assign]


class TestReportSerialization:
    def test_to_dict_shape(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [("L21_V001", 100, [(0, 99)])])
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        payload = report.to_dict()
        assert set(payload) == {
            "schema_version",
            "meta",
            "total_videos",
            "shots",
            "shot_duration_seconds",
            "estimated_keyframes",
            "outliers",
        }
        assert set(payload["shots"]) == {"total", "avg", "min", "median", "p90", "max"}
        assert set(payload["shot_duration_seconds"]) == {"mean", "median", "p90", "max"}
        assert set(payload["estimated_keyframes"]) == {"total", "avg", "min", "median", "p90", "max"}
        assert isinstance(payload["outliers"], list)
        assert isinstance(report, ShotDistributionReport)

    def test_write_report(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [("L21_V001", 100, [(0, 99)])])
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        output = tmp_path / "out" / "report.json"
        write_report(output, report)
        assert output.is_file()
        restored = json.loads(output.read_text())
        assert restored["total_videos"] == 1

    def test_format_report(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [("L21_V001", 100, [(0, 99)])])
        report = analyze_shot_distribution(path, SyntheticShotDetector(), {})
        text = format_report(report)
        assert "total shots:" in text
        assert "estimated keyframes/video:" in text


class TestAnalyzeCli:
    def test_help(self) -> None:
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframe_extraction.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--videos" in result.stdout
        assert "--detector" in result.stdout
        assert "--sampler" in result.stdout
        assert "--device" in result.stdout
        assert "--output" in result.stdout

    def test_run_end_to_end(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        path = _manifest(tmp_path, [("L21_V001", 500, [(0, 99), (100, 199), (200, 499)])])
        out = tmp_path / "report.json"
        script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframe_extraction.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--videos",
                str(path),
                "--detector",
                "synthetic",
                "--output",
                str(out),
                "--progress",
                "off",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert out.is_file()
        payload = json.loads(out.read_text())
        assert payload["total_videos"] == 1
        assert payload["shots"]["total"] == 3
