"""Integration tests for the OmniShotCut adaptive keyframe extraction pipeline.

Uses a single synthetic video descriptor per video that embeds both ``frames``
and ``shots``, so the whole pipeline (shot detection -> adaptive sampling ->
canonical artifact) runs with no ffmpeg and no model. Deterministic and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.keyframe_extraction import (
    ExtractionConfig,
    ExtractionManifest,
    VideoEntry,
    VideoManifest,
    extract_keyframes,
    save_video_manifest,
)
from aic2026.keyframe_extractor import (
    KeyframeExtractionError,
    SyntheticFrameReader,
    write_synthetic_video,
)
from aic2026.omnishotcut_extractor import (
    OmniShotCutAdaptiveExtractor,
    extract_video_keyframes_adaptive,
)
from aic2026.shot_detection.synthetic import SyntheticShotDetector


def _write_video(
    tmp_path: Path,
    name: str,
    num_frames: int,
    *,
    fps: float = 25.0,
    shots: list[tuple[int, int]] | None = None,
    scene_length: int = 40,
) -> Path:
    """Write a synthetic video descriptor with embedded shot ranges."""
    if shots is None:
        shots = [(0, num_frames - 1)]
    values = [float((i // scene_length) * 100.0) for i in range(num_frames)]
    return write_synthetic_video(
        tmp_path / f"{name}.json",
        values,
        fps=fps,
        shots=shots,
    )


def _adaptive_config(**overrides: object) -> ExtractionConfig:
    kwargs: dict[str, object] = {
        "extractor": "omnishotcut_adaptive",
        "frame_reader": "synthetic",
        "detector": "synthetic",
        "sampler": "adaptive_v1",
    }
    kwargs.update(overrides)
    return ExtractionConfig(**kwargs)


class TestOmniShotCutAdaptiveExtractor:
    def test_short_shot_only_middle(self, tmp_path: Path) -> None:
        path = _write_video(tmp_path, "vid", num_frames=100, shots=[(0, 99)])
        detector = SyntheticShotDetector()
        keyframes = extract_video_keyframes_adaptive(
            "L21_V001",
            path,
            shot_detector=detector,
            frame_reader=SyntheticFrameReader(),
        )
        assert [kf.frame_index for kf in keyframes] == [49]
        assert keyframes[0].video_id == "L21_V001"
        assert keyframes[0].pts_time == 49 / 25.0

    def test_medium_shot_uniform_plus_middle(self, tmp_path: Path) -> None:
        path = _write_video(tmp_path, "vid", num_frames=500, shots=[(0, 499)])
        detector = SyntheticShotDetector()
        keyframes = extract_video_keyframes_adaptive(
            "L21_V001",
            path,
            shot_detector=detector,
            frame_reader=SyntheticFrameReader(),
        )
        indices = [kf.frame_index for kf in keyframes]
        assert 249 in indices
        assert set(indices) == {0, 249, 250}

    def test_long_shot_adds_high_change_frames(self, tmp_path: Path) -> None:
        # 100s at 25fps -> 2500 frames. Synthetic scene boundaries create
        # strong change peaks every 40 frames, so the long-shot high-change
        # path must add frames beyond the uniform sample.
        path = _write_video(tmp_path, "vid", num_frames=2500, shots=[(0, 2499)])
        detector = SyntheticShotDetector()
        keyframes = extract_video_keyframes_adaptive(
            "L21_V001",
            path,
            shot_detector=detector,
            frame_reader=SyntheticFrameReader(),
        )
        indices = [kf.frame_index for kf in keyframes]
        uniform = set(range(0, 2500, 250))
        assert len(indices) == len(set(indices))
        assert indices == sorted(indices)
        assert len(indices) > len(uniform)
        assert uniform < set(indices)

    def test_missing_video(self, tmp_path: Path) -> None:
        detector = SyntheticShotDetector()
        with pytest.raises(FileNotFoundError):
            extract_video_keyframes_adaptive(
                "L21_V001",
                tmp_path / "nope.json",
                shot_detector=detector,
                frame_reader=SyntheticFrameReader(),
            )

    def test_invalid_descriptor_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"schema_version": "1.0", "fps": 25.0}', encoding="utf-8")
        detector = SyntheticShotDetector()
        with pytest.raises(KeyframeExtractionError):
            extract_video_keyframes_adaptive(
                "L21_V001",
                path,
                shot_detector=detector,
                frame_reader=SyntheticFrameReader(),
            )


class TestExtractKeyframesAdaptive:
    def test_full_pipeline_manifest(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        path = _write_video(video_dir, "L21_V001", num_frames=2500, shots=[(0, 2499)])
        videos_path = video_dir / "videos.json"
        save_video_manifest(
            videos_path,
            VideoManifest(
                schema_version="1.0",
                videos=(VideoEntry(video_id="L21_V001", path=str(path), num_frames=2500),),
            ),
        )
        output_dir = tmp_path / "out"
        manifest = extract_keyframes(videos_path, output_dir, _adaptive_config())

        assert isinstance(manifest, ExtractionManifest)
        assert manifest.extractor == "omnishotcut_adaptive"
        assert manifest.detector == "synthetic"
        assert manifest.sampler == "adaptive_v1"

        records = _load_records(output_dir / "frames.ndjson")
        assert records
        for record in records:
            assert record["keyframe"]["video_id"]["value"] == "L21_V001"
            assert isinstance(record["frame_index"], int)
            assert record["frame_index"] >= 0

        csv_path = output_dir / "map-keyframes" / "L21_V001.csv"
        assert csv_path.is_file()
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "n,pts_time,fps,frame_idx"
        assert len(lines) - 1 == len(records)

    def test_two_videos_two_detector_fps(self, tmp_path: Path) -> None:
        # Two videos with different frame rates exercise the per-video
        # fps-from-detector path and produce distinct artifacts.
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        entries = []
        for vid, fps in [("L21_V001", 25.0), ("L21_V002", 30.0)]:
            path = _write_video(video_dir, vid, num_frames=500, fps=fps, shots=[(0, 499)])
            entries.append(VideoEntry(video_id=vid, path=str(path), num_frames=500))
        videos_path = video_dir / "videos.json"
        save_video_manifest(videos_path, VideoManifest(schema_version="1.0", videos=tuple(entries)))
        output_dir = tmp_path / "out"
        manifest = extract_keyframes(videos_path, output_dir, _adaptive_config())
        assert manifest.num_videos == 2
        for vid in ("L21_V001", "L21_V002"):
            csv = output_dir / "map-keyframes" / f"{vid}.csv"
            assert csv.is_file()
            assert csv.read_text(encoding="utf-8").strip().splitlines()[0] == "n,pts_time,fps,frame_idx"


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
