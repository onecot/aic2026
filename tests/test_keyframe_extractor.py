"""Tests for the DAKE-inspired keyframe extractor.

Uses synthetic video descriptors only (never ffmpeg), so the tests are fast,
deterministic, and cross-platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeTiming
from aic2026.keyframe_extractor import (
    DakeKeyframeExtractor,
    FFmpegFrameReader,
    FrameDifferenceScorer,
    Keyframe,
    KeyframeExtractionError,
    SyntheticFrameReader,
    VideoFrame,
    candidate_peaks,
    create_frame_reader,
    create_scorer,
    extract_video_keyframes,
    select_keyframes,
    temporal_suppression,
    write_synthetic_video,
)


def make_video(
    tmp_path: Path,
    name: str,
    values: list[float],
    *,
    fps: float = 25.0,
    pts_times: list[float] | None = None,
) -> Path:
    """Write a synthetic video descriptor and return its path."""
    return write_synthetic_video(
        tmp_path / f"{name}.json",
        values,
        fps=fps,
        pts_times=pts_times,
    )


def _scene_values(num_frames: int, scene_length: int = 40) -> list[float]:
    """A sequence with a strong change at every scene boundary."""
    return [float((i // scene_length) * 100.0) for i in range(num_frames)]


# ---------------------------------------------------------------------- frames


class TestVideoFrame:
    def test_valid_frame(self) -> None:
        frame = VideoFrame(
            frame_index=0,
            pts_time=0.0,
            fps=25.0,
            image=np.zeros((4, 4), dtype=np.float32),
        )
        assert frame.frame_index == 0

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoFrame(frame_index=-1, pts_time=0.0, fps=25.0, image=None)

    def test_negative_pts_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoFrame(frame_index=0, pts_time=-0.1, fps=25.0, image=None)

    def test_non_positive_fps_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoFrame(frame_index=0, pts_time=0.0, fps=0.0, image=None)


class TestKeyframe:
    def test_to_dict_is_canonical(self) -> None:
        keyframe = Keyframe(
            video_id="L21_V001",
            keyframe_ordinal=1,
            pts_time=1.2,
            fps=25.0,
            frame_index=30,
        )
        record = keyframe.to_dict()
        assert record == {
            "keyframe": {
                "video_id": {"value": "L21_V001"},
                "keyframe_ordinal": 1,
            },
            "pts_time": 1.2,
            "fps": 25.0,
            "frame_index": 30,
        }
        # Must round-trip through the canonical contract.
        timing = KeyframeTiming.from_dict(record)
        assert timing.keyframe.video_id.value == "L21_V001"
        assert timing.keyframe.keyframe_ordinal == 1
        assert timing.frame_index == 30

    def test_to_timing(self) -> None:
        keyframe = Keyframe(
            video_id="L21_V002",
            keyframe_ordinal=3,
            pts_time=2.0,
            fps=30.0,
            frame_index=60,
        )
        timing = keyframe.to_timing()
        assert isinstance(timing, KeyframeTiming)
        assert Keyframe.from_timing(timing) == keyframe

    def test_invalid_video_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            Keyframe(video_id="bad", keyframe_ordinal=1, pts_time=0.0, fps=25.0, frame_index=0)


# ------------------------------------------------------------------ frame readers


class TestSyntheticFrameReader:
    def test_reads_descriptor(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "vid", [0.0, 50.0, 50.0, 100.0], fps=10.0)
        reader = SyntheticFrameReader()
        frames = list(reader.read(path))
        assert len(frames) == 4
        assert frames[0].frame_index == 0
        assert frames[0].pts_time == 0.0
        assert frames[1].pts_time == 0.1
        assert frames[1].fps == 10.0
        assert frames[1].image.shape == (8, 8)
        assert frames[1].image[0, 0] == 50.0

    def test_explicit_pts_times(self, tmp_path: Path) -> None:
        path = make_video(
            tmp_path,
            "vid",
            [0.0, 1.0],
            pts_times=[5.0, 6.0],
        )
        frames = list(SyntheticFrameReader().read(path))
        assert [f.pts_time for f in frames] == [5.0, 6.0]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(SyntheticFrameReader().read(tmp_path / "nope.json"))

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(KeyframeExtractionError):
            list(SyntheticFrameReader().read(path))


class TestWriteSyntheticVideo:
    def test_empty_values_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_synthetic_video(tmp_path / "x.json", [])

    def test_pts_mismatch_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_synthetic_video(tmp_path / "x.json", [0.0, 1.0], pts_times=[0.0])

    def test_round_trip(self, tmp_path: Path) -> None:
        path = write_synthetic_video(tmp_path / "x.json", [1.0, 2.0, 3.0])
        frames = list(SyntheticFrameReader().read(path))
        assert [float(f.image[0, 0]) for f in frames] == [1.0, 2.0, 3.0]


# -------------------------------------------------------------------- scorer


class TestFrameDifferenceScorer:
    def test_first_frame_scores_zero(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", [5.0, 5.0, 5.0])
        scores = FrameDifferenceScorer().compute(SyntheticFrameReader().read(path))
        assert scores[0] == 0.0

    def test_constant_video_scores_zero(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", [7.0] * 10)
        scores = FrameDifferenceScorer().compute(SyntheticFrameReader().read(path))
        assert np.all(scores == 0.0)

    def test_single_change(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", [0.0, 0.0, 80.0, 80.0])
        scores = FrameDifferenceScorer().compute(SyntheticFrameReader().read(path))
        assert scores[2] == 80.0

    def test_missing_image_raises(self) -> None:
        frames = [
            VideoFrame(frame_index=0, pts_time=0.0, fps=25.0, image=None),
            VideoFrame(frame_index=1, pts_time=0.04, fps=25.0, image=None),
        ]
        with pytest.raises(KeyframeExtractionError):
            FrameDifferenceScorer().compute(iter(frames))

    def test_name(self) -> None:
        assert FrameDifferenceScorer().name == "frame_difference"


# ------------------------------------------------------------------- selection


class TestCandidatePeaks:
    def test_detects_local_maxima(self) -> None:
        scores = np.array([0.0, 5.0, 40.0, 10.0, 0.0, 30.0, 0.0])
        peaks = candidate_peaks(scores, peak_threshold=8.0)
        assert peaks == [2, 5]

    def test_threshold_filters(self) -> None:
        scores = np.array([0.0, 5.0, 40.0, 10.0, 0.0])
        assert candidate_peaks(scores, peak_threshold=50.0) == []

    def test_edges_never_peaks(self) -> None:
        scores = np.array([100.0, 5.0, 5.0, 100.0])
        assert candidate_peaks(scores, peak_threshold=1.0) == []


class TestTemporalSuppression:
    def test_respects_min_distance(self) -> None:
        scores = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 45.0, 0.0, 40.0, 0.0])
        # peaks at 2, 5, 7
        accepted = temporal_suppression([2, 5, 7], scores, min_temporal_distance=3)
        assert accepted == [2, 5]

    def test_max_keyframes_caps(self) -> None:
        scores = np.array([0.0, 40.0, 0.0, 30.0, 0.0])
        accepted = temporal_suppression(
            [1, 3], scores, min_temporal_distance=1, max_keyframes=1
        )
        # highest score first -> index 1
        assert accepted == [1]

    def test_invalid_distance_rejected(self) -> None:
        with pytest.raises(ValueError):
            temporal_suppression([], np.zeros(3), min_temporal_distance=0)


class TestSelectKeyframes:
    def test_empty_scores(self) -> None:
        assert select_keyframes(np.array([]), min_temporal_distance=5, peak_threshold=0.0) == []

    def test_always_includes_first_frame(self) -> None:
        scores = np.zeros(50)
        indices = select_keyframes(scores, min_temporal_distance=5, peak_threshold=0.0)
        assert indices == [0]

    def test_target_caps_total(self) -> None:
        # 5 well-separated peaks plus frame 0
        scores = np.array([0.0, 50.0, 0.0, 40.0, 0.0, 30.0, 0.0, 20.0, 0.0])
        indices = select_keyframes(scores, min_temporal_distance=1, peak_threshold=0.0, target=3)
        assert len(indices) <= 3
        assert indices[0] == 0

    def test_deterministic(self) -> None:
        scores = np.array([0.0, 50.0, 0.0, 40.0, 0.0, 30.0, 0.0])
        a = select_keyframes(scores, min_temporal_distance=2, peak_threshold=0.0)
        b = select_keyframes(scores, min_temporal_distance=2, peak_threshold=0.0)
        assert a == b

    def test_invalid_target_rejected(self) -> None:
        with pytest.raises(ValueError):
            select_keyframes(np.zeros(10), min_temporal_distance=1, peak_threshold=0.0, target=0)


# ------------------------------------------------------------------ extractor


class TestDakeKeyframeExtractor:
    def test_extracts_ordered_keyframes(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", _scene_values(200, scene_length=40))
        extractor = DakeKeyframeExtractor(
            "L21_V001",
            SyntheticFrameReader(),
            min_temporal_distance=10,
            peak_threshold=10.0,
            target=10,
        )
        keyframes = extractor.extract(path)
        assert extractor.last_frames_scanned == 200
        assert [k.video_id for k in keyframes] == ["L21_V001"] * len(keyframes)
        # ordinals 1..N, frame indices strictly ascending
        assert [k.keyframe_ordinal for k in keyframes] == list(range(1, len(keyframes) + 1))
        assert [k.frame_index for k in keyframes] == sorted(k.frame_index for k in keyframes)
        # first keyframe is frame 0
        assert keyframes[0].frame_index == 0
        # no two keyframes closer than min_temporal_distance
        frames = [k.frame_index for k in keyframes]
        for a, b in zip(frames, frames[1:]):
            assert b - a >= 10

    def test_frames_scanned_tracks_reader(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", [0.0] * 12)
        extractor = DakeKeyframeExtractor("L21_V002", SyntheticFrameReader(), target=3)
        extractor.extract(path)
        assert extractor.last_frames_scanned == 12

    def test_sampling_ratio(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", _scene_values(200, scene_length=10))
        extractor = DakeKeyframeExtractor(
            "L21_V003",
            SyntheticFrameReader(),
            min_temporal_distance=1,
            peak_threshold=0.0,
            sampling_ratio=0.05,
        )
        keyframes = extractor.extract(path)
        # 0.05 * 200 = 10 max keyframes
        assert len(keyframes) <= 10

    def test_target_and_sampling_ratio_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError):
            DakeKeyframeExtractor("L21_V001", SyntheticFrameReader(), target=5, sampling_ratio=0.1)

    def test_invalid_video_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            DakeKeyframeExtractor("nope", SyntheticFrameReader(), target=5)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        extractor = DakeKeyframeExtractor("L21_V001", SyntheticFrameReader(), target=5)
        with pytest.raises(FileNotFoundError):
            extractor.extract(tmp_path / "missing.json")

    def test_constant_video_keeps_first_frame(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", [42.0] * 60)
        keyframes = DakeKeyframeExtractor("L21_V001", SyntheticFrameReader(), target=5).extract(path)
        assert [k.frame_index for k in keyframes] == [0]


class TestExtractVideoKeyframes:
    def test_wrapper(self, tmp_path: Path) -> None:
        path = make_video(tmp_path, "v", _scene_values(80, scene_length=20))
        keyframes = extract_video_keyframes(
            "L21_V001",
            path,
            frame_reader=SyntheticFrameReader(),
            min_temporal_distance=5,
            peak_threshold=5.0,
            target=4,
        )
        assert len(keyframes) >= 1
        assert keyframes[0].video_id == "L21_V001"


# ------------------------------------------------------------------ factories


class TestFactories:
    def test_create_frame_reader_ffmpeg(self) -> None:
        reader = create_frame_reader("ffmpeg", max_side=32)
        assert isinstance(reader, FFmpegFrameReader)

    def test_create_frame_reader_synthetic(self) -> None:
        assert isinstance(create_frame_reader("synthetic"), SyntheticFrameReader)

    def test_create_frame_reader_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_frame_reader("nope")

    def test_create_scorer(self) -> None:
        assert isinstance(create_scorer("frame_difference"), FrameDifferenceScorer)

    def test_create_scorer_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_scorer("nope")