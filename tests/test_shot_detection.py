"""Tests for the shot-detection package (contract + synthetic detector).

Uses synthetic shot descriptors only; the OmniShotCut model is never loaded
in these tests (it requires CUDA/MPS and a large checkpoint download).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.shot_detection import (
    DEFAULT_OMNISHOTCUT_FILENAME,
    SHOT_DETECTOR_NAMES,
    Shot,
    ShotDetectionError,
    create_shot_detector,
    normalize_shots,
)
from aic2026.shot_detection.synthetic import (
    SyntheticShotDetector,
    write_synthetic_shots,
)


def _write(tmp_path: Path, name: str, shots: list[tuple[int, int]], fps: float = 25.0) -> Path:
    return write_synthetic_shots(tmp_path / f"{name}.json", shots, fps=fps)


class TestShot:
    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            Shot(start_frame=-1, end_frame=0)
        with pytest.raises(ValueError):
            Shot(start_frame=10, end_frame=5)
        with pytest.raises(ValueError):
            Shot(start_frame=1.5, end_frame=2)

    def test_num_frames(self) -> None:
        assert Shot(0, 0).num_frames == 1
        assert Shot(100, 199).num_frames == 100

    def test_round_trip(self) -> None:
        shot = Shot(5, 99)
        assert Shot.from_dict(shot.to_dict()) == shot


class TestNormalizeShots:
    def test_empty_video(self) -> None:
        assert normalize_shots([], 0) == []
        assert normalize_shots([Shot(0, 10)], 0) == []

    def test_empty_shots_full_video(self) -> None:
        assert normalize_shots([], 100) == [Shot(0, 99)]

    def test_preserves_contiguous_boundaries(self) -> None:
        shots = [Shot(0, 99), Shot(100, 299), Shot(300, 499)]
        assert normalize_shots(shots, 500) == shots

    def test_sorts_and_merges_overlap(self) -> None:
        result = normalize_shots([Shot(350, 499), Shot(0, 199), Shot(150, 399)], 500)
        assert result == [Shot(0, 499)]

    def test_adjacent_boundaries_preserved(self) -> None:
        result = normalize_shots([Shot(200, 399), Shot(0, 199), Shot(400, 499)], 500)
        assert result == [Shot(0, 199), Shot(200, 399), Shot(400, 499)]

    def test_fills_gaps(self) -> None:
        result = normalize_shots([Shot(100, 199)], 500)
        assert result == [Shot(0, 99), Shot(100, 199), Shot(200, 499)]

    def test_clamps_to_bounds(self) -> None:
        result = normalize_shots([Shot(90, 600)], 100)
        assert result == [Shot(0, 89), Shot(90, 99)]


class TestSyntheticShotDetector:
    def test_detect_returns_shots(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "vid", [(0, 99), (100, 299)], fps=25.0)
        detector = SyntheticShotDetector()
        shots = detector.detect(path)
        assert shots == [Shot(0, 99), Shot(100, 299)]
        assert detector.last_fps == 25.0
        assert detector.last_num_frames == 300

    def test_detect_unsorted_input(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "vid", [(100, 199), (0, 99)])
        detector = SyntheticShotDetector()
        assert detector.detect(path) == [Shot(0, 99), Shot(100, 199)]

    def test_missing_file(self, tmp_path: Path) -> None:
        detector = SyntheticShotDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        detector = SyntheticShotDetector()
        with pytest.raises(ShotDetectionError):
            detector.detect(path)

    def test_invalid_fps(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "vid", [(0, 10)], fps=0.0)
        detector = SyntheticShotDetector()
        with pytest.raises(ShotDetectionError):
            detector.detect(path)

    def test_invalid_range(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            '{"schema_version": "1.0", "fps": 25.0, "shots": [{"start_frame": 5, "end_frame": 3}]}',
            encoding="utf-8",
        )
        detector = SyntheticShotDetector()
        with pytest.raises(ShotDetectionError):
            detector.detect(path)

    def test_empty_shots_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_synthetic_shots(tmp_path / "empty.json", [])


class TestCreateShotDetector:
    def test_synthetic(self) -> None:
        detector = create_shot_detector("synthetic")
        assert detector.name == "synthetic"
        assert isinstance(detector, SyntheticShotDetector)

    def test_unknown_name(self) -> None:
        with pytest.raises(ValueError):
            create_shot_detector("nope")

    def test_omnishotcut_returns_detector(self) -> None:
        detector = create_shot_detector(
            "omnishotcut",
            checkpoint="uva-cv-lab/OmniShotCut",
            filename=DEFAULT_OMNISHOTCUT_FILENAME,
            mode="clean_shot",
            overlap=20,
            device="auto",
        )
        assert detector.name == "omnishotcut"
        assert detector.last_fps is None
        assert detector.last_num_frames is None
        assert detector.model_load_count == 0


class TestOmniShotCutModelLoadCount:
    def test_model_loads_once_and_is_reused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OmniShotCut model is loaded a single time per detector.

        Substitutes the Torch/OmniShotCut load path so the detector's cache
        behavior is observable without a GPU or checkpoint download.
        """
        from omnishotcut.architecture import backbone, model as osc_model, transformer
        from aic2026.shot_detection import omnishotcut

        detector = create_shot_detector(
            "omnishotcut",
            checkpoint="uva-cv-lab/OmniShotCut",
            filename=DEFAULT_OMNISHOTCUT_FILENAME,
            mode="clean_shot",
            overlap=20,
            device="cpu",
        )

        class _Stub:
            pass

        loaded = {"count": 0}

        def fake_load(checkpoint_path: str, **kwargs: object) -> dict[str, object]:
            args = _Stub()
            args.max_process_window_length = 2
            args.num_intra_relation_classes = 1
            args.num_inter_relation_classes = 1
            args.num_queries = 1
            args.aux_loss = False
            model = _Stub()
            model.load_state_dict = lambda *a, **k: None
            model.to = lambda *a, **k: model
            model.eval = lambda: None
            loaded["count"] += 1
            return {"args": args, "model": model}

        monkeypatch.setattr(omnishotcut, "resolve_device", lambda requested: "cpu")
        # Point the detector at a local checkpoint file so no HF download runs.
        ckpt = tmp_path / "ckpt.pth"
        ckpt.write_bytes(b"placeholder")
        monkeypatch.setattr(
            omnishotcut.OmniShotCutShotDetector,
            "_resolve_checkpoint_path",
            lambda self: str(ckpt),
        )
        monkeypatch.setattr("torch.load", fake_load)
        monkeypatch.setattr(backbone, "build_backbone", lambda args: _Stub())
        monkeypatch.setattr(transformer, "build_transformer", lambda args: _Stub())

        def fake_omnishotcut(*args: object, **kwargs: object) -> _Stub:
            model = _Stub()
            model.load_state_dict = lambda *a, **k: None
            model.to = lambda *a, **k: model
            model.eval = lambda: None
            return model

        monkeypatch.setattr(osc_model, "OmniShotCut", fake_omnishotcut)

        detector._ensure_model()
        detector._ensure_model()

        assert loaded["count"] == 1
        assert detector.model_load_count == 1

    def test_names(self) -> None:
        assert "omnishotcut" in SHOT_DETECTOR_NAMES
        assert "synthetic" in SHOT_DETECTOR_NAMES
