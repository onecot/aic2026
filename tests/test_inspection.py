"""Tests for the dataset inspection and validation layer."""

from pathlib import Path

import numpy as np
import pytest

from aic2026.contracts import (
    OBJECT_PAYLOAD_KEYS,
    video_id_from_collection_name,
)
from aic2026.entities import (
    parse_keyframe_map,
    parse_media_info,
    parse_object_detections,
    validate_object_payload,
)
from aic2026.inspection import (
    collection_video_ids,
    discover_video_ids,
    inspect_dataset,
    inspect_video,
    validate_video,
)


def test_video_id_extraction():
    """Video ids are extracted across all expected filename suffixes."""
    assert video_id_from_collection_name("L21_V001") == "L21_V001"
    assert video_id_from_collection_name("L21_V001.npy") == "L21_V001"
    assert video_id_from_collection_name("L21_V001.csv") == "L21_V001"
    assert video_id_from_collection_name("L21_V001.json") == "L21_V001"
    assert video_id_from_collection_name(".DS_Store") is None
    assert video_id_from_collection_name("not_a_video") is None


def test_discover_video_ids(sample_dataset: Path):
    """Discovery returns the union of ids across collections."""
    ids = discover_video_ids(sample_dataset)
    assert ids == ["L21_V001", "L21_V002", "L22_V001"]


def test_collection_video_ids(sample_dataset: Path):
    """Per-collection video discovery works for a file-based collection."""
    assert collection_video_ids(sample_dataset, "clip-features-32") == {
        "L21_V001",
        "L21_V002",
        "L22_V001",
    }


def test_inspect_video_consistent(sample_dataset: Path):
    """A clean video produces no problems and consistent counts."""
    status = inspect_video(sample_dataset, "L21_V001", sample_objects=2)
    assert status.n_keyframes == 3
    assert status.n_objects == 3
    assert status.clip_rows == 3
    assert status.clip_dim == 512
    assert status.clip_dtype == "float16"
    assert status.fps == 25.0
    assert list(status.keyframe_indices) == [1, 2, 3]
    assert status.missing_files == []
    assert validate_video(status) == []


def test_inspect_video_reports_missing_collection(tmp_path: Path):
    """A partial video is reported without raising."""
    root = tmp_path / "raw"
    (root / "keyframes").mkdir(parents=True)
    (root / "keyframes" / "L21_V999").mkdir()
    (root / "keyframes" / "L21_V999" / "001.jpg").write_bytes(b"\xff\xd8\xff")

    status = inspect_video(root, "L21_V999", sample_objects=0)
    assert status.n_keyframes == 1
    assert status.n_objects == 0
    assert status.clip_rows is None
    assert "objects/L21_V999/" in status.missing_files
    assert "clip-features-32/L21_V999.npy" in status.missing_files
    assert "map-keyframes/L21_V999.csv" in status.missing_files
    assert "media-info/L21_V999.json" in status.missing_files
    problems = validate_video(status)
    assert problems, "a partial video must report consistency problems"


def test_inspect_dataset_report(sample_dataset: Path):
    """The report aggregates counts and consistency status."""
    report = inspect_dataset(sample_dataset, max_videos=10, sample_objects=2)
    assert report.video_ids == ["L21_V001", "L21_V002", "L22_V001"]
    assert report.collection_counts["clip-features-32"] == 3
    assert report.collection_counts["keyframes"] == 3
    assert report.total_keyframes == 6
    assert report.total_objects == 6
    assert report.map_rows_total == 6
    assert report.embedding_shapes[(3, 512)] == 1
    assert report.embedding_shapes[(2, 512)] == 1
    assert report.embedding_shapes[(1, 512)] == 1
    assert report.embedding_dtypes["float16"] == 3
    assert report.fps_histogram[25.0] == 3
    assert report.inconsistent_videos == {}
    assert "embedding dimensionality: 512 (uniform)" in report.observations


def test_inspect_dataset_detects_inconsistency(tmp_path: Path):
    """A mismatch between keyframes and objects is reported."""
    root = tmp_path / "raw"
    for collection in (
        "clip-features-32",
        "keyframes",
        "map-keyframes",
        "media-info",
        "objects",
    ):
        (root / collection).mkdir(parents=True, exist_ok=True)
    # keyframes only (no objects, no npy, no csv)
    kf_dir = root / "keyframes" / "L21_V007"
    kf_dir.mkdir()
    (kf_dir / "001.jpg").write_bytes(b"\xff\xd8\xff")
    (kf_dir / "002.jpg").write_bytes(b"\xff\xd8\xff")

    report = inspect_dataset(root, sample_objects=0)
    assert report.inconsistent_videos["L21_V007"]
    assert report.missing_files_total >= 2  # objects/ and clip + map + media


def test_parse_keyframe_map(sample_dataset: Path):
    """map-keyframes CSV parses to the KeyframeMapEntry contract."""
    rows = parse_keyframe_map(sample_dataset / "map-keyframes" / "L21_V001.csv")
    assert [row.keyframe_no for row in rows] == [1, 2, 3]
    assert all(row.fps == 25.0 for row in rows)
    assert [row.frame_idx for row in rows] == [0, 10, 20]


def test_parse_object_detections(sample_dataset: Path):
    """Objects JSON parses to the ObjectDetectionFrame contract."""
    frame = parse_object_detections(
        sample_dataset / "objects" / "L21_V001" / "001.json",
        video_id="L21_V001",
    )
    assert frame.video_id == "L21_V001"
    assert frame.keyframe_no == 1
    assert frame.detection_scores == ["0.99"]


def test_validate_object_payload():
    """Payload validation flags missing and unequal-length keys."""
    assert validate_object_payload(
        {key: ["x"] for key in OBJECT_PAYLOAD_KEYS}
    ) == []
    problems = validate_object_payload({"detection_scores": ["0.5"]})
    assert any("missing keys" in problem for problem in problems)
    payload = {key: ["x"] for key in OBJECT_PAYLOAD_KEYS}
    payload["detection_boxes"] = ["x", "y"]
    assert any("unequal lengths" in problem for problem in validate_object_payload(payload))


def test_parse_media_info(sample_dataset: Path):
    """media-info JSON parses and exposes expected keys."""
    payload = parse_media_info(sample_dataset / "media-info" / "L21_V001.json")
    assert payload["title"] == "Test video"
    assert payload["length"] == 42


def test_inspect_dataset_missing_root(tmp_path: Path):
    """Inspection fails fast on a nonexistent raw root."""
    with pytest.raises(FileNotFoundError):
        inspect_dataset(tmp_path / "does-not-exist")


def test_npy_shape_inspection(sample_dataset: Path):
    """CLIP npy files are read with mmap and expose shape/dtype."""
    report = inspect_dataset(sample_dataset, sample_objects=0)
    assert report.embedding_shapes[(3, 512)] == 1
    assert report.embedding_shapes[(2, 512)] == 1
    assert report.embedding_shapes[(1, 512)] == 1