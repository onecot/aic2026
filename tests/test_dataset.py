"""Tests for the read-only dataset loading layer.

These cover discovery, per-video access, lazy streaming of embeddings
and detections, explicit error behavior for missing/corrupt records,
and cross-platform path handling.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import (
    BoundingBox,
    ClipEmbedding,
    KeyframeDetections,
    KeyframeTiming,
    VideoId,
    VideoMetadata,
)
from aic2026.dataset import (
    CorruptRecordError,
    Dataset,
    DatasetError,
    MissingRecordError,
)
from aic2026.contracts import COLLECTION_NAMES

_REAL_RAW_ROOT = Path(__file__).resolve().parents[1] / "data/raw/aic2026_batch1"

pytestmark = []


def test_video_ids(sample_dataset: Path):
    """Discovery returns the sorted union of video ids."""
    dataset = Dataset(sample_dataset)
    assert dataset.video_ids() == ["L21_V001", "L21_V002", "L22_V001"]


def test_iter_video_ids(sample_dataset: Path):
    """iter_video_ids yields ids in sorted order."""
    dataset = Dataset(sample_dataset)
    assert list(dataset.iter_video_ids()) == ["L21_V001", "L21_V002", "L22_V001"]


def test_iter_yields_video_ids(sample_dataset: Path):
    """__iter__ yields canonical VideoId records."""
    dataset = Dataset(sample_dataset)
    assert list(dataset) == [VideoId("L21_V001"), VideoId("L21_V002"), VideoId("L22_V001")]


def test_len_and_contains(sample_dataset: Path):
    """len and membership behave over discovered videos."""
    dataset = Dataset(sample_dataset)
    assert len(dataset) == 3
    assert "L21_V001" in dataset
    assert "L21_V999" not in dataset
    assert "not_a_video" not in dataset


def test_has_video(sample_dataset: Path):
    """has_video reflects presence in any collection."""
    dataset = Dataset(sample_dataset)
    assert dataset.has_video("L21_V001")
    assert not dataset.has_video("L21_V999")
    assert not dataset.has_video(".DS_Store")


def test_missing_root_raises(tmp_path: Path):
    """A nonexistent root fails fast at construction."""
    with pytest.raises(FileNotFoundError):
        Dataset(tmp_path / "does-not-exist")


def test_keyframe_ordinals(sample_dataset: Path):
    """keyframe_ordinals returns the map-derived ordered list."""
    dataset = Dataset(sample_dataset)
    assert dataset.keyframe_ordinals("L21_V001") == [1, 2, 3]
    assert dataset.keyframe_ordinals("L21_V002") == [1, 2]
    assert dataset.keyframe_ordinals("L22_V001") == [1]


def test_iter_keyframes(sample_dataset: Path):
    """iter_keyframes yields canonical KeyframeKey records in order."""
    dataset = Dataset(sample_dataset)
    keys = list(dataset.iter_keyframes("L21_V001"))
    assert [k.keyframe_ordinal for k in keys] == [1, 2, 3]
    assert all(k.video_id == VideoId("L21_V001") for k in keys)


def test_metadata(sample_dataset: Path):
    """metadata returns canonical VideoMetadata."""
    dataset = Dataset(sample_dataset)
    meta = dataset.metadata("L21_V001")
    assert isinstance(meta, VideoMetadata)
    assert meta.video_id == VideoId("L21_V001")
    assert meta.title == "Test video"
    assert meta.length == 42
    assert meta.keywords == ("k1",)


def test_timings(sample_dataset: Path):
    """timings returns one KeyframeTiming per row in map order."""
    dataset = Dataset(sample_dataset)
    timings = dataset.timings("L21_V001")
    assert isinstance(timings, list)
    assert len(timings) == 3
    assert timings[0].keyframe.keyframe_ordinal == 1
    assert timings[0].pts_time == 0.0
    assert timings[0].frame_index == 0
    assert timings[1].pts_time == 0.4
    assert timings[1].frame_index == 10
    assert all(t.fps == 25.0 for t in timings)


def test_iter_embeddings_lazy(sample_dataset: Path):
    """Embeddings stream one record per matrix row with correct keys."""
    dataset = Dataset(sample_dataset)
    embeddings = list(dataset.iter_embeddings("L21_V001"))
    assert len(embeddings) == 3
    for index, emb in enumerate(embeddings, start=1):
        assert isinstance(emb, ClipEmbedding)
        assert emb.keyframe.keyframe_ordinal == index
        assert emb.embedding_dim() == 512
    np.testing.assert_array_equal(embeddings[0].vector, np.zeros(512, dtype=np.float16))


def test_iter_embeddings_consumes_incrementally(sample_dataset: Path):
    """Pull one embedding at a time without materializing the video first."""
    dataset = Dataset(sample_dataset)
    iterator = dataset.iter_embeddings("L21_V001")
    first = next(iterator)
    assert first.keyframe.keyframe_ordinal == 1
    # Exhausting is unnecessary; partial consumption must not error.
    del iterator


def test_iter_detections(sample_dataset: Path):
    """Detections stream one KeyframeDetections per keyframe."""
    dataset = Dataset(sample_dataset)
    detections = list(dataset.iter_detections("L21_V001"))
    assert len(detections) == 3
    for index, kd in enumerate(detections, start=1):
        assert isinstance(kd, KeyframeDetections)
        assert kd.keyframe.keyframe_ordinal == index
        assert len(kd.detections) == 1
        det = kd.detections[0]
        assert det.score == 0.99
        assert det.class_label == 84
        assert det.box == BoundingBox(0.0, 0.0, 1.0, 1.0)


# ------------------------------------------------------------------ errors


def test_missing_media_info(sample_dataset: Path):
    """A missing media-info file raises MissingRecordError."""
    (sample_dataset / "media-info" / "L21_V001.json").unlink()
    dataset = Dataset(sample_dataset)
    with pytest.raises(MissingRecordError):
        dataset.metadata("L21_V001")


def test_missing_map_csv(sample_dataset: Path):
    """A missing map-keyframes CSV raises MissingRecordError."""
    (sample_dataset / "map-keyframes" / "L21_V001.csv").unlink()
    dataset = Dataset(sample_dataset)
    with pytest.raises(MissingRecordError):
        dataset.timings("L21_V001")


def test_missing_embedding_matrix(sample_dataset: Path):
    """A missing .npy raises MissingRecordError."""
    (sample_dataset / "clip-features-32" / "L21_V001.npy").unlink()
    dataset = Dataset(sample_dataset)
    with pytest.raises(MissingRecordError):
        list(dataset.iter_embeddings("L21_V001"))


def test_missing_object_detections(sample_dataset: Path):
    """A missing per-keyframe object JSON raises MissingRecordError."""
    (sample_dataset / "objects" / "L21_V001" / "002.json").unlink()
    dataset = Dataset(sample_dataset)
    with pytest.raises(MissingRecordError):
        list(dataset.iter_detections("L21_V001"))


def test_corrupt_media_info(sample_dataset: Path):
    """Malformed media-info JSON raises CorruptRecordError."""
    (sample_dataset / "media-info" / "L21_V001.json").write_text("{not valid")
    dataset = Dataset(sample_dataset)
    with pytest.raises(CorruptRecordError):
        dataset.metadata("L21_V001")


def test_corrupt_embedding_matrix(sample_dataset: Path):
    """An unreadable/malformed .npy raises CorruptRecordError."""
    (sample_dataset / "clip-features-32" / "L21_V001.npy").write_bytes(b"not an npy file")
    dataset = Dataset(sample_dataset)
    with pytest.raises(CorruptRecordError):
        list(dataset.iter_embeddings("L21_V001"))


def test_corrupt_object_detections(sample_dataset: Path):
    """Malformed object JSON raises CorruptRecordError."""
    (sample_dataset / "objects" / "L21_V001" / "003.json").write_text("{broken")
    dataset = Dataset(sample_dataset)
    with pytest.raises(CorruptRecordError):
        list(dataset.iter_detections("L21_V001"))


def test_error_hierarchy():
    """Both error types share the DatasetError base."""
    assert issubclass(MissingRecordError, DatasetError)
    assert issubclass(CorruptRecordError, DatasetError)
    assert issubclass(DatasetError, Exception)


# -------------------------------------------------------- cross-platform paths


def test_accepts_str_path(sample_dataset: Path):
    """Dataset accepts a string path as well as a Path."""
    dataset = Dataset(str(sample_dataset))
    assert dataset.raw_root == sample_dataset
    assert dataset.video_ids() == ["L21_V001", "L21_V002", "L22_V001"]


def test_raw_root_is_pathlib_path(sample_dataset: Path):
    """raw_root is exposed as a pathlib.Path, never a bare string."""
    dataset = Dataset(sample_dataset)
    assert isinstance(dataset.raw_root, Path)


def test_paths_with_spaces(tmp_path: Path):
    """A raw root whose path contains spaces works via pathlib.

    Only pathlib-based concatenation can locate the collections safely.
    """
    root = tmp_path / "raw data dir"
    for collection in COLLECTION_NAMES:
        (root / collection).mkdir(parents=True, exist_ok=True)
    (root / "media-info").mkdir(parents=True, exist_ok=True)
    # A single video with one keyframe across all collections.
    import csv

    obj_dir = root / "objects" / "L21_V001"
    kf_dir = root / "keyframes" / "L21_V001"
    obj_dir.mkdir()
    kf_dir.mkdir()
    (kf_dir / "001.jpg").write_bytes(b"\xff\xd8\xff")
    (obj_dir / "001.json").write_text(
        json.dumps(
            {
                "detection_scores": ["0.5"],
                "detection_class_names": ["/m/01jfsr"],
                "detection_class_entities": ["Lantern"],
                "detection_class_labels": ["84"],
                "detection_boxes": [["0.0", "0.0", "1.0", "1.0"]],
            }
        )
    )
    with (root / "map-keyframes" / "L21_V001.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        writer.writerow([1, 0.4, 25.0, 10])
    matrix = np.zeros((1, 512), dtype=np.float16)
    np.save(root / "clip-features-32" / "L21_V001.npy", matrix)
    (root / "media-info" / "L21_V001.json").write_text(
        json.dumps(
            {
                "author": "A",
                "channel_id": "C",
                "channel_url": "U",
                "description": "D",
                "keywords": ["k"],
                "length": 42,
                "publish_date": "d",
                "thumbnail_url": "T",
                "title": "T",
                "watch_url": "W",
            }
        )
    )

    dataset = Dataset(root)
    assert dataset.video_ids() == ["L21_V001"]
    assert len(list(dataset.iter_embeddings("L21_V001"))) == 1
    assert len(list(dataset.iter_detections("L21_V001"))) == 1
    assert dataset.metadata("L21_V001").title == "T"


# ------------------------------------------------------- integration (real data)


def test_integration_loads_tiny_real_subset():
    """Load one small real video end-to-end (skipped when data is absent)."""
    if not _REAL_RAW_ROOT.is_dir():
        pytest.skip("real raw dataset not present")
    dataset = Dataset(_REAL_RAW_ROOT)
    assert "L30_V041" in dataset

    timings = dataset.timings("L30_V041")
    assert len(timings) == 24
    assert timings[0].keyframe.keyframe_ordinal == 1
    assert timings[0].frame_index == 0

    embeddings = list(dataset.iter_embeddings("L30_V041"))
    assert len(embeddings) == len(timings)
    assert all(emb.embedding_dim() == 512 for emb in embeddings)

    detections = list(dataset.iter_detections("L30_V041"))
    assert len(detections) == len(timings)
    for kd in detections:
        assert len(kd.detections) == 100
        assert all(0.0 <= det.score <= 1.0 for det in kd.detections)

    metadata = dataset.metadata("L30_V041")
    assert metadata.video_id == VideoId("L30_V041")
    assert metadata.length > 0