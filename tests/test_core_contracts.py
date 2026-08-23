"""Core contracts: canonical model, raw dataset layer, inspection, evaluation sets."""

from __future__ import annotations


# ==========================================================================
# Section: test_canonical
# ==========================================================================

import json
from pathlib import Path

import numpy as np
import pytest

import aic2026.canonical as canon
from aic2026.canonical import (
    CANONICAL_SCHEMA_VERSION,
    BoundingBox,
    ClipEmbedding,
    Detection,
    KeyframeDetections,
    KeyframeKey,
    KeyframeTiming,
    VideoId,
    VideoMetadata,
)
from aic2026.converters import (
    clip_matrix_to_embeddings,
    keyframe_map_rows_to_timings,
    media_dict_to_metadata,
    object_frame_to_detections,
)
from aic2026.entities import (
    KeyframeMapEntry,
    ObjectDetectionFrame,
    parse_keyframe_map,
    parse_object_detections,
)


def make_key(video_id="L21_V001", ordinal=1) -> KeyframeKey:
    """Build a minimal valid canonical keyframe key."""
    return KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=ordinal)


# ---------------------------------------------------------------- VideoId


def test_video_id_valid():
    """A well-formed video id constructs successfully."""
    assert VideoId("L21_V001").value == "L21_V001"
    assert VideoId("L26_V499").value == "L26_V499"


def test_video_id_invalid():
    """Malformed video ids raise ValueError."""
    for bad in ("", "21_V001", "L21V001", "L21_V001_extra", "l21_v001", ".DS_Store"):
        with pytest.raises(ValueError):
            VideoId(bad)


def test_video_id_round_trip():
    """VideoId survives to_dict/from_dict and JSON serialization."""
    vid = VideoId("L21_V001")
    blob = json.dumps(vid.to_dict())
    assert VideoId.from_dict(json.loads(blob)) == vid


# -------------------------------------------------------------- KeyframeKey


def test_keyframe_key_rejects_non_positive_ordinal():
    """Ordinal must be a positive integer (1-based)."""
    for bad in (0, -1, 1.5, "1"):
        with pytest.raises(ValueError):
            KeyframeKey(video_id=VideoId("L21_V001"), keyframe_ordinal=bad)


def test_keyframe_key_round_trip():
    """KeyframeKey survives serialization."""
    key = make_key(ordinal=7)
    blob = json.dumps(key.to_dict())
    assert KeyframeKey.from_dict(json.loads(blob)) == key


# ----------------------------------------------------------- KeyframeTiming


def test_keyframe_timing_valid():
    """A valid temporal mapping constructs successfully."""
    timing = KeyframeTiming(
        keyframe=make_key(ordinal=1),
        pts_time=3.6,
        fps=25.0,
        frame_index=90,
    )
    assert timing.pts_time == 3.6
    assert timing.fps == 25.0
    assert timing.frame_index == 90


def test_keyframe_timing_invalid_values():
    """Invalid timing values are rejected."""
    for bad_pts in (-0.1, -5):
        with pytest.raises(ValueError):
            KeyframeTiming(keyframe=make_key(), pts_time=bad_pts, fps=25.0, frame_index=0)
    for bad_fps in (0.0, -25.0):
        with pytest.raises(ValueError):
            KeyframeTiming(keyframe=make_key(), pts_time=0.0, fps=bad_fps, frame_index=0)
    for bad_frame in (-1, 1.5, "0"):
        with pytest.raises(ValueError):
            KeyframeTiming(keyframe=make_key(), pts_time=0.0, fps=25.0, frame_index=bad_frame)


def test_ordinal_vs_frame_index_distinct():
    """keyframe_ordinal and frame_index remain distinct concepts.

    A keyframe with ordinal 1 can sit at video frame 90; the two fields
    must be stored separately and never conflated.
    """
    timing = KeyframeTiming(
        keyframe=KeyframeKey(video_id=VideoId("L21_V001"), keyframe_ordinal=1),
        pts_time=3.6,
        fps=25.0,
        frame_index=90,
    )
    assert timing.keyframe.keyframe_ordinal == 1
    assert timing.frame_index == 90
    assert timing.keyframe.keyframe_ordinal != timing.frame_index


def test_keyframe_timing_round_trip():
    """KeyframeTiming survives serialization including fps."""
    timing = KeyframeTiming(
        keyframe=make_key(ordinal=3),
        pts_time=10.6,
        fps=25.0,
        frame_index=265,
    )
    blob = json.dumps(timing.to_dict())
    restored = KeyframeTiming.from_dict(json.loads(blob))
    assert restored == timing
    assert restored.fps == 25.0


# ------------------------------------------------------------- ClipEmbedding


def test_clip_embedding_valid_no_dim_hardcode():
    """ClipEmbedding accepts any 1-D vector, preserving dtype/shape.

    The canonical entity must not hardcode dimension 512.
    """
    v512 = np.zeros(512, dtype=np.float16)
    v10 = np.ones(10, dtype=np.float32)
    emb512 = ClipEmbedding(keyframe=make_key(), vector=v512)
    emb10 = ClipEmbedding(keyframe=make_key(ordinal=2), vector=v10)
    assert emb512.embedding_dim() == 512
    assert emb10.embedding_dim() == 10
    assert emb512.vector.dtype == np.float16
    assert emb10.vector.dtype == np.float32


def test_clip_embedding_rejects_non_1d():
    """ClipEmbedding only accepts 1-D numpy arrays."""
    for bad in (np.zeros((2, 2)), [1.0, 2.0], "abc", None):
        with pytest.raises(ValueError):
            ClipEmbedding(keyframe=make_key(), vector=bad)


def test_clip_embedding_round_trip_preserves_dtype_shape():
    """Base64 serialization preserves float16 dtype, shape, and values."""
    vector = np.arange(8, dtype=np.float16).reshape(8)
    emb = ClipEmbedding(keyframe=make_key(), vector=vector)
    blob = json.dumps(emb.to_dict())
    restored = ClipEmbedding.from_dict(json.loads(blob))
    assert restored.vector.dtype == np.float16
    assert restored.vector.shape == (8,)
    np.testing.assert_array_equal(restored.vector, vector)
    assert restored.keyframe == emb.keyframe


def test_clip_embedding_is_immutable_copy():
    """Mutating the source array does not mutate the canonical record."""
    source = np.zeros(4, dtype=np.float16)
    emb = ClipEmbedding(keyframe=make_key(), vector=source)
    source[0] = 1.0
    assert emb.vector[0] == 0.0


# --------------------------------------------------------------- BoundingBox


def test_bounding_box_valid():
    """A fully normalised box constructs."""
    box = BoundingBox(y_min=0.1, x_min=0.2, y_max=0.8, x_max=0.9)
    assert box.y_min == 0.1


def test_bounding_box_invalid_out_of_range():
    """Box coordinates outside [0, 1] are rejected."""
    for bad in (-0.1, 1.1, 2.0):
        with pytest.raises(ValueError):
            BoundingBox(y_min=bad, x_min=0.0, y_max=1.0, x_max=1.0)


def test_bounding_box_round_trip():
    """BoundingBox survives serialization."""
    box = BoundingBox(0.0, 0.0, 1.0, 1.0)
    assert BoundingBox.from_dict(json.loads(json.dumps(box.to_dict()))) == box


# ---------------------------------------------------------------- Detection


def test_detection_valid():
    """A complete detection record constructs."""
    det = Detection(
        box=BoundingBox(0.0, 0.0, 1.0, 1.0),
        score=0.88,
        class_name="/m/01jfsr",
        class_entity="Lantern",
        class_label=84,
    )
    assert det.class_label == 84


def test_detection_invalid():
    """Scores outside [0, 1] and non-int labels are rejected."""
    for bad_score in (-0.1, 1.1):
        with pytest.raises(ValueError):
            Detection(
                box=BoundingBox(0.0, 0.0, 1.0, 1.0),
                score=bad_score,
                class_name="/m/01jfsr",
                class_entity="Lantern",
                class_label=84,
            )
    with pytest.raises(ValueError):
        Detection(
            box=BoundingBox(0.0, 0.0, 1.0, 1.0),
            score=0.5,
            class_name="/m/01jfsr",
            class_entity="Lantern",
            class_label="84",
        )


def test_detection_round_trip():
    """Detection survives serialization."""
    det = Detection(
        box=BoundingBox(0.1, 0.2, 0.8, 0.9),
        score=0.99,
        class_name="/m/01jfsr",
        class_entity="Lantern",
        class_label=84,
    )
    restored = Detection.from_dict(json.loads(json.dumps(det.to_dict())))
    assert restored == det


# --------------------------------------------------------- KeyframeDetections


def test_keyframe_detections_round_trip():
    """KeyframeDetections survives serialization."""
    dets = KeyframeDetections(
        keyframe=make_key(),
        detections=(
            Detection(BoundingBox(0.0, 0.0, 1.0, 1.0), 0.5, "/m/a", "A", 1),
            Detection(BoundingBox(0.1, 0.1, 0.9, 0.9), 0.7, "/m/b", "B", 2),
        ),
    )
    restored = KeyframeDetections.from_dict(json.loads(json.dumps(dets.to_dict())))
    assert restored == dets
    assert len(restored.detections) == 2


# ------------------------------------------------------------ VideoMetadata


def _meta_payload() -> dict:
    keywords = ("k1", "k2")
    return {
        "author": "Author",
        "channel_id": "UCHANNEL",
        "channel_url": "https://youtube.com/channel/UCHANNEL",
        "description": "desc",
        "keywords": keywords,
        "length": 42,
        "publish_date": "01/01/2024",
        "thumbnail_url": "https://example.com/t.jpg",
        "title": "Title",
        "watch_url": "https://youtube.com/watch?v=TEST",
    }


def test_video_metadata_valid_required_fields():
    """VideoMetadata requires all media-info fields."""
    meta = VideoMetadata(video_id=VideoId("L21_V001"), **_meta_payload())
    assert meta.title == "Title"
    assert meta.keywords == ("k1", "k2")
    assert meta.length == 42


def test_video_metadata_rejects_missing_required_field():
    """A missing required key in media dict raises ValueError via converter."""
    payload = _meta_payload()
    payload.pop("title")
    with pytest.raises(ValueError):
        media_dict_to_metadata("L21_V001", payload)


def test_video_metadata_round_trip():
    """VideoMetadata survives serialization."""
    meta = VideoMetadata(video_id=VideoId("L21_V001"), **_meta_payload())
    restored = VideoMetadata.from_dict(json.loads(json.dumps(meta.to_dict())))
    assert restored == meta


# --------------------------------------------------------------- converters


def _sample_map_height(sample_dataset: Path) -> list[KeyframeMapEntry]:
    return parse_keyframe_map(sample_dataset / "map-keyframes" / "L21_V001.csv")


def test_convert_map_rows_to_timings(sample_dataset: Path):
    """Raw map rows convert to canonical timings preserving fps."""
    rows = _sample_map_height(sample_dataset)
    timings = keyframe_map_rows_to_timings("L21_V001", rows)
    assert len(timings) == 3
    assert [t.keyframe.keyframe_ordinal for t in timings] == [1, 2, 3]
    assert all(t.fps == 25.0 for t in timings)
    assert [t.frame_index for t in timings] == [0, 10, 20]


def test_convert_clip_matrix_to_embeddings(sample_dataset: Path):
    """A CLIP matrix converts to one embedding per keyframe ordinal."""
    matrix = np.arange(6, dtype=np.float16).reshape(3, 2)
    embeddings = clip_matrix_to_embeddings("L21_V001", matrix)
    assert len(embeddings) == 3
    assert [e.keyframe.keyframe_ordinal for e in embeddings] == [1, 2, 3]
    assert [e.embedding_dim() for e in embeddings] == [2, 2, 2]
    np.testing.assert_array_equal(embeddings[1].vector, matrix[1])


def test_convert_clip_matrix_requires_2d():
    """Conversion rejects non-2-D matrices."""
    with pytest.raises(ValueError):
        clip_matrix_to_embeddings("L21_V001", np.zeros(4))


def test_convert_object_frame_to_detections(sample_dataset: Path):
    """Raw object frame converts to typed canonical detections."""
    frame = parse_object_detections(
        sample_dataset / "objects" / "L21_V001" / "001.json",
        video_id="L21_V001",
    )
    detections = object_frame_to_detections(frame)
    assert detections.keyframe.keyframe_ordinal == 1
    assert len(detections.detections) == 1
    det = detections.detections[0]
    assert det.score == 0.99
    assert det.class_label == 84
    assert det.box == BoundingBox(0.0, 0.0, 1.0, 1.0)
    assert det.class_name == "/m/01jfsr"
    assert det.class_entity == "Lantern"


def test_convert_media_to_metadata():
    """Raw media-info dict converts to canonical metadata."""
    meta = media_dict_to_metadata("L21_V001", _meta_payload())
    assert meta.video_id == VideoId("L21_V001")
    assert meta.title == "Title"


def test_convert_media_missing_key_raises():
    """Missing media-info key propagates as a ValueError."""
    payload = _meta_payload()
    payload.pop("author")
    with pytest.raises(ValueError):
        media_dict_to_metadata("L21_V001", payload)


def test_convert_object_frame_unequal_lengths_raises():
    """Unequal parallel lists are rejected by the converter."""
    frame = ObjectDetectionFrame(
        video_id="L21_V001",
        keyframe_no=1,
        detection_scores=["0.99"],
        detection_class_names=["/m/01jfsr"],
        detection_class_entities=["Lantern"],
        detection_class_labels=["84"],
        detection_boxes=[["0.0", "0.0", "1.0", "1.0"], ["0.0", "0.0", "1.0", "1.0"]],
    )
    with pytest.raises(ValueError):
        object_frame_to_detections(frame)


# ------------------------------------------------------------------ schema


def test_schema_version_constant():
    """The canonical schema version is defined and stable."""
    assert isinstance(CANONICAL_SCHEMA_VERSION, str)
    assert CANONICAL_SCHEMA_VERSION == "1.0"


def test_serialized_embedding_carries_version():
    """The embedding serialization carries the schema version."""
    emb = ClipEmbedding(keyframe=make_key(), vector=np.zeros(4, dtype=np.float16))
    assert emb.to_dict()["embedding_version"] == CANONICAL_SCHEMA_VERSION


# ==========================================================================
# Section: test_dataset
# ==========================================================================

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


# ==========================================================================
# Section: test_inspection
# ==========================================================================

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


# ==========================================================================
# Section: test_evaluation
# ==========================================================================

import csv
from pathlib import Path

import pytest

from aic2026.canonical import VideoId
from aic2026.evaluation import (
    AnnotationMetadata,
    EvaluationDatasetError,
    EvaluationQuery,
    EvaluationTask,
    EventAnnotation,
    FrameRange,
    load_eval_set,
)

_HEADERS = [
    "query_id",
    "task",
    "query_text_vi",
    "query_text_en",
    "gt_video_id",
    "gt_frame_start",
    "gt_frame_end",
    "gt_answer",
    "gt_events",
    "annotator",
    "note",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "query_id": "kis_001",
        "task": "kis",
        "query_text_vi": "Tìm cảnh một chiếc thuyền.",
        "query_text_en": "Find a boat scene.",
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], headers: list[str] | None = None) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or _HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_kis_rows_normalizes_csv_whitespace_and_preserves_metadata(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [_row(gt_video_id=" L21_V001\r ", note=" useful annotation ")],
    )

    evaluation_set = load_eval_set(path)
    query = evaluation_set.get_query("kis_001")

    assert len(evaluation_set) == 1
    assert query.task is EvaluationTask.KIS
    assert query.query_text_vi == "Tìm cảnh một chiếc thuyền."
    assert query.query_text_en == "Find a boat scene."
    assert query.ground_truth_video_id.value == "L21_V001"
    assert query.ground_truth_frame_range == FrameRange(10, 20)
    assert query.answer is None
    assert query.event is None
    assert query.annotation == AnnotationMetadata(annotator="P1", note="useful annotation")


def test_load_qa_and_trake_task_specific_values(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [
            _row(
                query_id="qa_001",
                task="QA",
                gt_answer="Đáp án tiếng Việt",
            ),
            _row(
                query_id="trake_001",
                task="trake",
                gt_events="người bước vào phòng",
            ),
        ],
    )

    evaluation_set = load_eval_set(path)
    qa_query = evaluation_set.get_query("qa_001")
    trake_query = evaluation_set.get_query("trake_001")

    assert qa_query.task is EvaluationTask.QA
    assert qa_query.answer == "Đáp án tiếng Việt"
    assert qa_query.event is None
    assert trake_query.task is EvaluationTask.TRAKE
    assert trake_query.answer is None
    assert trake_query.event == EventAnnotation("người bước vào phòng")
    assert list(evaluation_set.iter_queries("QA")) == [qa_query]
    assert evaluation_set.task_counts() == {EvaluationTask.QA: 1, EvaluationTask.TRAKE: 1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "other"}, "task must be one of"),
        ({"query_id": ""}, "query_id must be a non-empty string"),
        ({"gt_video_id": "bad-id"}, "invalid video id"),
        ({"gt_frame_start": "21", "gt_frame_end": "20"}, "frame range end"),
        ({"gt_frame_start": "not-a-frame"}, "gt_frame_start must be an integer"),
        ({"task": "qa"}, "QA queries require a non-empty answer"),
        ({"task": "trake"}, "TRAKE queries require a non-empty event annotation"),
        ({"task": "kis", "gt_answer": "unexpected"}, "KIS queries must not define"),
    ],
)
def test_invalid_task_specific_or_ground_truth_values_fail(tmp_path: Path, overrides: dict[str, str], message: str) -> None:
    path = _write_csv(tmp_path / "eval_set.csv", [_row(**overrides)])

    with pytest.raises(EvaluationDatasetError, match=message):
        load_eval_set(path)


def test_duplicate_query_ids_and_invalid_headers_fail(tmp_path: Path) -> None:
    duplicate_path = _write_csv(tmp_path / "duplicates.csv", [_row(), _row()])
    with pytest.raises(EvaluationDatasetError, match="query_ids must be unique"):
        load_eval_set(duplicate_path)

    invalid_header_path = _write_csv(
        tmp_path / "bad_header.csv",
        [_row()],
        headers=_HEADERS[:-1],
    )
    with pytest.raises(EvaluationDatasetError, match="columns must be"):
        load_eval_set(invalid_header_path)


def test_query_contract_rejects_invalid_optional_task_fields() -> None:
    with pytest.raises(ValueError, match="TRAKE queries require"):
        EvaluationQuery(
            query_id="trake_001",
            task=EvaluationTask.TRAKE,
            query_text_vi="câu hỏi",
            query_text_en=None,
            ground_truth_video_id=VideoId("L21_V001"),
            ground_truth_frame_range=FrameRange(0, 1),
            answer=None,
            event=None,
            annotation=AnnotationMetadata("P1", None),
        )


def test_trake_row_with_empty_frame_fields_and_gt_events_loads(tmp_path: Path) -> None:
    """A valid TRAKE row may leave gt_frame_start/gt_frame_end empty.

    TRAKE ground truth identifies an event (gt_events) within a video; the
    frame range is optional and maps to ``FrameRange(0, 0)``.
    """
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [
            _row(
                query_id="trake_001",
                task="trake",
                gt_frame_start="",
                gt_frame_end="",
                gt_events="2366-2442;3334-3410;4130-4206;6306-6382",
            ),
        ],
    )

    evaluation_set = load_eval_set(path)
    query = evaluation_set.get_query("trake_001")

    assert len(evaluation_set) == 1
    assert query.task is EvaluationTask.TRAKE
    assert query.ground_truth_frame_range == FrameRange(0, 0)
    assert query.answer is None
    assert query.event == EventAnnotation("2366-2442;3334-3410;4130-4206;6306-6382")


def test_trake_row_without_gt_events_fails_even_with_empty_frames(tmp_path: Path) -> None:
    """TRAKE rows must define gt_events even when frame fields are empty."""
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [
            _row(
                query_id="trake_001",
                task="trake",
                gt_frame_start="",
                gt_frame_end="",
                gt_events="",
            ),
        ],
    )

    with pytest.raises(EvaluationDatasetError, match="TRAKE queries require a non-empty event annotation"):
        load_eval_set(path)


def test_kis_and_qa_still_require_frame_fields(tmp_path: Path) -> None:
    """KIS and QA rows must still provide gt_frame_start/gt_frame_end."""
    for task in ("kis", "qa"):
        overrides = {"query_id": f"{task}_001", "task": task, "gt_frame_start": "", "gt_frame_end": ""}
        if task == "qa":
            overrides["gt_answer"] = "Đáp án"
        path = _write_csv(tmp_path / f"{task}.csv", [_row(**overrides)])
        with pytest.raises(EvaluationDatasetError, match="gt_frame_start must be a non-empty string"):
            load_eval_set(path)


def test_real_eval_set_loads_when_available() -> None:
    path = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    if not path.is_file():
        pytest.skip("team evaluation CSV not available")

    evaluation_set = load_eval_set(path)

    assert len(evaluation_set) == 101
    assert evaluation_set.task_counts() == {EvaluationTask.KIS: 101}
    assert all(query.answer is None and query.event is None for query in evaluation_set.queries)
    assert all(
        query.ground_truth_frame_range.start <= query.ground_truth_frame_range.end
        for query in evaluation_set.queries
    )
