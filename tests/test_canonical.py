"""Tests for the canonical internal data contracts and converters."""

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