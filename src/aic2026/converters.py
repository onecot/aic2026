"""Raw-to-canonical conversion helpers.

These functions bridge the raw BTC parsing layer (``aic2026.entities``)
to the canonical internal contract (``aic2026.canonical``). They are
purely in-memory transforms over already-parsed raw records; they do
not read or write files. Downstream pipeline stages should consume the
canonical entities, not the raw layouts.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from aic2026.contracts import MEDIA_INFO_KEYS  # noqa: E402
from aic2026.canonical import (
    BoundingBox,
    ClipEmbedding,
    Detection,
    KeyframeDetections,
    KeyframeKey,
    KeyframeTiming,
    VideoId,
    VideoMetadata,
)
from aic2026.entities import (
    KeyframeMapEntry,
    ObjectDetectionFrame,
    validate_object_payload,
)


def keyframe_key_from_parts(video_id: str, keyframe_ordinal: int) -> KeyframeKey:
    """Build a canonical ``KeyframeKey`` from a raw video id and ordinal.

    Expected input:
        - video_id: the ``L*_V*`` video id string.
        - keyframe_ordinal: 1-based keyframe sequence number.

    Returns:
        - The canonical :class:`KeyframeKey`.

    Raises:
        - ValueError: if the video id or ordinal is invalid.
    """
    return KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=keyframe_ordinal)


def keyframe_map_rows_to_timings(
    video_id: str, rows: Iterable[KeyframeMapEntry]
) -> list[KeyframeTiming]:
    """Convert raw map-keyframes rows to canonical timings.

    Expected input:
        - video_id: the ``L*_V*`` video id the rows belong to.
        - rows: iterable of raw :class:`KeyframeMapEntry` records.

    Returns:
        - List of canonical :class:`KeyframeTiming` records in the same
          order as the input rows.

    Raises:
        - ValueError: if any value violates the canonical contract.
    """
    timings: list[KeyframeTiming] = []
    for row in rows:
        timings.append(
            KeyframeTiming(
                keyframe=keyframe_key_from_parts(video_id, row.keyframe_no),
                pts_time=row.pts_time,
                fps=row.fps,
                frame_index=row.frame_idx,
            )
        )
    return timings


def clip_matrix_to_embeddings(video_id: str, matrix: np.ndarray) -> list[ClipEmbedding]:
    """Convert a CLIP feature matrix into per-keyframe canonical embeddings.

    Expected input:
        - video_id: the ``L*_V*`` video id the matrix belongs to.
        - matrix: 2-D numpy array with shape ``(num_keyframes, dim)``;
          row ``i`` (0-based) maps to keyframe ordinal ``i + 1``.

    Returns:
        - List of :class:`ClipEmbedding` records, one per matrix row.

    Raises:
        - ValueError: if the matrix is not 2-D.
        - ValueError: if any row is not 1-D (should not happen after the
          2-D check).
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"CLIP feature matrix must be 2-D, got ndim={matrix.ndim} shape={matrix.shape}"
        )
    embeddings: list[ClipEmbedding] = []
    for ordinal, vector in enumerate(matrix, start=1):
        embeddings.append(
            ClipEmbedding(
                keyframe=keyframe_key_from_parts(video_id, ordinal),
                vector=np.asarray(vector),
            )
        )
    return embeddings


def object_frame_to_detections(frame: ObjectDetectionFrame) -> KeyframeDetections:
    """Convert one raw object detection frame into canonical detections.

    Expected input:
        - frame: raw :class:`ObjectDetectionFrame` record.

    Returns:
        - Canonical :class:`KeyframeDetections` for that keyframe, with
          str-encoded raw values converted to typed values (float score,
          int label, float box coordinates).

    Raises:
        - ValueError: if the payload violates schema consistency or any
          value violates the canonical contract.
    """
    problems = _validate_object_frame(frame)
    if problems:
        raise ValueError(f"invalid object detection payload: {problems}")
    detections: list[Detection] = []
    for i in range(len(frame.detection_scores)):
        box_raw = frame.detection_boxes[i]
        detections.append(
            Detection(
                box=BoundingBox(
                    y_min=float(box_raw[0]),
                    x_min=float(box_raw[1]),
                    y_max=float(box_raw[2]),
                    x_max=float(box_raw[3]),
                ),
                score=float(frame.detection_scores[i]),
                class_name=frame.detection_class_names[i],
                class_entity=frame.detection_class_entities[i],
                class_label=int(frame.detection_class_labels[i]),
            )
        )
    return KeyframeDetections(
        keyframe=keyframe_key_from_parts(frame.video_id, frame.keyframe_no),
        detections=tuple(detections),
    )


def _validate_object_frame(frame: ObjectDetectionFrame) -> list[str]:
    """Validate the parallel lists of a raw object frame.

    Expected input:
        - frame: raw :class:`ObjectDetectionFrame`.

    Returns:
        - List of problems; empty when consistent.

    Raises:
        - No exceptions.
    """
    payload = {
        "detection_scores": frame.detection_scores,
        "detection_class_names": frame.detection_class_names,
        "detection_class_entities": frame.detection_class_entities,
        "detection_boxes": frame.detection_boxes,
        "detection_class_labels": frame.detection_class_labels,
    }
    return validate_object_payload(payload)


def media_dict_to_metadata(video_id: str, payload: dict) -> VideoMetadata:
    """Convert a raw ``media-info`` JSON dict into canonical metadata.

    Expected input:
        - video_id: the ``L*_V*`` video id the payload belongs to.
        - payload: the parsed ``media-info`` JSON dict.

    Returns:
        - Canonical :class:`VideoMetadata`.

    Raises:
        - ValueError: if a required key is missing or any value violates
          the canonical contract.
    """
    missing = [key for key in MEDIA_INFO_KEYS if key not in payload]
    if missing:
        raise ValueError(f"media-info payload missing keys: {missing}")
    for key in MEDIA_INFO_KEYS:
        val = payload[key]
        if key in (
            "author",
            "channel_id",
            "channel_url",
            "description",
            "publish_date",
            "thumbnail_url",
            "title",
            "watch_url",
        ):
            if not (isinstance(val, str) and val != ""):
                raise ValueError(f"field {key!r} must be a non-empty string")
        elif key == "keywords":
            if not (isinstance(val, (list, tuple)) and all(isinstance(k, str) for k in val)):
                raise ValueError(f"field keywords must be a tuple/list of strings")
        elif key == "length":
            if not isinstance(val, int):
                raise ValueError(f"field length must be an int")
        else:
            raise ValueError(f"unexpected media-info key {key!r}")
    return VideoMetadata(
        video_id=VideoId(video_id),
        author=payload["author"],
        channel_id=payload["channel_id"],
        channel_url=payload["channel_url"],
        description=payload["description"],
        keywords=tuple(payload["keywords"]),
        length=int(payload["length"]),
        publish_date=payload["publish_date"],
        thumbnail_url=payload["thumbnail_url"],
        title=payload["title"],
        watch_url=payload["watch_url"],
    )