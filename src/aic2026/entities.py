"""Typed models and parsers for the discovered raw dataset entities.

These model the records as they actually occur in the BTC-provided
files. They are the explicit contracts that other stages may rely on.

Field meanings and constraints are documented on each model. The
vocabulary (patterns, collection names, required keys) lives in
:mod:`aic2026.contracts`.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from aic2026.contracts import (
    MEDIA_INFO_KEYS,
    OBJECT_FILENAME_PATTERN,
    OBJECT_PAYLOAD_KEYS,
)
from aic2026.contracts import video_id_from_collection_name  # re-exported helper


@dataclass(frozen=True)
class KeyframeMapEntry:
    """One row of a ``map-keyframes`` CSV.

    Maps a keyframe sequence number to its position in the original
    video. The CSV columns are ``n, pts_time, fps, frame_idx``.

    Fields:
        - keyframe_no: 1-based keyframe index, matches the ``n`` column
          and the keyframe file index in
          ``keyframes/<video_id>/NNN.jpg``.
        - pts_time: presentation timestamp in seconds.
        - fps: frame rate of the original video.
        - frame_idx: 0-based index of the keyframe in the original video.
    """

    keyframe_no: int
    pts_time: float
    fps: float
    frame_idx: int


@dataclass(frozen=True)
class ObjectDetectionFrame:
    """Object detections for one keyframe (one ``objects/*.json`` file).

    Fields:
        - video_id: the video the keyframe belongs to.
        - keyframe_no: 1-based keyframe index, matches the JSON filename.
        - detection_scores: parallel list of detection scores.
        - detection_class_names: parallel list of class identifiers
          (OWL-ViT ``/m/...`` names) aligned with scores.
        - detection_class_entities: parallel list of human-readable
          entity names aligned with scores.
        - detection_class_labels: parallel list of class label ids
          (strings) aligned with scores.
        - detection_boxes: parallel list of normalised bounding boxes;
          each box is a list of four floats in [0, 1].
    """

    video_id: str
    keyframe_no: int
    detection_scores: list[str]
    detection_class_names: list[str]
    detection_class_entities: list[str]
    detection_class_labels: list[str]
    detection_boxes: list[list[str]]


def parse_keyframe_map(csv_path: Path) -> list[KeyframeMapEntry]:
    """Parse a ``map-keyframes`` CSV into rows of the data contract.

    Expected input:
        - csv_path: path to a per-video ``map-keyframes`` CSV.

    Returns:
        - Ordered list of :class:`KeyframeMapEntry` matching the CSV
          header order ``n, pts_time, fps, frame_idx``.

    Raises:
        - FileNotFoundError: if the CSV does not exist.
        - KeyError: if a required column is missing.
        - ValueError: if a cell cannot be converted to its type.
    """
    entries: list[KeyframeMapEntry] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                KeyframeMapEntry(
                    keyframe_no=int(row["n"]),
                    pts_time=float(row["pts_time"]),
                    fps=float(row["fps"]),
                    frame_idx=int(row["frame_idx"]),
                )
            )
    return entries


def parse_object_detections(json_path: Path, video_id: str) -> ObjectDetectionFrame:
    """Parse an ``objects`` JSON file into the object detection contract.

    Expected input:
        - json_path: path to one per-keyframe object detection JSON.
        - video_id: the video the keyframe belongs to.

    Returns:
        - An :class:`ObjectDetectionFrame` with all parallel fields
          intact, in file order.

    Raises:
        - FileNotFoundError: if the JSON does not exist.
        - json.JSONDecodeError: if the JSON is malformed.
        - KeyError: if a required key is missing.
    """
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    match = OBJECT_FILENAME_PATTERN.match(json_path.name)
    if match is None:
        raise ValueError(f"object file does not follow NNN.json: {json_path.name}")
    return ObjectDetectionFrame(
        video_id=video_id,
        keyframe_no=int(match.group(1)),
        detection_scores=payload["detection_scores"],
        detection_class_names=payload["detection_class_names"],
        detection_class_entities=payload["detection_class_entities"],
        detection_class_labels=payload["detection_class_labels"],
        detection_boxes=payload["detection_boxes"],
    )


def parse_media_info(json_path: Path) -> dict[str, object]:
    """Parse a ``media-info`` JSON and validate its top-level keys.

    Expected input:
        - json_path: path to one per-video ``media-info`` JSON.

    Returns:
        - The parsed JSON dict, guaranteed to contain all keys in
          ``MEDIA_INFO_KEYS``.

    Raises:
        - FileNotFoundError: if the JSON does not exist.
        - json.JSONDecodeError: if the JSON is malformed.
        - KeyError: if a required top-level key is missing.
    """
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = [key for key in MEDIA_INFO_KEYS if key not in payload]
    if missing:
        raise KeyError(f"media-info missing keys: {missing}")
    return payload


def validate_object_payload(payload: dict) -> list[str]:
    """Check an object-detection payload for schema consistency.

    Expected input:
        - payload: parsed ``objects`` JSON dict.

    Returns:
        - List of problems found; empty list when the payload is valid.

    Raises:
        - No exceptions.
    """
    problems: list[str] = []
    missing = [key for key in OBJECT_PAYLOAD_KEYS if key not in payload]
    if missing:
        problems.append(f"missing keys: {missing}")
        return problems
    lengths = {key: len(payload[key]) for key in OBJECT_PAYLOAD_KEYS}
    if len(set(lengths.values())) > 1:
        problems.append(f"parallel lists have unequal lengths: {lengths}")
    return problems