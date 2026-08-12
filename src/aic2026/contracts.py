"""Explicit data contracts for core entities in the AIC 2026 dataset.

These are derived from the actual BTC-provided data under
``data/raw/``. Field names, meanings, types, and ordering guarantees
documented here are the contracts that other pipeline stages rely on.

This module is intentionally free of I/O; it defines the vocabulary.
Per-file parsing and validation live in :mod:`aic2026.entities`.
"""

import re

#: A video id looks like ``L21_V001``: lane ``L<lane>`` + underscore + ``V`` + number.
VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")

#: Keyframe filenames are zero-padded 3-digit indices with an image extension.
KEYFRAME_FILENAME_PATTERN = re.compile(r"^(\d{3})\.(jpg|jpeg|png)$")

#: Object metadata filenames are zero-padded 3-digit keyframe indices.
OBJECT_FILENAME_PATTERN = re.compile(r"^(\d{3})\.json$")

#: Known collection directories under the raw dataset root.
COLLECTION_NAMES = (
    "clip-features-32",
    "keyframes",
    "map-keyframes",
    "media-info",
    "objects",
)

#: Per-video file suffixes per collection (dirs have no suffix).
COLLECTION_FILE_SUFFIXES = {
    "clip-features-32": ".npy",
    "map-keyframes": ".csv",
    "media-info": ".json",
}

#: Required parallel keys for a per-keyframe object detection payload.
OBJECT_PAYLOAD_KEYS = (
    "detection_scores",
    "detection_class_names",
    "detection_class_entities",
    "detection_boxes",
    "detection_class_labels",
)

#: Required top-level keys for a per-video media-info payload.
MEDIA_INFO_KEYS = (
    "author",
    "channel_id",
    "channel_url",
    "description",
    "keywords",
    "length",
    "publish_date",
    "thumbnail_url",
    "title",
    "watch_url",
)


def video_id_from_collection_name(name: str) -> str | None:
    """Extract the ``L*_V*`` video id from a per-video collection entry.

    Expected input:
        - name: a directory or filename found directly under one of the
          collection folders (e.g. ``L21_V001``, ``L21_V001.npy``,
          ``L21_V001.csv``, ``L21_V001.json``).

    Returns:
        - The video id (e.g. ``L21_V001``) if it matches the expected
          pattern; ``None`` otherwise (e.g. ``.DS_Store``, a non-video
          file).

    Raises:
        - No exceptions.
    """
    for suffix in COLLECTION_FILE_SUFFIXES.values():
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if VIDEO_ID_PATTERN.fullmatch(name):
        return name
    return None


def is_expected_collection(name: str) -> bool:
    """Return whether ``name`` is one of the expected collection dirs."""
    return name in COLLECTION_NAMES