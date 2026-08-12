"""Canonical internal data contracts.

This module defines the stable, storage-agnostic representation of the
core AIC 2026 entities that downstream stages rely on. It is
intentionally independent from the BTC raw directory/file layout:
downstream code should never need to know filenames such as
``001.jpg`` or the raw collection names.

Relation to the raw layer:
    - ``aic2026.contracts`` and ``aic2026.entities`` describe the raw
      BTC layout and its file parsers.
    - ``aic2026.canonical`` is the typed, validated internal model.
    - ``aic2026.converters`` bridges raw records to canonical records.

Cardinal distinction of the four identifiers (never interchangeable):
    - video_id: identity of a video, e.g. ``L21_V001``.
    - keyframe_ordinal: 1-based sequential keyframe number within a
      video (matches the raw ``n`` / keyframe filename index).
    - frame_index: 0-based index of the keyframe within the original
      encoded video (raw ``frame_idx``).
    - pts_time: presentation timestamp of the keyframe in seconds.

Every canonical entity supports ``to_dict``/``from_dict`` so records
can be passed across Python code boundaries and serialized for future
API use. Schema versioning is handled via ``CANONICAL_SCHEMA_VERSION``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aic2026.contracts import VIDEO_ID_PATTERN, MEDIA_INFO_KEYS

#: Version of the canonical schema. Bump the minor number on
#: additive/backward-compatible changes, the major number on breaking
#: changes (see ``docs/data-contracts.md``).
CANONICAL_SCHEMA_VERSION = "1.0"

#: Serialized records carry this key with the schema version.
_VERSION_KEY = "version"
_EMBEDDING_VERSION_KEY = "embedding_version"


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class VideoId:
    """Identity of a video.

    Fields:
        - value: the video id string, e.g. ``L21_V001``, matching
          ``^L\\d+_V\\d+$``.
    """

    value: str

    def __post_init__(self) -> None:
        _require(
            VIDEO_ID_PATTERN.fullmatch(self.value) is not None,
            f"invalid video id: {self.value!r} (expected L<lane>_V<number>)",
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict.

        Returns:
            - Dict with the ``value`` field.
        """
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoId":
        """Deserialize from a :meth:`to_dict` dict.

        Expected input:
            - data: dict with a ``value`` key.

        Returns:
            - A :class:`VideoId`.

        Raises:
            - KeyError: if ``value`` is missing.
            - ValueError: if the value is malformed.
        """
        return cls(value=data["value"])

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class KeyframeKey:
    """Identity of one keyframe within a video.

    Fields:
        - video_id: the video the keyframe belongs to.
        - keyframe_ordinal: 1-based sequential keyframe number within
          the video (not the original video frame index).
    """

    video_id: VideoId
    keyframe_ordinal: int

    def __post_init__(self) -> None:
        _require(
            isinstance(self.keyframe_ordinal, int) and self.keyframe_ordinal >= 1,
            f"keyframe_ordinal must be a positive integer, got {self.keyframe_ordinal!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "video_id": self.video_id.to_dict(),
            "keyframe_ordinal": self.keyframe_ordinal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeKey":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            video_id=VideoId.from_dict(data["video_id"]),
            keyframe_ordinal=int(data["keyframe_ordinal"]),
        )


@dataclass(frozen=True)
class KeyframeTiming:
    """Temporal mapping of a keyframe to its original-video position.

    Fields:
        - keyframe: identity of the keyframe.
        - pts_time: presentation timestamp of the keyframe in seconds.
        - fps: frame rate of the original video (kept faithful to the
          raw map-keyframes row; not normalized at this layer).
        - frame_index: 0-based index of the keyframe in the original
          encoded video. Distinct from ``keyframe.keyframe_ordinal``.
    """

    keyframe: KeyframeKey
    pts_time: float
    fps: float
    frame_index: int

    def __post_init__(self) -> None:
        _require(float(self.pts_time) >= 0.0, f"pts_time must be >= 0, got {self.pts_time!r}")
        _require(float(self.fps) > 0.0, f"fps must be > 0, got {self.fps!r}")
        _require(
            isinstance(self.frame_index, int) and self.frame_index >= 0,
            f"frame_index must be a non-negative integer, got {self.frame_index!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "keyframe": self.keyframe.to_dict(),
            "pts_time": self.pts_time,
            "fps": self.fps,
            "frame_index": self.frame_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeTiming":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            keyframe=KeyframeKey.from_dict(data["keyframe"]),
            pts_time=float(data["pts_time"]),
            fps=float(data["fps"]),
            frame_index=int(data["frame_index"]),
        )


@dataclass(frozen=True)
class ClipEmbedding:
    """A single CLIP embedding vector for one keyframe.

    The canonical entity does not enforce any dataset-specific
    dimensionality; it only requires a 1-D numpy array and preserves
    its dtype and shape. Dataset/model-specific dimensional constraints
    (e.g. 512 for the current BTC CLIP features) are the responsibility
    of the dataset or configuration validation layer.

    Fields:
        - keyframe: identity of the keyframe the embedding belongs to.
        - vector: 1-D numpy array (typically float16, shape ``(512,)``
          for the current data; dtype and shape are preserved).
    """

    keyframe: KeyframeKey
    vector: np.ndarray

    def __post_init__(self) -> None:
        _require(
            isinstance(self.vector, np.ndarray) and self.vector.ndim == 1,
            f"embedding vector must be a 1-D numpy array, got {type(self.vector).__name__!r} "
            f"ndim={getattr(self.vector, 'ndim', 'n/a')}",
        )
        # Ensure the vector is owned by the record (no shared views) so
        # serialization is deterministic.
        object.__setattr__(self, "vector", np.array(self.vector, copy=True))

    def embedding_dim(self) -> int:
        """Return the embedding vector dimension.

        Returns:
            - Number of elements in the vector.
        """
        return int(self.vector.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with base64-encoded vector.

        The vector is encoded with shape, dtype, and raw bytes so the
        float dtype/shape are preserved losslessly. This is the JSON
        interchange representation; efficient on-disk array storage is
        handled elsewhere.

        Returns:
            - Dict with ``keyframe``, ``vector`` (base64), ``shape``,
              ``dtype``, and ``embedding_version``.
        """
        return {
            "keyframe": self.keyframe.to_dict(),
            "vector": base64.b64encode(self.vector.tobytes()).decode("ascii"),
            "shape": list(self.vector.shape),
            "dtype": self.vector.dtype.str,
            _EMBEDDING_VERSION_KEY: CANONICAL_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipEmbedding":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if shape/dtype/encoding is inconsistent.
        """
        vector = np.frombuffer(
            base64.b64decode(data["vector"]),
            dtype=np.dtype(data["dtype"]),
        ).reshape(tuple(data["shape"]))
        return cls(
            keyframe=KeyframeKey.from_dict(data["keyframe"]),
            vector=np.array(vector, copy=True),
        )


@dataclass(frozen=True)
class BoundingBox:
    """A normalised axis-aligned bounding box.

    Fields:
        - y_min: top coordinate (normalised, 0..1).
        - x_min: left coordinate (normalised, 0..1).
        - y_max: bottom coordinate (normalised, 0..1).
        - x_max: right coordinate (normalised, 0..1).
    """

    y_min: float
    x_min: float
    y_max: float
    x_max: float

    def __post_init__(self) -> None:
        for name, value in (
            ("y_min", self.y_min),
            ("x_min", self.x_min),
            ("y_max", self.y_max),
            ("x_max", self.x_max),
        ):
            _require(
                0.0 <= float(value) <= 1.0,
                f"box coordinate {name} must be in [0, 1], got {value!r}",
            )

    def to_dict(self) -> dict[str, float]:
        """Serialize to a JSON-safe dict."""
        return {
            "y_min": self.y_min,
            "x_min": self.x_min,
            "y_max": self.y_max,
            "x_max": self.x_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            y_min=float(data["y_min"]),
            x_min=float(data["x_min"]),
            y_max=float(data["y_max"]),
            x_max=float(data["x_max"]),
        )


@dataclass(frozen=True)
class Detection:
    """A single object detection record.

    Fields:
        - box: normalised bounding box.
        - score: detection confidence in [0, 1].
        - class_name: class identifier (OWL-ViT ``/m/...`` name) for the
          current data.
        - class_entity: human-readable entity name.
        - class_label: class label id (integer).
    """

    box: BoundingBox
    score: float
    class_name: str
    class_entity: str
    class_label: int

    def __post_init__(self) -> None:
        _require(
            0.0 <= float(self.score) <= 1.0,
            f"detection score must be in [0, 1], got {self.score!r}",
        )
        _require(
            isinstance(self.class_label, int),
            f"class_label must be an integer, got {self.class_label!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "box": self.box.to_dict(),
            "score": self.score,
            "class_name": self.class_name,
            "class_entity": self.class_entity,
            "class_label": self.class_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            box=BoundingBox.from_dict(data["box"]),
            score=float(data["score"]),
            class_name=data["class_name"],
            class_entity=data["class_entity"],
            class_label=int(data["class_label"]),
        )


@dataclass(frozen=True)
class KeyframeDetections:
    """All object detections for one keyframe.

    Fields:
        - keyframe: identity of the keyframe.
        - detections: tuple of per-detection records in the raw file
          order.
    """

    keyframe: KeyframeKey
    detections: tuple[Detection, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "keyframe": self.keyframe.to_dict(),
            "detections": [det.to_dict() for det in self.detections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeDetections":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            keyframe=KeyframeKey.from_dict(data["keyframe"]),
            detections=tuple(Detection.from_dict(det) for det in data["detections"]),
        )


@dataclass(frozen=True)
class VideoMetadata:
    """Video-level metadata.

    Fields:
        - video_id: the video identity.
        - author: channel author name.
        - channel_id: YouTube channel id.
        - channel_url: YouTube channel URL.
        - description: video description string.
        - keywords: list of keyword strings.
        - length: video length in seconds.
        - publish_date: publish date string (kept as-is, format not
          interpreted by this layer).
        - thumbnail_url: thumbnail URL.
        - title: video title.
        - watch_url: watch URL.
    """

    video_id: VideoId
    author: str
    channel_id: str
    channel_url: str
    description: str
    keywords: tuple[str, ...]
    length: int
    publish_date: str
    thumbnail_url: str
    title: str
    watch_url: str

    def __post_init__(self) -> None:
        _require(isinstance(self.length, int), f"length must be an int, got {self.length!r}")
        _require(
            isinstance(self.keywords, tuple) and all(isinstance(k, str) for k in self.keywords),
            "keywords must be a tuple of strings",
        )
        required_fields = {
            "author": self.author,
            "channel_id": self.channel_id,
            "channel_url": self.channel_url,
            "description": self.description,
            "publish_date": self.publish_date,
            "thumbnail_url": self.thumbnail_url,
            "title": self.title,
            "watch_url": self.watch_url,
        }
        for name, value in required_fields.items():
            _require(
                isinstance(value, str) and value != "",
                f"required metadata field {name!r} must be a non-empty string",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "video_id": self.video_id.to_dict(),
            "author": self.author,
            "channel_id": self.channel_id,
            "channel_url": self.channel_url,
            "description": self.description,
            "keywords": list(self.keywords),
            "length": self.length,
            "publish_date": self.publish_date,
            "thumbnail_url": self.thumbnail_url,
            "title": self.title,
            "watch_url": self.watch_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            video_id=VideoId.from_dict(data["video_id"]),
            author=data["author"],
            channel_id=data["channel_id"],
            channel_url=data["channel_url"],
            description=data["description"],
            keywords=tuple(data["keywords"]),
            length=int(data["length"]),
            publish_date=data["publish_date"],
            thumbnail_url=data["thumbnail_url"],
            title=data["title"],
            watch_url=data["watch_url"],
        )


#: Required keys for a canonical ``VideoMetadata`` payload (used by
#: ``VideoMetadata.from_dict`` validation at the converters layer).
CANONICAL_METADATA_KEYS = tuple(MEDIA_INFO_KEYS)