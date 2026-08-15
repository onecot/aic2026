"""Deterministic merge of completed keyframe-extraction shards.

Combines multiple completed extraction output directories (each produced by
``aic2026.keyframe_extraction.extract_keyframes``) into a single canonical
keyframe artifact::

    merged/
        frames.ndjson             canonical keyframe records (row order)
        map-keyframes/
            L21_V001.csv          per-video ``n,pts_time,fps,frame_idx`` maps
            ...
        manifest.json             versioned merge metadata

Merge guarantees:

- every shard must be complete (a ``manifest.json`` exists) and internally
  consistent;
- all shards must share the same extractor, scorer, frame reader, and
  extraction configuration (mixing is rejected);
- no keyframe may appear more than once across shards; no video may be
  duplicated across shards (each video must live in exactly one shard);
- the merge order is deterministic: shards are ordered by their manifest, and
  within a shard records keep file order;
- the merged ``map-keyframes/*.csv`` files are rebuilt deterministically from
  the merged records, so they are always consistent with ``frames.ndjson``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.keyframe_extraction import (
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    MAP_KEYFRAMES_DIRNAME,
    ExtractionManifest,
    write_video_map,
)
from aic2026.keyframe_extractor import Keyframe

#: Schema version of the merge manifest.
MERGE_MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename of the merge manifest inside the output directory.
MERGE_MANIFEST_FILENAME = "manifest.json"

logger = logging.getLogger(__name__)


class MergeError(RuntimeError):
    """Raised when shards cannot be merged (incomplete, inconsistent, or duplicated)."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_timings(path: Path) -> list[dict[str, Any]]:
    """Read canonical keyframe records from a ``frames.ndjson`` file."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class MergeManifest:
    """Versioned metadata describing one merged keyframe artifact.

    Fields:
        - schema_version: merge schema version
          (``MERGE_MANIFEST_SCHEMA_VERSION``).
        - extractor / extractor_version / scorer / frame_reader: shared
          extraction identity across all merged shards.
        - detector / sampler: shared shot-detector and sampler names
          (additive; ``"n/a"`` for the ``dake`` extractor).
        - parameters: shared extraction configuration dict.
        - config_hash: sha256 identity of the shared configuration.
        - num_videos / num_keyframes: merged artifact counts.
        - source_shards: the merged shard directory paths, in merge order.
        - created_at: ISO-8601 timestamp of merge completion.
    """

    schema_version: str
    extractor: str
    extractor_version: str
    scorer: str
    frame_reader: str
    parameters: dict[str, Any]
    config_hash: str
    num_videos: int
    num_keyframes: int
    source_shards: tuple[str, ...]
    created_at: str
    detector: str = "n/a"
    sampler: str = "n/a"

    def __post_init__(self) -> None:
        _require(
            self.schema_version == MERGE_MANIFEST_SCHEMA_VERSION,
            f"unsupported merge schema version {self.schema_version!r}",
        )
        _require(isinstance(self.extractor, str) and self.extractor != "", "extractor required")
        _require(isinstance(self.extractor_version, str) and self.extractor_version != "", "extractor_version required")
        _require(isinstance(self.scorer, str) and self.scorer != "", "scorer required")
        _require(isinstance(self.frame_reader, str) and self.frame_reader != "", "frame_reader required")
        _require(isinstance(self.detector, str) and self.detector != "", "detector required")
        _require(isinstance(self.sampler, str) and self.sampler != "", "sampler required")
        _require(isinstance(self.parameters, dict), "parameters must be a dict")
        _require(isinstance(self.config_hash, str) and self.config_hash != "", "config_hash required")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.num_keyframes, int) and self.num_keyframes >= 0, "num_keyframes must be >= 0")
        _require(len(self.source_shards) >= 1, "source_shards must not be empty")
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "scorer": self.scorer,
            "frame_reader": self.frame_reader,
            "detector": self.detector,
            "sampler": self.sampler,
            "parameters": self.parameters,
            "config_hash": self.config_hash,
            "num_videos": self.num_videos,
            "num_keyframes": self.num_keyframes,
            "source_shards": list(self.source_shards),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeManifest":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            schema_version=data["schema_version"],
            extractor=data["extractor"],
            extractor_version=data["extractor_version"],
            scorer=data["scorer"],
            frame_reader=data["frame_reader"],
            detector=data.get("detector", "n/a"),
            sampler=data.get("sampler", "n/a"),
            parameters=data["parameters"],
            config_hash=data["config_hash"],
            num_videos=int(data["num_videos"]),
            num_keyframes=int(data["num_keyframes"]),
            source_shards=tuple(data["source_shards"]),
            created_at=data["created_at"],
        )


def _load_shard(
    shard_dir: Path,
    reference: ExtractionManifest,
) -> tuple[ExtractionManifest, list[dict[str, Any]]]:
    """Load and validate one shard against a reference manifest.

    Raises:
        - MergeError: if the shard is incomplete or inconsistent with the
          reference (extractor, scorer, frame reader, or config hash).
    """
    manifest_path = shard_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MergeError(f"shard is not complete (missing {MANIFEST_FILENAME}): {shard_dir}")
    manifest = ExtractionManifest.from_dict(_read_json(manifest_path))

    for field_name, value, expected in (
        ("extractor", manifest.extractor, reference.extractor),
        ("extractor_version", manifest.extractor_version, reference.extractor_version),
        ("scorer", manifest.scorer, reference.scorer),
        ("frame_reader", manifest.frame_reader, reference.frame_reader),
        ("detector", manifest.detector, reference.detector),
        ("sampler", manifest.sampler, reference.sampler),
        ("config_hash", manifest.config_hash, reference.config_hash),
    ):
        if value != expected:
            raise MergeError(
                f"shard {field_name} {value!r} != merged {expected!r}: {shard_dir}"
            )

    frames_path = shard_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        raise MergeError(f"shard is missing {FRAMES_FILENAME}: {shard_dir}")
    frames = _read_timings(frames_path)
    if manifest.num_keyframes != len(frames):
        raise MergeError(
            f"shard frames rows {len(frames)} != manifest {manifest.num_keyframes}: {shard_dir}"
        )
    return manifest, frames


def _collect_keys(frames: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], set[str]]:
    """Return ``(frame_keys, video_ids)`` for a shard's keyframe records.

    A frame key is ``(video_id, frame_index)`` (the original-video position);
    the video set is the distinct video ids present.
    """
    frame_keys: set[tuple[str, int]] = set()
    video_ids: set[str] = set()
    for row in frames:
        kf = row["keyframe"]
        video_id = kf["video_id"]["value"]
        video_ids.add(video_id)
        frame_keys.add((video_id, int(row["frame_index"])))
    return frame_keys, video_ids


def merge_shards(
    shard_dirs: list[Path | str],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> MergeManifest:
    """Merge completed extraction shards into a single canonical artifact.

    Expected input:
        - shard_dirs: completed extraction directories (each with
          ``manifest.json`` and ``frames.ndjson``).
        - output_dir: target directory for the merged artifact (created if
          absent).
        - overwrite: if True, replace an existing merged artifact; otherwise
          raise ``MergeError`` when the output already exists.

    Returns:
        - A :class:`MergeManifest` describing the merged artifact.

    Raises:
        - MergeError: on incomplete/inconsistent shards, mixed configurations,
          duplicate or missing videos, overlapping keyframes, or a
          pre-existing non-overwritten output.
        - FileNotFoundError: if a shard directory is missing.
        - OSError: if outputs cannot be written.
    """
    shard_dirs = [Path(d) for d in shard_dirs]
    if not shard_dirs:
        raise MergeError("at least one shard directory is required")
    for directory in shard_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"shard directory does not exist: {directory}")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise MergeError(
            f"merge output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )

    ordered = sorted(shard_dirs, key=lambda d: str(d))
    # Read the reference from the first shard after confirming it is complete
    # (a missing manifest raises MergeError, not FileNotFoundError).
    reference_path = ordered[0] / MANIFEST_FILENAME
    if not reference_path.is_file():
        raise MergeError(f"shard is not complete (missing {MANIFEST_FILENAME}): {ordered[0]}")
    reference = ExtractionManifest.from_dict(_read_json(reference_path))

    all_frames: list[list[dict[str, Any]]] = []
    seen_frame_keys: set[tuple[str, int]] = set()
    seen_videos: set[str] = set()
    for directory in ordered:
        manifest, frames = _load_shard(directory, reference)
        frame_keys, video_ids = _collect_keys(frames)
        overlap = frame_keys & seen_frame_keys
        if overlap:
            raise MergeError(f"duplicate keyframe across shards: {sorted(overlap)[:5]}")
        duplicate_videos = video_ids & seen_videos
        if duplicate_videos:
            raise MergeError(f"video appears in more than one shard: {sorted(duplicate_videos)[:5]}")
        seen_frame_keys |= frame_keys
        seen_videos |= video_ids
        all_frames.append(frames)

    total_frames = sum(len(f) for f in all_frames)
    total_videos = len(seen_videos)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_out = output_dir / FRAMES_FILENAME
    tmp_frames = frames_out.with_name(frames_out.name + ".tmp")
    with tmp_frames.open("w", encoding="utf-8") as handle:
        for frames in all_frames:
            for row in frames:
                handle.write(json.dumps(row) + "\n")
    tmp_frames.replace(frames_out)

    # Rebuild per-video maps deterministically from the merged records so the
    # CSV files are always consistent with frames.ndjson.
    video_rows: dict[str, list[Keyframe]] = {}
    for frames in all_frames:
        for row in frames:
            kf = row["keyframe"]
            video_id = kf["video_id"]["value"]
            keyframe = Keyframe(
                video_id=video_id,
                keyframe_ordinal=int(kf["keyframe_ordinal"]),
                pts_time=float(row["pts_time"]),
                fps=float(row["fps"]),
                frame_index=int(row["frame_index"]),
            )
            video_rows.setdefault(video_id, []).append(keyframe)

    maps_dir = output_dir / MAP_KEYFRAMES_DIRNAME
    maps_dir.mkdir(parents=True, exist_ok=True)
    for video_id, keyframes in video_rows.items():
        keyframes.sort(key=lambda k: k.keyframe_ordinal)
        write_video_map(maps_dir / f"{video_id}.csv", keyframes)

    manifest = MergeManifest(
        schema_version=MERGE_MANIFEST_SCHEMA_VERSION,
        extractor=reference.extractor,
        extractor_version=reference.extractor_version,
        scorer=reference.scorer,
        frame_reader=reference.frame_reader,
        detector=reference.detector,
        sampler=reference.sampler,
        parameters=reference.parameters,
        config_hash=reference.config_hash,
        num_videos=total_videos,
        num_keyframes=total_frames,
        source_shards=tuple(str(d) for d in ordered),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_atomic(output_dir / MERGE_MANIFEST_FILENAME, manifest.to_dict())
    logger.info("merged %d shards -> %d keyframes for %d videos", len(ordered), total_frames, total_videos)
    return manifest