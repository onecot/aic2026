"""Materialization of adaptive keyframe images from the original videos.

Given a video manifest (``videos.json``) and a canonical keyframe artifact's
``frames.ndjson`` (produced by extraction or merge), this module decodes the
selected frames from the original mp4 files and writes them as JPEG images.
The result is our own adaptive keyframe image collection (artifact id
``adaptive_sbd_v1``), removing the pipeline's dependency on the BTC-provided
keyframe images for the embedding step.

Output artifact::

    <output-dir>/
        images/
            <video_id>/
                000001.jpg            (keyframe_ordinal 1)
                000002.jpg            (keyframe_ordinal 2)
                ...
        frames.ndjson                copy of the input records (validated)
        manifest.json                versioned materialization metadata

Ordinal contract: the ``frames.ndjson`` record with ``keyframe_ordinal N``
maps to ``images/<video_id>/<N:06d>.jpg``. This is the same key the embedder
resolves via ``aic2026.frame_images.find_keyframe_image`` (canonical
layout, with a legacy ``keyframes/`` fallback).

Behavior:

- deterministic: identical inputs produce identical image names and the same
  ``frames.ndjson`` (a byte copy); image bytes are produced by ffmpeg.
- batched decoding: each video's selected frame indices are grouped into
  batches of at most ``DEFAULT_MAX_FRAMES_PER_BATCH`` (64), decoded by one
  ffmpeg pass per batch (keeping the ``select`` filter expression within
  ffmpeg's term-count limit for long videos), and merged preserving keyframe
  ordinal order.
- safe: the artifact is assembled in a sibling staging directory and moved
  into place only after it validates; the ``manifest.json`` is written last
  and atomically, so a partial run is never exposed as a valid artifact.
- validated: the finished artifact is re-read and checked (every frame record
  has an image, no image is missing, image count matches ``frames.ndjson``)
  before it is marked complete.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.artifact_io import read_json_object, write_json_atomic
from aic2026.canonical import KeyframeTiming
from aic2026.embedding_artifact import iter_frames
from aic2026.assets.keyframe_extraction import (
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    load_video_manifest,
    require_media_tools,
    resolve_video_path,
)
from aic2026.progress import ProgressReporter, create_progress_reporter

logger = logging.getLogger(__name__)

#: Schema version of the materialization manifest.
MATERIALIZE_MANIFEST_SCHEMA_VERSION = "1.0"

#: Identifier of the adaptive keyframe image artifact.
ARTIFACT_ID = "adaptive_sbd_v1"

#: Directory (relative to the output dir) holding the per-video images.
#: Canonical definition lives in :mod:`aic2026.frame_images` so runtime code
#: never imports this offline module.
from aic2026.frame_images import IMAGES_DIRNAME as IMAGES_DIRNAME

#: Image extension written by materialization.
IMAGE_EXTENSION = "jpg"

#: Zero-padded width of ``keyframe_ordinal`` in materialized image filenames.
IMAGE_ORDINAL_DIGITS = 6

#: Suffix of the sibling staging directory used for safe assembly.
STAGING_SUFFIX = ".staging"

#: Maximum frame indices grouped into a single ffmpeg ``select`` filter
#: invocation. ffmpeg's expression parser rejects ``select`` expressions with
#: too many ``eq(n,<idx>)`` terms (observed to fail at ~100 terms in ffmpeg
#: 8.1; the exact cap varies by build). 64 keeps well below the parser limit
#: while staying large enough that long videos need only a few passes; larger
#: videos are decoded in several ffmpeg passes and merged in ordinal order.
DEFAULT_MAX_FRAMES_PER_BATCH = 64


class MaterializeError(RuntimeError):
    """Raised when materialization fails or produces an invalid artifact."""


from aic2026.frame_images import image_filename  # canonical definition

def image_relpath(video_id: str, keyframe_ordinal: int) -> str:
    """Return the canonical relative path of a materialized image.

    Expected input:
        - video_id: the ``L*_V*`` video id.
        - keyframe_ordinal: the 1-based keyframe ordinal.

    Returns:
        - ``images/<video_id>/<ordinal:06d>.jpg``.
    """
    return f"{IMAGES_DIRNAME}/{video_id}/{image_filename(keyframe_ordinal)}"


def materialized_image_path(
    keyframes_root: Path | str,
    video_id: str,
    keyframe_ordinal: int,
) -> Path:
    """Resolve the canonical materialized image path for one record.

    Expected input:
        - keyframes_root: root directory containing an ``images/`` collection.
        - video_id: the ``L*_V*`` id.
        - keyframe_ordinal: the 1-based keyframe ordinal.

    Returns:
        - The ``images/<video_id>/<ordinal:06d>.jpg`` path.
    """
    return Path(keyframes_root) / image_relpath(video_id, keyframe_ordinal)


@dataclass(frozen=True)
class MaterializeManifest:
    """Versioned metadata describing one materialized image artifact.

    Fields:
        - schema_version: materialization schema version
          (``MATERIALIZE_MANIFEST_SCHEMA_VERSION``).
        - artifact_id: identifier of this artifact type
          (:data:`ARTIFACT_ID`).
        - source_frames: resolved ``frames.ndjson`` input path.
        - source_videos: resolved ``videos.json`` input path.
        - num_videos / num_keyframes: artifact counts.
        - image_format: image extension written (``jpg``).
        - jpeg_quality: ffmpeg mjpeg quality (``-q:v`` scale, lower is better).
        - created_at: ISO-8601 timestamp of completion.
    """

    schema_version: str
    artifact_id: str
    source_frames: str
    source_videos: str
    num_videos: int
    num_keyframes: int
    image_format: str
    jpeg_quality: int
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != MATERIALIZE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported materialize schema {self.schema_version!r}")
        if not isinstance(self.artifact_id, str) or self.artifact_id == "":
            raise ValueError("artifact_id required")
        if not isinstance(self.source_frames, str) or self.source_frames == "":
            raise ValueError("source_frames required")
        if not isinstance(self.source_videos, str) or self.source_videos == "":
            raise ValueError("source_videos required")
        if not isinstance(self.num_videos, int) or self.num_videos < 0:
            raise ValueError("num_videos must be >= 0")
        if not isinstance(self.num_keyframes, int) or self.num_keyframes < 0:
            raise ValueError("num_keyframes must be >= 0")
        if not isinstance(self.image_format, str) or self.image_format == "":
            raise ValueError("image_format required")
        if not isinstance(self.jpeg_quality, int) or not (1 <= self.jpeg_quality <= 31):
            raise ValueError("jpeg_quality must be in [1, 31]")
        if not isinstance(self.created_at, str) or self.created_at == "":
            raise ValueError("created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source_frames": self.source_frames,
            "source_videos": self.source_videos,
            "num_videos": self.num_videos,
            "num_keyframes": self.num_keyframes,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterializeManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            source_frames=data["source_frames"],
            source_videos=data["source_videos"],
            num_videos=int(data["num_videos"]),
            num_keyframes=int(data["num_keyframes"]),
            image_format=data["image_format"],
            jpeg_quality=int(data["jpeg_quality"]),
            created_at=data["created_at"],
        )


# ------------------------------------------------------------------ helpers


def _require(condition: bool, message: str) -> None:
    """Raise MaterializeError unless ``condition`` holds."""
    if not condition:
        raise MaterializeError(message)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a same-directory temporary file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _group_frame_batches(
    records: list[KeyframeTiming],
    max_frames_per_batch: int,
) -> list[list[KeyframeTiming]]:
    """Group records into ffmpeg ``select`` batches in frame order.

    Records are sorted by ``frame_index`` (the order ffmpeg emits frames) and
    split into contiguous batches of at most ``max_frames_per_batch`` entries.
    Each batch is decoded by one ffmpeg pass; concatenating the batches in
    order reproduces the full frame-ordered sequence.

    Expected input:
        - records: keyframe records for one video.
        - max_frames_per_batch: maximum records per batch.

    Returns:
        - A list of batches, each preserving global frame order.
    """
    ordered = sorted(records, key=lambda record: record.frame_index)
    return [
        ordered[i : i + max_frames_per_batch]
        for i in range(0, len(ordered), max_frames_per_batch)
    ]


def _extract_video_images(
    video_path: Path,
    records: list[KeyframeTiming],
    dest_dir: Path,
    jpeg_quality: int,
    *,
    max_frames_per_batch: int = DEFAULT_MAX_FRAMES_PER_BATCH,
) -> None:
    """Decode the selected frames of one video and write ordinal-named JPEGs.

    Uses one ffmpeg pass per batch of frame indices: a ``select`` filter keeps
    exactly the requested indices of that batch (mirroring the
    ``-fps_mode passthrough`` frame indexing used by the extractor), and
    ``-vsync vfr`` emits only those frames, sequentially numbered in frame
    order. Each output is renamed to ``<ordinal:06d>.jpg`` based on its
    record's ``keyframe_ordinal``. Batching keeps the filter expression bounded
    for long videos with thousands of keyframes; batch order preserves the
    global frame order.

    Expected input:
        - video_path: original video file.
        - records: the video's keyframe records.
        - dest_dir: directory to write the renamed images into.
        - jpeg_quality: ffmpeg mjpeg quality ``-q:v``.
        - max_frames_per_batch: maximum records per ffmpeg pass.

    Raises:
        - ValueError: if ``max_frames_per_batch`` is not a positive integer.
        - MaterializeError: if a batch decodes fewer frames than requested, or
          if ffmpeg fails.
    """
    if (
        not isinstance(max_frames_per_batch, int)
        or isinstance(max_frames_per_batch, bool)
        or max_frames_per_batch < 1
    ):
        raise ValueError(
            f"max_frames_per_batch must be a positive integer, got {max_frames_per_batch!r}"
        )
    for batch_no, batch in enumerate(_group_frame_batches(records, max_frames_per_batch)):
        select_expr = "+".join(f"eq(n,{record.frame_index})" for record in batch)
        tmp_dir = dest_dir / f".batch_{batch_no}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select='{select_expr}'",
            "-fps_mode",
            "passthrough",
            "-vsync",
            "vfr",
            "-f",
            "image2",
            "-q:v",
            str(jpeg_quality),
            "-start_number",
            "0",
            str(tmp_dir / "%d.jpg"),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise MaterializeError(
                f"ffmpeg failed to decode frames from {video_path}: {exc.stderr.strip()}"
            ) from exc
        produced = sorted(tmp_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        if len(produced) != len(batch):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise MaterializeError(
                f"decoded {len(produced)}/{len(batch)} requested frames from {video_path}"
            )
        for image_path, record in zip(produced, batch):
            image_path.rename(dest_dir / image_filename(record.keyframe.keyframe_ordinal))
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------------------------------------------------------ pipeline


def materialize_keyframes(
    videos_path: Path | str,
    frames_path: Path | str,
    output_dir: Path | str,
    *,
    video_root: Path | str | None = None,
    jpeg_quality: int = 2,
    max_frames_per_batch: int = DEFAULT_MAX_FRAMES_PER_BATCH,
    overwrite: bool = False,
    progress: ProgressReporter | str = "auto",
) -> MaterializeManifest:
    """Decode the selected keyframes from the original videos into JPEG images.

    Reads a ``videos.json`` manifest and a canonical ``frames.ndjson`` file,
    decodes the selected ``frame_index``/``pts_time`` frames from each original
    mp4, and writes ``images/<video_id>/<ordinal:06d>.jpg`` plus a validated
    copy of ``frames.ndjson`` and a ``manifest.json``.

    Expected input:
        - videos_path: path to a ``videos.json`` manifest (``video_id`` +
          ``path`` entries).
        - frames_path: path to a canonical artifact ``frames.ndjson``.
        - output_dir: target artifact directory (created if absent).
        - video_root: optional root directory that relative video paths in the
          manifest are resolved against (overrides ``source_dir``); absolute
          paths are used unchanged.
        - jpeg_quality: ffmpeg mjpeg quality in ``[1, 31]`` (lower is better;
          default 2).
        - max_frames_per_batch: maximum frame indices decoded per ffmpeg pass
          (default ``DEFAULT_MAX_FRAMES_PER_BATCH``); larger videos are decoded
          in several passes that are merged preserving ordinal order.        - overwrite: if True, replace an existing artifact; otherwise an
          existing non-empty output without a valid matching manifest raises.
        - progress: progress reporter or a CLI-style mode (``auto``/``on``/
          ``off``).

    Returns:
        - The :class:`MaterializeManifest` describing the produced artifact.

    Raises:
        - FileNotFoundError: if ``videos_path`` or ``frames_path`` does not
          exist.
        - ValueError: if the manifest or records are invalid.
        - MaterializeError: on a failed decode, a frame count mismatch, an
          invalid assembled artifact, or a refused non-empty output.
        - OSError: if outputs cannot be written.
    """
    videos_path = Path(videos_path)
    frames_path = Path(frames_path)
    output_dir = Path(output_dir)
    if not videos_path.is_file():
        raise FileNotFoundError(f"video manifest not found: {videos_path}")
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames.ndjson not found: {frames_path}")

    videos = load_video_manifest(videos_path)
    records = list(iter_frames(frames_path))
    by_id = videos.by_id()
    unknown = sorted({record.keyframe.video_id.value for record in records} - set(by_id))
    if unknown:
        raise MaterializeError(
            f"frames.ndjson references videos not in the manifest: {unknown[:5]}"
        )
    num_videos = len({record.keyframe.video_id.value for record in records})
    num_keyframes = len(records)

    manifest_path = output_dir / MANIFEST_FILENAME
    # A complete, matching artifact is idempotently reused.
    if manifest_path.is_file() and not overwrite:
        existing = MaterializeManifest.from_dict(read_json_object(manifest_path))
        if existing.artifact_id == ARTIFACT_ID and (output_dir / FRAMES_FILENAME).is_file():
            report = verify_materialization(output_dir)
            if report.valid:
                logger.info("artifact already complete at %s; skipping materialization", output_dir)
                return existing
            raise MaterializeError(
                f"existing artifact is incomplete or invalid: {output_dir}; use --overwrite to rebuild"
            )

    require_media_tools(ffmpeg=True, ffprobe=False)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise MaterializeError(
            f"materialization output already exists (and is not empty): {output_dir}; "
            "use --overwrite to replace it"
        )

    # Assemble in a sibling staging directory so a partial run is never exposed
    # as a valid artifact; the final move happens only after validation.
    staging = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    images_dir = staging / IMAGES_DIRNAME
    images_dir.mkdir(parents=True)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    reporter = create_progress_reporter(
        progress, unit="videos"
    ) if isinstance(progress, str) else progress
    if reporter is not None:
        reporter.set_total(num_videos)
        reporter.begin()

    done = 0
    per_video: dict[str, list[KeyframeTiming]] = {}
    for record in records:
        per_video.setdefault(record.keyframe.video_id.value, []).append(record)
    try:
        for video_id, video_records in per_video.items():
            entry = by_id[video_id]
            video_path = resolve_video_path(
                entry.path,
                source_dir=videos.source_dir,
                video_root=video_root,
            )
            dest_dir = images_dir / video_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            _extract_video_images(
                video_path,
                video_records,
                dest_dir,
                jpeg_quality,
                max_frames_per_batch=max_frames_per_batch,
            )
            done += 1
            if reporter is not None:
                reporter.update(done, note=f"video={video_id} keyframes={len(video_records)}")
            logger.info(
                "materialized %d images for %s", len(video_records), video_id
            )
    except Exception:
        raise
    finally:
        if reporter is not None:
            reporter.finish()

    frames_out = staging / FRAMES_FILENAME
    shutil.copyfile(frames_path, frames_out)
    manifest = MaterializeManifest(
        schema_version=MATERIALIZE_MANIFEST_SCHEMA_VERSION,
        artifact_id=ARTIFACT_ID,
        source_frames=str(frames_path.resolve()),
        source_videos=str(videos_path.resolve()),
        num_videos=num_videos,
        num_keyframes=num_keyframes,
        image_format=IMAGE_EXTENSION,
        jpeg_quality=jpeg_quality,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(staging / MANIFEST_FILENAME, manifest.to_dict())

    report = verify_materialization(staging)
    if not report.valid:
        shutil.rmtree(staging, ignore_errors=True)
        messages = "; ".join(f"[{issue.section}] {issue.message}" for issue in report.issues)
        raise MaterializeError(f"assembled artifact failed validation: {messages}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(staging), str(output_dir))
    logger.info(
        "materialized %d keyframe images for %d videos -> %s",
        num_keyframes,
        num_videos,
        output_dir,
    )
    return manifest


# ------------------------------------------------------------------ verification


@dataclass(frozen=True)
class MaterializationIssue:
    """One problem found while verifying a materialized image artifact.

    Fields:
        - section: which artifact component the issue belongs to
          (``artifact``, ``manifest``, ``images``).
        - message: human-readable description of the problem.
    """

    section: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"section": self.section, "message": self.message}


@dataclass(frozen=True)
class MaterializationVerificationReport:
    """Result of verifying one materialized image artifact.

    Fields:
        - output_dir: the artifact directory that was verified.
        - valid: True when no issues were found.
        - issues: every problem found (empty when ``valid``).
        - num_videos / num_keyframes: artifact counts from the manifest.
        - images_present: number of image files found under ``images/``.
        - missing_images: number of frame records without a corresponding
          image.
    """

    output_dir: str
    valid: bool
    issues: tuple[MaterializationIssue, ...]
    num_videos: int
    num_keyframes: int
    images_present: int
    missing_images: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "output_dir": self.output_dir,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "num_videos": self.num_videos,
            "num_keyframes": self.num_keyframes,
            "images_present": self.images_present,
            "missing_images": self.missing_images,
        }


def verify_materialization(output_dir: Path | str) -> MaterializationVerificationReport:
    """Verify a materialized image artifact for internal consistency.

    Checks that the manifest exists and parses, every ``frames.ndjson`` record
    has a corresponding ``images/<video_id>/<ordinal:06d>.jpg`` image, no image
    is missing, and the image count matches ``frames.ndjson``.

    Expected input:
        - output_dir: directory of a materialized artifact
          (``images/`` + ``frames.ndjson`` + ``manifest.json``).

    Returns:
        - A :class:`MaterializationVerificationReport`.
    """
    output_dir = Path(output_dir)
    issues: list[MaterializationIssue] = []
    images_present = 0
    missing_images = 0
    num_videos = 0
    num_keyframes = 0

    manifest_path = output_dir / MANIFEST_FILENAME
    frames_path = output_dir / FRAMES_FILENAME
    if not manifest_path.is_file():
        issues.append(MaterializationIssue("artifact", f"manifest.json missing: {manifest_path}"))
    if not frames_path.is_file():
        issues.append(MaterializationIssue("artifact", f"frames.ndjson missing: {frames_path}"))
        return MaterializationVerificationReport(
            output_dir=str(output_dir),
            valid=not issues,
            issues=tuple(issues),
            num_videos=0,
            num_keyframes=0,
            images_present=0,
            missing_images=0,
        )

    if manifest_path.is_file():
        try:
            manifest = MaterializeManifest.from_dict(read_json_object(manifest_path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(MaterializationIssue("manifest", f"invalid manifest.json: {exc}"))
        else:
            num_videos = manifest.num_videos
            num_keyframes = manifest.num_keyframes

    records: list[KeyframeTiming] = []
    try:
        records = list(iter_frames(frames_path))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        issues.append(MaterializationIssue("frames", f"invalid frames.ndjson: {exc}"))

    images_dir = output_dir / IMAGES_DIRNAME
    if not images_dir.is_dir():
        issues.append(MaterializationIssue("artifact", f"images directory missing: {images_dir}"))
    else:
        images_present = sum(
            1 for _ in images_dir.glob(f"*/*.{IMAGE_EXTENSION}")
        )

    for record in records:
        image_path = materialized_image_path(
            output_dir, record.keyframe.video_id.value, record.keyframe.keyframe_ordinal
        )
        if not image_path.is_file():
            missing_images += 1
            issues.append(
                MaterializationIssue(
                    "images",
                    f"missing image for {record.keyframe.video_id.value} "
                    f"ordinal {record.keyframe.keyframe_ordinal}: {image_path}",
                )
            )

    if manifest_path.is_file() and not any(issue.section == "manifest" for issue in issues):
        found_videos = len({record.keyframe.video_id.value for record in records})
        if num_videos != found_videos:
            issues.append(
                MaterializationIssue(
                    "manifest",
                    f"manifest num_videos {num_videos} != videos in frames.ndjson ({found_videos})",
                )
            )
        if num_keyframes != len(records):
            issues.append(
                MaterializationIssue(
                    "manifest",
                    f"manifest num_keyframes {num_keyframes} != records in frames.ndjson ({len(records)})",
                )
            )
        if num_keyframes != images_present:
            issues.append(
                MaterializationIssue(
                    "manifest",
                    f"manifest num_keyframes {num_keyframes} != images found ({images_present})",
                )
            )

    if len(records) != images_present:
        issues.append(
            MaterializationIssue(
                "images",
                f"frames.ndjson records ({len(records)}) != images found ({images_present})",
            )
        )

    return MaterializationVerificationReport(
        output_dir=str(output_dir),
        valid=not issues,
        issues=tuple(issues),
        num_videos=num_videos,
        num_keyframes=num_keyframes,
        images_present=images_present,
        missing_images=missing_images,
    )
