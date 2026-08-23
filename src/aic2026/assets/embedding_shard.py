"""Resumable, hardware-aware shard embedding pipeline.

Builds a bounded "shard" of an embedding artifact from a canonical
``frames.ndjson`` file using a reusable :class:`ImageEmbedder`. The shard is
designed to be run on a single machine (e.g. one laptop) against a subset of
the full keyframe set; later the shards are merged into a full embedding
artifact. This module performs no model-specific work itself.

Behavior:

- deterministic input order: the first ``max_frames`` records of ``frames.ndjson``
  in file (row) order, or the whole file when ``max_frames`` is omitted. Row ``i``
  of ``embeddings.npy`` always corresponds to line ``i`` of the shard's
  ``frames.ndjson``.
- resumable via ``checkpoint.json``; already-embedded frames are skipped on
  resume and the final artifact is considered complete only after
  ``manifest.json`` is written atomically. Partial outputs are never exposed
  as valid (no ``manifest.json`` until the shard completes).
- safe writes: embeddings are stored via memory-mapped files and the manifest
  and checkpoint are written atomically.

Shard directory layout::

    <output-dir>/
        checkpoint.json   resume state (completed count + video ids + identity)
        embeddings.npy    dense (N, D) matrix; row i <-> frames[i]
        frames.ndjson     the shard's keyframe records (row order)
        manifest.json     versioned shard metadata (written on completion)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.artifact_io import read_json_object, write_json_atomic
from aic2026.canonical import KeyframeTiming
from aic2026.embedding_artifact import iter_frames
from aic2026.image_embedder import default_image_loader
from aic2026.embeddings import EmbeddingSpace
from aic2026.frame_images import find_keyframe_image
from aic2026.image_embedder import ImageEmbedder
from aic2026.assets.keyframe_materialize import IMAGE_EXTENSION
from aic2026.progress import ProgressReporter, create_progress_reporter

#: Schema version of the shard manifest.
SHARD_MANIFEST_SCHEMA_VERSION = "1.0"

#: Relative filenames within a shard directory.
CHECKPOINT_FILENAME = "checkpoint.json"
MANIFEST_FILENAME = "manifest.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
FRAMES_FILENAME = "frames.ndjson"

logger = logging.getLogger(__name__)


class ShardEmbeddingError(RuntimeError):
    """Raised when a shard build is interrupted (leaving a resumable state)."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------ manifest


@dataclass(frozen=True)
class ShardManifest:
    """Versioned metadata describing one completed embedding shard.

    Fields:
        - schema_version: shard schema version
          (``SHARD_MANIFEST_SCHEMA_VERSION``).
        - worker_id: identifier of the worker machine that produced the shard.
        - model: stable model identity (e.g. ``openclip:ViT-L-14:laion2b...``).
        - embedding_dim: dimensionality ``D`` of each embedding vector.
        - dtype: numpy dtype name of the stored matrix, e.g. ``float32``.
        - device: concrete runtime device that produced the embeddings.
        - num_embeddings: number ``N`` of encoded keyframes (rows).
        - num_videos: number of distinct videos covered.
        - source_frames: the input ``frames.ndjson`` path this shard came from.
        - embedding_space: full embedding-space identity of the vectors.
        - created_at: ISO-8601 timestamp of shard completion.
    """

    schema_version: str
    worker_id: int
    model: str
    embedding_dim: int
    dtype: str
    device: str
    num_embeddings: int
    num_videos: int
    source_frames: str
    embedding_space: EmbeddingSpace
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == SHARD_MANIFEST_SCHEMA_VERSION,
            f"unsupported shard schema version {self.schema_version!r}",
        )
        _require(isinstance(self.worker_id, int) and self.worker_id >= 0, "worker_id must be >= 0")
        _require(isinstance(self.model, str) and self.model != "", "model required")
        _require(isinstance(self.embedding_dim, int) and self.embedding_dim >= 1, "embedding_dim must be >= 1")
        _require(isinstance(self.dtype, str) and self.dtype != "", "dtype required")
        _require(isinstance(self.device, str) and self.device != "", "device required")
        _require(isinstance(self.num_embeddings, int) and self.num_embeddings >= 0, "num_embeddings must be >= 0")
        _require(isinstance(self.num_videos, int) and self.num_videos >= 0, "num_videos must be >= 0")
        _require(isinstance(self.source_frames, str) and self.source_frames != "", "source_frames required")
        _require(isinstance(self.embedding_space, EmbeddingSpace), "embedding_space required")
        _require(
            self.embedding_dim == self.embedding_space.embedding_dim,
            "embedding_dim must match embedding_space.embedding_dim",
        )
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"invalid dtype {self.dtype!r}") from exc
        _require(np.issubdtype(dtype, np.floating), "dtype must be a floating numpy dtype")
        _require(
            dtype == np.dtype(self.embedding_space.output_dtype),
            "dtype must match embedding_space.output_dtype",
        )
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype,
            "device": self.device,
            "num_embeddings": self.num_embeddings,
            "num_videos": self.num_videos,
            "source_frames": self.source_frames,
            "embedding_space": self.embedding_space.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid or the schema version is
              unsupported.
        """
        return cls(
            schema_version=data["schema_version"],
            worker_id=int(data["worker_id"]),
            model=data["model"],
            embedding_dim=int(data["embedding_dim"]),
            dtype=data["dtype"],
            device=data["device"],
            num_embeddings=int(data["num_embeddings"]),
            num_videos=int(data["num_videos"]),
            source_frames=data["source_frames"],
            embedding_space=EmbeddingSpace.from_dict(data["embedding_space"]),
            created_at=data["created_at"],
        )


# ------------------------------------------------------------------ checkpoint


@dataclass(frozen=True)
class ShardCheckpoint:
    """Resume state persisted after each completed batch.

    Fields:
        - completed_frame_count: number of rows fully embedded and written.
        - completed_video_ids: video ids whose every keyframe in the shard is
          fully embedded (subset of the shard's videos).
        - model: stable model identity of the run.
        - device: concrete device of the run (informational; a device change
          on resume is allowed and only logged as a warning).
        - precision: output precision of the run (informational; a precision
          change on resume is allowed and only logged as a warning).
        - embedding_space: full embedding-space identity of the run (strict:
          a resumed run must produce the identical space).
        - timestamp: ISO-8601 timestamp of the last checkpoint write.
        - total_frames: total number of rows in this shard (used to reject a
          resumed run whose ``max_frames``/source changed).
        - worker_id: worker machine identifier.
        - source_frames: the resolved input ``frames.ndjson`` path.
    """

    completed_frame_count: int
    completed_video_ids: tuple[str, ...]
    model: str
    device: str
    precision: str | None
    embedding_space: EmbeddingSpace
    timestamp: str
    total_frames: int
    worker_id: int
    source_frames: str

    def __post_init__(self) -> None:
        _require(isinstance(self.completed_frame_count, int) and self.completed_frame_count >= 0, "completed_frame_count must be >= 0")
        _require(isinstance(self.total_frames, int) and self.total_frames >= 0, "total_frames must be >= 0")
        _require(0 <= self.completed_frame_count <= self.total_frames, "completed_frame_count must be within total_frames")
        _require(isinstance(self.model, str) and self.model != "", "model required")
        _require(isinstance(self.device, str) and self.device != "", "device required")
        _require(
            self.precision is None or (isinstance(self.precision, str) and self.precision != ""),
            "precision must be a non-empty string or None",
        )
        _require(isinstance(self.embedding_space, EmbeddingSpace), "embedding_space required")
        _require(isinstance(self.worker_id, int) and self.worker_id >= 0, "worker_id must be >= 0")
        _require(isinstance(self.source_frames, str) and self.source_frames != "", "source_frames required")
        _require(isinstance(self.timestamp, str) and self.timestamp != "", "timestamp required")
        _require(all(isinstance(vid, str) and vid != "" for vid in self.completed_video_ids), "completed_video_ids must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "completed_frame_count": self.completed_frame_count,
            "completed_video_ids": list(self.completed_video_ids),
            "model": self.model,
            "device": self.device,
            "precision": self.precision,
            "embedding_space": self.embedding_space.to_dict(),
            "timestamp": self.timestamp,
            "total_frames": self.total_frames,
            "worker_id": self.worker_id,
            "source_frames": self.source_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardCheckpoint":
        """Deserialize from a :meth:`to_dict` dict.

        ``precision`` and ``embedding_space`` are optional for backward
        compatibility with checkpoints written before they were introduced;
        they default to ``None`` and the current space default respectively
        when absent. When an older checkpoint lacks ``embedding_space``, the
        returned value is ``None`` so a resume treats the space as unknown.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            completed_frame_count=int(data["completed_frame_count"]),
            completed_video_ids=tuple(data["completed_video_ids"]),
            model=data["model"],
            device=data["device"],
            precision=data.get("precision"),
            embedding_space=(
                EmbeddingSpace.from_dict(data["embedding_space"])
                if data.get("embedding_space") is not None
                else None
            ),
            timestamp=data["timestamp"],
            total_frames=int(data["total_frames"]),
            worker_id=int(data["worker_id"]),
            source_frames=data["source_frames"],
        )


# ------------------------------------------------------------------ helpers


def read_frames(
    frames_path: Path | str,
    limit: int | None = None,
) -> list[KeyframeTiming]:
    """Load canonical keyframe records from a ``frames.ndjson`` file.

    Expected input:
        - frames_path: path to a canonical artifact ``frames.ndjson``.
        - limit: optional positive row limit; only the first ``limit`` records
          are read.

    Returns:
        - The records in file (row) order (at most ``limit`` when given).

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
    """
    iterator = iter_frames(Path(frames_path))
    if limit is not None:
        iterator = islice(iterator, limit)
    return list(iterator)


def select_frames(
    frames: Sequence[KeyframeTiming],
    max_frames: int | None,
) -> list[KeyframeTiming]:
    """Select the shard's frames: the first ``max_frames``, or all.

    Expected input:
        - frames: canonical keyframe records in row order.
        - max_frames: optional positive row limit (``None`` means all rows).

    Returns:
        - The selected records in the original order.

    Raises:
        - ValueError: if ``max_frames`` is not a non-negative integer.
    """
    if max_frames is None:
        return list(frames)
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 0:
        raise ValueError(f"max_frames must be a non-negative integer, got {max_frames!r}")
    return list(frames[:max_frames])


def _frames_line_count(path: Path) -> int:
    """Return the number of non-empty lines in an NDJSON frame file."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() != "":
                count += 1
    return count


def _trim_frames_overflow(path: Path, keep: int) -> None:
    """Rewrite ``path`` keeping only the first ``keep`` lines when longer.

    A crash between writing frames and the checkpoint write can leave more
    lines than the checkpoint records; those trailing lines are dropped.
    """
    if not path.is_file():
        return
    kept: list[str] = []
    overflow = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(kept) < keep:
                kept.append(line)
            else:
                overflow = True
                break
    if not overflow:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept)


def _last_index_by_video(frames: Sequence[KeyframeTiming]) -> dict[str, int]:
    """Return ``video_id -> last row index`` in the shard's row order."""
    last: dict[str, int] = {}
    for i, frame in enumerate(frames):
        last[frame.keyframe.video_id.value] = i
    return last


def _completed_video_ids(
    last_index: dict[str, int],
    completed_rows: int,
) -> tuple[str, ...]:
    """Return sorted video ids fully completed by ``completed_rows`` rows."""
    return tuple(sorted(vid for vid, last in last_index.items() if last < completed_rows))


# ------------------------------------------------------------------ pipeline


def embed_shard(
    frames_path: Path | str,
    output_dir: Path | str,
    embedder: ImageEmbedder,
    *,
    batch_size: int = 32,
    max_frames: int | None = None,
    worker_id: int = 0,
    precision: str | None = None,
    resume: bool = True,
    keyframes_root: Path | str | None = None,
    image_loader: Callable[[Path], object] | None = None,
    stop_after_frames: int | None = None,
    progress: ProgressReporter | str = "auto",
) -> ShardManifest:
    """Build a resumable embedding shard from canonical keyframe records.

    Expected input:
        - frames_path: path to a canonical artifact ``frames.ndjson``.
        - output_dir: target shard directory (created if absent).
        - embedder: an :class:`ImageEmbedder` used to encode keyframe images.
        - batch_size: number of keyframe images encoded per model invocation
          (and rows written between checkpoint flushes).
        - max_frames: optional row limit selecting the first ``max_frames``
          records (for testing / sharding).
        - worker_id: worker machine identifier recorded in the manifest.
        - precision: optional output precision label (e.g. ``fp32``/``fp16``)
          recorded in the checkpoint; a changed precision on resume is allowed
          and only logged as a warning.
        - resume: if True, continue a partial shard when present; an
          already-complete shard is validated and returned without recompute.
        - keyframes_root: root directory containing the ``keyframes/``
          collection used to resolve image paths.
        - image_loader: callable ``path -> backend-native image``; defaults to
          a Pillow-based loader.
        - stop_after_frames: optional testing hook to stop after writing N rows
          (no final manifest is written; the shard remains resumable).
        - progress: progress reporting mode ``auto``/``on``/``off`` or a shared
          :class:`ProgressReporter` (e.g. from the CLI, to merge model-load
          stage timing). Reporting is throttled and never affects outputs.

    Returns:
        - Final :class:`ShardManifest`.

    Raises:
        - ValueError: on invalid inputs or an inconsistent/conflicting partial
          shard (model, embedding_space, source, worker_id, or shard size
          changed). A changed device or precision is allowed (warning only).
        - ShardEmbeddingError: when ``stop_after_frames`` interrupts the build.
        - FileNotFoundError: if the source frames or a keyframe image is missing.
    """
    frames_path = Path(frames_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0:
        raise ValueError(f"worker_id must be a non-negative integer, got {worker_id!r}")
    if precision is not None and (not isinstance(precision, str) or precision == ""):
        raise ValueError("precision must be a non-empty string or None")
    if keyframes_root is None:
        raise ValueError("keyframes_root is required to resolve keyframe images")

    if isinstance(progress, ProgressReporter):
        reporter = progress
    elif progress is None:
        reporter = ProgressReporter("off")
    elif isinstance(progress, str):
        reporter = create_progress_reporter(progress)
    else:
        raise ValueError(
            f"progress must be a ProgressReporter or a mode string, got {progress!r}"
        )
    reporter.begin()
    stages = reporter.stages

    space = embedder.embedding_space
    embedding_dim = embedder.embedding_dim
    emb_dtype = np.dtype(space.output_dtype)
    model = embedder.model_name
    device = embedder.device
    load_image = image_loader if image_loader is not None else default_image_loader()
    source = str(frames_path.resolve())

    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    frames_out_path = output_dir / FRAMES_FILENAME

    with stages.time("load_frames"):
        frames = read_frames(frames_path, limit=max_frames)
        selected = select_frames(frames, max_frames)
        total_frames = len(selected)

    # ------------------------------------------------------------ complete fast path
    if resume and manifest_path.is_file():
        existing = ShardManifest.from_dict(read_json_object(manifest_path))
        if existing.model != model:
            raise ValueError(
                f"existing shard model {existing.model!r} does not match {model!r}"
            )
        if existing.embedding_space != space:
            raise ValueError("existing shard embedding_space does not match the embedder")
        if existing.embedding_dim != embedding_dim or existing.dtype != emb_dtype.name:
            raise ValueError("existing shard embedding_dim/dtype does not match the embedder")
        if existing.source_frames != source:
            raise ValueError(
                f"existing shard source_frames {existing.source_frames!r} != current {source!r}"
            )
        if existing.num_embeddings != total_frames:
            raise ValueError(
                f"existing shard num_embeddings {existing.num_embeddings} != selected rows {total_frames}"
            )
        return existing

    # ------------------------------------------------------------ decide the resume point
    start_row = 0
    completed_videos: set[str] = set()
    if resume and checkpoint_path.is_file():
        ckpt = ShardCheckpoint.from_dict(read_json_object(checkpoint_path))
        if ckpt.model != model:
            raise ValueError(f"checkpoint model {ckpt.model!r} does not match {model!r}")
        if ckpt.embedding_space is not None and ckpt.embedding_space != space:
            raise ValueError("checkpoint embedding_space does not match the embedder")
        if ckpt.worker_id != worker_id:
            raise ValueError(f"checkpoint worker_id {ckpt.worker_id} != {worker_id}")
        if ckpt.source_frames != source:
            raise ValueError(f"checkpoint source_frames {ckpt.source_frames!r} != current {source!r}")
        if ckpt.total_frames != total_frames:
            raise ValueError(
                f"checkpoint total_frames {ckpt.total_frames} != current {total_frames}; "
                "source or max_frames changed"
            )
        if ckpt.device != device:
            logger.warning(
                "resuming shard on device %r (checkpoint was device %r); continuing",
                device,
                ckpt.device,
            )
        if ckpt.precision is not None and ckpt.precision != precision:
            logger.warning(
                "resuming shard with precision %r (checkpoint was precision %r); continuing",
                precision,
                ckpt.precision,
            )
        start_row = ckpt.completed_frame_count
        completed_videos = set(ckpt.completed_video_ids)

    if start_row > 0:
        if not embeddings_path.is_file():
            raise ValueError("checkpoint exists but embeddings.npy is missing; cannot resume")
        if not frames_out_path.is_file():
            raise ValueError("checkpoint exists but frames.ndjson is missing; cannot resume")
        _trim_frames_overflow(frames_out_path, start_row)
        if _frames_line_count(frames_out_path) != start_row:
            raise ValueError(
                f"frames.ndjson lines do not match checkpoint rows ({start_row}); cannot resume"
            )

    logger.info(
        "embedding shard: %d rows (worker=%d, device=%s, resume_from=%d)",
        total_frames,
        worker_id,
        device,
        start_row,
    )
    reporter.set_total(total_frames, completed=start_row)

    # ------------------------------------------------------------ allocate row files
    if start_row > 0 and embeddings_path.is_file():
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != (total_frames, embedding_dim):
            raise ValueError(
                f"existing embeddings.npy shape {embeddings.shape} != expected "
                f"{(total_frames, embedding_dim)}"
            )
        if embeddings.dtype != emb_dtype:
            raise ValueError(
                f"existing embeddings.npy dtype {embeddings.dtype} != expected {emb_dtype}"
            )
    else:
        embeddings = np.lib.format.open_memmap(
            str(embeddings_path), mode="w+", dtype=emb_dtype, shape=(total_frames, embedding_dim)
        )
    frames_file = frames_out_path.open("a" if start_row > 0 else "w", encoding="utf-8")

    last_index = _last_index_by_video(selected)

    def write_checkpoint(completed: int) -> None:
        write_json_atomic(
            checkpoint_path,
            ShardCheckpoint(
                completed_frame_count=completed,
                completed_video_ids=_completed_video_ids(last_index, completed),
                model=model,
                device=device,
                precision=precision,
                embedding_space=space,
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_frames=total_frames,
                worker_id=worker_id,
                source_frames=source,
            ).to_dict(),
        )

    # ------------------------------------------------------------ stream keyframes through the embedder
    next_row = start_row
    pending_frames: list[KeyframeTiming] = []
    pending_images: list[object] = []

    def write_pending_batch() -> None:
        nonlocal next_row, pending_frames, pending_images
        if not pending_frames:
            return
        images = list(pending_images)
        timings = list(pending_frames)
        with stages.time("encode"):
            vectors = np.asarray(embedder.encode_images(images), dtype=emb_dtype)
        if vectors.shape != (len(timings), embedding_dim):
            raise ValueError(
                f"embedder returned shape {vectors.shape}, expected {(len(timings), embedding_dim)}"
            )
        video_id = timings[0].keyframe.video_id.value
        with stages.time("write_outputs"):
            for idx, timing in enumerate(timings):
                if stop_after_frames is not None and next_row >= stop_after_frames:
                    embeddings.flush()
                    write_checkpoint(next_row)
                    raise ShardEmbeddingError(
                        f"interrupted shard at row {next_row}; checkpoint written to {checkpoint_path}"
                    )
                embeddings[next_row, :] = vectors[idx]
                frames_file.write(json.dumps(timing.to_dict()) + "\n")
                next_row += 1
            embeddings.flush()
            write_checkpoint(next_row)
        pending_frames = []
        pending_images = []
        reporter.update(next_row, note=f"video={video_id}")
        logger.debug("shard batch done: %d/%d rows video=%s", next_row, total_frames, video_id)

    try:
        for frame in selected[start_row:]:
            if stop_after_frames is not None and next_row >= stop_after_frames:
                embeddings.flush()
                write_checkpoint(next_row)
                raise ShardEmbeddingError(
                    f"interrupted shard at row {next_row}; checkpoint written to {checkpoint_path}"
                )
            with stages.time("load_images"):
                path = find_keyframe_image(
                    keyframes_root, frame.keyframe.video_id.value, frame.keyframe.keyframe_ordinal
                )
                pending_images.append(load_image(path))
            pending_frames.append(frame)
            if len(pending_frames) >= batch_size:
                write_pending_batch()
        write_pending_batch()
        embeddings.flush()
        write_checkpoint(next_row)
    except Exception:
        raise
    finally:
        frames_file.close()

    if next_row != total_frames:
        raise ValueError(f"written rows ({next_row}) != expected ({total_frames})")

    # ------------------------------------------------------------ validate then write the final manifest
    with stages.time("write_outputs"):
        if _frames_line_count(frames_out_path) != total_frames:
            raise ValueError("frames.ndjson rows do not match the shard row count")
        manifest = ShardManifest(
            schema_version=SHARD_MANIFEST_SCHEMA_VERSION,
            worker_id=worker_id,
            model=model,
            embedding_dim=embedding_dim,
            dtype=emb_dtype.name,
            device=device,
            num_embeddings=total_frames,
            num_videos=len({frame.keyframe.video_id.value for frame in selected}),
            source_frames=source,
            embedding_space=space,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json_atomic(manifest_path, manifest.to_dict())
    reporter.finish()
    return manifest