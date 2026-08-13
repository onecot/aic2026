"""Reusable image-embedding runner for the AIC 2026 pipeline.

Builds a model-specific embedding artifact from canonical dataset keyframes
using an injected image encoder. The runner is model/backend-agnostic: any
object satisfying the ``EmbeddingEncoder`` protocol from
:mod:`aic2026.embeddings` works, and batched encoding uses the encoder's
``encode_image_batch`` when available (falling back to per-image
``encode_image``).

Public API:
- ``build_embeddings(dataset, encoder, artifact_dir, artifact_id, ...)``

Behavior:
- deterministic input order: videos in ``Dataset.video_ids()`` order, then
  keyframes in ``Dataset.timings()`` (canonical) order; row ``i`` of
  ``embeddings.npy`` always corresponds to line ``i`` of ``frames.ndjson``
- two-pass build: (1) inspect the dataset for per-video keyframe counts and
  total rows; (2) stream keyframe images through the encoder in batches and
  write rows incrementally into a memory-mapped matrix
- resumable via ``.checkpoint.json`` and ``.build_meta.json``; the artifact
  is considered complete only after ``manifest.json`` is written atomically
- the raw ``keyframes/`` layout is never referenced here: image paths come
  from ``Dataset.keyframe_image_path`` and images are loaded through an
  injected ``image_loader``
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from aic2026.canonical import CANONICAL_SCHEMA_VERSION, KeyframeTiming
from aic2026.dataset import Dataset
from aic2026.embedding_artifact import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    EmbeddingManifest,
    validate_embedding_artifact,
    write_manifest,
)
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace

#: A loader turns a keyframe image path into a backend-native image object.
ImageLoader = Callable[[Path], object]

_CHECKPOINT_FILENAME = ".checkpoint.json"
_BUILD_META_FILENAME = ".build_meta.json"
_MANIFEST_FILENAME = "manifest.json"


class EmbeddingBuildError(RuntimeError):
    """Raised when an embedding build is interrupted or inconsistent."""


@dataclass
class _BuildMeta:
    num_rows: int
    num_videos: int
    embedding_dim: int
    embedding_dtype: str
    videos: list[dict]
    embedding_space: dict
    artifact_id: str


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via a same-directory temporary file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    """Read a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_build_meta(path: Path) -> _BuildMeta:
    """Deserialize a persisted build-metadata payload."""
    payload = _read_json(path)
    return _BuildMeta(
        num_rows=int(payload["num_rows"]),
        num_videos=int(payload["num_videos"]),
        embedding_dim=int(payload["embedding_dim"]),
        embedding_dtype=str(payload["embedding_dtype"]),
        videos=payload["videos"],
        embedding_space=payload["embedding_space"],
        artifact_id=str(payload["artifact_id"]),
    )


def _persist_build_meta(path: Path, meta: _BuildMeta) -> None:
    """Serialize build metadata so a resumed build can be validated."""
    _write_json_atomic(
        path,
        {
            "num_rows": meta.num_rows,
            "num_videos": meta.num_videos,
            "embedding_dim": meta.embedding_dim,
            "embedding_dtype": meta.embedding_dtype,
            "videos": meta.videos,
            "embedding_space": meta.embedding_space,
            "artifact_id": meta.artifact_id,
        },
    )


def default_image_loader() -> ImageLoader:
    """Return a Pillow-based image loader (lazy optional dependency).

    Returns:
        - A callable ``path -> fully-loaded RGB PIL image``.

    Raises:
        - ImportError: if Pillow is unavailable.
    """
    def load(path: Path) -> object:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "image loading requires Pillow; install pillow or pass image_loader=..."
            ) from exc
        with Image.open(path) as img:
            return img.convert("RGB")

    return load


def _inspect_dataset(dataset: Dataset, video_ids: Iterable[str]) -> _BuildMeta:
    """First pass: determine total rows and per-video keyframe counts.

    Expected input:
        - dataset: the canonical dataset accessor.
        - video_ids: video ids in the deterministic processing order.

    Returns:
        - A build-metadata structure saved to disk to guide the second pass.
    """
    total_rows = 0
    videos_meta: list[dict] = []
    for vid in video_ids:
        count = len(dataset.timings(vid))
        videos_meta.append({"video_id": vid, "num_keyframes": count})
        total_rows += count
    return _BuildMeta(
        num_rows=total_rows,
        num_videos=len(videos_meta),
        embedding_dim=0,
        embedding_dtype="",
        videos=videos_meta,
        embedding_space={},
        artifact_id="",
    )


def _encode_images(encoder: EmbeddingEncoder, images: list[object]) -> list[Embedding]:
    """Encode a batch of images through the given encoder.

    Uses ``encode_image_batch`` when the encoder exposes it; otherwise
    falls back to per-image ``encode_image`` so any future encoder works.

    Expected input:
        - encoder: an object satisfying the ``EmbeddingEncoder`` protocol.
        - images: backend-native image objects to encode.

    Returns:
        - One image embedding per input, in the same order.

    Raises:
        - ValueError: if the encoder rejects the input batch.
    """
    batch_method = getattr(encoder, "encode_image_batch", None)
    if callable(batch_method):
        return list(batch_method(images))
    return [encoder.encode_image(image) for image in images]


def _validate_embedding(
    embedding: Embedding,
    expected_dim: int,
    expected_dtype: np.dtype,
    expected_space: EmbeddingSpace,
    video_id: str,
    keyframe_ordinal: int,
) -> None:
    """Require one embedding to satisfy the build contract.

    Expected input:
        - embedding: the embedding produced for one keyframe.
        - expected_dim: required 1-D vector length.
        - expected_dtype: required vector dtype.
        - expected_space: required embedding-space identity.
        - video_id / keyframe_ordinal: identity for error messages.

    Raises:
        - ValueError: if the embedding violates any contract field.
    """
    context = f"{video_id} keyframe {keyframe_ordinal}"
    if not isinstance(embedding, Embedding):
        raise ValueError(f"encoder returned {type(embedding).__name__} for {context}, expected Embedding")
    if embedding.kind != "image":
        raise ValueError(f"encoder returned kind {embedding.kind!r} for {context}, expected 'image'")
    if embedding.vector.ndim != 1:
        raise ValueError(f"embedding vector must be 1-D for {context}, got ndim={embedding.vector.ndim}")
    if int(embedding.vector.shape[0]) != expected_dim:
        raise ValueError(
            f"embedding dim mismatch for {context}: {embedding.vector.shape[0]} != {expected_dim}"
        )
    if embedding.vector.dtype != expected_dtype:
        raise ValueError(
            f"embedding dtype mismatch for {context}: {embedding.vector.dtype} != {expected_dtype}"
        )
    if embedding.space != expected_space:
        raise ValueError(
            f"embedding space mismatch for {context}: {embedding.space.space_id!r} != "
            f"{expected_space.space_id!r}"
        )


def _frames_line_count(path: Path) -> int:
    """Return the number of non-empty lines in an NDJSON frame file."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() != "":
                count += 1
    return count


def _trim_frames_overflow(path: Path, keep: int) -> None:
    """Rewrite ``path`` keeping only the first ``keep`` lines when it is longer.

    A partially written trailing batch can leave more frame lines than the
    checkpoint records; those lines are rewritten from the resume point.

    Expected input:
        - path: the frames NDJSON file.
        - keep: number of lines to keep.

    Returns:
        - None.
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


def build_embeddings(
    dataset: Dataset,
    encoder: EmbeddingEncoder,
    artifact_dir: Path | str,
    artifact_id: str,
    batch_size: int = 16,
    resume: bool = True,
    stop_after_rows: Optional[int] = None,
    video_ids: Optional[Iterable[str]] = None,
    image_loader: Optional[ImageLoader] = None,
) -> EmbeddingManifest:
    """Build a model-specific image-embedding artifact from a canonical dataset.

    Expected input:
        - dataset: canonical :class:`Dataset` accessor.
        - encoder: object satisfying the :class:`EmbeddingEncoder` protocol.
        - artifact_dir: target artifact directory (created if absent).
        - artifact_id: unique id for this artifact (run identity).
        - batch_size: number of keyframe images encoded per model invocation
          (and rows written between checkpoint flushes).
        - resume: if True, continue a partial build when present; an
          already-complete artifact is validated and returned without recompute.
        - stop_after_rows: optional testing hook to stop after writing N rows
          (no final manifest is written).
        - video_ids: optional iterable restricting the build to a subset;
          videos are always processed in the dataset's sorted video-id order.
        - image_loader: callable ``path -> backend-native image``; defaults
          to a Pillow-based loader.

    Returns:
        - Final :class:`EmbeddingManifest`.

    Raises:
        - ValueError: on invalid inputs, embedding-space mismatch with a
          resumable/complete build, or an inconsistent partial build.
        - EmbeddingBuildError: when ``stop_after_rows`` interrupts the build
          (the partial build remains resumable).
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(artifact_id, str) or artifact_id == "":
        raise ValueError("artifact_id must be a non-empty string")

    space = encoder.embedding_space
    expected_dim = space.embedding_dim
    expected_dtype = np.dtype(space.output_dtype)
    load_image = image_loader if image_loader is not None else default_image_loader()

    checkpoint_path = artifact_dir / _CHECKPOINT_FILENAME
    meta_path = artifact_dir / _BUILD_META_FILENAME
    manifest_path = artifact_dir / _MANIFEST_FILENAME
    embeddings_path = artifact_dir / EMBEDDINGS_FILENAME
    frames_path = artifact_dir / FRAMES_FILENAME

    # ------------------------------------------------------------------ complete artifact fast path
    if resume and manifest_path.is_file():
        existing = EmbeddingManifest.from_dict(_read_json(manifest_path))
        if existing.artifact_id != artifact_id:
            raise ValueError(
                f"artifact_id mismatch: existing artifact {existing.artifact_id!r} does not "
                f"match requested {artifact_id!r}"
            )
        if existing.embedding_space != space:
            raise ValueError(
                "existing artifact embedding space does not match the encoder's space: "
                f"{existing.embedding_space.space_id!r} != {space.space_id!r}"
            )
        problems = validate_embedding_artifact(existing, artifact_dir)
        if problems:
            raise ValueError(f"existing artifact is invalid: {problems}")
        return existing

    # ------------------------------------------------------------------ inspect / load build metadata
    resuming = resume and meta_path.exists()
    if resuming:
        meta = _load_build_meta(meta_path)
        if meta.artifact_id != artifact_id:
            raise ValueError(
                f"artifact_id mismatch: resumable build {meta.artifact_id!r} does not "
                f"match requested {artifact_id!r}"
            )
        stored_space = EmbeddingSpace.from_dict(meta.embedding_space)
        if stored_space != space:
            raise ValueError(
                "encoder embedding space does not match the resumable build metadata: "
                f"{space.space_id!r} != {stored_space.space_id!r}"
            )
        if meta.embedding_dim != expected_dim:
            raise ValueError(
                f"encoder dim {expected_dim} != resumable build dim {meta.embedding_dim}"
            )
        if np.dtype(meta.embedding_dtype) != expected_dtype:
            raise ValueError(
                f"encoder dtype {expected_dtype} != resumable build dtype {meta.embedding_dtype}"
            )
    else:
        if video_ids is None:
            vids = dataset.video_ids()
        else:
            requested = set(video_ids)
            available = set(dataset.video_ids())
            missing = sorted(requested - available)
            if missing:
                raise ValueError(f"video_ids not found in dataset: {missing}")
            vids = [vid for vid in dataset.video_ids() if vid in requested]
        meta = _inspect_dataset(dataset, vids)
        meta.embedding_dim = expected_dim
        meta.embedding_dtype = expected_dtype.name
        meta.embedding_space = space.to_dict()
        meta.artifact_id = artifact_id
        _persist_build_meta(meta_path, meta)

    N = meta.num_rows
    D = meta.embedding_dim
    emb_dtype = np.dtype(meta.embedding_dtype)

    # ------------------------------------------------------------------ decide the resume point
    start_row = 0
    if resume and checkpoint_path.is_file():
        try:
            start_row = int(_read_json(checkpoint_path).get("next_row", 0))
        except (OSError, ValueError):
            start_row = 0
    elif resume and frames_path.is_file():
        start_row = _frames_line_count(frames_path)

    if start_row > 0:
        if not embeddings_path.is_file():
            raise ValueError("checkpoint exists but embeddings.npy is missing; cannot resume")
        if not frames_path.is_file():
            raise ValueError("checkpoint exists but frames.ndjson is missing; cannot resume")
        _trim_frames_overflow(frames_path, start_row)
        if _frames_line_count(frames_path) != start_row:
            raise ValueError(
                f"frames.ndjson lines do not match checkpoint rows ({start_row}); cannot resume"
            )

    # ------------------------------------------------------------------ allocate row files
    if embeddings_path.is_file() and resume and start_row > 0:
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != (N, D):
            raise ValueError(f"existing embeddings.npy shape {embeddings.shape} != expected {(N, D)}")
    else:
        embeddings = np.lib.format.open_memmap(
            str(embeddings_path), mode="w+", dtype=emb_dtype, shape=(N, D)
        )
    frames_file = frames_path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------ stream keyframes through the encoder
    row = 0
    next_row = start_row
    pending_timings: list[KeyframeTiming] = []
    pending_images: list[object] = []

    def write_pending_batch() -> None:
        """Encode and write one pending batch, honoring the row limit."""
        nonlocal next_row, row, pending_timings, pending_images
        if not pending_timings:
            return
        images = list(pending_images)
        timings = list(pending_timings)
        for idx, embedding in enumerate(_encode_images(encoder, images)):
            if stop_after_rows is not None and next_row >= stop_after_rows:
                embeddings.flush()
                _write_json_atomic(checkpoint_path, {"next_row": next_row})
                raise EmbeddingBuildError(
                    f"interrupted build at row {next_row}; checkpoint written to {checkpoint_path}"
                )
            timing = timings[idx]
            _validate_embedding(
                embedding, D, emb_dtype, space, timing.keyframe.video_id.value,
                timing.keyframe.keyframe_ordinal,
            )
            embeddings[next_row, :] = np.asarray(embedding.vector, dtype=emb_dtype)
            frames_file.write(json.dumps(timing.to_dict()) + "\n")
            row += 1
            next_row += 1
        embeddings.flush()
        _write_json_atomic(checkpoint_path, {"next_row": next_row})
        pending_timings = []
        pending_images = []

    try:
        for vmeta in meta.videos:
            vid = vmeta["video_id"]
            k = int(vmeta["num_keyframes"])
            if k == 0:
                continue
            if row + k <= start_row:
                row += k
                continue
            offset_in_video = max(0, start_row - row)
            row += offset_in_video
            timings = dataset.timings(vid)
            if len(timings) != k:
                raise ValueError(
                    f"timings count for {vid} ({len(timings)}) != recorded count ({k}); "
                    "dataset changed since the build metadata was written"
                )
            for timing in timings[offset_in_video:]:
                if stop_after_rows is not None and next_row >= stop_after_rows:
                    embeddings.flush()
                    _write_json_atomic(checkpoint_path, {"next_row": next_row})
                    raise EmbeddingBuildError(
                        f"interrupted build at row {next_row}; checkpoint written to {checkpoint_path}"
                    )
                pending_images.append(
                    load_image(
                        dataset.keyframe_image_path(
                            timing.keyframe.video_id.value, timing.keyframe.keyframe_ordinal
                        )
                    )
                )
                pending_timings.append(timing)
                if len(pending_timings) >= batch_size:
                    write_pending_batch()
        write_pending_batch()
        embeddings.flush()
        _write_json_atomic(checkpoint_path, {"next_row": next_row})
    finally:
        frames_file.close()

    if next_row != N:
        raise ValueError(f"written rows ({next_row}) != expected ({N})")

    # ------------------------------------------------------------------ validate then write the final manifest
    manifest = EmbeddingManifest(
        schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        artifact_id=artifact_id,
        embedding_space=space,
        embedding_dim=D,
        embedding_dtype=emb_dtype.name,
        num_rows=N,
        num_videos=meta.num_videos,
        source_dataset_id=str(dataset.raw_root),
        canonical_schema_version=CANONICAL_SCHEMA_VERSION,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    tmp_manifest = artifact_dir / (_MANIFEST_FILENAME + ".tmp")
    write_manifest(tmp_manifest, manifest)
    tmp_manifest.replace(manifest_path)

    for leftover in (checkpoint_path, meta_path):
        try:
            leftover.unlink()
        except FileNotFoundError:
            pass

    return manifest
