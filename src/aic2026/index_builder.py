"""Index builder: construct IndexArtifact from a canonical Dataset.

Public API:
- build_index(dataset, artifact_dir, artifact_id, score_metric='cosine', batch_flush=1000, stop_after_rows=None)

Behavior:
- two-pass build: (1) inspect dataset to collect per-video row counts and embedding dim/dtype; (2) allocate memmaps and stream embeddings/frames into files
- resumable via .checkpoint.json and .build_meta.json; final artifact is considered valid only after manifest.json is written
- safe writes: manifest is written atomically; data files are written directly but manifest absence indicates incomplete

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from aic2026.dataset import Dataset
from aic2026.index_artifact import (
    IndexManifest,
    write_manifest,
    FRAMES_FILENAME,
    EMBEDDINGS_FILENAME,
    NORMS_FILENAME,
)
from aic2026.canonical import ClipEmbedding, CANONICAL_SCHEMA_VERSION


@dataclass
class _BuildMeta:
    num_rows: int
    num_videos: int
    embedding_dim: int
    embedding_dtype: str
    videos: list[dict]


_CHECKPOINT_FILENAME = ".checkpoint.json"
_BUILD_META_FILENAME = ".build_meta.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _compute_norm(v: np.ndarray) -> float:
    # compute L2 norm in float32 for stability and store as float32
    return float(np.linalg.norm(v.astype(np.float32)))


def _inspect_dataset(dataset: Dataset, video_ids: Optional[Iterable[str]] = None) -> _BuildMeta:
    """First pass: determine total rows, per-video counts, embedding dim/dtype.

    Returns a small meta structure saved to disk to guide the second pass.
    """
    total_rows = 0
    videos_meta: list[dict] = []
    embedding_dim = None
    embedding_dtype = None
    vids = list(video_ids) if video_ids is not None else list(dataset.iter_video_ids())
    for vid in vids:
        count = 0
        first_vec = None
        for emb in dataset.iter_embeddings(vid):
            if first_vec is None:
                first_vec = emb.vector
            count += 1
        if first_vec is None:
            # video with zero keyframes is allowed; skip but record
            dim = 0
            dtype = "float32"
        else:
            dim = int(first_vec.shape[0])
            dtype = str(first_vec.dtype)
        if embedding_dim is None and dim > 0:
            embedding_dim = dim
            embedding_dtype = dtype
        elif dim > 0:
            if dim != embedding_dim:
                raise ValueError(f"inconsistent embedding dim for {vid}: {dim} != {embedding_dim}")
            if dtype != embedding_dtype:
                raise ValueError(f"inconsistent embedding dtype for {vid}: {dtype} != {embedding_dtype}")
        videos_meta.append({"video_id": vid, "num_keyframes": count})
        total_rows += count
    return _BuildMeta(
        num_rows=total_rows,
        num_videos=len(videos_meta),
        embedding_dim=int(embedding_dim) if embedding_dim is not None else 0,
        embedding_dtype=embedding_dtype or "float32",
        videos=videos_meta,
    )


def build_index(
    dataset: Dataset,
    artifact_dir: Path,
    artifact_id: str,
    score_metric: str = "cosine",
    batch_flush: int = 1000,
    resume: bool = True,
    stop_after_rows: Optional[int] = None,
    video_ids: Optional[Iterable[str]] = None,
) -> IndexManifest:
    """Build an index artifact from a canonical Dataset.

    Args:
        dataset: Dataset loader instance.
        artifact_dir: target artifact directory (will be created).
        artifact_id: unique id for this artifact.
        score_metric: similarity metric (default: 'cosine').
        batch_flush: how many rows between memmap.flush() + checkpoint write.
        resume: if True, attempt to resume a partial build when present.
        stop_after_rows: optional testing hook to stop after writing N rows (no manifest written).
        video_ids: optional iterable of video ids to restrict the build.

    Returns:
        IndexManifest (final manifest). If stop_after_rows is set and reached, no manifest is written and the function returns a manifest-like object not persisted.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifact_dir / _CHECKPOINT_FILENAME
    meta_path = artifact_dir / _BUILD_META_FILENAME

    # Step 1: inspect or load meta
    if resume and meta_path.exists():
        meta_payload = _read_json(meta_path)
        meta = _BuildMeta(
            num_rows=int(meta_payload["num_rows"]),
            num_videos=int(meta_payload["num_videos"]),
            embedding_dim=int(meta_payload["embedding_dim"]),
            embedding_dtype=str(meta_payload["embedding_dtype"]),
            videos=meta_payload["videos"],
        )
    else:
        meta = _inspect_dataset(dataset, video_ids=video_ids)
        _write_json_atomic(meta_path, {
            "num_rows": meta.num_rows,
            "num_videos": meta.num_videos,
            "embedding_dim": meta.embedding_dim,
            "embedding_dtype": meta.embedding_dtype,
            "videos": meta.videos,
        })

    # Prepare memmaps
    N = meta.num_rows
    D = meta.embedding_dim
    emb_dtype = np.dtype(meta.embedding_dtype)

    embeddings_path = artifact_dir / EMBEDDINGS_FILENAME
    norms_path = artifact_dir / NORMS_FILENAME
    frames_path = artifact_dir / FRAMES_FILENAME

    # Decide resume point
    start_row = 0
    if resume and checkpoint_path.exists():
        try:
            chk = _read_json(checkpoint_path)
            start_row = int(chk.get("next_row", 0))
        except Exception:
            start_row = 0
    else:
        # If frames.ndjson exists, use its line count as start
        if frames_path.exists():
            try:
                start_row = sum(1 for _ in open(frames_path, encoding="utf-8"))
            except Exception:
                start_row = 0

    # Allocate memmaps (create or open existing)
    if embeddings_path.exists() and resume:
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != (N, D):
            raise ValueError(f"existing embeddings.npy shape {embeddings.shape} != expected {(N,D)}")
    else:
        # create new memmap file
        # use open_memmap for safe on-disk allocation
        embeddings = np.lib.format.open_memmap(
            str(embeddings_path), mode="w+", dtype=emb_dtype, shape=(N, D)
        )
    if norms_path.exists() and resume:
        norms = np.load(norms_path, mmap_mode="r+")
        if norms.shape != (N,):
            raise ValueError(f"existing norms.npy shape {norms.shape} != expected {(N,)}")
    else:
        norms = np.lib.format.open_memmap(str(norms_path), mode="w+", dtype=np.float32, shape=(N,))

    # Open frames file in append mode; ensure it's consistent with start_row
    frames_file = frames_path.open("a", encoding="utf-8")
    # If file newly created and start_row > 0, that's inconsistent
    if start_row > 0:
        # verify frames line count matches start_row
        with frames_path.open(encoding="utf-8") as fh:
            n_lines = sum(1 for _ in fh)
        if n_lines != start_row:
            raise ValueError(f"frames.ndjson lines ({n_lines}) != checkpoint ({start_row})")

    # Iterate videos and fill rows
    row = 0
    next_row = start_row
    flush_counter = 0

    try:
        for vmeta in meta.videos:
            vid = vmeta["video_id"]
            k = int(vmeta["num_keyframes"])
            # skip whole video if its rows are already done
            if k == 0:
                continue
            if row + k <= start_row:
                row += k
                continue
            # Determine offset within video to resume
            offset_in_video = max(0, start_row - row)
            emb_iter = dataset.iter_embeddings(vid)
            # skip offset_in_video embeddings
            for _ in range(offset_in_video):
                next(emb_iter)
            # now stream remaining
            timings = dataset.timings(vid)
            for emb in emb_iter:
                if stop_after_rows is not None and next_row >= stop_after_rows:
                    embeddings.flush()
                    norms.flush()
                    _write_json_atomic(checkpoint_path, {"next_row": next_row})
                    raise RuntimeError(
                        f"interrupted build at row {next_row}; checkpoint written to {checkpoint_path}"
                    )
                # validation
                if not isinstance(emb, ClipEmbedding):
                    raise ValueError(f"unexpected embedding record type for {vid}: {type(emb)}")
                vec = emb.vector
                if vec.ndim != 1:
                    raise ValueError(f"embedding vector must be 1-D for {vid}")
                if int(vec.shape[0]) != D:
                    raise ValueError(f"embedding dim mismatch for {vid}: {vec.shape[0]} != {D}")
                if str(vec.dtype) != str(emb_dtype):
                    raise ValueError(f"embedding dtype mismatch for {vid}: {vec.dtype} != {emb_dtype}")
                # write
                embeddings[next_row, :] = np.asarray(vec, dtype=emb_dtype)
                norms[next_row] = np.float32(_compute_norm(vec))
                # frames: need timing record for this keyframe; we can derive via dataset.timings
                # but to avoid reloading timings per keyframe, reuse dataset.timings when needed
                # Simpler: load timings once per video
                # We'll load timings here lazily
                timing = timings[emb.keyframe.keyframe_ordinal - 1]
                if timing.keyframe != emb.keyframe:
                    raise ValueError(
                        f"frame/embedding key mismatch for {vid}: "
                        f"{timing.keyframe.to_dict()} != {emb.keyframe.to_dict()}"
                    )
                frames_file.write(json.dumps(timing.to_dict()) + "\n")
                row += 1
                next_row += 1
                flush_counter += 1
                if flush_counter >= batch_flush:
                    embeddings.flush()
                    # write checkpoint
                    _write_json_atomic(checkpoint_path, {"next_row": next_row})
                    flush_counter = 0
        # final flush and checkpoint
        embeddings.flush()
        _write_json_atomic(checkpoint_path, {"next_row": next_row})
    finally:
        frames_file.close()

    # Post-checks
    if next_row != N:
        raise ValueError(f"written rows ({next_row}) != expected ({N})")

    # Build and write manifest atomically
    manifest = IndexManifest(
        schema_version="1.0",
        artifact_id=artifact_id,
        score_metric=score_metric,
        embedding_dim=D,
        embedding_dtype=str(emb_dtype),
        num_rows=N,
        num_videos=meta.num_videos,
        created_at=datetime.utcnow().isoformat() + "Z",
        source_dataset_id=str(getattr(dataset, "raw_root", "unknown")),
        canonical_schema_version=CANONICAL_SCHEMA_VERSION,
    )
    # atomic write
    tmp_manifest = artifact_dir / ("manifest.json.tmp")
    write_manifest(tmp_manifest, manifest)
    tmp_manifest.replace(artifact_dir / "manifest.json")

    # cleanup checkpoint and meta
    try:
        checkpoint_path.unlink()
    except Exception:
        pass
    try:
        meta_path.unlink()
    except Exception:
        pass

    return manifest
