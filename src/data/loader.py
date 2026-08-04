"""Load per-video CLIP features into a flat searchable index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def list_videos(features_dir: Path) -> list[str]:
    """List feature video identifiers in deterministic order.

    Expected input: ``features_dir`` is a directory containing one ``.npy``
    feature file per video, named ``<video_id>.npy``.
    Returns: Sorted video identifier stems.
    Raises: FileNotFoundError if ``features_dir`` does not exist.
    """
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory does not exist: {features_dir}")
    return sorted(path.stem for path in features_dir.glob("*.npy"))


def load_one_video(
    video_id: str,
    features_dir: Path,
    mapping_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one video's CLIP embeddings and their source-video frame numbers.

    Expected input: ``video_id`` identifies matching ``.npy`` and ``.csv`` files;
    the feature array has shape ``(n_keyframes, 512)`` and CSV row ``i`` matches
    feature row ``i``.
    Returns: ``(emb, frame_idx)`` with float32 embeddings and real source frame indices.
    Raises: FileNotFoundError if an input file is missing; AssertionError if row counts differ;
        KeyError if the mapping lacks ``frame_idx``.
    """
    emb = np.load(features_dir / f"{video_id}.npy").astype(np.float32)
    mapping = pd.read_csv(mapping_dir / f"{video_id}.csv")

    # Feature row i aligns with mapping row i; mismatches would corrupt submissions.
    assert emb.shape[0] == len(mapping)
    frame_idx = mapping["frame_idx"].to_numpy()
    return emb, frame_idx


def build_index(
    features_dir: Path,
    mapping_dir: Path,
    normalize: bool = True,
    limit: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one flat embedding index from the per-video feature files.

    Expected input: Feature and mapping directories contain matching per-video files;
    ``limit`` is zero for all videos or a positive count for the first sorted videos.
    Returns: ``(emb, vids, frame_idx)`` where embeddings have shape ``(N, 512)``,
        video IDs have shape ``(N,)``, and source frame indices have shape ``(N,)``.
    Raises: FileNotFoundError if an input directory or file is missing; AssertionError if
        feature and mapping row counts differ; ValueError if no videos are selected.
    """
    video_ids = list_videos(features_dir)
    if limit:
        video_ids = video_ids[:limit]
    if not video_ids:
        raise ValueError("No videos selected for index construction.")

    embeddings: list[np.ndarray] = []
    video_ids_by_frame: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    for video_id in video_ids:
        emb, frame_idx = load_one_video(video_id, features_dir, mapping_dir)
        embeddings.append(emb)
        video_ids_by_frame.append(np.full(emb.shape[0], video_id))
        frame_indices.append(frame_idx)

    emb = np.concatenate(embeddings).astype(np.float32, copy=False)
    vids = np.concatenate(video_ids_by_frame)
    frame_idx = np.concatenate(frame_indices)

    if normalize:
        # Normalize each embedding so a dot product is equivalent to cosine similarity.
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8

    return emb, vids, frame_idx


def save_index(
    out_path: Path,
    emb: np.ndarray,
    vids: np.ndarray,
    frame_idx: np.ndarray,
) -> None:
    """Save a flat embedding index as an NPZ artifact.

    Expected input: ``emb`` has shape ``(N, 512)``, while ``vids`` and ``frame_idx``
    are parallel arrays of shape ``(N,)``; ``out_path`` is the target NPZ path.
    Returns: None.
    Raises: OSError if the index cannot be written.
    """
    np.savez(out_path, emb=emb, vids=vids, frame_idx=frame_idx)


def load_index(index_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a flat embedding index from an NPZ artifact.

    Expected input: ``index_path`` points to an NPZ file containing ``emb``, ``vids``,
    and ``frame_idx`` arrays.
    Returns: ``(emb, vids, frame_idx)`` in the same order as ``build_index``.
    Raises: FileNotFoundError if the index is missing; KeyError if a required array is absent.
    """
    with np.load(index_path, allow_pickle=True) as index:
        emb = index["emb"]
        vids = index["vids"].astype(str)
        frame_idx = index["frame_idx"]
    return emb, vids, frame_idx
