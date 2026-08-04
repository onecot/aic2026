"""Shared cosine-similarity retrieval over a flat keyframe index."""

from __future__ import annotations

import numpy as np


def search(
    query_vecs: np.ndarray,
    emb: np.ndarray,
    vids: np.ndarray,
    frame_idx: np.ndarray,
    top_k: int = 100,
) -> list[list[dict[str, str | int | float]]]:
    """Return the highest-scoring keyframes for each normalized query vector.

    Expected input: ``query_vecs`` has shape ``(Q, 512)`` float32 and ``emb`` has
    shape ``(N, 512)`` float32, with both row-normalized; ``vids`` and ``frame_idx``
    are parallel arrays of shape ``(N,)``; ``top_k`` is a non-negative result count.
    Returns: A length-``Q`` list of score-descending result lists containing video ID,
        real source frame index, and cosine-similarity score dictionaries.
    Raises: ValueError if ``top_k`` is negative or the index arrays have inconsistent lengths.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if emb.shape[0] != vids.shape[0] or emb.shape[0] != frame_idx.shape[0]:
        raise ValueError("Index embeddings, video IDs, and frame indices must align.")

    sims = query_vecs @ emb.T
    result_count = min(top_k, emb.shape[0])
    results: list[list[dict[str, str | int | float]]] = []
    for query_index in range(query_vecs.shape[0]):
        # Stable sorting makes equal scores deterministic by ascending index row position.
        ranked_indices = np.argsort(-sims[query_index], kind="stable")[:result_count]
        results.append(
            [
                {
                    "video_id": str(vids[index]),
                    "frame_idx": int(frame_idx[index]),
                    "score": float(sims[query_index, index]),
                }
                for index in ranked_indices
            ]
        )
    return results
