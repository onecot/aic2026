"""Temporal-keyframe task formatting placeholder."""

from __future__ import annotations

from typing import Any


def format_trake(
    results_per_query: list[list[dict[str, str | int | float]]],
    query_ids: list[str],
    num_keyframes: int,
) -> list[dict[str, Any]]:
    """Format temporal-keyframe retrieval results for a future localization stage.

    Expected input: ``results_per_query`` contains ranked keyframes per query,
    ``query_ids`` is parallel to it, and ``num_keyframes`` requests the semantic frames.
    Returns: Future output will contain a ``video_id`` and temporally ordered ``frame_idx``
        list for each query.
    Raises: NotImplementedError because temporal keyframe localization is not implemented.
    """
    raise NotImplementedError("TRAKE formatting requires future temporal localization.")
