"""Temporal/frame-level refinement ceiling analysis (Phase 4A).

Diagnostic analysis answering: can cheap, deterministic refinement over the
EXISTING visual index recover frame-level failures without new frames,
embeddings, or models?

Pipeline position of the analyzed methods::

    global visual retrieval (existing index, exact cosine)
        |
    failure decomposition (VIDEO_MISS / VIDEO_HIT_FRAME_HIT /
                           VIDEO_HIT_FRAME_MISS)
        |
    distance-to-GT of retrieved GT-video frames
        |
    candidate-video oracle ceiling (perfect localizer over indexed frames)
        |
    cheap baselines: score pooling / temporal smoothing / local peak

All functions are pure and deterministic. Ranking derivation mirrors
``aic2026.retrieval.IndexRetriever`` exactly (float32 cosine with
lexicographic tie-breaking by row index) so baseline metrics reproduce the
production retriever; callers can verify equality directly.

Candidate videos for a query are derived from the global row ranking by
first-appearance order of distinct ``video_id`` values ("top-N distinct
candidate videos"). Refinement methods only reselect representative frames
for those videos; no new candidates are ever created except where a pooling
method explicitly reorders videos by pooled evidence (reported separately).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

#: KIS cutoffs reported throughout this analysis.
REPORT_K = (1, 5, 20, 50, 100)

#: Candidate-video cutoffs for the oracle ceiling.
VIDEO_CUTOFFS = (1, 5, 10, 20, 50, 100)

#: Fixed distance buckets (frames). Defined once, reported honestly.
DISTANCE_BUCKETS: tuple[tuple[str, int], ...] = (
    ("inside_gt", 0),
    ("1-50", 50),
    ("51-100", 100),
    ("101-250", 250),
    ("251-500", 500),
    ("501-1000", 1000),
    (">1000", -1),
)

#: Outcome categories.
VIDEO_MISS = "VIDEO_MISS"
FRAME_HIT = "VIDEO_HIT_FRAME_HIT"
FRAME_MISS = "VIDEO_HIT_FRAME_MISS"


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def derive_ranking(scores: np.ndarray) -> np.ndarray:
    """Return row indices ordered by score descending, ties by row index.

    Mirrors the deterministic ordering of
    :func:`aic2026.retrieval.IndexRetriever.search`.

    Expected input:
        - scores: 1-D float array of cosine scores (row-aligned with the
          index artifact).

    Returns:
        - int64 indices of all rows in ranking order.

    Raises:
        - ValueError: if ``scores`` is not 1-D.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got ndim={scores.ndim}")
    n = scores.shape[0]
    return np.lexsort((np.arange(n, dtype=np.int64), -scores.astype(np.float32)))


def frame_in_gt(gt_video_id: str, gt_start: int, gt_end: int, video_id: str, frame_index: int) -> bool:
    """Return whether one frame satisfies the existing GT correctness rule.

    Mirrors :func:`aic2026.kis_scoring.is_kis_correct` (video match +
    inclusive original-frame range); kept as an explicit local predicate so
    the analysis needs no prediction dataclass round-trips. The CLI asserts
    agreement with ``is_kis_correct`` on real queries.
    """
    if video_id != gt_video_id:
        return False
    return gt_start <= frame_index <= gt_end


def outcome_at_k(
    ranked_rows: Sequence[int],
    video_ids: Sequence[str],
    frame_indices: Sequence[int],
    gt_video_id: str,
    gt_start: int,
    gt_end: int,
    k: int,
) -> str:
    """Classify one query's outcome at cutoff ``k``.

    Expected input:
        - ranked_rows: row indices in ranking order (at least ``k`` rows).
        - video_ids / frame_indices: per-row metadata arrays.

    Returns:
        - ``VIDEO_MISS``, ``VIDEO_HIT_FRAME_HIT``, or
          ``VIDEO_HIT_FRAME_MISS``.

    Raises:
        - ValueError: if fewer than ``k`` rows are supplied or ``k < 1``.
    """
    _require(k >= 1, f"k must be >= 1, got {k!r}")
    _require(len(ranked_rows) >= k, f"need at least {k} ranked rows, got {len(ranked_rows)}")
    seen_video = False
    for row in ranked_rows[:k]:
        if video_ids[row] != gt_video_id:
            continue
        seen_video = True
        if frame_in_gt(gt_video_id, gt_start, gt_end, video_ids[row], int(frame_indices[row])):
            return FRAME_HIT
    return VIDEO_MISS if not seen_video else FRAME_MISS


def distinct_videos_in_order(ranked_rows: Sequence[int], video_ids: Sequence[str]) -> list[str]:
    """Return distinct candidate videos in first-appearance ranking order.

    This is the explicit definition of "top-N distinct candidate videos"
    used by every video-cutoff metric in this module.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for row in ranked_rows:
        video = video_ids[row]
        if video not in seen:
            seen.add(video)
            ordered.append(video)
    return ordered


def min_distance_to_range(frame_index: int, gt_start: int, gt_end: int) -> int:
    """Return 0 inside the inclusive range, else distance to nearest boundary."""
    if frame_index < gt_start:
        return gt_start - frame_index
    if frame_index > gt_end:
        return frame_index - gt_end
    return 0


def bucket_for_distance(distance: int) -> str:
    """Map a non-negative distance to the fixed reporting bucket."""
    _require(distance >= 0, f"distance must be >= 0, got {distance!r}")
    for label, upper in DISTANCE_BUCKETS:
        if upper == -1 or distance <= upper:
            return label
    return DISTANCE_BUCKETS[-1][0]  # pragma: no cover - unreachable


@dataclass(frozen=True)
class QueryAnalysis:
    """Per-query analysis record.

    Fields:
        - query_id: evaluation query id.
        - outcomes: outcome category at each cutoff in :data:`REPORT_K`.
        - gt_video_rank_rows: number of rows before the first GT-video row
          (None when absent from the full ranking).
        - gt_video_candidate_index: 1-based position among distinct
          candidate videos (None when absent).
        - gt_has_correct_indexed_frame: whether ANY indexed keyframe of the
          GT video satisfies the correctness rule (oracle feasibility).
        - min_distance: smallest distance of any retrieved GT-video frame
          (within the analysis pool) to the GT range; None when the video
          was not retrieved.
        - distance_bucket: bucket of ``min_distance`` (or None).
    """

    query_id: str
    outcomes: dict[int, str]
    gt_video_rank_rows: int | None
    gt_video_candidate_index: int | None
    gt_has_correct_indexed_frame: bool
    min_distance: int | None
    distance_bucket: str | None


def analyze_query(
    query_id: str,
    gt_video_id: str,
    gt_start: int,
    gt_end: int,
    ranked_rows: np.ndarray,
    video_ids: Sequence[str],
    frame_indices: Sequence[int],
) -> QueryAnalysis:
    """Decompose one query's outcome and measure GT-frame distances.

    Expected input:
        - ranked_rows: full ranking (all index rows) for the query.
        - video_ids / frame_indices: per-row metadata arrays aligned with
          the ranking source.

    Returns:
        - The :class:`QueryAnalysis` record.

    Raises:
        - ValueError: on empty ranking.
    """
    _require(len(ranked_rows) > 0, "ranking must be non-empty")
    outcomes = {
        k: outcome_at_k(ranked_rows, video_ids, frame_indices, gt_video_id, gt_start, gt_end, k)
        for k in REPORT_K
    }

    gt_rows_mask = np.fromiter(
        (video_ids[row] == gt_video_id for row in ranked_rows), dtype=bool, count=len(ranked_rows)
    )
    gt_rank_rows: int | None = None
    min_distance: int | None = None
    if gt_rows_mask.any():
        first_position = int(np.argmax(gt_rows_mask))
        gt_rank_rows = first_position + 1
        distances = [
            min_distance_to_range(int(frame_indices[row]), gt_start, gt_end)
            for row in ranked_rows[first_position:]
            if video_ids[row] == gt_video_id
        ]
        min_distance = min(distances)

    candidate_index: int | None = None
    for position, video in enumerate(
        distinct_videos_in_order(ranked_rows, video_ids), start=1
    ):
        if video == gt_video_id:
            candidate_index = position
            break

    has_correct = False
    for row, video in enumerate(video_ids):
        if video == gt_video_id and frame_in_gt(
            gt_video_id, gt_start, gt_end, video, int(frame_indices[row])
        ):
            has_correct = True
            break

    return QueryAnalysis(
        query_id=query_id,
        outcomes=outcomes,
        gt_video_rank_rows=gt_rank_rows,
        gt_video_candidate_index=candidate_index,
        gt_has_correct_indexed_frame=has_correct,
        min_distance=min_distance,
        distance_bucket=bucket_for_distance(min_distance) if min_distance is not None else None,
    )


def oracle_hit_at_video_cutoffs(
    analysis: QueryAnalysis,
    *,
    frame_miss_only: bool = False,
) -> dict[int, bool]:
    """Return oracle success per candidate-video cutoff for one query.

    The oracle succeeds at cutoff N when the GT video appears within the
    first N distinct candidate videos AND at least one indexed keyframe of
    that video satisfies the correctness rule (i.e. a perfect localizer
    restricted to already-indexed frames could produce a hit).

    Expected input:
        - analysis: result of :func:`analyze_query`.
        - frame_miss_only: when True, only count queries whose baseline
          outcome at the same cutoff was ``VIDEO_HIT_FRAME_MISS`` (isolates
          localization upside rather than total recall).

    Returns:
        - Mapping from each cutoff in :data:`VIDEO_CUTOFFS` to oracle hit.
    """
    results: dict[int, bool] = {}
    for n in VIDEO_CUTOFFS:
        recalled = (
            analysis.gt_video_candidate_index is not None
            and analysis.gt_video_candidate_index <= n
        )
        ok = bool(recalled and analysis.gt_has_correct_indexed_frame)
        if frame_miss_only:
            # Baseline outcomes exist only at REPORT_K cutoffs; for other n,
            # use the next larger measured cutoff (conservative: a query that
            # hits by that larger cutoff is not counted as fixable at n).
            measured = min((k for k in REPORT_K if k >= n), default=max(REPORT_K))
            ok = ok and analysis.outcomes[measured] == FRAME_MISS
        results[n] = ok
    return results


# ------------------------------------------------------------------ refinement


def _video_row_groups(
    video_ids: Sequence[str],
) -> dict[str, list[int]]:
    """Group all row indices by video id (ascending row order)."""
    groups: dict[str, list[int]] = {}
    for row, video in enumerate(video_ids):
        groups.setdefault(video, []).append(row)
    return groups


def smoothed_scores(
    scores: np.ndarray,
    same_video_rows: Sequence[int],
    window: int,
) -> dict[int, float]:
    """Compute temporal neighborhood-smoothed scores for one video's rows.

    Rows are ordered by their index (which equals keyframe ordinal order in
    the canonical artifact). The smoothed score of row ``rows[i]`` is the
    unweighted mean of its own score and the scores of up to ``window``
    neighbors on each side *within the same video* (boundaries are clipped;
    no cross-video smoothing ever happens).

    Expected input:
        - scores: full per-row score array.
        - same_video_rows: ascending row indices belonging to one video.
        - window: neighbor radius in indexed keyframes (>= 0; 0 disables).

    Returns:
        - Mapping from row index to smoothed score.

    Raises:
        - ValueError: if ``window < 0``.
    """
    _require(window >= 0, f"window must be >= 0, got {window!r}")
    values = [float(scores[row]) for row in same_video_rows]
    smoothed: dict[int, float] = {}
    for i, row in enumerate(same_video_rows):
        lo = max(0, i - window)
        hi = min(len(values), i + window + 1)
        smoothed[row] = sum(values[lo:hi]) / (hi - lo)
    return smoothed


def select_local_peak(
    scores: np.ndarray,
    same_video_rows: Sequence[int],
    window: int = 1,
) -> int:
    """Select the highest-scoring temporally supported local peak of a video.

    A row is a local peak when its own score is greater than or equal to
    every neighbor's score within ``window`` positions on each side (same
    video, boundaries clipped). Among peaks, the one with the highest own
    score wins; ties break toward the earliest row. When the video has no
    peak (impossible mathematically since the maximum always qualifies),
    the global argmax is returned as a documented fallback.

    Expected input:
        - scores: full per-row score array.
        - same_video_rows: ascending row indices of one video.
        - window: peak neighborhood radius (>= 1).

    Returns:
        - The selected row index.

    Raises:
        - ValueError: if the video has no rows or ``window < 1``.
    """
    _require(window >= 1, f"window must be >= 1, got {window!r}")
    _require(len(same_video_rows) > 0, "video must have at least one row")
    best_row: int | None = None
    best_score = -np.inf
    for i, row in enumerate(same_video_rows):
        own = float(scores[row])
        lo = max(0, i - window)
        hi = min(len(same_video_rows), i + window + 1)
        neighbors = [float(scores[same_video_rows[j]]) for j in range(lo, hi) if j != i]
        if own >= max(neighbors, default=-np.inf) and own > best_score:
            best_score = own
            best_row = row
    assert best_row is not None  # the video maximum always qualifies
    return best_row


@dataclass(frozen=True)
class RefinementResult:
    """Outcome of one refinement method for one query.

    Fields:
        - ranked_rows: the refined full ranking.
        - selected_frame_by_video: representative frame chosen per candidate
          video (empty when the method keeps raw rows).
    """

    ranked_rows: list[int]
    selected_frame_by_video: dict[str, int]


def refine_pooling(
    scores: np.ndarray,
    video_ids: Sequence[str],
    baseline_ranked_rows: Sequence[int],
    *,
    pool: str,
    max_candidates: int,
) -> RefinementResult:
    """Reorder candidate videos by pooled same-video evidence.

    Every candidate video's pooled score is computed over ALL its indexed
    frames (max, mean of top-2, or mean of top-3 frame scores). Videos are
    then ranked by pooled score descending (ties by baseline first-appearance
    order); each video contributes its highest-scoring indexed frame. Rows of
    non-candidate videos keep baseline order after the candidates.

    This method intentionally changes video ordering; report its effect on
    video recall separately from frame localization.

    Expected input:
        - pool: ``"max"``, ``"top2-mean"``, or ``"top3-mean"``.
        - max_candidates: number of distinct candidate videos to keep from
          the baseline ranking.

    Returns:
        - The refined ranking.

    Raises:
        - ValueError: on unknown pooling method or invalid inputs.
    """
    if pool not in ("max", "top2-mean", "top3-mean"):
        raise ValueError(f"unknown pooling method: {pool!r}")
    _require(max_candidates >= 1, "max_candidates must be >= 1")

    groups = _video_row_groups(video_ids)
    candidate_videos = distinct_videos_in_order(baseline_ranked_rows, video_ids)[:max_candidates]
    baseline_position = {video: i for i, video in enumerate(candidate_videos)}

    pooled: list[tuple[float, int, str]] = []
    representatives: dict[str, int] = {}
    for video in candidate_videos:
        rows = groups[video]
        top = sorted((float(scores[row]) for row in rows), reverse=True)
        if pool == "max":
            value = top[0]
        elif pool == "top2-mean":
            value = sum(top[:2]) / min(2, len(top))
        else:
            value = sum(top[:3]) / min(3, len(top))
        pooled.append((-value, baseline_position[video], video))
        representatives[video] = max(rows, key=lambda row: (float(scores[row]), -row))

    pooled.sort()
    ordered_videos = [video for _, _, video in pooled]

    ranked: list[int] = []
    for video in ordered_videos:
        ranked.append(representatives[video])
    used = set(ranked)
    for row in baseline_ranked_rows:
        if row not in used:
            ranked.append(row)
            used.add(row)
    return RefinementResult(ranked_rows=ranked, selected_frame_by_video=representatives)


def refine_frame_selection(
    scores: np.ndarray,
    video_ids: Sequence[str],
    baseline_ranked_rows: Sequence[int],
    *,
    method: str,
    window: int,
    max_candidates: int,
) -> RefinementResult:
    """Keep baseline video order but reselect each candidate video's frame.

    Methods:
        - ``"argmax"``: each candidate video contributes its globally
          highest-scoring indexed frame (upper bound of pure reselection).
        - ``"smooth-argmax"``: selection uses temporally smoothed scores
          (:func:`smoothed_scores`) with radius ``window``.
        - ``"local-peak"``: selection uses :func:`select_local_peak` with
          radius ``window``.

    Video ordering is the baseline first-appearance order, so candidate-video
    recall is unchanged by construction; only frame choice varies.

    Expected input:
        - method: one of the three above.
        - window: smoothing/peak radius (>= 1 for local-peak, >= 0 for
          smooth-argmax).
        - max_candidates: number of distinct candidate videos to keep.

    Returns:
        - The refined ranking.

    Raises:
        - ValueError: on unknown method or invalid inputs.
    """
    if method not in ("argmax", "smooth-argmax", "local-peak"):
        raise ValueError(f"unknown frame-selection method: {method!r}")
    _require(max_candidates >= 1, "max_candidates must be >= 1")

    groups = _video_row_groups(video_ids)
    candidate_videos = distinct_videos_in_order(baseline_ranked_rows, video_ids)[:max_candidates]

    representatives: dict[str, int] = {}
    for video in candidate_videos:
        rows = groups[video]
        if method == "argmax":
            representatives[video] = max(rows, key=lambda row: (float(scores[row]), -row))
        elif method == "smooth-argmax":
            smoothed = smoothed_scores(scores, rows, window)
            representatives[video] = max(smoothed, key=lambda row: (smoothed[row], -row))
        else:
            representatives[video] = select_local_peak(scores, rows, window=max(window, 1))

    candidate_set = set(candidate_videos)
    ranked: list[int] = []
    for video in candidate_videos:
        ranked.append(representatives[video])
    used = set(ranked)
    for row in baseline_ranked_rows:
        if row not in used:
            ranked.append(row)
            used.add(row)
    return RefinementResult(ranked_rows=ranked, selected_frame_by_video=representatives)


def rerank_by_smoothed(
    scores: np.ndarray,
    video_ids: Sequence[str],
    baseline_ranked_rows: Sequence[int],
    *,
    window: int,
) -> list[int]:
    """Reorder existing pool rows by temporally smoothed scores.

    Unlike the one-representative-per-video methods, this keeps every row of
    the pool (the hit set cannot shrink) and only changes their order: rows
    are sorted by smoothed score descending, ties broken by baseline
    position. Smoothing itself never crosses video boundaries
    (:func:`smoothed_scores`).

    Expected input:
        - scores: full per-row score array.
        - video_ids: per-row video ids.
        - baseline_ranked_rows: pool rows in baseline order.
        - window: smoothing radius (>= 0).

    Returns:
        - The reordered row list (same multiset as the input pool).
    """
    _require(len(baseline_ranked_rows) > 0, "pool must be non-empty")
    by_video: dict[str, list[int]] = {}
    for row in baseline_ranked_rows:
        by_video.setdefault(video_ids[row], []).append(row)
    smoothed: dict[int, float] = {}
    for rows in by_video.values():
        smoothed.update(smoothed_scores(scores, sorted(rows), window))
    return sorted(baseline_ranked_rows, key=lambda row: (-smoothed[row], row))


def hits_at_k(
    ranked_rows: Sequence[int],
    video_ids: Sequence[int | str],
    frame_indices: Sequence[int],
    gt_video_id: str,
    gt_start: int,
    gt_end: int,
    cutoffs: Sequence[int] = REPORT_K,
) -> dict[int, bool]:
    """Return whether the ranking contains a correct frame at each cutoff."""
    return {
        k: any(
            frame_in_gt(gt_video_id, gt_start, gt_end, video_ids[row], int(frame_indices[row]))
            for row in ranked_rows[:k]
        )
        for k in cutoffs
    }


def summarize_outcomes(outcomes_by_k: dict[int, Counter]) -> dict[int, dict[str, int]]:
    """Return per-cutoff counts of each outcome category.

    Expected input:
        - outcomes_by_k: mapping cutoff -> Counter of outcome categories
          over all analyzed queries.

    Returns:
        - Mapping cutoff -> {category: count} including totals.
    """
    summary: dict[int, dict[str, int]] = {}
    for k, counter in sorted(outcomes_by_k.items()):
        total = sum(counter.values())
        summary[k] = {
            VIDEO_MISS: counter.get(VIDEO_MISS, 0),
            FRAME_HIT: counter.get(FRAME_HIT, 0),
            FRAME_MISS: counter.get(FRAME_MISS, 0),
            "total": total,
        }
    return summary
