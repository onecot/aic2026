"""Read-only KIS retrieval error analysis for manual inspection.

This module categorizes each query's top-``k`` ranked predictions into
``HIT`` / ``NEAR_VIDEO`` / ``FAIL`` and computes lightweight statistics that
help a human understand where retrieval fails. It only reads predictions and
ground truth; it never runs retrieval, modifies artifacts, or evaluates.

It reuses existing contracts:

- :class:`aic2026.evaluation.EvaluationQuery` for ground truth.
- :class:`aic2026.kis_scoring.KisPrediction` for ranked predictions
  (the scorer-compatible form produced by ``load_kis_predictions_for_scoring``).

Category definitions (over the top-``k`` predictions for one query):

- ``HIT``: at least one prediction whose ``video_id`` matches the ground-truth
  video and whose ``frame_index`` falls inside the inclusive ground-truth
  frame range.
- ``NEAR_VIDEO``: no ``HIT``, but at least one prediction has the correct
  ``video_id``.
- ``FAIL``: no prediction has the correct ``video_id``.

This module contains no I/O; the CLI and tests supply typed inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aic2026.evaluation import EvaluationQuery, EvaluationTask
from aic2026.kis_scoring import KisPrediction

CATEGORY_HIT = "HIT"
CATEGORY_NEAR_VIDEO = "NEAR_VIDEO"
CATEGORY_FAIL = "FAIL"

ErrorCategory = Literal["HIT", "NEAR_VIDEO", "FAIL"]

#: Default prediction cutoff inspected per query.
DEFAULT_TOP_K = 100


@dataclass(frozen=True)
class QueryErrorAnalysis:
    """Categorized retrieval result for one KIS query.

    Expected input:
        - category: one of ``HIT`` / ``NEAR_VIDEO`` / ``FAIL``.
        - query_id: evaluation query identifier.
        - query_text_en: English query text, if present.
        - query_text_vi: Vietnamese query text.
        - gt_video_id: ground-truth video id.
        - gt_frame_start: inclusive ground-truth start frame.
        - gt_frame_end: inclusive ground-truth end frame.
        - first_gt_video_rank: rank of the first prediction with the correct
          video id, or ``None`` when the correct video never appears.
        - closest_frame_distance: for ``NEAR_VIDEO``, the smallest distance of
          a correct-video prediction frame to the nearest ground-truth
          boundary; ``0`` for ``HIT``; ``None`` for ``FAIL``.
        - top5: up to five ``(rank, video_id, frame_index, score)`` tuples.

    Returns:
        - A validated per-query analysis record.

    Raises:
        - ValueError: if ``top5`` tuples are not in strictly increasing rank
          order or carry an invalid category.
    """

    category: ErrorCategory
    query_id: str
    query_text_en: str | None
    query_text_vi: str
    gt_video_id: str
    gt_frame_start: int
    gt_frame_end: int
    first_gt_video_rank: int | None
    closest_frame_distance: int | None
    top5: tuple[tuple[int, str, int, float], ...]

    def __post_init__(self) -> None:
        if self.category not in (CATEGORY_HIT, CATEGORY_NEAR_VIDEO, CATEGORY_FAIL):
            raise ValueError(f"invalid category {self.category!r}")
        previous_rank = 0
        for rank, _video_id, _frame_index, _score in self.top5:
            if rank <= previous_rank:
                raise ValueError("top5 tuples must be in strictly increasing rank order")
            previous_rank = rank
        if self.first_gt_video_rank is not None and self.first_gt_video_rank < 1:
            raise ValueError("first_gt_video_rank must be a positive rank or None")
        if self.closest_frame_distance is not None and self.closest_frame_distance < 0:
            raise ValueError("closest_frame_distance must be non-negative or None")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this analysis record to a JSON-safe dictionary.

        Expected input:
            - No arguments.

        Returns:
            - A dictionary with scalar fields plus ``ground_truth`` and ``top5``
              sub-structures. Missing scores are represented as ``null``.

        Raises:
            - No exceptions.
        """
        return {
            "category": self.category,
            "query_id": self.query_id,
            "query_text_en": self.query_text_en,
            "query_text_vi": self.query_text_vi,
            "ground_truth": {
                "video_id": self.gt_video_id,
                "frame_start": self.gt_frame_start,
                "frame_end": self.gt_frame_end,
            },
            "first_gt_video_rank": self.first_gt_video_rank,
            "closest_frame_distance": self.closest_frame_distance,
            "top5": [
                {
                    "rank": rank,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "score": None if math.isnan(score) else float(score),
                }
                for rank, video_id, frame_index, score in self.top5
            ],
        }


def frame_distance_to_interval(frame: int, start: int, end: int) -> int:
    """Return the distance from a frame to the nearest point of a closed interval.

    Expected input:
        - frame: non-negative original-video frame index.
        - start: inclusive ground-truth start frame.
        - end: inclusive ground-truth end frame at or after ``start``.

    Returns:
        - ``0`` when ``start <= frame <= end``; otherwise the number of frames
          between ``frame`` and the nearer boundary (``start`` or ``end``).

    Raises:
        - No exceptions.
    """
    if frame < start:
        return start - frame
    if frame > end:
        return frame - end
    return 0


def classify_query(
    predictions: Sequence[KisPrediction],
    gt_video_id: str,
    frame_start: int,
    frame_end: int,
) -> ErrorCategory:
    """Classify a query's ranked predictions into HIT / NEAR_VIDEO / FAIL.

    Expected input:
        - predictions: ranked predictions in rank order.
        - gt_video_id: ground-truth video id.
        - frame_start: inclusive ground-truth start frame.
        - frame_end: inclusive ground-truth end frame.

    Returns:
        - ``HIT``, ``NEAR_VIDEO``, or ``FAIL`` per the category definitions.

    Raises:
        - No exceptions.
    """
    if any(
        prediction.video_id == gt_video_id
        and frame_start <= prediction.frame_index <= frame_end
        for prediction in predictions
    ):
        return CATEGORY_HIT
    if any(prediction.video_id == gt_video_id for prediction in predictions):
        return CATEGORY_NEAR_VIDEO
    return CATEGORY_FAIL


def analyze_query(
    query: EvaluationQuery,
    predictions: Sequence[KisPrediction],
    scores: Mapping[tuple[str, int], float] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> QueryErrorAnalysis:
    """Analyze one KIS query's top-``k`` predictions.

    Expected input:
        - query: canonical KIS query.
        - predictions: ranked predictions for the query, in any order (they are
          sorted by rank internally).
        - scores: optional mapping from ``(video_id, frame_index)`` to the
          retrieval score, used only for the displayed top-5; absent scores
          appear as ``nan``.
        - top_k: number of predictions to inspect, ordered by rank.

    Returns:
        - A :class:`QueryErrorAnalysis` for the query.

    Raises:
        - ValueError: if ``query`` is not a KIS query.
    """
    _require_kis(query)
    ordered = tuple(sorted(predictions, key=lambda prediction: prediction.rank))[:top_k]
    gt_video_id = query.ground_truth_video_id.value
    start = query.ground_truth_frame_range.start
    end = query.ground_truth_frame_range.end

    category = classify_query(ordered, gt_video_id, start, end)
    first_gt_video_rank = next(
        (prediction.rank for prediction in ordered if prediction.video_id == gt_video_id),
        None,
    )
    if category == CATEGORY_NEAR_VIDEO:
        closest_frame_distance = min(
            frame_distance_to_interval(prediction.frame_index, start, end)
            for prediction in ordered
            if prediction.video_id == gt_video_id
        )
    elif category == CATEGORY_HIT:
        closest_frame_distance = 0
    else:
        closest_frame_distance = None

    top5 = tuple(
        (
            prediction.rank,
            prediction.video_id,
            prediction.frame_index,
            float(scores.get((prediction.video_id, prediction.frame_index), float("nan")))
            if scores is not None
            else float("nan"),
        )
        for prediction in ordered[:5]
    )
    return QueryErrorAnalysis(
        category=category,
        query_id=query.query_id,
        query_text_en=query.query_text_en,
        query_text_vi=query.query_text_vi,
        gt_video_id=gt_video_id,
        gt_frame_start=start,
        gt_frame_end=end,
        first_gt_video_rank=first_gt_video_rank,
        closest_frame_distance=closest_frame_distance,
        top5=top5,
    )


def analyze_kis_errors(
    queries: Sequence[EvaluationQuery],
    predictions_by_query: Mapping[str, Sequence[KisPrediction]],
    scores_by_query: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[QueryErrorAnalysis]:
    """Analyze a complete KIS prediction set against canonical KIS queries.

    Expected input:
        - queries: canonical KIS queries in deterministic (CSV) order.
        - predictions_by_query: ranked predictions per query id (the form
          returned by ``load_kis_predictions_for_scoring``).
        - scores_by_query: optional per-query ``(video_id, frame_index) ->
          score`` mappings for the displayed top-5.
        - top_k: number of predictions to inspect per query.

    Returns:
        - One :class:`QueryErrorAnalysis` per KIS query, in query order.

    Raises:
        - ValueError: if any query is not a KIS query.
    """
    analyses: list[QueryErrorAnalysis] = []
    for query in queries:
        _require_kis(query)
        predictions = predictions_by_query.get(query.query_id, ())
        scores = scores_by_query.get(query.query_id) if scores_by_query is not None else None
        analyses.append(analyze_query(query, predictions, scores, top_k=top_k))
    return analyses


def select_analyses(
    analyses: Sequence[QueryErrorAnalysis],
    category: ErrorCategory | None = None,
    sort: Literal["distance", "rank", "none"] = "none",
) -> list[QueryErrorAnalysis]:
    """Filter and sort analyses for reporting or export.

    Expected input:
        - analyses: per-query error analyses, in query order.
        - category: optional category to keep; ``None`` keeps all categories.
        - sort: ``"distance"`` sorts by ``closest_frame_distance`` ascending
          (records without a distance, i.e. ``FAIL``, sort last);
          ``"rank"`` sorts by ``first_gt_video_rank`` ascending (records
          without a rank sort last); ``"none"`` keeps the original order.
          Sorting is stable, so records that compare equal keep their original
          order.

    Returns:
        - The filtered analyses in the requested order. The input sequence is
          never mutated.

    Raises:
        - ValueError: if ``sort`` is not one of ``"distance"``, ``"rank"``, or
          ``"none"``.
    """
    if sort not in ("distance", "rank", "none"):
        raise ValueError("sort must be one of: 'distance', 'rank', 'none'")
    selected = list(analyses)
    if category is not None:
        selected = [analysis for analysis in selected if analysis.category == category]
    if sort == "distance":
        return sorted(
            selected,
            key=lambda analysis: (
                analysis.closest_frame_distance is None,
                analysis.closest_frame_distance if analysis.closest_frame_distance is not None else 0,
            ),
        )
    if sort == "rank":
        return sorted(
            selected,
            key=lambda analysis: (
                analysis.first_gt_video_rank is None,
                analysis.first_gt_video_rank if analysis.first_gt_video_rank is not None else 0,
            ),
        )
    return selected


@dataclass(frozen=True)
class ErrorSummary:
    """Aggregate error-analysis statistics across queries."""

    total: int
    hits: int
    near_video: int
    fails: int
    gt_video_in_top_k: tuple[tuple[int, float], ...]
    avg_first_gt_video_rank: float | None
    avg_near_frame_distance: float | None


def summarize(
    analyses: Sequence[QueryErrorAnalysis],
    top_k_values: Sequence[int] = (5, 20, 50, 100),
) -> ErrorSummary:
    """Summarize per-query analyses into aggregate statistics.

    Expected input:
        - analyses: per-query error analyses, in query order.
        - top_k_values: cutoffs for the "ground-truth video appears in top-k"
          statistic.

    Returns:
        - An :class:`ErrorSummary` with category counts, the percentage of
          queries whose ground-truth video appears within each top-k cutoff,
          the average rank of the first ground-truth video occurrence (over
          queries where it appears), and the average frame distance to the
          ground-truth segment for ``NEAR_VIDEO`` queries.

    Raises:
        - ValueError: if ``analyses`` is empty or ``top_k_values`` are not
          positive and ascending.
    """
    if not analyses:
        raise ValueError("cannot summarize an empty analysis sequence")
    normalized = tuple(top_k_values)
    if any(not isinstance(value, int) or value < 1 for value in normalized) or tuple(sorted(normalized)) != normalized:
        raise ValueError("top_k_values must be positive integers in ascending order")

    total = len(analyses)
    hits = sum(1 for analysis in analyses if analysis.category == CATEGORY_HIT)
    near_video = sum(1 for analysis in analyses if analysis.category == CATEGORY_NEAR_VIDEO)
    fails = sum(1 for analysis in analyses if analysis.category == CATEGORY_FAIL)

    present_ranks = [analysis.first_gt_video_rank for analysis in analyses if analysis.first_gt_video_rank is not None]
    avg_first_gt_video_rank = sum(present_ranks) / len(present_ranks) if present_ranks else None

    near_distances = [
        analysis.closest_frame_distance
        for analysis in analyses
        if analysis.category == CATEGORY_NEAR_VIDEO and analysis.closest_frame_distance is not None
    ]
    avg_near_frame_distance = sum(near_distances) / len(near_distances) if near_distances else None

    gt_video_in_top_k = tuple(
        (
            cutoff,
            sum(1 for rank in present_ranks if rank <= cutoff) / total * 100.0,
        )
        for cutoff in normalized
    )

    return ErrorSummary(
        total=total,
        hits=hits,
        near_video=near_video,
        fails=fails,
        gt_video_in_top_k=gt_video_in_top_k,
        avg_first_gt_video_rank=avg_first_gt_video_rank,
        avg_near_frame_distance=avg_near_frame_distance,
    )


def format_report(
    analyses: Sequence[QueryErrorAnalysis],
    summary: ErrorSummary,
) -> str:
    """Render a human-readable error-analysis report.

    Expected input:
        - analyses: per-query error analyses, in query order.
        - summary: aggregate statistics from :func:`summarize`.

    Returns:
        - A multi-line report with one block per query and a trailing summary.

    Raises:
        - No exceptions.
    """
    lines: list[str] = ["KIS retrieval error analysis", "=" * 80]
    for analysis in analyses:
        lines.append(f"[{analysis.category}] {analysis.query_id}")
        lines.append(f"  query_text_en: {analysis.query_text_en or ''}")
        lines.append(f"  query_text_vi: {analysis.query_text_vi}")
        lines.append(
            f"  ground_truth: video={analysis.gt_video_id} "
            f"frames=[{analysis.gt_frame_start},{analysis.gt_frame_end}]"
        )
        if analysis.first_gt_video_rank is not None:
            lines.append(f"  first GT video at rank: {analysis.first_gt_video_rank}")
        if analysis.closest_frame_distance is not None:
            lines.append(f"  closest GT boundary distance: {analysis.closest_frame_distance}")
        lines.append("  top-5:")
        for rank, video_id, frame_index, score in analysis.top5:
            lines.append(f"    rank={rank} video={video_id} frame={frame_index} score={score:.4f}")
        lines.append("")

    lines.append("Summary:")
    lines.append(f"  HIT: {summary.hits} / {summary.total}")
    lines.append(f"  NEAR_VIDEO: {summary.near_video} / {summary.total}")
    lines.append(f"  FAIL: {summary.fails} / {summary.total}")
    for cutoff, percent in summary.gt_video_in_top_k:
        lines.append(f"  GT video in top-{cutoff}: {percent:.1f}%")
    if summary.avg_first_gt_video_rank is not None:
        lines.append(f"  avg rank of first GT video occurrence: {summary.avg_first_gt_video_rank:.2f}")
    if summary.avg_near_frame_distance is not None:
        lines.append(f"  avg frame distance (NEAR_VIDEO): {summary.avg_near_frame_distance:.2f}")
    return "\n".join(lines) + "\n"


def _require_kis(query: EvaluationQuery) -> None:
    """Fail unless the canonical query represents the supported KIS task."""
    if query.task is not EvaluationTask.KIS:
        raise ValueError(f"error analysis requires task='kis', got {query.task.value!r}")