"""Task-specific KIS scoring over canonical evaluation queries and predictions.

This module deliberately does not run retrieval or encode queries. Callers
supply ranked original-video frame predictions, which are compared to the KIS
ground-truth video and inclusive original-frame range.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum

from aic2026.canonical import VideoId
from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask

KIS_TOP_K = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class KisPrediction:
    """One ranked KIS prediction expressed in original-video frame identity.

    Expected input:
        - rank: positive rank assigned by the prediction source.
        - video_id: predicted canonical video identifier.
        - frame_index: non-negative original-video frame index, not keyframe
          ordinal or presentation timestamp.

    Returns:
        - A validated prediction identity.

    Raises:
        - ValueError: if rank, video id, or frame index is invalid.
    """

    rank: int
    video_id: str
    frame_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("prediction rank must be a positive integer")
        VideoId(self.video_id)
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("prediction frame_index must be a non-negative integer")


@dataclass(frozen=True)
class KisQueryScore:
    """R-Score values for one KIS query at the configured top-k cutoffs."""

    query_id: str
    r_at: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id == "":
            raise ValueError("query_id must be a non-empty string")
        _validate_cutoff_scores(self.r_at)

    def score_at(self, k: int) -> float:
        """Return the R-Score at a configured positive cutoff.

        Expected input:
            - k: a cutoff represented by this per-query score.

        Returns:
            - The binary maximum R-Score among ranks up to ``k``.

        Raises:
            - KeyError: if ``k`` is not present.
        """
        for cutoff, score in self.r_at:
            if cutoff == k:
                return score
        raise KeyError(f"cutoff not available for query {self.query_id!r}: {k}")


@dataclass(frozen=True)
class TopKScore:
    """Aggregate mean R-Score across KIS queries at one cutoff."""

    k: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
            raise ValueError("k must be a positive integer")
        if self.score < 0.0 or self.score > 1.0:
            raise ValueError("top-k score must be in [0, 1]")


@dataclass(frozen=True)
class KisEvaluationResult:
    """Structured KIS per-query scores, aggregate R@k values, and final score."""

    query_scores: tuple[KisQueryScore, ...]
    top_k_scores: tuple[TopKScore, ...]
    final_score: float

    def __post_init__(self) -> None:
        if not self.query_scores:
            raise ValueError("KIS evaluation requires at least one query")
        if not self.top_k_scores:
            raise ValueError("KIS evaluation requires at least one top-k score")
        if self.final_score < 0.0 or self.final_score > 1.0:
            raise ValueError("final score must be in [0, 1]")

    def score_at(self, k: int) -> float:
        """Return the aggregate R@k value.

        Expected input:
            - k: a cutoff represented by this result.

        Returns:
            - Mean R-Score at ``k``.

        Raises:
            - KeyError: if ``k`` is not present.
        """
        for metric in self.top_k_scores:
            if metric.k == k:
                return metric.score
        raise KeyError(f"top-k score not available: {k}")


def is_kis_correct(query: EvaluationQuery, prediction: KisPrediction) -> bool:
    """Return whether one prediction satisfies the KIS ground truth.

    Expected input:
        - query: canonical KIS query with an original-video frame range.
        - prediction: ranked video/frame prediction.

    Returns:
        - True only when video ids match and ``prediction.frame_index`` is in
          the query's inclusive original-video frame range.

    Raises:
        - ValueError: if ``query`` is not a KIS query.
    """
    _require_kis(query)
    frame_range = query.ground_truth_frame_range
    return (
        prediction.video_id == query.ground_truth_video_id.value
        and frame_range.start <= prediction.frame_index <= frame_range.end
    )


def score_kis_query(
    query: EvaluationQuery,
    predictions: Sequence[KisPrediction],
    cutoffs: Sequence[int] = KIS_TOP_K,
) -> KisQueryScore:
    """Compute binary KIS R-Score values for one ranked prediction list.

    Expected input:
        - query: canonical KIS query.
        - predictions: ordered predictions with unique, increasing ranks.
        - cutoffs: positive R@k cutoffs.

    Returns:
        - Per-query maximum R-Score at every cutoff. Missing ranks/candidates
          contribute no correct result.

    Raises:
        - ValueError: if the query, ranks, or cutoffs are invalid.
    """
    _require_kis(query)
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    _validate_predictions(predictions)
    return KisQueryScore(
        query_id=query.query_id,
        r_at=tuple(
            (
                cutoff,
                float(
                    any(
                        prediction.rank <= cutoff and is_kis_correct(query, prediction)
                        for prediction in predictions
                    )
                ),
            )
            for cutoff in normalized_cutoffs
        ),
    )


def aggregate_top_k_scores(query_scores: Sequence[KisQueryScore]) -> tuple[TopKScore, ...]:
    """Average matching per-query R-Scores into deterministic R@k metrics.

    Expected input:
        - query_scores: non-empty per-query scores with identical cutoff sets.

    Returns:
        - Aggregate R@k metrics in the shared cutoff order.

    Raises:
        - ValueError: if scores are absent or use inconsistent cutoffs.
    """
    if not query_scores:
        raise ValueError("cannot aggregate an empty query score sequence")
    cutoffs = tuple(cutoff for cutoff, _ in query_scores[0].r_at)
    if not cutoffs:
        raise ValueError("query scores must include at least one cutoff")
    expected = set(cutoffs)
    for query_score in query_scores[1:]:
        if {cutoff for cutoff, _ in query_score.r_at} != expected:
            raise ValueError("all query scores must use identical cutoffs")
    count = len(query_scores)
    return tuple(
        TopKScore(
            k=cutoff,
            score=fsum(query_score.score_at(cutoff) for query_score in query_scores) / count,
        )
        for cutoff in cutoffs
    )


def calculate_final_score(top_k_scores: Sequence[TopKScore]) -> float:
    """Return the arithmetic mean of aggregate R@k metrics.

    Expected input:
        - top_k_scores: non-empty aggregate metrics, normally R@1/R@5/R@20/
          R@50/R@100.

    Returns:
        - Arithmetic mean of the supplied aggregate R@k values.

    Raises:
        - ValueError: if no metrics are supplied or cutoffs are duplicated.
    """
    if not top_k_scores:
        raise ValueError("cannot calculate a final score from no top-k scores")
    if len({metric.k for metric in top_k_scores}) != len(top_k_scores):
        raise ValueError("top-k score cutoffs must be unique")
    return fsum(metric.score for metric in top_k_scores) / len(top_k_scores)


def evaluate_kis(
    queries: Iterable[EvaluationQuery] | EvaluationSet,
    predictions_by_query: Mapping[str, Sequence[KisPrediction]],
    cutoffs: Sequence[int] = KIS_TOP_K,
) -> KisEvaluationResult:
    """Evaluate a complete KIS prediction set against canonical KIS queries.

    Expected input:
                - queries: non-empty canonical KIS queries with unique ids, either as
                    an iterable of :class:`EvaluationQuery` or an :class:`EvaluationSet`.
        - predictions_by_query: one ranked prediction sequence per query id;
          an empty sequence is valid when a system returned no candidates.
        - cutoffs: positive R@k cutoffs; defaults to BTC's 1, 5, 20, 50, 100.

    Returns:
        - Per-query binary R-Scores, mean R@k values, and their final mean.

    Raises:
        - ValueError: if queries are not KIS, query ids are duplicated, or the
          prediction mapping is missing/contains unexpected query ids.
    """
    query_values = queries.queries if isinstance(queries, EvaluationSet) else tuple(queries)
    if not query_values:
        raise ValueError("KIS evaluation requires at least one query")
    query_ids = [query.query_id for query in query_values]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("KIS evaluation query_ids must be unique")
    for query in query_values:
        _require_kis(query)
    expected_ids = set(query_ids)
    supplied_ids = set(predictions_by_query)
    missing_ids = expected_ids - supplied_ids
    unexpected_ids = supplied_ids - expected_ids
    if missing_ids or unexpected_ids:
        problems: list[str] = []
        if missing_ids:
            problems.append(f"missing predictions for query ids: {sorted(missing_ids)}")
        if unexpected_ids:
            problems.append(f"unexpected prediction query ids: {sorted(unexpected_ids)}")
        raise ValueError("; ".join(problems))

    query_scores = tuple(
        score_kis_query(query, predictions_by_query[query.query_id], cutoffs=cutoffs)
        for query in query_values
    )
    top_k_scores = aggregate_top_k_scores(query_scores)
    return KisEvaluationResult(
        query_scores=query_scores,
        top_k_scores=top_k_scores,
        final_score=calculate_final_score(top_k_scores),
    )


def _require_kis(query: EvaluationQuery) -> None:
    """Fail unless the canonical query represents the supported KIS task."""
    if query.task is not EvaluationTask.KIS:
        raise ValueError(f"KIS scorer requires task='kis', got {query.task.value!r}")


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    """Validate and materialize unique, ascending positive score cutoffs."""
    values = tuple(cutoffs)
    if not values:
        raise ValueError("at least one positive cutoff is required")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("cutoffs must be positive integers")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("cutoffs must be unique and in ascending order")
    return values


def _validate_predictions(predictions: Sequence[KisPrediction]) -> None:
    """Require predictions to be typed and strictly increasing by rank."""
    previous_rank = 0
    for prediction in predictions:
        if not isinstance(prediction, KisPrediction):
            raise ValueError("predictions must contain only KisPrediction values")
        if prediction.rank <= previous_rank:
            raise ValueError("prediction ranks must be strictly increasing")
        previous_rank = prediction.rank


def _validate_cutoff_scores(scores: tuple[tuple[int, float], ...]) -> None:
    """Validate cutoff ordering and binary per-query R-Score values."""
    cutoffs = tuple(cutoff for cutoff, _ in scores)
    _validate_cutoffs(cutoffs)
    if any(score not in (0.0, 1.0) for _, score in scores):
        raise ValueError("per-query R-Scores must be binary 0.0 or 1.0")
