"""Video-level evaluation of the media-info lexical retrieval branch.

This module measures whether the media-info BM25 index (title / keywords /
description) retrieves the ground-truth *video* for KIS queries. It is a
retrieval-branch evaluation: it does not run or modify CLIP retrieval, the
KIS scorer, prediction artifacts, or evaluation code. It consumes:

- the canonical evaluation set (:func:`aic2026.evaluation.load_eval_set`);
- a built :class:`aic2026.media_info_retrieval.MediaInfoIndex`;
- optionally, a CLIP baseline prediction artifact
  (:func:`aic2026.kis_prediction_runner.load_kis_predictions_for_scoring`)
  for the error-analysis comparison.

Metrics are video-level only (ground-truth video id); they never score frames,
so they do not reuse the KIS scorer. Query texts are resolved from either
``query_text_en`` or ``query_text_vi``.

No fusion is implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask
from aic2026.kis_prediction_runner import QueryTextField
from aic2026.kis_scoring import KisPrediction
from aic2026.media_info_retrieval import MediaInfoHit, MediaInfoIndex

#: Default video-recall cutoffs, matching the requested metric list.
DEFAULT_TOP_K_VALUES = (1, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class VideoRetrievalQueryResult:
    """Video-level media-info retrieval result for one KIS query.

    Expected input:
        - query_id: evaluation query identifier.
        - query_text: the resolved query text used for retrieval.
        - gt_video_id: ground-truth video id.
        - gt_rank: rank (1-based) at which the ground-truth video appears in
          the media-info hits, or ``None`` when it is not retrieved.
        - hits: the ranked media-info hits returned for the query.

    Returns:
        - A validated per-query retrieval record.

    Raises:
        - ValueError: if ``query_id`` is empty or ``hits`` are not in
          strictly increasing rank order.
    """

    query_id: str
    query_text: str
    gt_video_id: str
    gt_rank: int | None
    hits: tuple[MediaInfoHit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        previous_rank = 0
        for hit in self.hits:
            if not isinstance(hit, MediaInfoHit):
                raise ValueError("hits must contain only MediaInfoHit values")
            if hit.rank <= previous_rank:
                raise ValueError("hits must be in strictly increasing rank order")
            previous_rank = hit.rank
        if self.gt_rank is not None and (not isinstance(self.gt_rank, int) or self.gt_rank < 1):
            raise ValueError("gt_rank must be a positive integer or None")


@dataclass(frozen=True)
class VideoRetrievalSummary:
    """Aggregate video-level metrics across KIS queries.

    Expected input:
        - total: number of evaluated queries.
        - recall_at: ``(k, fraction)`` tuples at the requested cutoffs.
        - avg_gt_rank: mean ground-truth video rank over queries where it was
          retrieved; ``None`` when no query retrieved it.
        - not_retrieved: query ids whose ground-truth video was not retrieved.

    Returns:
        - A validated aggregate summary.

    Raises:
        - ValueError: if ``total`` is not positive, cutoffs are not ascending
          with recall fractions in ``[0, 1]``, or ``not_retrieved`` is empty
          but ``avg_gt_rank`` is ``None``.
    """

    total: int
    recall_at: tuple[tuple[int, float], ...]
    avg_gt_rank: float | None
    not_retrieved: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.total, int) or isinstance(self.total, bool) or self.total < 1:
            raise ValueError("total must be a positive integer")
        cutoffs = tuple(k for k, _ in self.recall_at)
        if tuple(sorted(cutoffs)) != cutoffs or len(set(cutoffs)) != len(cutoffs):
            raise ValueError("recall_at cutoffs must be unique and ascending")
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in cutoffs):
            raise ValueError("recall_at cutoffs must be positive integers")
        if any(not 0.0 <= fraction <= 1.0 for _, fraction in self.recall_at):
            raise ValueError("recall fractions must be in [0, 1]")
        if self.avg_gt_rank is not None and self.avg_gt_rank < 0.0:
            raise ValueError("avg_gt_rank must be non-negative or None")
        if self.avg_gt_rank is None and not self.not_retrieved:
            raise ValueError("avg_gt_rank cannot be None when every query retrieved its ground truth")

    def recall_at_cutoff(self, k: int) -> float:
        """Return the video recall fraction at a recorded cutoff.

        Expected input:
            - k: a cutoff present in :attr:`recall_at`.

        Returns:
            - The recall fraction at ``k``.

        Raises:
            - KeyError: if ``k`` is not present.
        """
        for cutoff, fraction in self.recall_at:
            if cutoff == k:
                return fraction
        raise KeyError(f"recall cutoff not available: {k}")

    def to_dict(self) -> dict[str, object]:
        """Serialize this summary to a JSON-safe dictionary.

        Expected input:
            - No arguments.

        Returns:
            - A dictionary with scalar fields and ``recall_at`` tuples.

        Raises:
            - No exceptions.
        """
        return {
            "total": self.total,
            "recall_at": [
                {"k": k, "recall": fraction} for k, fraction in self.recall_at
            ],
            "avg_gt_rank": self.avg_gt_rank,
            "not_retrieved_count": len(self.not_retrieved),
            "not_retrieved_query_ids": list(self.not_retrieved),
        }


@dataclass(frozen=True)
class BaselineQueryRecord:
    """One query's video-level outcome under media-info and CLIP baseline.

    Expected input:
        - query_id: evaluation query identifier.
        - query_text_vi: Vietnamese query text.
        - query_text_en: English query text, if present.
        - gt_video_id: ground-truth video id.
        - media_info_gt_rank: media-info rank of the ground-truth video, or
          ``None`` when not retrieved.
        - clip_gt_rank: CLIP baseline rank of the ground-truth video, or
          ``None`` when not present in the baseline predictions.

    Returns:
        - A validated comparison record.

    Raises:
        - ValueError: if ``query_id`` is empty or either rank is invalid.
    """

    query_id: str
    query_text_vi: str
    query_text_en: str | None
    gt_video_id: str
    media_info_gt_rank: int | None
    clip_gt_rank: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        for name, value in (
            ("media_info_gt_rank", self.media_info_gt_rank),
            ("clip_gt_rank", self.clip_gt_rank),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

    def to_dict(self) -> dict[str, object]:
        """Serialize this record to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "query_text_vi": self.query_text_vi,
            "query_text_en": self.query_text_en,
            "gt_video_id": self.gt_video_id,
            "media_info_gt_rank": self.media_info_gt_rank,
            "clip_gt_rank": self.clip_gt_rank,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Queries where the media-info and CLIP baseline disagree at video level.

    Expected input:
        - cutoff: the top-k cutoff used for both "succeeds" checks.
        - media_info_only: queries where media-info retrieves the
          ground-truth video but the CLIP baseline does not.
        - clip_only: queries where the CLIP baseline retrieves the
          ground-truth video but media-info does not.
        - both: queries where both branches retrieve it.
        - neither: queries where neither branch retrieves it.

    Returns:
        - A validated comparison.

    Raises:
        - ValueError: if ``cutoff`` is not a positive integer or any record
          sequence contains invalid records.
    """

    cutoff: int
    media_info_only: tuple[BaselineQueryRecord, ...]
    clip_only: tuple[BaselineQueryRecord, ...]
    both: tuple[BaselineQueryRecord, ...]
    neither: tuple[BaselineQueryRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cutoff, int) or isinstance(self.cutoff, bool) or self.cutoff < 1:
            raise ValueError("cutoff must be a positive integer")
        for name, records in (
            ("media_info_only", self.media_info_only),
            ("clip_only", self.clip_only),
            ("both", self.both),
            ("neither", self.neither),
        ):
            if any(not isinstance(record, BaselineQueryRecord) for record in records):
                raise ValueError(f"{name} must contain only BaselineQueryRecord values")

    def to_dict(self) -> dict[str, object]:
        """Serialize this comparison to a JSON-safe dictionary."""
        return {
            "cutoff": self.cutoff,
            "media_info_only": [record.to_dict() for record in self.media_info_only],
            "clip_only": [record.to_dict() for record in self.clip_only],
            "both": [record.to_dict() for record in self.both],
            "neither": [record.to_dict() for record in self.neither],
        }


def retrieve_video_hits(
    index: MediaInfoIndex,
    text: str,
    top_k: int,
) -> tuple[MediaInfoHit, ...]:
    """Run one media-info query and return its ranked video hits.

    Expected input:
        - index: a built media-info index.
        - text: query text.
        - top_k: positive retrieval cutoff.

    Returns:
        - Ranked video hits in strictly increasing rank order.

    Raises:
        - ValueError: if ``top_k`` is not a positive integer.
    """
    return tuple(index.query(text, top_k=top_k))


def evaluate_media_info_video_retrieval(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    index: MediaInfoIndex,
    query_text_field: QueryTextField = "query_text_en",
    top_k: int = 100,
    cutoffs: Sequence[int] = DEFAULT_TOP_K_VALUES,
) -> tuple[tuple[VideoRetrievalQueryResult, ...], VideoRetrievalSummary]:
    """Evaluate video-level retrieval of every KIS query via media-info.

    Expected input:
        - queries: canonical KIS queries (an :class:`EvaluationSet` or an
          iterable of :class:`EvaluationQuery`).
        - index: a built media-info index.
        - query_text_field: ``"query_text_en"`` or ``"query_text_vi"``.
        - top_k: media-info retrieval cutoff; must be >= the largest cutoff.
        - cutoffs: positive, ascending video-recall cutoffs.

    Returns:
        - ``(results, summary)``: per-query results in query order and the
          aggregate summary.

    Raises:
        - ValueError: if a query is not KIS, ``query_text_field`` is invalid,
          a query is missing ``query_text_en`` when requested, or the cutoffs
          / ``top_k`` are invalid.
    """
    normalized = _normalize_queries(queries)
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if top_k < max(normalized_cutoffs):
        raise ValueError("top_k must be >= the largest cutoff")

    results: list[VideoRetrievalQueryResult] = []
    for query in normalized:
        text = _resolve_query_text(query, query_text_field)
        hits = retrieve_video_hits(index, text, top_k=top_k)
        gt_rank = next(
            (hit.rank for hit in hits if hit.video_id == query.ground_truth_video_id.value),
            None,
        )
        results.append(
            VideoRetrievalQueryResult(
                query_id=query.query_id,
                query_text=text,
                gt_video_id=query.ground_truth_video_id.value,
                gt_rank=gt_rank,
                hits=hits,
            )
        )
    return tuple(results), _summarize(results, normalized_cutoffs)


def compare_to_clip_baseline(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    results_by_query: Mapping[str, VideoRetrievalQueryResult],
    clip_predictions: Mapping[str, Sequence[KisPrediction]],
    cutoff: int = 100,
) -> BaselineComparison:
    """Compare media-info video retrieval with a CLIP baseline at video level.

    Expected input:
        - queries: canonical KIS queries.
        - results_by_query: media-info per-query results keyed by query id.
        - clip_predictions: scorer-compatible CLIP baseline predictions keyed
          by query id (the form returned by
          :func:`aic2026.kis_prediction_runner.load_kis_predictions_for_scoring`).
        - cutoff: top-k cutoff used for both "succeeds" checks.

    Returns:
        - A :class:`BaselineComparison` grouping queries by whether the
          ground-truth video is retrieved (rank <= cutoff) by media-info, the
          CLIP baseline, both, or neither.

    Raises:
        - ValueError: if any query is not KIS, a query is missing from either
          mapping, or ``cutoff`` is invalid.
    """
    if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    normalized = _normalize_queries(queries)
    media_info_only: list[BaselineQueryRecord] = []
    clip_only: list[BaselineQueryRecord] = []
    both: list[BaselineQueryRecord] = []
    neither: list[BaselineQueryRecord] = []
    for query in normalized:
        result = results_by_query.get(query.query_id)
        if result is None:
            raise ValueError(f"missing media-info result for query {query.query_id!r}")
        predictions = clip_predictions.get(query.query_id)
        if predictions is None:
            raise ValueError(f"missing CLIP predictions for query {query.query_id!r}")
        gt_video = query.ground_truth_video_id.value
        clip_rank = next(
            (
                prediction.rank
                for prediction in predictions
                if prediction.video_id == gt_video
            ),
            None,
        )
        media_info_succeeds = result.gt_rank is not None and result.gt_rank <= cutoff
        clip_succeeds = clip_rank is not None and clip_rank <= cutoff
        record = BaselineQueryRecord(
            query_id=query.query_id,
            query_text_vi=query.query_text_vi,
            query_text_en=query.query_text_en,
            gt_video_id=gt_video,
            media_info_gt_rank=result.gt_rank,
            clip_gt_rank=clip_rank,
        )
        if media_info_succeeds and clip_succeeds:
            both.append(record)
        elif media_info_succeeds:
            media_info_only.append(record)
        elif clip_succeeds:
            clip_only.append(record)
        else:
            neither.append(record)
    return BaselineComparison(
        cutoff=cutoff,
        media_info_only=tuple(media_info_only),
        clip_only=tuple(clip_only),
        both=tuple(both),
        neither=tuple(neither),
    )


def format_report(
    summary: VideoRetrievalSummary,
    results: Sequence[VideoRetrievalQueryResult] = (),
    comparison: BaselineComparison | None = None,
) -> str:
    """Render a human-readable video-level evaluation report.

    Expected input:
        - summary: aggregate metrics from :func:`evaluate_media_info_video_retrieval`.
        - results: optional per-query results (only aggregate metrics are used).
        - comparison: optional CLIP baseline comparison to render.

    Returns:
        - A multi-line report with recall cutoffs, average ground-truth video
          rank, and not-retrieved query ids.

    Raises:
        - No exceptions.
    """
    lines: list[str] = ["Media-info video-level retrieval evaluation", "=" * 80]
    lines.append(f"Queries: {summary.total}")
    for k, fraction in summary.recall_at:
        lines.append(f"  Video Recall@{k}: {fraction:.4f}")
    if summary.avg_gt_rank is not None:
        lines.append(f"  average GT video rank: {summary.avg_gt_rank:.2f}")
    lines.append(f"  queries where GT video not retrieved: {len(summary.not_retrieved)}")
    for query_id in summary.not_retrieved:
        lines.append(f"    - {query_id}")
    if comparison is not None:
        lines.append(f"Comparison vs CLIP baseline (cutoff top-{comparison.cutoff}):")
        lines.append(
            f"  media-info succeeds, CLIP fails: {len(comparison.media_info_only)}"
        )
        for record in comparison.media_info_only:
            lines.append(f"    - {record.query_id}")
        lines.append(
            f"  CLIP succeeds, media-info fails: {len(comparison.clip_only)}"
        )
        for record in comparison.clip_only:
            lines.append(f"    - {record.query_id}")
        lines.append(f"  both succeed: {len(comparison.both)}")
        lines.append(f"  neither succeeds: {len(comparison.neither)}")
    return "\n".join(lines) + "\n"


def _category_records(
    comparison: BaselineComparison,
) -> tuple[tuple[str, str, tuple[BaselineQueryRecord, ...]], ...]:
    """Return category name, description, and records in report order."""
    return (
        ("MEDIA_ONLY", "media-info retrieves GT video, CLIP does not", comparison.media_info_only),
        ("CLIP_ONLY", "CLIP retrieves GT video, media-info does not", comparison.clip_only),
        ("BOTH", "both retrieve the GT video", comparison.both),
        ("NEITHER", "neither retrieves the GT video", comparison.neither),
    )


def category_report_to_dict(comparison: BaselineComparison) -> dict[str, object]:
    """Serialize a categorized query report to a JSON-safe dictionary.

    Expected input:
        - comparison: a :class:`BaselineComparison` produced by
          :func:`compare_to_clip_baseline`.

    Returns:
        - A dictionary with ``cutoff``, summary ``counts`` per category,
          ``total`` queries, and per-category ``categories`` with each query's
          id, texts, GT video, and both ranks.

    Raises:
        - No exceptions.
    """
    categories = {
        name: [record.to_dict() for record in records]
        for name, _description, records in _category_records(comparison)
    }
    counts = {
        name: len(records) for name, _description, records in _category_records(comparison)
    }
    return {
        "cutoff": comparison.cutoff,
        "counts": counts,
        "total": sum(counts.values()),
        "categories": categories,
    }


def format_category_report(comparison: BaselineComparison) -> str:
    """Render a human-readable categorized query report.

    Expected input:
        - comparison: a :class:`BaselineComparison` produced by
          :func:`compare_to_clip_baseline`.

    Returns:
        - A multi-line report with summary counts and, for every query, its
          id, English and Vietnamese texts, GT video id, media-info rank, and
          CLIP first GT-video rank, grouped by category.

    Raises:
        - No exceptions.
    """
    lines: list[str] = [
        "Media-info vs CLIP baseline query categorization",
        "=" * 80,
        f"Cutoff: top-{comparison.cutoff}",
    ]
    for name, description, records in _category_records(comparison):
        lines.append(f"{name} ({len(records)}): {description}")
        for record in records:
            en = record.query_text_en if record.query_text_en is not None else "(none)"
            mi_rank = record.media_info_gt_rank if record.media_info_gt_rank is not None else "—"
            clip_rank = record.clip_gt_rank if record.clip_gt_rank is not None else "—"
            lines.append(
                f"  {record.query_id} | EN: {en} | VI: {record.query_text_vi} "
                f"| GT: {record.gt_video_id} | media-info rank: {mi_rank} "
                f"| CLIP rank: {clip_rank}"
            )
    counts = ", ".join(
        f"{name}={len(records)}" for name, _description, records in _category_records(comparison)
    )
    lines.append(f"Summary: {counts} | total={sum(len(r) for _, _, r in _category_records(comparison))}")
    return "\n".join(lines) + "\n"


def _normalize_queries(
    queries: EvaluationSet | Iterable[EvaluationQuery],
) -> tuple[EvaluationQuery, ...]:
    """Validate queries are unique KIS queries in input order."""
    values = queries.queries if isinstance(queries, EvaluationSet) else tuple(queries)
    if not values:
        raise ValueError("media-info evaluation requires at least one query")
    for query in values:
        if query.task is not EvaluationTask.KIS:
            raise ValueError(f"media-info evaluation requires task='kis', got {query.task.value!r}")
    query_ids = [query.query_id for query in values]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation query_ids must be unique")
    return values


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    """Validate unique, ascending, positive cutoffs."""
    values = tuple(cutoffs)
    if not values:
        raise ValueError("at least one cutoff is required")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("cutoffs must be positive integers")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("cutoffs must be unique and in ascending order")
    return values


def _resolve_query_text(query: EvaluationQuery, field: QueryTextField) -> str:
    """Return the requested query text field, validating availability."""
    if field not in ("query_text_vi", "query_text_en"):
        raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
    if field == "query_text_vi":
        return query.query_text_vi
    if query.query_text_en is None:
        raise ValueError("query_text_en was requested but is missing for at least one query")
    return query.query_text_en


def _summarize(
    results: Sequence[VideoRetrievalQueryResult],
    cutoffs: tuple[int, ...],
) -> VideoRetrievalSummary:
    """Aggregate per-query results into a summary."""
    total = len(results)
    present_ranks = [result.gt_rank for result in results if result.gt_rank is not None]
    avg_gt_rank = sum(present_ranks) / len(present_ranks) if present_ranks else None
    not_retrieved = tuple(result.query_id for result in results if result.gt_rank is None)
    recall_at = tuple(
        (
            cutoff,
            sum(1 for rank in present_ranks if rank <= cutoff) / total,
        )
        for cutoff in cutoffs
    )
    return VideoRetrievalSummary(
        total=total,
        recall_at=recall_at,
        avg_gt_rank=avg_gt_rank,
        not_retrieved=not_retrieved,
    )