"""OCR-only retrieval evaluation over canonical evaluation sets.

Measures how well OCR-chunk dense retrieval alone surfaces the ground-truth
video/frame region, without any visual scoring or answer generation. A chunk
hit is adapted from the existing KIS semantics
(:func:`aic2026.kis_scoring.is_kis_correct`): the predicted video must match
the ground-truth video and the predicted frame evidence must fall in the
inclusive ground-truth frame range. Two evidence levels are supported:

- range mode (default): the chunk's inclusive ``[frame_start, frame_end]``
  span intersects the ground-truth range;
- representative mode: only ``representative_frame_index`` counts.

This module does not run retrieval; callers supply ranked
:class:`aic2026.ocr_index.OcrSearchResult` lists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask
from aic2026.ocr_index import (
    OcrIndexHandle,
    OcrSearchResult,
    read_ocr_index,
    search_ocr_chunks,
)

#: Hit@k cutoffs reported by this evaluation.
OCR_TOP_K = (1, 5, 10, 20, 50, 100)


def is_ocr_chunk_correct(
    query: EvaluationQuery,
    result: OcrSearchResult,
    *,
    representative_only: bool = False,
) -> bool:
    """Return whether one OCR chunk hit satisfies the query's ground truth.

    Expected input:
        - query: canonical KIS or QA evaluation query.
        - result: one ranked OCR chunk hit.
        - representative_only: when True only the chunk's representative
          frame counts as evidence; otherwise the full chunk frame span
          intersecting the ground-truth range counts.

    Returns:
        - True when the video matches and the frame evidence overlaps the
          inclusive ground-truth frame range.

    Raises:
        - ValueError: if the query task is not KIS or QA.
    """
    if query.task not in (EvaluationTask.KIS, EvaluationTask.QA):
        raise ValueError(f"unsupported OCR retrieval task: {query.task}")
    if result.video_id != query.ground_truth_video_id.value:
        return False
    frame_range = query.ground_truth_frame_range
    if representative_only:
        return frame_range.start <= result.representative_frame_index <= frame_range.end
    return result.frame_start <= frame_range.end and result.frame_end >= frame_range.start


@dataclass(frozen=True)
class OcrQueryEvaluation:
    """Per-query OCR retrieval outcome.

    Fields:
        - query_id: canonical query id.
        - task: KIS or QA.
        - first_correct_rank: 1-based rank of the first correct chunk, or
          None when no chunk within the retrieved cutoff was correct.
    """

    query_id: str
    task: EvaluationTask
    first_correct_rank: int | None


@dataclass(frozen=True)
class OcrRetrievalReport:
    """Aggregate OCR-only retrieval metrics for one task.

    Fields:
        - task: evaluated task.
        - num_queries: number of evaluated queries.
        - top_k: retrieval cutoff used.
        - representative_only: evidence mode used.
        - hit_at_k: mean hit rate at each cutoff in :data:`OCR_TOP_K`.
        - query_evaluations: per-query outcomes in evaluation order.
    """

    task: EvaluationTask
    num_queries: int
    top_k: int
    representative_only: bool
    hit_at_k: tuple[tuple[int, float], ...]
    query_evaluations: tuple[OcrQueryEvaluation, ...]


def evaluate_ocr_query(
    query: EvaluationQuery,
    results: Sequence[OcrSearchResult],
    *,
    representative_only: bool = False,
) -> OcrQueryEvaluation:
    """Evaluate one ranked OCR result list against one query.

    Expected input:
        - query: canonical KIS or QA query.
        - results: ranked OCR chunk hits (already cut at the desired k).
        - representative_only: evidence mode (see :func:`is_ocr_chunk_correct`).

    Returns:
        - The per-query outcome including the first correct rank.

    Raises:
        - ValueError: if the query task is unsupported.
    """
    first_correct_rank: int | None = None
    for result in results:
        if is_ocr_chunk_correct(query, result, representative_only=representative_only):
            first_correct_rank = result.rank
            break
    return OcrQueryEvaluation(
        query_id=query.query_id,
        task=query.task,
        first_correct_rank=first_correct_rank,
    )


def evaluate_ocr_retrieval(
    eval_set: EvaluationSet,
    index: Path | str | OcrIndexHandle,
    encoder: Any,
    *,
    task: EvaluationTask | str = EvaluationTask.KIS,
    top_k: int = 100,
    representative_only: bool = False,
) -> OcrRetrievalReport:
    """Run OCR-only retrieval for every query of one task and aggregate hits.

    Expected input:
        - eval_set: loaded canonical evaluation set.
        - index: an :class:`OcrIndexHandle` or artifact directory.
        - encoder: text encoder whose space matches the index.
        - task: task to iterate (KIS or QA).
        - top_k: number of chunks retrieved per query.
        - representative_only: evidence mode (see :func:`is_ocr_chunk_correct`).

    Returns:
        - Aggregate report with Hit@k for :data:`OCR_TOP_K` (cutoffs above
          ``top_k`` report the rate at ``top_k``).

    Raises:
        - ValueError: if the task is unsupported or no queries exist.
    """
    handle = index if isinstance(index, OcrIndexHandle) else read_ocr_index(Path(index))
    queries = list(eval_set.iter_queries(task))
    if not queries:
        raise ValueError(f"evaluation set has no {task} queries")
    evaluations = []
    for query in queries:
        results = search_ocr_chunks(handle, query.query_text_vi, encoder=encoder, top_k=top_k)
        evaluations.append(
            evaluate_ocr_query(query, results, representative_only=representative_only)
        )
    hit_at_k = tuple(
        (
            k,
            sum(
                1 for e in evaluations if e.first_correct_rank is not None and e.first_correct_rank <= min(k, top_k)
            )
            / len(evaluations),
        )
        for k in OCR_TOP_K
    )
    return OcrRetrievalReport(
        task=EvaluationTask(task),
        num_queries=len(queries),
        top_k=top_k,
        representative_only=representative_only,
        hit_at_k=hit_at_k,
        query_evaluations=tuple(evaluations),
    )


def format_report(report: OcrRetrievalReport) -> str:
    """Render a human-readable one-table summary of a report."""
    lines = [
        f"OCR-only retrieval ({report.task.value}): "
        f"{report.num_queries} queries, top_k={report.top_k}, "
        f"evidence={'representative' if report.representative_only else 'chunk-range'}"
    ]
    for k, score in report.hit_at_k:
        lines.append(f"  Hit@{k:<3}: {score:.4f}")
    return "\n".join(lines)
