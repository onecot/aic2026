"""Read-only analysis of BTC-provided keyframe extraction quality for KIS.

This module measures how well the current BTC-provided keyframes cover the
KIS ground-truth events, before any new extractor is implemented. It performs
*analysis only*: it never runs or modifies retrieval, the embedding pipeline,
the scorer, or evaluation. It reuses the canonical evaluation contract
(``aic2026.evaluation``), the canonical keyframe timing records
(``aic2026.canonical.KeyframeTiming`` loaded from an artifact
``frames.ndjson`` via ``aic2026.embedding_artifact.iter_frames``), and the
scorer-compatible CLIP prediction loader
(``aic2026.kis_prediction_runner.load_kis_predictions_for_scoring``).

Report sections:

1. Keyframe statistics: per-video keyframe counts over the canonical keyframe
   set (total videos, total keyframes, mean, min/max, p50/p90/p95).
2. GT frame coverage: for each query, whether a keyframe falls inside the GT
   frame interval (exact hit) and the minimum absolute frame distance from any
   keyframe of the GT video to the interval (relaxed hit). Summaries include
   exact-hit count/percentage and relaxed-hit percentages at configurable
   thresholds.
3. Per-query error report: the misses (queries without an exact hit), each with
   the closest keyframe, its distance, and the number of keyframes in the GT
   video, sorted worst misses first.
4. CLIP baseline comparison: against a ``predict-kis`` prediction artifact,
   whether missing keyframe coverage can explain CLIP failures to retrieve the
   correct frame.
"""

from __future__ import annotations

import bisect
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.canonical import KeyframeTiming
from aic2026.embedding_artifact import iter_frames
from aic2026.evaluation import (
    EvaluationQuery,
    EvaluationSet,
    EvaluationTask,
    load_eval_set,
)
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring
from aic2026.kis_scoring import KisPrediction

logger = logging.getLogger(__name__)

#: Schema version of the keyframe-analysis report artifact.
ANALYSIS_SCHEMA_VERSION = "1.0"

#: Default relaxed-hit distance thresholds in original-video frames.
DEFAULT_RELAXED_THRESHOLDS = (10, 30, 60, 150, 300)

#: Default number of CLIP predictions analyzed per query.
DEFAULT_PREDICTION_TOP_K = 1000

#: Category label: the GT event has at least one keyframe inside its interval.
KEYFRAME_COVERED = "GT event covered by keyframes"

#: Category label: the GT event has no keyframe inside its interval.
KEYFRAME_NOT_COVERED = "GT event not covered by keyframes"

#: Category label: CLIP retrieved the GT video but not a GT frame, and the GT
#: event is not covered by a keyframe, so the coverage gap can explain the
#: frame-level failure.
VIDEO_RETRIEVED_COVERAGE_ISSUE = "GT video retrieved but frame coverage issue"


@dataclass(frozen=True)
class KeyframeStatistics:
    """Per-video keyframe-count statistics over a keyframe set.

    Counts are over unique original-video frames: when a ``frames.ndjson``
    contains multiple records for the same ``(video_id, frame_index)`` (the
    same original frame extracted more than once), the frame is counted once.
    The raw record counts are reported separately so the statistics reconcile
    with the source artifact row count.

    Expected input:
        - total_videos: number of distinct videos with at least one keyframe.
        - total_keyframes: number of unique keyframe frames across all videos.
        - avg_keyframes_per_video: mean unique keyframes per video.
        - min_keyframes: minimum unique keyframes in a single video.
        - max_keyframes: maximum unique keyframes in a single video.
        - p50: 50th percentile of unique keyframes per video.
        - p90: 90th percentile of unique keyframes per video.
        - p95: 95th percentile of unique keyframes per video.
        - total_keyframe_records: number of keyframe records in the source
          ``frames.ndjson`` (including duplicates).
        - duplicate_keyframe_records: number of records that repeat an
          already-present ``(video_id, frame_index)``.

    Returns:
        - A validated keyframe-statistics record.

    Raises:
        - ValueError: if a count is negative.
    """

    total_videos: int
    total_keyframes: int
    avg_keyframes_per_video: float
    min_keyframes: int
    max_keyframes: int
    p50: float
    p90: float
    p95: float
    total_keyframe_records: int
    duplicate_keyframe_records: int

    def __post_init__(self) -> None:
        if self.total_videos < 0 or self.total_keyframes < 0:
            raise ValueError("keyframe counts must be non-negative")
        if self.min_keyframes < 0 or self.max_keyframes < 0:
            raise ValueError("per-video keyframe counts must be non-negative")
        if self.total_keyframe_records < self.total_keyframes:
            raise ValueError("total_keyframe_records cannot be less than total_keyframes")
        if self.duplicate_keyframe_records < 0:
            raise ValueError("duplicate_keyframe_records must be non-negative")
        if self.total_keyframe_records - self.duplicate_keyframe_records < self.total_keyframes:
            raise ValueError(
                "unique records (records - duplicates) cannot be less than total_keyframes"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total_videos": self.total_videos,
            "total_keyframes": self.total_keyframes,
            "total_keyframe_records": self.total_keyframe_records,
            "duplicate_keyframe_records": self.duplicate_keyframe_records,
            "avg_keyframes_per_video": self.avg_keyframes_per_video,
            "min_keyframes": self.min_keyframes,
            "max_keyframes": self.max_keyframes,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
        }


@dataclass(frozen=True)
class GtCoverageRecord:
    """Keyframe-coverage result for one evaluation query.

    Expected input:
        - query_id: evaluation query identifier.
        - gt_video_id: ground-truth video id.
        - gt_frame_start: inclusive GT interval start (original-video frame).
        - gt_frame_end: inclusive GT interval end (original-video frame).
        - exact_hit: whether any keyframe falls inside the GT interval.
        - nearest_keyframe_frame_index: frame index of the keyframe nearest to
          the GT interval, or ``None`` when the GT video has no keyframes.
        - nearest_keyframe_distance: minimum absolute frame distance from a
          keyframe to the GT interval (0 when a keyframe is inside), or
          ``None`` when the GT video has no keyframes.
        - keyframes_in_video: number of keyframes of the GT video.
        - keyframes_in_range: number of keyframes inside the GT interval.

    Returns:
        - A validated per-query coverage record.

    Raises:
        - ValueError: if the nearest distance is negative or the exact-hit /
          distance values are inconsistent.
    """

    query_id: str
    gt_video_id: str
    gt_frame_start: int
    gt_frame_end: int
    exact_hit: bool
    nearest_keyframe_frame_index: int | None
    nearest_keyframe_distance: int | None
    keyframes_in_video: int
    keyframes_in_range: int

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        if self.keyframes_in_video < 0 or self.keyframes_in_range < 0:
            raise ValueError("keyframe counts must be non-negative")
        if self.keyframes_in_range > self.keyframes_in_video:
            raise ValueError("keyframes_in_range cannot exceed keyframes_in_video")
        if self.exact_hit and self.keyframes_in_range < 1:
            raise ValueError("exact_hit requires at least one keyframe in range")
        if self.nearest_keyframe_distance is not None and self.nearest_keyframe_distance < 0:
            raise ValueError("nearest_keyframe_distance must be non-negative")
        if self.nearest_keyframe_distance == 0 and not self.exact_hit:
            raise ValueError("zero nearest_keyframe_distance requires exact_hit")
        if (self.nearest_keyframe_frame_index is None) != (
            self.nearest_keyframe_distance is None
        ):
            raise ValueError(
                "nearest_keyframe_frame_index and nearest_keyframe_distance "
                "must both be None or both be defined"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "gt_video_id": self.gt_video_id,
            "gt_frame_start": self.gt_frame_start,
            "gt_frame_end": self.gt_frame_end,
            "exact_hit": self.exact_hit,
            "nearest_keyframe_frame_index": self.nearest_keyframe_frame_index,
            "nearest_keyframe_distance": self.nearest_keyframe_distance,
            "keyframes_in_video": self.keyframes_in_video,
            "keyframes_in_range": self.keyframes_in_range,
        }


@dataclass(frozen=True)
class RelaxedHit:
    """Relaxed-hit aggregate at one distance threshold.

    Expected input:
        - threshold: maximum distance (inclusive) for a relaxed hit.
        - count: number of queries whose nearest-keyframe distance is at most
          ``threshold``.
        - percentage: ``count`` as a percentage of the total query count.

    Returns:
        - A validated relaxed-hit record.

    Raises:
        - ValueError: if the threshold is not positive or the count is negative.
    """

    threshold: int
    count: int
    percentage: float

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, int) or isinstance(self.threshold, bool) or self.threshold < 1:
            raise ValueError("threshold must be a positive integer")
        if self.count < 0:
            raise ValueError("count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "threshold": self.threshold,
            "count": self.count,
            "percentage": self.percentage,
        }


@dataclass(frozen=True)
class GtCoverageSummary:
    """Aggregate keyframe coverage over all analyzed queries.

    Expected input:
        - total_queries: number of queries analyzed.
        - exact_hit_count: queries with at least one keyframe in the GT interval.
        - exact_hit_percentage: exact-hit count as a percentage of total.
        - relaxed_hits: per-threshold relaxed-hit aggregates.
        - queries_without_keyframes: queries whose GT video has no keyframes.
        - nearest_distance_stats: percentile/mean description of defined
          nearest-keyframe distances (empty when none are defined).

    Returns:
        - A validated coverage summary.

    Raises:
        - ValueError: if the counts do not reconcile with ``total_queries``.
    """

    total_queries: int
    exact_hit_count: int
    exact_hit_percentage: float
    relaxed_hits: tuple[RelaxedHit, ...]
    queries_without_keyframes: int
    nearest_distance_stats: dict[str, float]

    def __post_init__(self) -> None:
        if self.total_queries < 0 or self.exact_hit_count < 0:
            raise ValueError("coverage counts must be non-negative")
        if not 0 <= self.exact_hit_count <= self.total_queries:
            raise ValueError("exact_hit_count must be within [0, total_queries]")
        if not 0 <= self.queries_without_keyframes <= self.total_queries:
            raise ValueError("queries_without_keyframes must be within [0, total_queries]")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total_queries": self.total_queries,
            "exact_hit_count": self.exact_hit_count,
            "exact_hit_percentage": self.exact_hit_percentage,
            "relaxed_hits": [hit.to_dict() for hit in self.relaxed_hits],
            "queries_without_keyframes": self.queries_without_keyframes,
            "nearest_distance_stats": dict(self.nearest_distance_stats),
        }


@dataclass(frozen=True)
class KeyframeMissRecord:
    """One coverage miss (no exact keyframe hit) for the error report.

    Expected input:
        - query_id: evaluation query identifier.
        - gt_video_id: ground-truth video id.
        - gt_frame_start: inclusive GT interval start.
        - gt_frame_end: inclusive GT interval end.
        - closest_keyframe_frame_index: frame index of the nearest keyframe, or
          ``None`` when the GT video has no keyframes.
        - distance: minimum absolute frame distance to the GT interval, or
          ``None`` when the GT video has no keyframes.
        - keyframes_in_video: number of keyframes of the GT video.

    Returns:
        - A validated miss record.

    Raises:
        - ValueError: if the distance is negative.
    """

    query_id: str
    gt_video_id: str
    gt_frame_start: int
    gt_frame_end: int
    closest_keyframe_frame_index: int | None
    distance: int | None
    keyframes_in_video: int

    def __post_init__(self) -> None:
        if self.keyframes_in_video < 0:
            raise ValueError("keyframes_in_video must be non-negative")
        if self.distance is not None and self.distance < 0:
            raise ValueError("distance must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "gt_video_id": self.gt_video_id,
            "gt_frame_start": self.gt_frame_start,
            "gt_frame_end": self.gt_frame_end,
            "closest_keyframe_frame_index": self.closest_keyframe_frame_index,
            "distance": self.distance,
            "keyframes_in_video": self.keyframes_in_video,
        }


@dataclass(frozen=True)
class ClipComparisonRecord:
    """CLIP-baseline comparison result for one query.

    Expected input:
        - query_id: evaluation query identifier.
        - gt_video_id: ground-truth video id.
        - gt_frame_start: inclusive GT interval start.
        - gt_frame_end: inclusive GT interval end.
        - keyframe_covered: whether a keyframe lies inside the GT interval.
        - clip_retrieved_gt_video: whether the GT video appears in the CLIP
          top-k predictions.
        - clip_retrieved_gt_frame: whether any CLIP prediction has the GT video
          and a frame inside the GT interval.
        - clip_gt_video_best_rank: rank of the first GT-video prediction, or
          ``None`` when the video is absent.
        - clip_gt_frame_best_rank: rank of the first GT-frame prediction, or
          ``None`` when no such prediction exists.
        - category: one of ``KEYFRAME_COVERED``, ``KEYFRAME_NOT_COVERED``, or
          ``VIDEO_RETRIEVED_COVERAGE_ISSUE``.

    Returns:
        - A validated comparison record.

    Raises:
        - ValueError: if the category is inconsistent with the flags.
    """

    query_id: str
    gt_video_id: str
    gt_frame_start: int
    gt_frame_end: int
    keyframe_covered: bool
    clip_retrieved_gt_video: bool
    clip_retrieved_gt_frame: bool
    clip_gt_video_best_rank: int | None
    clip_gt_frame_best_rank: int | None
    category: str

    def __post_init__(self) -> None:
        expected = _classify_query(
            self.keyframe_covered,
            self.clip_retrieved_gt_video,
            self.clip_retrieved_gt_frame,
        )
        if self.category != expected:
            raise ValueError(f"category {self.category!r} inconsistent with flags (expected {expected!r})")
        if self.clip_retrieved_gt_frame and not self.clip_retrieved_gt_video:
            raise ValueError("clip_retrieved_gt_frame requires clip_retrieved_gt_video")
        if self.clip_retrieved_gt_video != (self.clip_gt_video_best_rank is not None):
            raise ValueError("clip_retrieved_gt_video must match clip_gt_video_best_rank presence")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "gt_video_id": self.gt_video_id,
            "gt_frame_start": self.gt_frame_start,
            "gt_frame_end": self.gt_frame_end,
            "keyframe_covered": self.keyframe_covered,
            "clip_retrieved_gt_video": self.clip_retrieved_gt_video,
            "clip_retrieved_gt_frame": self.clip_retrieved_gt_frame,
            "clip_gt_video_best_rank": self.clip_gt_video_best_rank,
            "clip_gt_frame_best_rank": self.clip_gt_frame_best_rank,
            "category": self.category,
        }


@dataclass(frozen=True)
class ClipComparisonSummary:
    """Aggregate CLIP-baseline comparison results.

    Expected input:
        - total_queries: number of queries analyzed.
        - clip_retrieved_gt_video_count: queries whose GT video appears in the
          CLIP top-k predictions.
        - clip_retrieved_gt_frame_count: queries with a GT-frame prediction.
        - category_counts: ``(category, count)`` pairs sorted by count desc then
          category.

    Returns:
        - A validated comparison summary.

    Raises:
        - ValueError: if the counts do not reconcile with ``total_queries``.
    """

    total_queries: int
    clip_retrieved_gt_video_count: int
    clip_retrieved_gt_frame_count: int
    category_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.total_queries < 0:
            raise ValueError("total_queries must be non-negative")
        if not 0 <= self.clip_retrieved_gt_video_count <= self.total_queries:
            raise ValueError("clip_retrieved_gt_video_count must be within [0, total_queries]")
        if not 0 <= self.clip_retrieved_gt_frame_count <= self.total_queries:
            raise ValueError("clip_retrieved_gt_frame_count must be within [0, total_queries]")
        if sum(count for _category, count in self.category_counts) != self.total_queries:
            raise ValueError("category_counts must sum to total_queries")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total_queries": self.total_queries,
            "clip_retrieved_gt_video_count": self.clip_retrieved_gt_video_count,
            "clip_retrieved_gt_frame_count": self.clip_retrieved_gt_frame_count,
            "categories": [
                {"category": category, "count": count} for category, count in self.category_counts
            ],
        }


@dataclass(frozen=True)
class KeyframeAnalysisReport:
    """Complete keyframe-analysis report.

    Expected input:
        - schema_version: report schema version.
        - keyframes_source: path to the ``frames.ndjson`` keyframe source.
        - evaluation_set_path: path to the evaluation CSV.
        - clip_predictions_path: path to the CLIP prediction artifact.
        - created_at: ISO-8601 UTC timestamp.
        - keyframe_statistics: :class:`KeyframeStatistics`.
        - coverage: :class:`GtCoverageSummary`.
        - per_query: per-query :class:`GtCoverageRecord` values.
        - error_report: :class:`KeyframeMissRecord` values, worst first.
        - clip_comparison: :class:`ClipComparisonSummary`.
        - clip_comparison_queries: per-query :class:`ClipComparisonRecord` values.

    Returns:
        - A validated report.

    Raises:
        - No exceptions.
    """

    schema_version: str
    keyframes_source: str
    evaluation_set_path: str
    clip_predictions_path: str
    created_at: str
    keyframe_statistics: KeyframeStatistics
    coverage: GtCoverageSummary
    per_query: tuple[GtCoverageRecord, ...]
    error_report: tuple[KeyframeMissRecord, ...]
    clip_comparison: ClipComparisonSummary
    clip_comparison_queries: tuple[ClipComparisonRecord, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the full report to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "keyframes_source": self.keyframes_source,
            "evaluation_set_path": self.evaluation_set_path,
            "clip_predictions_path": self.clip_predictions_path,
            "created_at": self.created_at,
            "keyframe_statistics": self.keyframe_statistics.to_dict(),
            "gt_coverage": self.coverage.to_dict(),
            "per_query": [record.to_dict() for record in self.per_query],
            "error_report": [record.to_dict() for record in self.error_report],
            "clip_comparison": {
                "summary": self.clip_comparison.to_dict(),
                "queries": [record.to_dict() for record in self.clip_comparison_queries],
            },
        }


def load_keyframes(frames_path: Path | str) -> dict[str, tuple[int, ...]]:
    """Load canonical keyframe frame indices grouped by video id.

    Expected input:
        - frames_path: path to a canonical artifact ``frames.ndjson`` file.

    Returns:
        - Mapping ``video_id -> sorted tuple of unique keyframe frame_index``.
          A video present in the file but with duplicate frame indices (the
          same original frame extracted more than once) is de-duplicated and
          sorted. Use :func:`load_keyframe_source_stats` to also obtain the
          raw record counts.

    Raises:
        - FileNotFoundError: if ``frames_path`` does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
    """
    keyframes_by_video, _total_records, _duplicates = _load_keyframes_and_stats(frames_path)
    return keyframes_by_video


def load_keyframe_source_stats(frames_path: Path | str) -> tuple[int, int]:
    """Return the raw record counts of a ``frames.ndjson`` file.

    Expected input:
        - frames_path: path to a canonical artifact ``frames.ndjson`` file.

    Returns:
        - ``(total_records, duplicate_records)`` where ``total_records`` is the
          number of keyframe records (the artifact row count) and
          ``duplicate_records`` is the number of records whose
          ``(video_id, frame_index)`` already appeared earlier in the file.

    Raises:
        - FileNotFoundError: if ``frames_path`` does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
    """
    _keyframes, total_records, duplicates = _load_keyframes_and_stats(frames_path)
    return total_records, duplicates


def analyze_keyframe_statistics(
    keyframes_by_video: Mapping[str, Sequence[int]],
    total_keyframe_records: int | None = None,
    duplicate_keyframe_records: int = 0,
) -> KeyframeStatistics:
    """Compute per-video keyframe-count statistics.

    Expected input:
        - keyframes_by_video: mapping ``video_id -> keyframe frame indices``
          (as returned by :func:`load_keyframes`).
        - total_keyframe_records: optional raw record count of the source
          ``frames.ndjson``; when omitted it defaults to the unique-frame
          count.
        - duplicate_keyframe_records: optional number of duplicate records
          (only meaningful together with ``total_keyframe_records``).

    Returns:
        - :class:`KeyframeStatistics` over the videos present.

    Raises:
        - No exceptions.
    """
    counts = [len(indices) for indices in keyframes_by_video.values()]
    total_videos = len(counts)
    total_keyframes = sum(counts)
    if counts:
        p50, p90, p95 = (float(value) for value in np.percentile(counts, [50, 90, 95]))
    else:
        p50, p90, p95 = 0.0, 0.0, 0.0
    records = (
        total_keyframe_records if total_keyframe_records is not None else total_keyframes
    )
    duplicates = duplicate_keyframe_records if total_keyframe_records is not None else 0
    return KeyframeStatistics(
        total_videos=total_videos,
        total_keyframes=total_keyframes,
        avg_keyframes_per_video=total_keyframes / total_videos if total_videos else 0.0,
        min_keyframes=min(counts) if counts else 0,
        max_keyframes=max(counts) if counts else 0,
        p50=p50,
        p90=p90,
        p95=p95,
        total_keyframe_records=records,
        duplicate_keyframe_records=duplicates,
    )


def analyze_gt_coverage(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    keyframes_by_video: Mapping[str, Sequence[int]],
    relaxed_thresholds: Sequence[int] = DEFAULT_RELAXED_THRESHOLDS,
) -> tuple[tuple[GtCoverageRecord, ...], GtCoverageSummary]:
    """Measure exact and relaxed keyframe coverage of GT events.

    Expected input:
        - queries: canonical KIS queries.
        - keyframes_by_video: mapping ``video_id -> keyframe frame indices``.
        - relaxed_thresholds: distance thresholds (inclusive) for relaxed hits.

    Returns:
        - ``(records, summary)``: per-query coverage records in query order and
          the aggregate summary.

    Raises:
        - ValueError: if a query is not KIS, queries are not unique, or the
          thresholds are invalid.
    """
    valid_thresholds = _validate_thresholds(relaxed_thresholds)
    records = tuple(
        _analyze_one_coverage(query, keyframes_by_video)
        for query in _normalize_queries(queries)
    )
    summary = _summarize_coverage(records, valid_thresholds)
    return records, summary


def build_error_report(
    coverage_records: Sequence[GtCoverageRecord],
) -> tuple[KeyframeMissRecord, ...]:
    """Return the coverage misses sorted worst first.

    Expected input:
        - coverage_records: per-query :class:`GtCoverageRecord` values.

    Returns:
        - :class:`KeyframeMissRecord` values for every query without an exact
          hit, sorted worst first: videos with no keyframes first, then by
          nearest distance descending.

    Raises:
        - No exceptions.
    """
    misses: list[KeyframeMissRecord] = []
    for record in coverage_records:
        if record.exact_hit:
            continue
        misses.append(
            KeyframeMissRecord(
                query_id=record.query_id,
                gt_video_id=record.gt_video_id,
                gt_frame_start=record.gt_frame_start,
                gt_frame_end=record.gt_frame_end,
                closest_keyframe_frame_index=record.nearest_keyframe_frame_index,
                distance=record.nearest_keyframe_distance,
                keyframes_in_video=record.keyframes_in_video,
            )
        )
    return tuple(
        sorted(
            misses,
            key=lambda record: (
                record.distance is not None,
                -(record.distance if record.distance is not None else 0),
            ),
        )
    )


def analyze_clip_comparison(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    keyframes_by_video: Mapping[str, Sequence[int]],
    predictions: Mapping[str, Sequence[KisPrediction]],
    top_k: int = DEFAULT_PREDICTION_TOP_K,
) -> tuple[tuple[ClipComparisonRecord, ...], ClipComparisonSummary]:
    """Categorize whether missing keyframe coverage explains CLIP failures.

    Expected input:
        - queries: canonical KIS queries.
        - keyframes_by_video: mapping ``video_id -> keyframe frame indices``.
        - predictions: scorer-compatible CLIP predictions keyed by query id
          (the form returned by
          :func:`aic2026.kis_prediction_runner.load_kis_predictions_for_scoring`).
        - top_k: number of predictions analyzed per query.

    Returns:
        - ``(records, summary)``: per-query comparison records and the
          aggregate summary.

    Raises:
        - ValueError: if a query is not KIS, ``top_k`` is invalid, or a query
          is missing from ``predictions``.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    records: list[ClipComparisonRecord] = []
    for query in _normalize_queries(queries):
        records.append(
            _analyze_one_clip_comparison(query, keyframes_by_video, predictions, top_k)
        )
    summary = _summarize_clip_comparison(records)
    return tuple(records), summary


def run_keyframe_analysis(
    evaluation_set_path: Path | str,
    frames_path: Path | str,
    clip_predictions_path: Path | str,
    relaxed_thresholds: Sequence[int] = DEFAULT_RELAXED_THRESHOLDS,
    prediction_top_k: int = DEFAULT_PREDICTION_TOP_K,
) -> KeyframeAnalysisReport:
    """Run the full keyframe-analysis pipeline and return the report.

    Expected input:
        - evaluation_set_path: path to ``eval_set.csv``.
        - frames_path: path to a canonical artifact ``frames.ndjson``.
        - clip_predictions_path: path to a ``predict-kis`` prediction artifact.
        - relaxed_thresholds: distance thresholds (inclusive) for relaxed hits.
        - prediction_top_k: CLIP predictions analyzed per query.

    Returns:
        - A complete :class:`KeyframeAnalysisReport`.

    Raises:
        - FileNotFoundError: if a required input path is absent.
        - ValueError: if inputs are invalid.
    """
    eval_set = load_eval_set(evaluation_set_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    if not queries:
        raise ValueError("evaluation set contains no KIS queries")

    keyframes, total_records, duplicate_records = _load_keyframes_and_stats(frames_path)
    statistics = analyze_keyframe_statistics(
        keyframes,
        total_keyframe_records=total_records,
        duplicate_keyframe_records=duplicate_records,
    )
    coverage_records, coverage = analyze_gt_coverage(
        queries, keyframes, relaxed_thresholds=relaxed_thresholds
    )
    error_report = build_error_report(coverage_records)
    predictions = load_kis_predictions_for_scoring(clip_predictions_path)
    clip_records, clip_summary = analyze_clip_comparison(
        queries, keyframes, predictions, top_k=prediction_top_k
    )
    return KeyframeAnalysisReport(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        keyframes_source=str(Path(frames_path).resolve()),
        evaluation_set_path=str(Path(evaluation_set_path).resolve()),
        clip_predictions_path=str(Path(clip_predictions_path).resolve()),
        created_at=_utc_now_iso(),
        keyframe_statistics=statistics,
        coverage=coverage,
        per_query=coverage_records,
        error_report=error_report,
        clip_comparison=clip_summary,
        clip_comparison_queries=clip_records,
    )


def format_report(report: KeyframeAnalysisReport) -> str:
    """Render a human-readable keyframe-analysis report.

    Expected input:
        - report: a :class:`KeyframeAnalysisReport`.

    Returns:
        - A multi-line human-readable report.

    Raises:
        - No exceptions.
    """
    lines: list[str] = ["BTC keyframe extraction analysis for KIS", "=" * 80]
    lines.append(f"Keyframes source: {report.keyframes_source}")
    lines.append(f"Evaluation set: {report.evaluation_set_path}")
    lines.append(f"CLIP predictions: {report.clip_predictions_path}")

    statistics = report.keyframe_statistics
    lines.append("1. Keyframe statistics")
    lines.append(f"  total videos: {statistics.total_videos}")
    lines.append(
        f"  total keyframes: {statistics.total_keyframes} unique "
        f"({statistics.total_keyframe_records} records, "
        f"{statistics.duplicate_keyframe_records} duplicate)"
    )
    lines.append(f"  average keyframes/video: {statistics.avg_keyframes_per_video:.2f}")
    lines.append(
        f"  distribution: min={statistics.min_keyframes} max={statistics.max_keyframes} "
        f"p50={statistics.p50:.1f} p90={statistics.p90:.1f} p95={statistics.p95:.1f}"
    )

    coverage = report.coverage
    lines.append("2. GT frame coverage")
    lines.append(
        f"  exact hit: {coverage.exact_hit_count}/{coverage.total_queries} "
        f"({coverage.exact_hit_percentage:.1f}%)"
    )
    for hit in coverage.relaxed_hits:
        lines.append(
            f"  relaxed <= {hit.threshold:>3} frames: {hit.count}/{coverage.total_queries} "
            f"({hit.percentage:.1f}%)"
        )
    lines.append(
        f"  queries with no keyframes in GT video: {coverage.queries_without_keyframes}"
    )
    if coverage.nearest_distance_stats:
        lines.append("  nearest-keyframe distance stats (defined distances):")
        for key, value in coverage.nearest_distance_stats.items():
            lines.append(f"    {key}: {value:.1f}")

    lines.append(f"3. Per-query error report (misses, worst first): {len(report.error_report)}")
    for record in report.error_report:
        lines.append(
            f"  {record.query_id} video={record.gt_video_id} range="
            f"[{record.gt_frame_start},{record.gt_frame_end}] "
            f"closest={record.closest_keyframe_frame_index} "
            f"distance={record.distance} keyframes_in_video={record.keyframes_in_video}"
        )

    comparison = report.clip_comparison
    lines.append("4. CLIP baseline comparison")
    lines.append(
        f"  GT video retrieved by CLIP: {comparison.clip_retrieved_gt_video_count}/{comparison.total_queries}"
    )
    lines.append(
        f"  GT frame retrieved by CLIP: {comparison.clip_retrieved_gt_frame_count}/{comparison.total_queries}"
    )
    for category, count in comparison.category_counts:
        lines.append(f"  {category}: {count}/{comparison.total_queries}")
    return "\n".join(lines) + "\n"


def write_report(path: Path | str, report: KeyframeAnalysisReport) -> None:
    """Write a keyframe-analysis report as JSON atomically.

    Expected input:
        - path: target JSON file path.
        - report: the report to persist.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(output_path)


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _load_keyframes_and_stats(
    frames_path: Path | str,
) -> tuple[dict[str, tuple[int, ...]], int, int]:
    """Load deduplicated keyframes plus raw record counts in one pass.

    Returns:
        - ``(keyframes_by_video, total_records, duplicate_records)`` where
          ``keyframes_by_video`` maps ``video_id`` to sorted unique
          ``frame_index`` values, ``total_records`` is the number of keyframe
          records, and ``duplicate_records`` counts records whose
          ``(video_id, frame_index)`` already appeared.

    Raises:
        - FileNotFoundError: if ``frames_path`` does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
    """
    keyframes_by_video: dict[str, set[int]] = {}
    total_records = 0
    duplicate_records = 0
    for frame in iter_frames(Path(frames_path)):
        video_id = frame.keyframe.video_id.value
        indices = keyframes_by_video.setdefault(video_id, set())
        total_records += 1
        if frame.frame_index in indices:
            duplicate_records += 1
        indices.add(frame.frame_index)
    mapping = {
        video_id: tuple(sorted(indices))
        for video_id, indices in sorted(keyframes_by_video.items())
    }
    return mapping, total_records, duplicate_records


def _normalize_queries(
    queries: EvaluationSet | Iterable[EvaluationQuery],
) -> tuple[EvaluationQuery, ...]:
    """Validate queries are unique KIS queries in input order."""
    values = queries.queries if isinstance(queries, EvaluationSet) else tuple(queries)
    if not values:
        raise ValueError("keyframe analysis requires at least one query")
    for query in values:
        if query.task is not EvaluationTask.KIS:
            raise ValueError(f"keyframe analysis requires task='kis', got {query.task.value!r}")
    query_ids = [query.query_id for query in values]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation query_ids must be unique")
    return values


def _validate_thresholds(thresholds: Sequence[int]) -> tuple[int, ...]:
    """Validate relaxed-hit thresholds are unique positive ints in order."""
    values = tuple(thresholds)
    if not values:
        raise ValueError("at least one relaxed-hit threshold is required")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("relaxed-hit thresholds must be positive integers")
        if value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized))


def _analyze_one_coverage(
    query: EvaluationQuery,
    keyframes_by_video: Mapping[str, Sequence[int]],
) -> GtCoverageRecord:
    """Compute the coverage record for one query."""
    video_id = query.ground_truth_video_id.value
    frame_indices = tuple(sorted(keyframes_by_video.get(video_id, ())))
    start = query.ground_truth_frame_range.start
    end = query.ground_truth_frame_range.end
    in_range = [frame for frame in frame_indices if start <= frame <= end]
    nearest_index, nearest_distance = _nearest_keyframe(frame_indices, start, end)
    return GtCoverageRecord(
        query_id=query.query_id,
        gt_video_id=video_id,
        gt_frame_start=start,
        gt_frame_end=end,
        exact_hit=bool(in_range),
        nearest_keyframe_frame_index=nearest_index,
        nearest_keyframe_distance=nearest_distance,
        keyframes_in_video=len(frame_indices),
        keyframes_in_range=len(in_range),
    )


def _nearest_keyframe(
    frame_indices: Sequence[int],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    """Return ``(nearest frame index, distance)`` to the ``[start, end]`` interval.

    Distance is 0 when a keyframe lies inside the interval, otherwise the
    minimum absolute frame offset from the interval. Returns ``(None, None)``
    when there are no keyframes.
    """
    if not frame_indices:
        return None, None
    position = bisect.bisect_left(frame_indices, start)
    candidates: list[int] = []
    if position < len(frame_indices):
        candidates.append(frame_indices[position])
    if position > 0:
        candidates.append(frame_indices[position - 1])
    nearest = min(candidates, key=lambda frame: _interval_distance(frame, start, end))
    return nearest, _interval_distance(nearest, start, end)


def _interval_distance(frame: int, start: int, end: int) -> int:
    """Return the absolute frame distance from a frame to an interval."""
    if start <= frame <= end:
        return 0
    if frame < start:
        return start - frame
    return frame - end


def _summarize_coverage(
    records: Sequence[GtCoverageRecord],
    thresholds: Sequence[int],
) -> GtCoverageSummary:
    """Aggregate per-query coverage records."""
    total = len(records)
    exact_hit_count = sum(1 for record in records if record.exact_hit)
    relaxed_hits = tuple(
        RelaxedHit(
            threshold=threshold,
            count=sum(
                1
                for record in records
                if record.nearest_keyframe_distance is not None
                and record.nearest_keyframe_distance <= threshold
            ),
            percentage=(
                sum(
                    1
                    for record in records
                    if record.nearest_keyframe_distance is not None
                    and record.nearest_keyframe_distance <= threshold
                )
                / total
                * 100.0
                if total
                else 0.0
            ),
        )
        for threshold in thresholds
    )
    distances = [
        record.nearest_keyframe_distance
        for record in records
        if record.nearest_keyframe_distance is not None
    ]
    distance_array = np.asarray(distances, dtype=np.float64)
    if distance_array.size:
        percentiles = np.percentile(distance_array, [50, 90, 95])
        distance_stats = {
            "min": float(np.min(distance_array)),
            "mean": float(np.mean(distance_array)),
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "max": float(np.max(distance_array)),
        }
    else:
        distance_stats = {}
    return GtCoverageSummary(
        total_queries=total,
        exact_hit_count=exact_hit_count,
        exact_hit_percentage=exact_hit_count / total * 100.0 if total else 0.0,
        relaxed_hits=relaxed_hits,
        queries_without_keyframes=sum(1 for record in records if record.keyframes_in_video == 0),
        nearest_distance_stats=distance_stats,
    )


def _analyze_one_clip_comparison(
    query: EvaluationQuery,
    keyframes_by_video: Mapping[str, Sequence[int]],
    predictions: Mapping[str, Sequence[KisPrediction]],
    top_k: int,
) -> ClipComparisonRecord:
    """Compute the CLIP-comparison record for one query."""
    video_id = query.ground_truth_video_id.value
    frame_indices = tuple(sorted(keyframes_by_video.get(video_id, ())))
    start = query.ground_truth_frame_range.start
    end = query.ground_truth_frame_range.end
    keyframe_covered = any(start <= frame <= end for frame in frame_indices)

    query_predictions = predictions.get(query.query_id)
    if query_predictions is None:
        raise ValueError(f"missing predictions for query {query.query_id!r}")
    video_rank: int | None = None
    frame_rank: int | None = None
    for prediction in query_predictions[:top_k]:
        if prediction.video_id != video_id:
            continue
        if video_rank is None:
            video_rank = prediction.rank
        if start <= prediction.frame_index <= end:
            frame_rank = prediction.rank
            break
    clip_video_ok = video_rank is not None
    clip_frame_ok = frame_rank is not None
    return ClipComparisonRecord(
        query_id=query.query_id,
        gt_video_id=video_id,
        gt_frame_start=start,
        gt_frame_end=end,
        keyframe_covered=keyframe_covered,
        clip_retrieved_gt_video=clip_video_ok,
        clip_retrieved_gt_frame=clip_frame_ok,
        clip_gt_video_best_rank=video_rank,
        clip_gt_frame_best_rank=frame_rank,
        category=_classify_query(keyframe_covered, clip_video_ok, clip_frame_ok),
    )


def _classify_query(
    keyframe_covered: bool,
    clip_video_ok: bool,
    clip_frame_ok: bool,
) -> str:
    """Return the exclusive CLIP-comparison category for one query.

    The coverage-issue category is the most specific: it applies when the GT
    video is retrieved by CLIP, the correct frame is not retrieved, and the GT
    event is not covered by a keyframe (so the coverage gap can explain the
    frame failure). It therefore implies ``KEYFRAME_NOT_COVERED``.
    """
    if not keyframe_covered and clip_video_ok and not clip_frame_ok:
        return VIDEO_RETRIEVED_COVERAGE_ISSUE
    if not keyframe_covered:
        return KEYFRAME_NOT_COVERED
    return KEYFRAME_COVERED


def _summarize_clip_comparison(
    records: Sequence[ClipComparisonRecord],
) -> ClipComparisonSummary:
    """Aggregate per-query CLIP-comparison records."""
    total = len(records)
    category_counts = tuple(
        sorted(
            {
                category: sum(1 for record in records if record.category == category)
                for category in (
                    KEYFRAME_COVERED,
                    KEYFRAME_NOT_COVERED,
                    VIDEO_RETRIEVED_COVERAGE_ISSUE,
                )
            }.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return ClipComparisonSummary(
        total_queries=total,
        clip_retrieved_gt_video_count=sum(
            1 for record in records if record.clip_retrieved_gt_video
        ),
        clip_retrieved_gt_frame_count=sum(
            1 for record in records if record.clip_retrieved_gt_frame
        ),
        category_counts=category_counts,
    )


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")