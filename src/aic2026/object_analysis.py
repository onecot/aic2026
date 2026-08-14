"""Read-only analysis of BTC object detections for KIS retrieval.

This module studies whether the per-keyframe object detections
(``objects/<VIDEO_ID>/<NNN>.json``) can support KIS retrieval. It performs
*analysis only*: it never runs or modifies retrieval, the scorer, evaluation,
or any prediction artifact. It computes four sections:

1. Object vocabulary statistics over the whole collection.
2. Query-object coverage: whether a query's object words appear in its
   ground-truth frames.
3. Ground-truth frame analysis: the object list, frequency, and confidence
   statistics inside each query's annotated frame range.
4. Retrieval potential: whether object overlap can separate ground-truth
   frames from false positives within the CLIP top-``k`` predictions.

The module reuses the existing raw parsers and canonical converters
(``aic2026.entities``, ``aic2026.converters``) and the canonical evaluation
contract (``aic2026.evaluation``). No fusion is implemented here.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from aic2026.converters import object_frame_to_detections
from aic2026.entities import OBJECT_FILENAME_PATTERN, parse_keyframe_map, parse_object_detections
from aic2026.evaluation import (
    EvaluationQuery,
    EvaluationSet,
    EvaluationTask,
    load_eval_set,
)
from aic2026.kis_prediction_runner import QueryTextField, load_kis_predictions_for_scoring
from aic2026.kis_scoring import KisPrediction

logger = logging.getLogger(__name__)

#: Default confidence thresholds whose effect on retained detections is reported.
DEFAULT_CONFIDENCE_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)

#: Default number of top object classes reported in the vocabulary section.
DEFAULT_TOP_CLASSES = 20

#: Default number of CLIP predictions analyzed per query.
DEFAULT_PREDICTION_TOP_K = 1000

#: Default per-section confidence used when computing GT / predicted object sets.
DEFAULT_MIN_CONFIDENCE = 0.3

#: Default number of matched / unmatched example queries reported.
DEFAULT_EXAMPLES = 5

CoverageStatus = Literal["MATCH", "NO_MATCH"]

_TOKENIZE_RE = re.compile(r"[^\w]+")


@dataclass(frozen=True)
class ThresholdEffect:
    """Vocabulary statistics at one confidence threshold.

    Expected input:
        - threshold: the confidence threshold applied to ``score >= threshold``.
        - kept_detections: number of detections retained.
        - kept_unique_entities: number of unique entities retained.
        - frames_with_detection: number of frames with at least one retained
          detection.
        - avg_detections_per_frame: mean retained detections per frame.

    Returns:
        - A validated threshold-effect record.

    Raises:
        - No exceptions.
    """

    threshold: float
    kept_detections: int
    kept_unique_entities: int
    frames_with_detection: int
    avg_detections_per_frame: float

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool):
            raise ValueError("threshold must be numeric")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if (
            self.kept_detections < 0
            or self.kept_unique_entities < 0
            or self.frames_with_detection < 0
        ):
            raise ValueError("threshold effect counts must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "threshold": self.threshold,
            "kept_detections": self.kept_detections,
            "kept_unique_entities": self.kept_unique_entities,
            "frames_with_detection": self.frames_with_detection,
            "avg_detections_per_frame": self.avg_detections_per_frame,
        }


@dataclass(frozen=True)
class ObjectVocabularyStats:
    """Aggregate vocabulary statistics over all object frames.

    Expected input:
        - total_frames: number of object frames scanned.
        - total_detections: number of detections scanned.
        - unique_entities: number of distinct entity names.
        - entity_frequency: ``(entity, count)`` pairs sorted by count desc,
          then entity.
        - confidence_stats: percentile description of all detection scores.
        - threshold_effects: per-threshold retention effects.

    Returns:
        - A validated vocabulary-statistics record.

    Raises:
        - No exceptions.
    """

    total_frames: int
    total_detections: int
    unique_entities: int
    entity_frequency: tuple[tuple[str, int], ...]
    confidence_stats: dict[str, float]
    threshold_effects: tuple[ThresholdEffect, ...]

    def avg_detections_per_frame(self) -> float:
        """Return mean detections per frame across all frames."""
        if self.total_frames == 0:
            return 0.0
        return self.total_detections / self.total_frames

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "unique_entities": self.unique_entities,
            "avg_detections_per_frame": self.avg_detections_per_frame(),
            "entity_frequency": [
                {"entity": entity, "count": count} for entity, count in self.entity_frequency
            ],
            "confidence_stats": dict(self.confidence_stats),
            "threshold_effects": [effect.to_dict() for effect in self.threshold_effects],
        }


@dataclass(frozen=True)
class QueryObjectCoverage:
    """Object-coverage result for one query.

    Expected input:
        - query_id: evaluation query identifier.
        - query_text_en: English query text, if present.
        - query_text_vi: Vietnamese query text.
        - object_words: query tokens that match the object vocabulary.
        - gt_entities: entities detected in the query's ground-truth frames.
        - matched_entities: subset of ``gt_entities`` sharing a token with
          ``object_words``.
        - status: ``MATCH`` when ``matched_entities`` is non-empty, else
          ``NO_MATCH``.

    Returns:
        - A validated per-query coverage record.

    Raises:
        - ValueError: if ``status`` is not consistent with the matched set.
    """

    query_id: str
    query_text_en: str | None
    query_text_vi: str
    object_words: tuple[str, ...]
    gt_entities: tuple[str, ...]
    matched_entities: tuple[str, ...]
    status: CoverageStatus

    def __post_init__(self) -> None:
        expected = "MATCH" if self.matched_entities else "NO_MATCH"
        if self.status != expected:
            raise ValueError(
                f"status {self.status!r} inconsistent with matched_entities "
                f"(expected {expected!r})"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "query_text_en": self.query_text_en,
            "query_text_vi": self.query_text_vi,
            "object_words": list(self.object_words),
            "gt_entities": list(self.gt_entities),
            "matched_entities": list(self.matched_entities),
            "status": self.status,
        }


@dataclass(frozen=True)
class ObjectCoverageSummary:
    """Aggregate coverage results and matched/unmatched examples.

    Expected input:
        - total: number of queries analyzed.
        - matched_count: number of ``MATCH`` queries.
        - no_match_count: number of ``NO_MATCH`` queries.
        - matched_examples: first ``example_limit`` matched queries.
        - unmatched_examples: first ``example_limit`` unmatched queries.

    Returns:
        - A validated coverage summary.

    Raises:
        - ValueError: if counts do not reconcile with ``total``.
    """

    total: int
    matched_count: int
    no_match_count: int
    matched_examples: tuple[QueryObjectCoverage, ...]
    unmatched_examples: tuple[QueryObjectCoverage, ...]

    def __post_init__(self) -> None:
        if self.total < 0 or self.matched_count < 0 or self.no_match_count < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.matched_count + self.no_match_count != self.total:
            raise ValueError("matched + no_match must equal total")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total": self.total,
            "matched_count": self.matched_count,
            "no_match_count": self.no_match_count,
            "matched_examples": [record.to_dict() for record in self.matched_examples],
            "unmatched_examples": [record.to_dict() for record in self.unmatched_examples],
        }


@dataclass(frozen=True)
class GtFrameObjectAnalysis:
    """Object analysis of one query's ground-truth frames.

    Expected input:
        - query_id: evaluation query identifier.
        - gt_video_id: ground-truth video id.
        - gt_keyframe_ordinals: keyframe ordinals inside the GT frame range.
        - gt_object_list: ``(entity, count)`` pairs over GT frames, sorted by
          count desc then entity.
        - confidence_stats: percentile description of GT-frame detection scores.
        - total_detections: number of detections in GT frames.

    Returns:
        - A validated per-query GT analysis record.

    Raises:
        - No exceptions.
    """

    query_id: str
    gt_video_id: str
    gt_keyframe_ordinals: tuple[int, ...]
    gt_object_list: tuple[tuple[str, int], ...]
    confidence_stats: dict[str, float]
    total_detections: int

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.gt_video_id, str) or self.gt_video_id.strip() == "":
            raise ValueError("gt_video_id must be a non-empty string")
        if len(set(self.gt_keyframe_ordinals)) != len(self.gt_keyframe_ordinals):
            raise ValueError("gt_keyframe_ordinals must be unique")
        if self.total_detections < 0:
            raise ValueError("total_detections must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "gt_video_id": self.gt_video_id,
            "gt_keyframe_ordinals": list(self.gt_keyframe_ordinals),
            "gt_object_list": [
                {"entity": entity, "count": count} for entity, count in self.gt_object_list
            ],
            "confidence_stats": dict(self.confidence_stats),
            "total_detections": self.total_detections,
        }


@dataclass(frozen=True)
class RetrievalPotentialQuery:
    """Retrieval-potential estimate for one query.

    Expected input:
        - query_id: evaluation query identifier.
        - gt_video_id: ground-truth video id.
        - predicted_frames: number of predictions analyzed.
        - gt_frame_predictions: predictions whose frame falls inside the GT
          range of the GT video.
        - gt_overlap_count: GT-frame predictions sharing a GT object entity.
        - false_positive_overlap_count: non-GT predictions sharing a GT object
          entity.
        - gt_overlap_rate: fraction of GT-frame predictions with overlap
          (``None`` when there are none).
        - false_positive_overlap_rate: fraction of non-GT predictions with
          overlap (``None`` when there are none).
        - separation: ``gt_overlap_rate - false_positive_overlap_rate`` when
          both rates are defined, else ``None``. Positive means objects can
          help separate GT frames from false positives.

    Returns:
        - A validated per-query potential record.

    Raises:
        - No exceptions.
    """

    query_id: str
    gt_video_id: str
    predicted_frames: int
    gt_frame_predictions: int
    gt_overlap_count: int
    false_positive_overlap_count: int
    gt_overlap_rate: float | None
    false_positive_overlap_rate: float | None
    separation: float | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "gt_video_id": self.gt_video_id,
            "predicted_frames": self.predicted_frames,
            "gt_frame_predictions": self.gt_frame_predictions,
            "gt_overlap_count": self.gt_overlap_count,
            "false_positive_overlap_count": self.false_positive_overlap_count,
            "gt_overlap_rate": self.gt_overlap_rate,
            "false_positive_overlap_rate": self.false_positive_overlap_rate,
            "separation": self.separation,
        }


@dataclass(frozen=True)
class RetrievalPotentialSummary:
    """Aggregate retrieval-potential estimates.

    Expected input:
        - total: number of queries analyzed.
        - queries_with_gt_predictions: queries having at least one GT-frame
          prediction.
        - queries_with_separation: queries with a defined separation.
        - queries_where_objects_help: queries with positive separation.
        - avg_separation: mean separation over defined values (``None`` when
          none).
        - mean_gt_overlap_rate: mean GT-frame overlap rate over defined
          values (``None`` when none).
        - mean_false_positive_overlap_rate: mean FP overlap rate over defined
          values (``None`` when none).

    Returns:
        - A validated summary.

    Raises:
        - No exceptions.
    """

    total: int
    queries_with_gt_predictions: int
    queries_with_separation: int
    queries_where_objects_help: int
    avg_separation: float | None
    mean_gt_overlap_rate: float | None
    mean_false_positive_overlap_rate: float | None

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if not 0 <= self.queries_with_gt_predictions <= self.total:
            raise ValueError("queries_with_gt_predictions must be within [0, total]")
        if not 0 <= self.queries_with_separation <= self.queries_with_gt_predictions:
            raise ValueError("queries_with_separation must be within [0, queries_with_gt_predictions]")
        if not 0 <= self.queries_where_objects_help <= self.queries_with_separation:
            raise ValueError("queries_where_objects_help must be within [0, queries_with_separation]")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total": self.total,
            "queries_with_gt_predictions": self.queries_with_gt_predictions,
            "queries_with_separation": self.queries_with_separation,
            "queries_where_objects_help": self.queries_where_objects_help,
            "avg_separation": self.avg_separation,
            "mean_gt_overlap_rate": self.mean_gt_overlap_rate,
            "mean_false_positive_overlap_rate": self.mean_false_positive_overlap_rate,
        }


@dataclass(frozen=True)
class ObjectAnalysisReport:
    """Complete object-analysis report.

    Expected input:
        - vocabulary: :class:`ObjectVocabularyStats`.
        - coverage: :class:`ObjectCoverageSummary`.
        - gt_analyses: per-query :class:`GtFrameObjectAnalysis` records.
        - retrieval: :class:`RetrievalPotentialSummary`.
        - retrieval_queries: per-query :class:`RetrievalPotentialQuery` records.
        - query_text_field: the query-text field analyzed.
        - min_confidence: the confidence used for GT / predicted object sets.

    Returns:
        - A validated report.

    Raises:
        - No exceptions.
    """

    vocabulary: ObjectVocabularyStats
    coverage: ObjectCoverageSummary
    gt_analyses: tuple[GtFrameObjectAnalysis, ...]
    retrieval: RetrievalPotentialSummary
    retrieval_queries: tuple[RetrievalPotentialQuery, ...]
    query_text_field: QueryTextField
    min_confidence: float

    def to_dict(self) -> dict[str, object]:
        """Serialize the full report to a JSON-safe dictionary."""
        return {
            "query_text_field": self.query_text_field,
            "min_confidence": self.min_confidence,
            "vocabulary": self.vocabulary.to_dict(),
            "coverage": self.coverage.to_dict(),
            "gt_frames": [record.to_dict() for record in self.gt_analyses],
            "retrieval_potential": {
                "summary": self.retrieval.to_dict(),
                "queries": [record.to_dict() for record in self.retrieval_queries],
            },
        }


def analyze_object_vocabulary(
    objects_dir: Path | str,
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    top_classes: int = DEFAULT_TOP_CLASSES,
    max_videos: int | None = None,
    video_ids_filter: set[str] | None = None,
) -> ObjectVocabularyStats:
    """Scan all object frames and compute vocabulary statistics.

    Expected input:
        - objects_dir: path to the raw ``objects`` collection directory.
        - thresholds: confidence thresholds for the retention effects.
        - top_classes: number of top entities retained in
          :attr:`entity_frequency`; 0 disables the cap.
        - max_videos: optional cap on the number of videos scanned (videos are
          scanned in sorted order).
        - video_ids_filter: optional set of video ids to restrict to.

    Returns:
        - :class:`ObjectVocabularyStats` over the scanned frames.

    Raises:
        - FileNotFoundError: if ``objects_dir`` does not exist.
    """
    directory = Path(objects_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"objects collection not found: {directory}")

    valid_thresholds = _validate_thresholds(thresholds)
    video_ids = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir() and _looks_like_video_id(entry.name)
    )
    if video_ids_filter is not None:
        video_ids = [vid for vid in video_ids if vid in video_ids_filter]
    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    entity_counter: Counter[str] = Counter()
    score_arrays: list[np.ndarray] = []
    total_frames = 0
    total_detections = 0

    threshold_counts = {t: 0 for t in valid_thresholds}
    threshold_entity_sets: dict[float, set[str]] = {t: set() for t in valid_thresholds}
    threshold_frames = {t: 0 for t in valid_thresholds}

    for video_id in video_ids:
        video_dir = directory / video_id
        for entry in sorted(video_dir.iterdir()):
            if OBJECT_FILENAME_PATTERN.match(entry.name) is None:
                continue
            try:
                frame = parse_object_detections(entry, video_id)
                detections = object_frame_to_detections(frame).detections
            except (OSError, ValueError) as exc:
                logger.warning("skipping unreadable object file %s: %s", entry, exc)
                continue
            total_frames += 1
            frame_scores = np.asarray(
                [detection.score for detection in detections], dtype=np.float32
            )
            score_arrays.append(frame_scores)
            for detection in detections:
                entity_counter[detection.class_entity] += 1
                total_detections += 1
                for t in valid_thresholds:
                    if detection.score >= t:
                        threshold_counts[t] += 1
                        threshold_entity_sets[t].add(detection.class_entity)
            for t in valid_thresholds:
                if frame_scores.size and bool(np.any(frame_scores >= t)):
                    threshold_frames[t] += 1

    entity_frequency = tuple(_sorted_counter(entity_counter, top_classes))
    confidence_stats = _percentile_stats(np.concatenate(score_arrays) if score_arrays else np.zeros(0))
    threshold_effects = tuple(
        ThresholdEffect(
            threshold=t,
            kept_detections=threshold_counts[t],
            kept_unique_entities=len(threshold_entity_sets[t]),
            frames_with_detection=threshold_frames[t],
            avg_detections_per_frame=(
                threshold_counts[t] / total_frames if total_frames else 0.0
            ),
        )
        for t in valid_thresholds
    )
    return ObjectVocabularyStats(
        total_frames=total_frames,
        total_detections=total_detections,
        unique_entities=len(entity_counter),
        entity_frequency=entity_frequency,
        confidence_stats=confidence_stats,
        threshold_effects=threshold_effects,
    )


def extract_object_words(query_text: str, entity_tokens: set[str]) -> tuple[str, ...]:
    """Return the query tokens present in the object-entity token vocabulary.

    Expected input:
        - query_text: query text to scan for object words.
        - entity_tokens: lowercased tokens of all object entity names.

    Returns:
        - Sorted unique query tokens that appear in ``entity_tokens``.

    Raises:
        - ValueError: if ``query_text`` is empty.
    """
    if not isinstance(query_text, str) or query_text.strip() == "":
        raise ValueError("query_text must be a non-empty string")
    query_tokens = _tokenize(query_text.lower())
    return tuple(sorted({token for token in query_tokens if token in entity_tokens}))


def build_entity_token_vocabulary(entity_frequency: Iterable[tuple[str, int]]) -> set[str]:
    """Build the lowercased token vocabulary from entity names.

    Expected input:
        - entity_frequency: ``(entity, count)`` pairs (e.g. the vocabulary's
          ``entity_frequency``).

    Returns:
        - Set of lowercased tokens across all entity names.

    Raises:
        - No exceptions.
    """
    tokens: set[str] = set()
    for entity, _count in entity_frequency:
        tokens.update(_tokenize(entity.lower()))
    return tokens


def analyze_query_object_coverage(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    objects_dir: Path | str,
    map_dir: Path | str,
    entity_tokens: set[str],
    query_text_field: QueryTextField = "query_text_en",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    example_limit: int = DEFAULT_EXAMPLES,
) -> tuple[tuple[QueryObjectCoverage, ...], ObjectCoverageSummary]:
    """Classify each query as having object coverage in its GT frames.

    Expected input:
        - queries: canonical KIS queries.
        - objects_dir: path to the raw ``objects`` collection directory.
        - map_dir: path to the raw ``map-keyframes`` collection directory.
        - entity_tokens: object-entity token vocabulary.
        - query_text_field: ``"query_text_en"`` (default) or ``"query_text_vi"``.
        - min_confidence: only entities detected with ``score >= min_confidence``
          count as GT entities.
        - example_limit: number of matched/unmatched examples to report.

    Returns:
        - ``(records, summary)``: per-query coverage in query order and the
          aggregate summary with examples.

    Raises:
        - FileNotFoundError: if a collection directory is absent.
        - ValueError: if a query is not KIS or the text field is invalid.
    """
    records = tuple(
        _analyze_one_coverage(query, objects_dir, map_dir, entity_tokens, query_text_field, min_confidence)
        for query in _normalize_queries(queries)
    )
    matched = tuple(record for record in records if record.status == "MATCH")
    unmatched = tuple(record for record in records if record.status == "NO_MATCH")
    summary = ObjectCoverageSummary(
        total=len(records),
        matched_count=len(matched),
        no_match_count=len(unmatched),
        matched_examples=matched[:example_limit],
        unmatched_examples=unmatched[:example_limit],
    )
    return records, summary


def analyze_gt_frames(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    objects_dir: Path | str,
    map_dir: Path | str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[GtFrameObjectAnalysis, ...]:
    """Analyze object detections inside every query's GT frame range.

    Expected input:
        - queries: canonical KIS queries.
        - objects_dir: path to the raw ``objects`` collection directory.
        - map_dir: path to the raw ``map-keyframes`` collection directory.
        - min_confidence: only detections with ``score >= min_confidence`` are
          retained in ``gt_object_list`` and ``confidence_stats``.

    Returns:
        - Per-query :class:`GtFrameObjectAnalysis` records in query order.

    Raises:
        - FileNotFoundError: if a collection directory is absent.
        - ValueError: if a query is not KIS.
    """
    return tuple(
        _analyze_one_gt_frame(query, objects_dir, map_dir, min_confidence)
        for query in _normalize_queries(queries)
    )


def analyze_retrieval_potential(
    queries: EvaluationSet | Iterable[EvaluationQuery],
    predictions: Mapping[str, Sequence[KisPrediction]],
    objects_dir: Path | str,
    map_dir: Path | str,
    top_k: int = DEFAULT_PREDICTION_TOP_K,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[tuple[RetrievalPotentialQuery, ...], RetrievalPotentialSummary]:
    """Estimate whether objects separate GT frames from false positives.

    Expected input:
        - queries: canonical KIS queries.
        - predictions: scorer-compatible CLIP predictions keyed by query id
          (the form returned by :func:`load_kis_predictions_for_scoring`).
        - objects_dir: path to the raw ``objects`` collection directory.
        - map_dir: path to the raw ``map-keyframes`` collection directory.
        - top_k: number of predictions analyzed per query.
        - min_confidence: minimum score for an entity to count toward overlap.

    Returns:
        - ``(records, summary)``: per-query potential estimates and the
          aggregate summary.

    Raises:
        - FileNotFoundError: if a collection directory is absent.
        - ValueError: if a query is not KIS, ``top_k`` is invalid, or a query
          is missing from ``predictions``.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    normalized = _normalize_queries(queries)
    frame_map_cache: dict[str, dict[int, int]] = {}
    detection_cache: dict[tuple[str, int], tuple[str, ...]] = {}

    records: list[RetrievalPotentialQuery] = []
    for query in normalized:
        query_predictions = predictions.get(query.query_id)
        if query_predictions is None:
            raise ValueError(f"missing predictions for query {query.query_id!r}")
        gt_entities = _gt_entity_set(
            query, objects_dir, map_dir, min_confidence, frame_map_cache, detection_cache
        )
        gt_start = query.ground_truth_frame_range.start
        gt_end = query.ground_truth_frame_range.end
        gt_video = query.ground_truth_video_id.value

        gt_frame_predictions = 0
        gt_overlap = 0
        fp_overlap = 0
        total = 0
        for prediction in query_predictions[:top_k]:
            total += 1
            in_gt_range = (
                prediction.video_id == gt_video
                and gt_start <= prediction.frame_index <= gt_end
            )
            entities = _frame_entities(
                prediction.video_id,
                prediction.frame_index,
                objects_dir,
                map_dir,
                frame_map_cache,
                detection_cache,
            )
            has_overlap = bool(gt_entities.intersection(entities))
            if in_gt_range:
                gt_frame_predictions += 1
                if has_overlap:
                    gt_overlap += 1
            elif has_overlap:
                fp_overlap += 1

        gt_overlap_rate = (
            gt_overlap / gt_frame_predictions if gt_frame_predictions else None
        )
        false_positive_count = total - gt_frame_predictions
        false_positive_overlap_rate = (
            fp_overlap / false_positive_count if false_positive_count else None
        )
        separation = (
            gt_overlap_rate - false_positive_overlap_rate
            if gt_overlap_rate is not None and false_positive_overlap_rate is not None
            else None
        )
        records.append(
            RetrievalPotentialQuery(
                query_id=query.query_id,
                gt_video_id=gt_video,
                predicted_frames=total,
                gt_frame_predictions=gt_frame_predictions,
                gt_overlap_count=gt_overlap,
                false_positive_overlap_count=fp_overlap,
                gt_overlap_rate=gt_overlap_rate,
                false_positive_overlap_rate=false_positive_overlap_rate,
                separation=separation,
            )
        )

    summary = _summarize_potential(records)
    return tuple(records), summary


def format_report(report: ObjectAnalysisReport) -> str:
    """Render a human-readable object-analysis report.

    Expected input:
        - report: an :class:`ObjectAnalysisReport`.

    Returns:
        - A multi-line human-readable report.

    Raises:
        - No exceptions.
    """
    lines: list[str] = ["BTC object detection analysis for KIS retrieval", "=" * 80]
    lines.append(f"Query text field: {report.query_text_field}")
    lines.append(f"GT / predicted object min confidence: {report.min_confidence:.2f}")

    vocabulary = report.vocabulary
    lines.append("1. Object vocabulary")
    lines.append(f"  total object frames: {vocabulary.total_frames}")
    lines.append(f"  total detections: {vocabulary.total_detections}")
    lines.append(f"  unique object entities: {vocabulary.unique_entities}")
    lines.append(f"  average detections per frame: {vocabulary.avg_detections_per_frame():.2f}")
    lines.append("  confidence stats:")
    for key, value in vocabulary.confidence_stats.items():
        lines.append(f"    {key}: {value:.4f}")
    lines.append("  effect of confidence thresholds:")
    for effect in vocabulary.threshold_effects:
        lines.append(
            f"    >= {effect.threshold:.1f}: {effect.kept_detections} detections "
            f"({effect.kept_unique_entities} entities, {effect.frames_with_detection} "
            f"frames, {effect.avg_detections_per_frame:.2f}/frame)"
        )
    lines.append("  top object classes:")
    for entity, count in vocabulary.entity_frequency[:DEFAULT_TOP_CLASSES]:
        lines.append(f"    {entity}: {count}")

    coverage = report.coverage
    lines.append("2. Query-object coverage")
    lines.append(f"  queries with object coverage (MATCH): {coverage.matched_count}/{coverage.total}")
    lines.append(f"  queries without object coverage (NO_MATCH): {coverage.no_match_count}/{coverage.total}")
    lines.append("  matched examples:")
    for record in coverage.matched_examples:
        lines.append(
            f"    {record.query_id} words={record.object_words} "
            f"matched={record.matched_entities}"
        )
    lines.append("  unmatched examples:")
    for record in coverage.unmatched_examples:
        lines.append(
            f"    {record.query_id} words={record.object_words} gt_entities={record.gt_entities}"
        )

    lines.append("3. Ground-truth frame analysis")
    for analysis in report.gt_analyses:
        lines.append(f"  {analysis.query_id} ({analysis.gt_video_id}):")
        lines.append(f"    keyframes in range: {analysis.gt_keyframe_ordinals}")
        lines.append(
            f"    objects ({len(analysis.gt_object_list)}): "
            + ", ".join(f"{e}:{c}" for e, c in analysis.gt_object_list[:10])
        )
        lines.append(f"    detections: {analysis.total_detections}")

    retrieval = report.retrieval
    lines.append("4. Retrieval potential (CLIP top-k)")
    lines.append(
        f"  queries with GT-frame predictions: {retrieval.queries_with_gt_predictions}/{retrieval.total}"
    )
    lines.append(
        f"  queries where object overlap separates (positive separation): "
        f"{retrieval.queries_where_objects_help}/{retrieval.queries_with_separation}"
    )
    if retrieval.avg_separation is not None:
        lines.append(f"  average separation: {retrieval.avg_separation:.4f}")
    if retrieval.mean_gt_overlap_rate is not None:
        lines.append(f"  mean GT-frame overlap rate: {retrieval.mean_gt_overlap_rate:.4f}")
    if retrieval.mean_false_positive_overlap_rate is not None:
        lines.append(
            f"  mean false-positive overlap rate: {retrieval.mean_false_positive_overlap_rate:.4f}"
        )
    for record in report.retrieval_queries:
        lines.append(
            f"  {record.query_id}: GT frames={record.gt_frame_predictions} "
            f"overlap={record.gt_overlap_count} "
            f"gt_rate={_fmt(record.gt_overlap_rate)} fp_rate={_fmt(record.false_positive_overlap_rate)} "
            f"separation={_fmt(record.separation)}"
        )
    return "\n".join(lines) + "\n"


def run_object_analysis(
    objects_dir: Path | str,
    map_dir: Path | str,
    evaluation_set_path: Path | str,
    clip_predictions_path: Path | str,
    query_text_field: QueryTextField = "query_text_en",
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    top_classes: int = DEFAULT_TOP_CLASSES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    prediction_top_k: int = DEFAULT_PREDICTION_TOP_K,
    example_limit: int = DEFAULT_EXAMPLES,
    max_videos: int | None = None,
    video_ids_filter: set[str] | None = None,
) -> ObjectAnalysisReport:
    """Run the full object-analysis pipeline and return the report.

    Expected input:
        - objects_dir: path to the raw ``objects`` collection directory.
        - map_dir: path to the raw ``map-keyframes`` collection directory.
        - evaluation_set_path: path to ``eval_set.csv``.
        - clip_predictions_path: path to a KIS prediction artifact (CLIP
          baseline) used for the retrieval-potential section.
        - query_text_field: ``"query_text_en"`` (default) or ``"query_text_vi"``.
        - thresholds: confidence thresholds for vocabulary retention effects.
        - top_classes: number of top entities retained in the vocabulary.
        - min_confidence: min score for GT / predicted object sets.
        - prediction_top_k: CLIP predictions analyzed per query.
        - example_limit: number of matched/unmatched examples reported.
        - max_videos: optional cap on videos scanned for vocabulary stats.
        - video_ids_filter: optional video id restriction for vocabulary stats.

    Returns:
        - A complete :class:`ObjectAnalysisReport`.

    Raises:
        - FileNotFoundError: if a required input path is absent.
        - ValueError: if inputs are invalid.
    """
    directory = Path(objects_dir)
    map_directory = Path(map_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"objects collection not found: {directory}")
    if not map_directory.is_dir():
        raise FileNotFoundError(f"map-keyframes collection not found: {map_directory}")
    eval_set = load_eval_set(evaluation_set_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    if not queries:
        raise ValueError("evaluation set contains no KIS queries")
    predictions = load_kis_predictions_for_scoring(clip_predictions_path)

    vocabulary = analyze_object_vocabulary(
        directory,
        thresholds=thresholds,
        top_classes=top_classes,
        max_videos=max_videos,
        video_ids_filter=video_ids_filter,
    )
    entity_tokens = build_entity_token_vocabulary(vocabulary.entity_frequency)
    coverage_records, coverage = analyze_query_object_coverage(
        queries,
        directory,
        map_directory,
        entity_tokens,
        query_text_field=query_text_field,
        min_confidence=min_confidence,
        example_limit=example_limit,
    )
    del coverage_records  # summary carries the examples
    gt_analyses = analyze_gt_frames(
        queries,
        directory,
        map_directory,
        min_confidence=min_confidence,
    )
    retrieval_queries, retrieval = analyze_retrieval_potential(
        queries,
        predictions,
        directory,
        map_directory,
        top_k=prediction_top_k,
        min_confidence=min_confidence,
    )
    return ObjectAnalysisReport(
        vocabulary=vocabulary,
        coverage=coverage,
        gt_analyses=gt_analyses,
        retrieval=retrieval,
        retrieval_queries=retrieval_queries,
        query_text_field=query_text_field,
        min_confidence=min_confidence,
    )


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _normalize_queries(
    queries: EvaluationSet | Iterable[EvaluationQuery],
) -> tuple[EvaluationQuery, ...]:
    """Validate queries are unique KIS queries in input order."""
    values = queries.queries if isinstance(queries, EvaluationSet) else tuple(queries)
    if not values:
        raise ValueError("object analysis requires at least one query")
    for query in values:
        if query.task is not EvaluationTask.KIS:
            raise ValueError(f"object analysis requires task='kis', got {query.task.value!r}")
    query_ids = [query.query_id for query in values]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation query_ids must be unique")
    return values


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    """Validate confidence thresholds are unique floats in [0, 1]."""
    values = tuple(thresholds)
    if not values:
        raise ValueError("at least one confidence threshold is required")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError("confidence thresholds must be numeric")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("confidence thresholds must be in [0, 1]")
    return values


def _looks_like_video_id(name: str) -> bool:
    """Return whether a collection entry name looks like a video id."""
    return bool(re.fullmatch(r"L\d+_V\d+", name))


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [token for token in _TOKENIZE_RE.split(text) if token]


def _sorted_counter(counter: Counter[str], top: int) -> list[tuple[str, int]]:
    """Return counter items sorted by count desc then entity, capped at top."""
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if top and top > 0:
        items = items[:top]
    return items


def _percentile_stats(scores: np.ndarray) -> dict[str, float]:
    """Compute a percentile description of a score array."""
    if scores.size == 0:
        return {
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    percentiles = np.percentile(scores, [1, 5, 10, 25, 50, 75, 95, 99])
    return {
        "min": float(np.min(scores)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "median": float(percentiles[4]),
        "p75": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "p99": float(percentiles[7]),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


def _load_frame_map(map_dir: Path, video_id: str) -> dict[int, int]:
    """Return ``frame_idx -> keyframe ordinal`` for one video."""
    map_path = map_dir / f"{video_id}.csv"
    rows = parse_keyframe_map(map_path)
    return {row.frame_idx: row.keyframe_no for row in rows}


def _keyframe_ordinals_in_range(
    frame_map: dict[int, int], start: int, end: int
) -> tuple[int, ...]:
    """Return sorted keyframe ordinals whose frame index falls in [start, end]."""
    return tuple(sorted(ordinal for frame, ordinal in frame_map.items() if start <= frame <= end))


def _load_frame_entities(
    objects_dir: Path,
    video_id: str,
    keyframe_ordinal: int,
    min_confidence: float,
) -> tuple[str, ...]:
    """Return entities detected in one keyframe at or above ``min_confidence``."""
    object_path = objects_dir / video_id / f"{keyframe_ordinal:03d}.json"
    frame = parse_object_detections(object_path, video_id)
    detections = object_frame_to_detections(frame).detections
    return tuple(
        sorted(
            {
                detection.class_entity
                for detection in detections
                if detection.score >= min_confidence
            }
        )
    )


def _frame_entities(
    video_id: str,
    frame_index: int,
    objects_dir: Path,
    map_dir: Path,
    frame_map_cache: dict[str, dict[int, int]],
    detection_cache: dict[tuple[str, int], tuple[str, ...]],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[str, ...]:
    """Return cached entities for a predicted ``(video_id, frame_index)``.

    Returns an empty tuple when the frame index is not a known keyframe.
    """
    frame_map = frame_map_cache.get(video_id)
    if frame_map is None:
        try:
            frame_map = _load_frame_map(map_dir, video_id)
        except FileNotFoundError:
            frame_map = {}
        frame_map_cache[video_id] = frame_map
    ordinal = frame_map.get(frame_index)
    if ordinal is None:
        return ()
    key = (video_id, ordinal)
    if key not in detection_cache:
        try:
            detection_cache[key] = _load_frame_entities(
                objects_dir, video_id, ordinal, min_confidence
            )
        except (OSError, ValueError):
            detection_cache[key] = ()
    return detection_cache[key]


def _gt_entity_set(
    query: EvaluationQuery,
    objects_dir: Path,
    map_dir: Path,
    min_confidence: float,
    frame_map_cache: dict[str, dict[int, int]],
    detection_cache: dict[tuple[str, int], tuple[str, ...]],
) -> set[str]:
    """Return the set of entities detected in a query's GT frames."""
    video_id = query.ground_truth_video_id.value
    frame_map = frame_map_cache.get(video_id)
    if frame_map is None:
        try:
            frame_map = _load_frame_map(map_dir, video_id)
        except FileNotFoundError:
            frame_map = {}
        frame_map_cache[video_id] = frame_map
    ordinals = _keyframe_ordinals_in_range(
        frame_map,
        query.ground_truth_frame_range.start,
        query.ground_truth_frame_range.end,
    )
    entities: set[str] = set()
    for ordinal in ordinals:
        key = (video_id, ordinal)
        if key not in detection_cache:
            try:
                detection_cache[key] = _load_frame_entities(
                    objects_dir, video_id, ordinal, min_confidence
                )
            except (OSError, ValueError):
                detection_cache[key] = ()
        entities.update(detection_cache[key])
    return entities


def _analyze_one_coverage(
    query: EvaluationQuery,
    objects_dir: Path,
    map_dir: Path,
    entity_tokens: set[str],
    query_text_field: QueryTextField,
    min_confidence: float,
) -> QueryObjectCoverage:
    """Compute the coverage record for one query."""
    if query_text_field not in ("query_text_vi", "query_text_en"):
        raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
    if query_text_field == "query_text_en":
        if query.query_text_en is None:
            raise ValueError("query_text_en was requested but is missing for at least one query")
        query_text = query.query_text_en
    else:
        query_text = query.query_text_vi
    object_words = extract_object_words(query_text, entity_tokens)
    frame_map_cache: dict[str, dict[int, int]] = {}
    detection_cache: dict[tuple[str, int], tuple[str, ...]] = {}
    gt_entities = tuple(
        sorted(
            _gt_entity_set(
                query,
                Path(objects_dir),
                Path(map_dir),
                min_confidence,
                frame_map_cache,
                detection_cache,
            )
        )
    )
    matched = tuple(
        sorted(
            {
                entity
                for entity in gt_entities
                if set(_tokenize(entity.lower())).intersection(object_words)
            }
        )
    )
    return QueryObjectCoverage(
        query_id=query.query_id,
        query_text_en=query.query_text_en,
        query_text_vi=query.query_text_vi,
        object_words=object_words,
        gt_entities=gt_entities,
        matched_entities=matched,
        status="MATCH" if matched else "NO_MATCH",
    )


def _analyze_one_gt_frame(
    query: EvaluationQuery,
    objects_dir: Path,
    map_dir: Path,
    min_confidence: float,
) -> GtFrameObjectAnalysis:
    """Analyze object detections in one query's GT frame range."""
    video_id = query.ground_truth_video_id.value
    frame_map_cache: dict[str, dict[int, int]] = {}
    detection_cache: dict[tuple[str, int], tuple[str, ...]] = {}
    frame_map = frame_map_cache.get(video_id)
    if frame_map is None:
        try:
            frame_map = _load_frame_map(map_dir, video_id)
        except FileNotFoundError:
            frame_map = {}
    ordinals = _keyframe_ordinals_in_range(
        frame_map,
        query.ground_truth_frame_range.start,
        query.ground_truth_frame_range.end,
    )
    counter: Counter[str] = Counter()
    score_values: list[float] = []
    for ordinal in ordinals:
        key = (video_id, ordinal)
        if key not in detection_cache:
            try:
                detection_cache[key] = _load_frame_entities(
                    Path(objects_dir), video_id, ordinal, min_confidence
                )
            except (OSError, ValueError):
                detection_cache[key] = ()
        for entity in detection_cache[key]:
            counter[entity] += 1
        try:
            object_path = Path(objects_dir) / video_id / f"{ordinal:03d}.json"
            frame = parse_object_detections(object_path, video_id)
            detections = object_frame_to_detections(frame).detections
            score_values.extend(
                detection.score for detection in detections if detection.score >= min_confidence
            )
        except (OSError, ValueError):
            continue
    return GtFrameObjectAnalysis(
        query_id=query.query_id,
        gt_video_id=video_id,
        gt_keyframe_ordinals=ordinals,
        gt_object_list=tuple(_sorted_counter(counter, top=0)),
        confidence_stats=_percentile_stats(np.asarray(score_values, dtype=np.float32)),
        total_detections=sum(counter.values()),
    )


def _summarize_potential(
    records: Sequence[RetrievalPotentialQuery],
) -> RetrievalPotentialSummary:
    """Aggregate per-query retrieval-potential records."""
    total = len(records)
    with_gt = sum(1 for record in records if record.gt_frame_predictions > 0)
    separations = [record.separation for record in records if record.separation is not None]
    gt_rates = [record.gt_overlap_rate for record in records if record.gt_overlap_rate is not None]
    fp_rates = [
        record.false_positive_overlap_rate
        for record in records
        if record.false_positive_overlap_rate is not None
    ]
    return RetrievalPotentialSummary(
        total=total,
        queries_with_gt_predictions=with_gt,
        queries_with_separation=len(separations),
        queries_where_objects_help=sum(1 for value in separations if value > 0.0),
        avg_separation=sum(separations) / len(separations) if separations else None,
        mean_gt_overlap_rate=sum(gt_rates) / len(gt_rates) if gt_rates else None,
        mean_false_positive_overlap_rate=sum(fp_rates) / len(fp_rates) if fp_rates else None,
    )


def _fmt(value: float | None) -> str:
    """Format an optional float for the report."""
    return f"{value:.4f}" if value is not None else "—"
