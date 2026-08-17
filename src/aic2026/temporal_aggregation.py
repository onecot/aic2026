"""Temporal aggregation as a retrieval post-processing stage.

Input is the ranked KIS prediction artifact JSON produced by ``predict-kis``.
For each query, predictions are grouped by ``video_id`` and sorted by
``frame_index``; every prediction's score is then replaced by a sliding-window
mean over nearby frames **from the same video** (``window_size`` frames on
each side). Predictions are re-ranked by the aggregated score with fresh
strictly increasing ranks, preserving ``video_id`` and ``frame_index``.

The output is a scorer-compatible ``{query_id: [KisPrediction, ...]}`` mapping
ready for :func:`aic2026.kis_scoring.evaluate_kis`.

Only the ``"mean"`` aggregation method is implemented. The window size and the
aggregation method are configurable through
:class:`TemporalAggregationConfig`.

This module reuses the shared retrieval foundation
(:mod:`aic2026.retrieval_candidates`) for video grouping; it does not modify
embedding, index, retrieval, or scoring code; it consumes an existing
prediction artifact and produces scoring input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Literal

from aic2026.kis_prediction_runner import (
    KisPredictionArtifact,
    RankedKisPrediction,
    read_kis_prediction_artifact,
)
from aic2026.kis_scoring import KisPrediction
from aic2026.retrieval_candidates import group_candidates_by_video

#: Aggregation methods currently implemented by this module.
_IMPLEMENTED_METHODS: frozenset[str] = frozenset({"mean"})

TemporalAggregationMethod = Literal["mean"]


@dataclass(frozen=True)
class TemporalAggregationConfig:
    """Configuration for temporal aggregation post-processing.

    Expected input:
        - window_size: number of neighboring frames to include on **each side**
          of a frame; the window covers at most ``2 * window_size + 1`` frame
          positions in original frame-index space.
        - method: aggregation method; only ``"mean"`` is implemented.

    Returns:
        - A validated temporal aggregation configuration.

    Raises:
        - ValueError: if ``window_size`` is not a positive integer or
          ``method`` is not an implemented aggregation method.
    """

    window_size: int = 1
    method: TemporalAggregationMethod = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if self.method not in _IMPLEMENTED_METHODS:
            raise ValueError(f"unsupported temporal aggregation method {self.method!r}; supported: 'mean'")


def aggregate_query_predictions(
    predictions: Sequence[RankedKisPrediction],
    config: TemporalAggregationConfig = TemporalAggregationConfig(),
) -> list[KisPrediction]:
    """Apply temporal aggregation to one query's ranked predictions.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - config: window size and aggregation method.

    Returns:
        - One re-ranked :class:`KisPrediction` per input prediction, with
          fresh strictly increasing ranks; ``video_id`` and ``frame_index``
          are preserved. The rank order follows the aggregated score in
          descending order, using stable ordering for ties.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    _require_ranked_predictions(predictions)

    by_video = group_candidates_by_video(predictions)

    aggregated: list[tuple[str, int, float]] = []
    for video_id, group in by_video.items():
        ordered = sorted(group, key=lambda item: item.frame_index)
        window_scores = _mean_window_scores(
            [(item.frame_index, float(item.score)) for item in ordered],
            config.window_size,
        )
        aggregated.extend(
            (video_id, item.frame_index, window_score)
            for item, window_score in zip(ordered, window_scores)
        )

    aggregated.sort(key=lambda entry: entry[2], reverse=True)
    return [
        KisPrediction(rank=rank, video_id=video_id, frame_index=frame_index)
        for rank, (video_id, frame_index, _window_score) in enumerate(aggregated, start=1)
    ]


def aggregate_artifact_predictions(
    artifact: KisPredictionArtifact,
    config: TemporalAggregationConfig = TemporalAggregationConfig(),
) -> dict[str, list[KisPrediction]]:
    """Apply temporal aggregation to every query of a prediction artifact.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - config: window size and aggregation method.

    Returns:
        - Mapping from query id to re-ranked scorer-compatible predictions,
          in the artifact's query order.

    Raises:
        - ValueError: if any query's predictions are not ranked predictions.
    """
    return {
        query_result.query_id: aggregate_query_predictions(query_result.predictions, config)
        for query_result in artifact.query_results
    }


def aggregate_prediction_artifact(
    path: Path | str,
    config: TemporalAggregationConfig = TemporalAggregationConfig(),
) -> dict[str, list[KisPrediction]]:
    """Load a prediction artifact JSON and return temporally aggregated predictions.

    Expected input:
        - path: path to a ``predict-kis`` prediction artifact JSON.
        - config: window size and aggregation method.

    Returns:
        - Mapping from query id to re-ranked scorer-compatible predictions,
          ready for :func:`aic2026.kis_scoring.evaluate_kis`.

    Raises:
        - FileNotFoundError: if ``path`` does not exist.
        - ValueError: if the artifact payload violates the prediction schema
          or aggregation cannot be applied.
    """
    return aggregate_artifact_predictions(read_kis_prediction_artifact(path), config)


def _mean_window_scores(
    sorted_frames: Sequence[tuple[int, float]],
    window_size: int,
) -> list[float]:
    """Return one sliding-window mean score per sorted frame.

    Expected input:
        - sorted_frames: ``(frame_index, score)`` pairs sorted by
          ``frame_index`` within a single video.
        - window_size: number of neighboring frames to include on each side.

    Returns:
        - A window-mean score per frame in the input order, computed over all
          frames whose frame index is within ``window_size`` frames of the
          current frame.

    Raises:
        - No exceptions.
    """
    count = len(sorted_frames)
    if count == 0:
        return []
    scores = [score for _, score in sorted_frames]
    window_scores: list[float] = []
    for current in range(count):
        low = current
        while low > 0 and sorted_frames[current][0] - sorted_frames[low - 1][0] <= window_size:
            low -= 1
        high = current
        while high < count - 1 and sorted_frames[high + 1][0] - sorted_frames[current][0] <= window_size:
            high += 1
        # fsum keeps mathematically equal window means bit-identical so that
        # the downstream stable re-ranking is deterministic regardless of the
        # order in which scores were accumulated.
        window_scores.append(fsum(scores[low : high + 1]) / (high - low + 1))
    return window_scores


def _require_ranked_predictions(predictions: Sequence[RankedKisPrediction]) -> None:
    """Fail unless every prediction is a typed ranked prediction."""
    for prediction in predictions:
        if not isinstance(prediction, RankedKisPrediction):
            raise ValueError("predictions must contain only RankedKisPrediction values")
