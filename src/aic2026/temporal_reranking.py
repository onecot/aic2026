"""Temporal bonus reranking as a retrieval post-processing method.

Consumes a ``predict-kis`` prediction artifact. For each query, predictions are
grouped by ``video_id`` and sorted by ``frame_index``; every prediction gains a
temporal bonus proportional to the number of neighboring predictions **from the
same video** within ``window_size`` frames (the current frame itself excluded):

    new_score = original_score + lambda_weight * temporal_support

Predictions are re-ranked by ``new_score`` in descending order with fresh
strictly increasing ranks, preserving ``video_id`` and ``frame_index`` and each
prediction's original identity.

The output is a scorer-compatible ``{query_id: [KisPrediction, ...]}`` mapping
ready for :func:`aic2026.kis_scoring.evaluate_kis`.

Only the temporal bonus method is implemented here; it does not modify the
embedding pipeline, index builder, scorer, evaluation code, or the existing
temporal mean aggregation module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aic2026.kis_prediction_runner import (
    KisPredictionArtifact,
    RankedKisPrediction,
    read_kis_prediction_artifact,
)
from aic2026.kis_scoring import KisPrediction


@dataclass(frozen=True)
class TemporalBonusConfig:
    """Configuration for temporal bonus reranking.

    Expected input:
        - window_size: number of neighboring frames on each side, in original
          frame-index space, that count toward temporal support.
        - lambda_weight: non-negative weight applied to temporal support when
          computing the boosted score.

    Returns:
        - A validated temporal bonus configuration.

    Raises:
        - ValueError: if ``window_size`` is not a positive integer or
          ``lambda_weight`` is not a non-negative number.
    """

    window_size: int
    lambda_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if (
            not isinstance(self.lambda_weight, (int, float))
            or isinstance(self.lambda_weight, bool)
            or self.lambda_weight < 0
        ):
            raise ValueError("lambda_weight must be a non-negative number")


def rerank_query_predictions(
    predictions: Sequence[RankedKisPrediction],
    config: TemporalBonusConfig,
) -> list[KisPrediction]:
    """Rerank one query's ranked predictions with a temporal bonus.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - config: window size and temporal bonus weight.

    Returns:
        - One re-ranked :class:`KisPrediction` per input prediction, ordered by
          boosted score in descending order with fresh strictly increasing
          ranks; ``video_id`` and ``frame_index`` are preserved. Ties are
          resolved by the input (original ranking) order.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    _require_ranked_predictions(predictions)

    by_video: dict[str, list[RankedKisPrediction]] = {}
    for prediction in predictions:
        by_video.setdefault(prediction.video_id, []).append(prediction)

    boosted: list[tuple[float, RankedKisPrediction]] = []
    for video_id, group in by_video.items():
        ordered = sorted(group, key=lambda item: item.frame_index)
        support = _temporal_support([item.frame_index for item in ordered], config.window_size)
        for item, neighbors in zip(ordered, support):
            boosted.append((float(item.score) + config.lambda_weight * neighbors, item))

    boosted.sort(key=lambda entry: entry[0], reverse=True)
    return [
        KisPrediction(rank=rank, video_id=item.video_id, frame_index=item.frame_index)
        for rank, (_score, item) in enumerate(boosted, start=1)
    ]


def rerank_artifact_predictions(
    artifact: KisPredictionArtifact,
    config: TemporalBonusConfig,
) -> dict[str, list[KisPrediction]]:
    """Apply temporal bonus reranking to every query of a prediction artifact.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - config: window size and temporal bonus weight.

    Returns:
        - Mapping from query id to re-ranked scorer-compatible predictions, in
          the artifact's query order.

    Raises:
        - ValueError: if any query's predictions are not ranked predictions.
    """
    return {
        query_result.query_id: rerank_query_predictions(query_result.predictions, config)
        for query_result in artifact.query_results
    }


def rerank_prediction_artifact(
    path: Path | str,
    config: TemporalBonusConfig,
) -> dict[str, list[KisPrediction]]:
    """Load a prediction artifact JSON and return temporally reranked predictions.

    Expected input:
        - path: path to a ``predict-kis`` prediction artifact JSON.
        - config: window size and temporal bonus weight.

    Returns:
        - Mapping from query id to re-ranked scorer-compatible predictions,
          ready for :func:`aic2026.kis_scoring.evaluate_kis`.

    Raises:
        - FileNotFoundError: if ``path`` does not exist.
        - ValueError: if the artifact payload violates the prediction schema
          or reranking cannot be applied.
    """
    return rerank_artifact_predictions(read_kis_prediction_artifact(path), config)


def _temporal_support(
    sorted_frames: Sequence[int],
    window_size: int,
) -> list[int]:
    """Return per-frame neighbor counts within the window, excluding each frame itself.

    Expected input:
        - sorted_frames: frame indexes of one video, sorted ascending.
        - window_size: number of neighboring frames to include on each side.

    Returns:
        - One neighbor count per frame in the input order: how many other
          frames have a frame index within ``window_size`` of the current one.

    Raises:
        - No exceptions.
    """
    count = len(sorted_frames)
    if count == 0:
        return []
    support: list[int] = []
    for current in range(count):
        low = current
        while low > 0 and sorted_frames[current] - sorted_frames[low - 1] <= window_size:
            low -= 1
        high = current
        while high < count - 1 and sorted_frames[high + 1] - sorted_frames[current] <= window_size:
            high += 1
        support.append(high - low)
    return support


def _require_ranked_predictions(predictions: Sequence[RankedKisPrediction]) -> None:
    """Fail unless every prediction is a typed ranked prediction."""
    for prediction in predictions:
        if not isinstance(prediction, RankedKisPrediction):
            raise ValueError("predictions must contain only RankedKisPrediction values")
