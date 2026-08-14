"""Video-level reranking as a retrieval post-processing stage.

Consumes a ``predict-kis`` prediction artifact. For each query, predictions are
grouped by ``video_id``; every video receives a video-level score computed from
its frame candidates (either the ``"max"`` of the frame scores or the mean of
the top ``top_n`` frame scores). Videos are ranked by video score in descending
order, and the frame ranking is reconstructed as: videos ordered by video
score, with each video's frames ordered by their original CLIP score.

The output is a scorer-compatible ``{query_id: [KisPrediction, ...]}`` mapping
ready for :func:`aic2026.kis_scoring.evaluate_kis`.

This module does not modify the embedding pipeline, index builder, scorer, or
evaluation code; it consumes an existing prediction artifact and produces
scoring input.
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

#: Video-level aggregation methods implemented by this module.
_IMPLEMENTED_AGGREGATIONS: frozenset[str] = frozenset({"max", "mean_top_n"})

VideoAggregationMethod = Literal["max", "mean_top_n"]


@dataclass(frozen=True)
class VideoRerankingConfig:
    """Configuration for video-level reranking.

    Expected input:
        - aggregation: video-level aggregation of frame scores;
          ``"max"`` (default) or ``"mean_top_n"``.
        - top_n: number of highest-scoring frames used by ``"mean_top_n"``;
          videos with fewer frames are averaged over what they have.

    Returns:
        - A validated video reranking configuration.

    Raises:
        - ValueError: if ``aggregation`` is not implemented or ``top_n`` is
          not a positive integer.
    """

    aggregation: VideoAggregationMethod = "max"
    top_n: int = 5

    def __post_init__(self) -> None:
        if self.aggregation not in _IMPLEMENTED_AGGREGATIONS:
            raise ValueError(
                f"unsupported video aggregation {self.aggregation!r}; supported: 'max', 'mean_top_n'"
            )
        if not isinstance(self.top_n, int) or isinstance(self.top_n, bool) or self.top_n < 1:
            raise ValueError("top_n must be a positive integer")


def rerank_query_predictions(
    predictions: Sequence[RankedKisPrediction],
    config: VideoRerankingConfig = VideoRerankingConfig(),
) -> list[KisPrediction]:
    """Apply video-level reranking to one query's ranked predictions.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - config: video aggregation method and ``top_n`` for ``"mean_top_n"``.

    Returns:
        - One re-ranked :class:`KisPrediction` per input prediction. Videos are
          ordered by video score in descending order (stable for ties), frames
          inside each video are ordered by original CLIP score in descending
          order (stable for ties), and fresh strictly increasing ranks are
          assigned. ``video_id`` and ``frame_index`` are preserved.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    _require_ranked_predictions(predictions)

    by_video: dict[str, list[RankedKisPrediction]] = {}
    for prediction in predictions:
        by_video.setdefault(prediction.video_id, []).append(prediction)

    video_order: list[tuple[str, float]] = [
        (video_id, _video_score([float(item.score) for item in group], config))
        for video_id, group in by_video.items()
    ]
    video_order.sort(key=lambda entry: entry[1], reverse=True)

    ranked_frames: list[RankedKisPrediction] = []
    for video_id, _score in video_order:
        ordered = sorted(by_video[video_id], key=lambda item: item.score, reverse=True)
        ranked_frames.extend(ordered)

    return [
        KisPrediction(rank=rank, video_id=item.video_id, frame_index=item.frame_index)
        for rank, item in enumerate(ranked_frames, start=1)
    ]


def rerank_artifact_predictions(
    artifact: KisPredictionArtifact,
    config: VideoRerankingConfig = VideoRerankingConfig(),
) -> dict[str, list[KisPrediction]]:
    """Apply video-level reranking to every query of a prediction artifact.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - config: video aggregation method and ``top_n`` for ``"mean_top_n"``.

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
    config: VideoRerankingConfig = VideoRerankingConfig(),
) -> dict[str, list[KisPrediction]]:
    """Load a prediction artifact JSON and return video-level reranked predictions.

    Expected input:
        - path: path to a ``predict-kis`` prediction artifact JSON.
        - config: video aggregation method and ``top_n`` for ``"mean_top_n"``.

    Returns:
        - Mapping from query id to re-ranked scorer-compatible predictions,
          ready for :func:`aic2026.kis_scoring.evaluate_kis`.

    Raises:
        - FileNotFoundError: if ``path`` does not exist.
        - ValueError: if the artifact payload violates the prediction schema
          or reranking cannot be applied.
    """
    return rerank_artifact_predictions(read_kis_prediction_artifact(path), config)


def _video_score(
    frame_scores: Sequence[float],
    config: VideoRerankingConfig,
) -> float:
    """Return one video-level score from its frame candidate scores.

    Expected input:
        - frame_scores: retrieval scores of one video's frame candidates.
        - config: aggregation method and ``top_n`` for ``"mean_top_n"``.

    Returns:
        - The ``"max"`` frame score, or the mean of the top ``top_n`` frame
          scores (averaged over all frames when the video has fewer than
          ``top_n`` candidates).

    Raises:
        - ValueError: if ``frame_scores`` is empty.
    """
    if not frame_scores:
        raise ValueError("video score requires at least one frame score")
    if config.aggregation == "max":
        return max(frame_scores)
    ordered = sorted(frame_scores, reverse=True)
    count = min(config.top_n, len(ordered))
    return fsum(ordered[:count]) / count


def _require_ranked_predictions(predictions: Sequence[RankedKisPrediction]) -> None:
    """Fail unless every prediction is a typed ranked prediction."""
    for prediction in predictions:
        if not isinstance(prediction, RankedKisPrediction):
            raise ValueError("predictions must contain only RankedKisPrediction values")