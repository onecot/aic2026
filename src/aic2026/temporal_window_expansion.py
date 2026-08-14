"""Temporal window expansion as a retrieval post-processing stage.

Input is a ranked KIS prediction artifact plus the frame mapping of the index
artifact it references. For every prediction ``(video_id, frame_index)`` the
stage generates neighboring candidate frames at ``frame_index + offset`` for
every ``offset`` that is a multiple of ``step`` within ``[-window_size,
window_size]`` (the original frame, offset ``0``, is always included). Only
candidates whose ``(video_id, frame_index)`` actually exists in the index
frame mapping are kept; all other generated frames are dropped.

Scoring strategy:

- the original frame keeps its retrieval score unchanged;
- every propagated candidate receives ``original_score * decay``.

Candidates are then ranked deterministically: by score descending, with ties
resolved by generation order (input prediction order, then offset ascending),
and fresh strictly increasing ranks are assigned.

Two output contracts are provided:

- ``expand_*_predictions`` functions return a scorer-compatible
  ``{query_id: [KisPrediction, ...]}`` mapping ready for
  :func:`aic2026.kis_scoring.evaluate_kis`;
- :func:`build_expanded_artifact` /
  :func:`expand_prediction_artifact_to_artifact` build a full
  ``KisPredictionArtifact`` that preserves the source artifact metadata and
  carries the expanded ranked predictions (with their computed scores), so it
  can be persisted and reloaded with the standard prediction-artifact
  utilities.

This module does not modify embedding, index, retrieval, or scoring code; it
consumes an existing prediction artifact and produces scoring input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from aic2026.index_artifact import FRAMES_FILENAME, iter_frames
from aic2026.kis_prediction_runner import (
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    _utc_now_iso,
    read_kis_prediction_artifact,
    write_kis_prediction_artifact,
)
from aic2026.kis_scoring import KisPrediction


@dataclass(frozen=True)
class TemporalWindowConfig:
    """Configuration for temporal window expansion.

    Expected input:
        - window_size: maximum absolute offset, in original frame-index
          space, of a generated candidate around each retrieved frame.
        - step: spacing between candidate offsets; offsets are multiples of
          ``step`` within ``[-window_size, window_size]`` (the original frame
          is always included even when the window is not an exact multiple).
        - decay: score multiplier applied to propagated candidates; must be in
          ``(0, 1]`` (``1`` keeps propagated scores equal to the original).

    Returns:
        - A validated temporal window expansion configuration.

    Raises:
        - ValueError: if ``window_size`` or ``step`` is not a positive integer,
          or ``decay`` is not a number in ``(0, 1]``.
    """

    window_size: int
    step: int
    decay: float

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 1:
            raise ValueError("step must be a positive integer")
        if (
            not isinstance(self.decay, (int, float))
            or isinstance(self.decay, bool)
            or not (0 < self.decay <= 1)
        ):
            raise ValueError("decay must be a number in (0, 1]")


def expand_query_predictions(
    predictions: Sequence[RankedKisPrediction],
    existing_frames: Mapping[str, set[int]],
    config: TemporalWindowConfig,
) -> list[KisPrediction]:
    """Expand one query's ranked predictions with temporal window candidates.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - existing_frames: mapping from video id to the set of frame indexes
          that exist in the index for that video; candidates outside this
          mapping are dropped.
        - config: window size, candidate spacing, and decay factor.

    Returns:
        - All generated candidates as scorer-compatible :class:`KisPrediction`
          values: the original frame of every input prediction plus its
          propagated neighbors, ranked by score descending (ties resolved by
          input prediction order and then offset ascending) with fresh
          strictly increasing ranks. Scores are not part of the returned
          values; use :func:`expand_query_predictions_ranked` to keep them.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    return [
        prediction.to_kis_prediction()
        for prediction in _expand_ranked_predictions(predictions, existing_frames, config)
    ]


def expand_query_predictions_ranked(
    predictions: Sequence[RankedKisPrediction],
    existing_frames: Mapping[str, set[int]],
    config: TemporalWindowConfig,
) -> list[RankedKisPrediction]:
    """Expand one query's predictions, returning ranked predictions with scores.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - existing_frames: mapping from video id to the set of frame indexes
          that exist in the index for that video.
        - config: window size, candidate spacing, and decay factor.

    Returns:
        - All generated candidates as :class:`RankedKisPrediction` values
          carrying the computed candidate scores, ranked by score descending
          (ties resolved by input prediction order and then offset ascending)
          with fresh strictly increasing ranks.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    return _expand_ranked_predictions(predictions, existing_frames, config)


def expand_artifact_predictions(
    artifact: KisPredictionArtifact,
    config: TemporalWindowConfig,
    existing_frames: Mapping[str, set[int]],
) -> dict[str, list[KisPrediction]]:
    """Apply temporal window expansion to every query of a prediction artifact.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - config: window size, candidate spacing, and decay factor.
        - existing_frames: mapping from video id to the set of frame indexes
          that exist in the index for that video.

    Returns:
        - Mapping from query id to expanded scorer-compatible predictions, in
          the artifact's query order.

    Raises:
        - ValueError: if any query's predictions are not ranked predictions.
    """
    return {
        query_result.query_id: expand_query_predictions(
            query_result.predictions,
            existing_frames,
            config,
        )
        for query_result in artifact.query_results
    }


def expand_prediction_artifact(
    path: Path | str,
    config: TemporalWindowConfig,
    index_artifact_dir: Path | str | None = None,
) -> dict[str, list[KisPrediction]]:
    """Load a prediction artifact and return temporally expanded predictions.

    Expected input:
        - path: path to a ``predict-kis`` prediction artifact JSON.
        - config: window size, candidate spacing, and decay factor.
        - index_artifact_dir: directory of the index artifact whose frame
          mapping bounds the expansion; defaults to the artifact's recorded
          ``index_artifact_path``.

    Returns:
        - Mapping from query id to expanded scorer-compatible predictions,
          ready for :func:`aic2026.kis_scoring.evaluate_kis`.

    Raises:
        - FileNotFoundError: if ``path`` does not exist or the resolved index
          artifact has no ``frames.ndjson``.
        - ValueError: if the artifact payload violates the prediction schema
          or expansion cannot be applied.
    """
    artifact = read_kis_prediction_artifact(path)
    index_dir = Path(index_artifact_dir) if index_artifact_dir is not None else Path(artifact.index_artifact_path)
    existing_frames = load_existing_frames(index_dir)
    return expand_artifact_predictions(artifact, config, existing_frames)


def build_expanded_artifact(
    artifact: KisPredictionArtifact,
    config: TemporalWindowConfig,
    existing_frames: Mapping[str, set[int]],
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Build a new prediction artifact from expanded predictions.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - config: window size, candidate spacing, and decay factor.
        - existing_frames: mapping from video id to the set of frame indexes
          that exist in the index for that video.
        - run_id: optional explicit run id for the output artifact; defaults
          to a deterministic id derived from the source run id and the
          expansion configuration.

    Returns:
        - A new :class:`KisPredictionArtifact` that preserves the source
          artifact metadata (evaluation set, index reference, model metadata,
          query text field) and carries the expanded ranked predictions with
          their computed scores, so it remains loadable by the standard
          prediction-artifact utilities.

    Raises:
        - ValueError: if any query's predictions are not ranked predictions.
    """
    query_results = tuple(
        QueryPredictionResult(
            query_id=query_result.query_id,
            query_text=query_result.query_text,
            predictions=tuple(
                _expand_ranked_predictions(
                    query_result.predictions,
                    existing_frames,
                    config,
                )
            ),
        )
        for query_result in artifact.query_results
    )
    return KisPredictionArtifact(
        schema_version=artifact.schema_version,
        run_id=run_id or _expanded_run_id(artifact.run_id, config),
        created_at=_utc_now_iso(),
        evaluation_set_path=artifact.evaluation_set_path,
        evaluation_set_query_count=artifact.evaluation_set_query_count,
        processed_query_count=len(query_results),
        query_text_field=artifact.query_text_field,
        top_k=artifact.top_k,
        embedding_space_compatible=artifact.embedding_space_compatible,
        embedding_space_override_used=artifact.embedding_space_override_used,
        index_artifact_path=artifact.index_artifact_path,
        index_manifest=artifact.index_manifest,
        model_embedding_space=artifact.model_embedding_space,
        model_config=artifact.model_config,
        selected_query_ids=artifact.selected_query_ids,
        query_results=query_results,
    )


def expand_prediction_artifact_to_artifact(
    path: Path | str,
    config: TemporalWindowConfig,
    index_artifact_dir: Path | str | None = None,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Load a prediction artifact and build an expanded prediction artifact.

    Expected input:
        - path: path to a ``predict-kis`` prediction artifact JSON.
        - config: window size, candidate spacing, and decay factor.
        - index_artifact_dir: directory of the index artifact whose frame
          mapping bounds the expansion; defaults to the artifact's recorded
          ``index_artifact_path``.
        - run_id: optional explicit run id for the output artifact.

    Returns:
        - A new :class:`KisPredictionArtifact` with expanded predictions,
          preserving source metadata; ready for persistence with
          :func:`write_kis_prediction_artifact` and reloading with
          :func:`load_kis_predictions_for_scoring`.

    Raises:
        - FileNotFoundError: if ``path`` does not exist or the resolved index
          artifact has no ``frames.ndjson``.
        - ValueError: if the artifact payload violates the prediction schema
          or expansion cannot be applied.
    """
    artifact = read_kis_prediction_artifact(path)
    index_dir = Path(index_artifact_dir) if index_artifact_dir is not None else Path(artifact.index_artifact_path)
    existing_frames = load_existing_frames(index_dir)
    return build_expanded_artifact(artifact, config, existing_frames, run_id=run_id)


def write_expanded_prediction_artifact(
    output_path: Path | str,
    source_path: Path | str,
    config: TemporalWindowConfig,
    index_artifact_dir: Path | str | None = None,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Expand a prediction artifact and persist the expanded artifact atomically.

    Expected input:
        - output_path: target path for the expanded prediction artifact JSON.
        - source_path: path to the source ``predict-kis`` prediction artifact.
        - config: window size, candidate spacing, and decay factor.
        - index_artifact_dir: directory of the index artifact whose frame
          mapping bounds the expansion; defaults to the artifact's recorded
          ``index_artifact_path``.
        - run_id: optional explicit run id for the output artifact.

    Returns:
        - The expanded :class:`KisPredictionArtifact` that was written to
          ``output_path`` using the standard atomic artifact writer.

    Raises:
        - FileNotFoundError: if ``source_path`` does not exist or the resolved
          index artifact has no ``frames.ndjson``.
        - ValueError: if the artifact payload violates the prediction schema
          or expansion cannot be applied.
        - OSError: if writing fails.
    """
    artifact = expand_prediction_artifact_to_artifact(
        source_path,
        config,
        index_artifact_dir=index_artifact_dir,
        run_id=run_id,
    )
    write_kis_prediction_artifact(output_path, artifact)
    return artifact


def load_existing_frames(index_artifact_dir: Path | str) -> dict[str, set[int]]:
    """Return the set of indexed frame indexes for every video.

    Expected input:
        - index_artifact_dir: directory of one index artifact.

    Returns:
        - Mapping from video id to the set of frame indexes recorded in the
          artifact's ``frames.ndjson``.

    Raises:
        - FileNotFoundError: if the index artifact has no ``frames.ndjson``.
        - ValueError: if a frame record fails the canonical contract.
    """
    frames_path = Path(index_artifact_dir) / FRAMES_FILENAME
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"index artifact missing {FRAMES_FILENAME}: {frames_path}"
        )
    existing: dict[str, set[int]] = {}
    for timing in iter_frames(frames_path):
        video_id = timing.keyframe.video_id.value
        existing.setdefault(video_id, set()).add(timing.frame_index)
    return existing


def _window_offsets(window_size: int, step: int) -> tuple[int, ...]:
    """Return sorted candidate offsets, always including the original frame.

    Expected input:
        - window_size: maximum absolute offset.
        - step: spacing between offsets.

    Returns:
        - Tuple of ``k * step`` for integer ``k`` with ``|k * step| <=
          window_size``, in ascending order. Offset ``0`` is always present.

    Raises:
        - No exceptions.
    """
    limit = window_size // step
    return tuple(sorted({k * step for k in range(-limit, limit + 1)}))


def _expand_ranked_predictions(
    predictions: Sequence[RankedKisPrediction],
    existing_frames: Mapping[str, set[int]],
    config: TemporalWindowConfig,
) -> list[RankedKisPrediction]:
    """Generate expanded candidates and return them as ranked predictions.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores.
        - existing_frames: mapping from video id to the set of existing
          frame indexes.
        - config: window size, candidate spacing, and decay factor.

    Returns:
        - All generated candidates as :class:`RankedKisPrediction` values
          carrying the computed candidate scores, ranked by score descending
          (ties resolved by input prediction order and then offset ascending)
          with fresh strictly increasing ranks.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    _require_ranked_predictions(predictions)

    offsets = _window_offsets(config.window_size, config.step)
    candidates: list[tuple[float, str, int]] = []
    for prediction in predictions:
        video_frames = existing_frames.get(prediction.video_id, set())
        original_score = float(prediction.score)
        for offset in offsets:
            candidate_frame = prediction.frame_index + offset
            if offset != 0 and candidate_frame not in video_frames:
                continue
            score = original_score if offset == 0 else original_score * config.decay
            candidates.append((score, prediction.video_id, candidate_frame))

    ranked = sorted(
        enumerate(candidates),
        key=lambda entry: (-entry[1][0], entry[0]),
    )
    return [
        RankedKisPrediction(
            rank=rank,
            video_id=video_id,
            frame_index=frame_index,
            score=score,
        )
        for rank, (_index, (score, video_id, frame_index)) in enumerate(ranked, start=1)
    ]


def _expanded_run_id(source_run_id: str, config: TemporalWindowConfig) -> str:
    """Derive a deterministic run id for an expanded artifact."""
    return f"expand-{source_run_id}-w{config.window_size}-s{config.step}-d{config.decay:g}"


def _require_ranked_predictions(predictions: Sequence[RankedKisPrediction]) -> None:
    """Fail unless every prediction is a typed ranked prediction."""
    for prediction in predictions:
        if not isinstance(prediction, RankedKisPrediction):
            raise ValueError("predictions must contain only RankedKisPrediction values")
