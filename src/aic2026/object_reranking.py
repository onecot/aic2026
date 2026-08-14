"""Object-based reranking on top of existing KIS prediction artifacts.

This module re-ranks an existing ``predict-kis`` prediction artifact using BTC
OWL-ViT object detections. For every prediction it keeps the original CLIP
score, computes an object score from the objects detected in the predicted
frame, and combines them:

    final_score = clip_score + lambda_weight * object_score

Predictions are then re-ranked by ``final_score`` in descending order with
fresh strictly increasing ranks, preserving each prediction's ``video_id`` and
``frame_index``. Ties are resolved deterministically by the original ranking
order (stable sort).

The object score is query-dependent: query tokens are extracted and normalized,
then matched against the entity names detected in the predicted frame
(``detection_class_entities``). Only entities detected at or above a confidence
threshold participate. The object score is the maximum (or sum) confidence
among matched entities.

The output is a new :class:`KisPredictionArtifact` that preserves the source
artifact schema and metadata (evaluation set, index reference, model metadata,
query text field) while carrying re-ranked predictions, so it remains loadable
by the standard prediction-artifact utilities and consumable by the scorer.

This module does not modify retrieval, the scorer, evaluation, the embedding or
index pipeline, or the prediction-artifact schema.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from aic2026.converters import object_frame_to_detections
from aic2026.entities import parse_keyframe_map, parse_object_detections
from aic2026.kis_prediction_runner import (
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    read_kis_prediction_artifact,
    write_kis_prediction_artifact,
)

logger = logging.getLogger(__name__)

#: Default minimum confidence for an object detection to count toward the
#: frame-level object representation and object score.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

#: Default weight of the object score in the combined score.
DEFAULT_LAMBDA_WEIGHT = 1.0

#: Default query token normalization method.
DEFAULT_NORMALIZATION_METHOD = "lower"

#: Default object-score aggregation over matched entities.
DEFAULT_OBJECT_SCORE_METHOD = "max"

NormalizationMethod = Literal["none", "lower", "stem"]
ObjectScoreMethod = Literal["max", "sum"]

_TOKENIZE_RE = re.compile(r"\w+", re.UNICODE)

#: Suffix-stripping rules for the naive ``"stem"`` normalization.
_STEM_SUFFIXES = ("ing", "ed", "es", "s")


@dataclass(frozen=True)
class ObjectRerankingConfig:
    """Configuration for object-based reranking.

    Expected input:
        - confidence_threshold: only detections with ``score >=
          confidence_threshold`` count toward the frame representation and the
          object score.
        - lambda_weight: non-negative weight of the object score in
          ``final_score = clip_score + lambda_weight * object_score``.
        - normalization_method: query/entity token normalization:
          ``"none"`` (raw tokens), ``"lower"`` (lowercase), or ``"stem"``
          (lowercase plus naive suffix stripping).
        - object_score_method: how matched entities aggregate into the object
          score: ``"max"`` (maximum confidence among matched entities) or
          ``"sum"`` (sum of matched confidences).

    Returns:
        - A validated object-reranking configuration.

    Raises:
        - ValueError: if ``confidence_threshold`` is not in [0, 1],
          ``lambda_weight`` is negative, or a method is not one of the
          supported values.
    """

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT
    normalization_method: NormalizationMethod = DEFAULT_NORMALIZATION_METHOD
    object_score_method: ObjectScoreMethod = DEFAULT_OBJECT_SCORE_METHOD

    def __post_init__(self) -> None:
        if (
            not isinstance(self.confidence_threshold, (int, float))
            or isinstance(self.confidence_threshold, bool)
        ):
            raise ValueError("confidence_threshold must be numeric")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if (
            not isinstance(self.lambda_weight, (int, float))
            or isinstance(self.lambda_weight, bool)
            or self.lambda_weight < 0
        ):
            raise ValueError("lambda_weight must be a non-negative number")
        if self.normalization_method not in ("none", "lower", "stem"):
            raise ValueError("normalization_method must be 'none', 'lower', or 'stem'")
        if self.object_score_method not in ("max", "sum"):
            raise ValueError("object_score_method must be 'max' or 'sum'")


@dataclass(frozen=True)
class FrameObjects:
    """Frame-level object representation from one keyframe's detections.

    Expected input:
        - video_id: canonical video identifier.
        - keyframe_ordinal: keyframe ordinal (``None`` when the predicted
          frame is not a keyframe and has no object data).
        - entities: ``(entity, confidence)`` pairs sorted by confidence
          descending then entity, after the confidence threshold is applied
          and duplicate entities are merged by maximum confidence.

    Returns:
        - A validated frame object representation.

    Raises:
        - No exceptions.
    """

    video_id: str
    keyframe_ordinal: int | None
    entities: tuple[tuple[str, float], ...]

    def confidence(self, entity: str) -> float | None:
        """Return the confidence of one entity, or ``None`` when absent."""
        for name, score in self.entities:
            if name == entity:
                return score
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "video_id": self.video_id,
            "keyframe_ordinal": self.keyframe_ordinal,
            "entities": [
                {"entity": entity, "confidence": score} for entity, score in self.entities
            ],
        }


@dataclass
class FrameObjectSource:
    """Resolves original-video frame indices to frame-level object data.

    A prediction references an original-video ``frame_index``; object JSON
    files are keyed by keyframe ordinal. This source maps ``frame_index`` to a
    keyframe ordinal through the ``map-keyframes`` CSV and loads the matching
    object file, caching per-video frame maps.

    Expected input:
        - objects_dir: path to the raw ``objects`` collection directory.
        - map_dir: path to the ``map-keyframes`` collection directory;
          defaults to ``<objects_dir>/../map-keyframes``.

    Returns:
        - A validated frame object source.

    Raises:
        - FileNotFoundError: if a required collection directory is absent.
    """

    objects_dir: Path
    map_dir: Path | None = None
    _map_cache: dict[str, dict[int, int]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        directory = Path(self.objects_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"objects collection not found: {directory}")
        map_directory = (
            Path(self.map_dir) if self.map_dir is not None else directory.parent / "map-keyframes"
        )
        if not map_directory.is_dir():
            raise FileNotFoundError(f"map-keyframes collection not found: {map_directory}")
        self.objects_dir = directory
        self.map_dir = map_directory

    def frame_objects(
        self,
        video_id: str,
        frame_index: int,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> FrameObjects:
        """Return the object representation of one predicted frame.

        Expected input:
            - video_id: predicted canonical video identifier.
            - frame_index: predicted original-video frame index.
            - confidence_threshold: minimum score for entities to be retained.

        Returns:
            - :class:`FrameObjects` for the predicted frame. When the frame is
              not a keyframe, or its object file is unreadable, the returned
              representation has no entities rather than failing the rerank.

        Raises:
            - No exceptions.
        """
        frame_map = self._frame_map(video_id)
        ordinal = frame_map.get(frame_index)
        if ordinal is None:
            return FrameObjects(video_id=video_id, keyframe_ordinal=None, entities=())
        try:
            return load_frame_objects(self.objects_dir, video_id, ordinal, confidence_threshold)
        except (OSError, ValueError) as exc:
            logger.warning("skipping unreadable object frame %s kf %s: %s", video_id, ordinal, exc)
            return FrameObjects(video_id=video_id, keyframe_ordinal=ordinal, entities=())

    def _frame_map(self, video_id: str) -> dict[int, int]:
        """Return and cache ``frame_idx -> keyframe ordinal`` for one video."""
        if video_id not in self._map_cache:
            try:
                rows = parse_keyframe_map(self.map_dir / f"{video_id}.csv")
                self._map_cache[video_id] = {row.frame_idx: row.keyframe_no for row in rows}
            except (OSError, ValueError) as exc:
                logger.warning("skipping frame map for %s: %s", video_id, exc)
                self._map_cache[video_id] = {}
        return self._map_cache[video_id]


def load_frame_objects(
    objects_dir: Path | str,
    video_id: str,
    keyframe_ordinal: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> FrameObjects:
    """Load one keyframe's object file into a frame-level representation.

    Expected input:
        - objects_dir: path to the raw ``objects`` collection directory.
        - video_id: canonical video identifier.
        - keyframe_ordinal: 1-based keyframe ordinal (the ``NNN.json`` index).
        - confidence_threshold: minimum score for entities to be retained.

    Returns:
        - :class:`FrameObjects` with entities at or above the threshold,
          merged by entity (maximum confidence wins) and sorted by confidence
          descending then entity.

    Raises:
        - FileNotFoundError: if the object JSON does not exist.
        - ValueError: if the payload violates the object schema.
    """
    object_path = Path(objects_dir) / video_id / f"{keyframe_ordinal:03d}.json"
    frame = parse_object_detections(object_path, video_id)
    detections = object_frame_to_detections(frame).detections
    best: dict[str, float] = {}
    for detection in detections:
        if detection.score >= confidence_threshold:
            entity = detection.class_entity
            best[entity] = max(best.get(entity, 0.0), detection.score)
    entities = tuple(sorted(best.items(), key=lambda item: (-item[1], item[0])))
    return FrameObjects(video_id=video_id, keyframe_ordinal=keyframe_ordinal, entities=entities)


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into alphanumeric word tokens in order.

    Expected input:
        - text: text to tokenize.

    Returns:
        - Word tokens (underscore included; other punctuation and whitespace
          split tokens).

    Raises:
        - No exceptions.
    """
    return tuple(_TOKENIZE_RE.findall(text))


def _stem(token: str) -> str:
    """Naive deterministic suffix stripping for the ``"stem"`` method."""
    for suffix in _STEM_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def normalize_token(token: str, method: NormalizationMethod) -> str:
    """Normalize one token by the configured method.

    Expected input:
        - token: a single word token.
        - method: ``"none"``, ``"lower"``, or ``"stem"``.

    Returns:
        - The normalized token.

    Raises:
        - ValueError: if ``method`` is not supported.
    """
    if method == "none":
        return token
    lowered = token.lower()
    if method == "lower":
        return lowered
    if method == "stem":
        return _stem(lowered)
    raise ValueError(f"unsupported normalization method: {method!r}")


def normalize_tokens(text: str, method: NormalizationMethod) -> tuple[str, ...]:
    """Tokenize and normalize a piece of text.

    Expected input:
        - text: text to tokenize and normalize.
        - method: normalization method.

    Returns:
        - Unique normalized tokens in first-appearance order.

    Raises:
        - ValueError: if ``method`` is not supported.
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for token in tokenize(text):
        value = normalize_token(token, method)
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def match_entities(
    query_tokens: Sequence[str],
    frame: FrameObjects,
    normalization_method: NormalizationMethod = DEFAULT_NORMALIZATION_METHOD,
) -> tuple[str, ...]:
    """Return frame entities whose normalized tokens overlap query tokens.

    Expected input:
        - query_tokens: normalized query tokens.
        - frame: frame-level object representation.
        - normalization_method: normalization used to build ``query_tokens``;
          entity names are normalized with the same method before matching.

    Returns:
        - Matched entity names in ``frame`` order (confidence descending then
          entity).

    Raises:
        - No exceptions.
    """
    query_set = set(query_tokens)
    matched: list[str] = []
    for entity, _score in frame.entities:
        entity_tokens = {normalize_token(token, normalization_method) for token in tokenize(entity)}
        if entity_tokens.intersection(query_set):
            matched.append(entity)
    return tuple(matched)


def compute_object_score(
    query_tokens: Sequence[str],
    frame: FrameObjects,
    config: ObjectRerankingConfig,
) -> float:
    """Compute the object score for one predicted frame and query.

    Expected input:
        - query_tokens: normalized query tokens.
        - frame: frame-level object representation.
        - config: reranking configuration (confidence threshold is already
          applied when ``frame`` was loaded; ``object_score_method`` selects
          the aggregation).

    Returns:
        - ``0.0`` when nothing matches, otherwise the maximum (or sum) of the
          confidences of matched entities.

    Raises:
        - No exceptions.
    """
    matched = match_entities(query_tokens, frame, config.normalization_method)
    if not matched:
        return 0.0
    scores = [score for entity, score in frame.entities if entity in matched]
    if config.object_score_method == "max":
        return max(scores)
    return sum(scores)


def rerank_query_predictions(
    predictions: Sequence[RankedKisPrediction],
    query_text: str,
    object_source: FrameObjectSource,
    config: ObjectRerankingConfig,
) -> tuple[RankedKisPrediction, ...]:
    """Rerank one query's predictions with the object bonus.

    Expected input:
        - predictions: ranked predictions carrying retrieval scores, typically
          the ``predictions`` field of one artifact ``QueryPredictionResult``.
        - query_text: the query text used for object matching.
        - object_source: resolves predicted frames to object data.
        - config: reranking configuration.

    Returns:
        - One :class:`RankedKisPrediction` per input prediction with
          ``score = clip_score + lambda_weight * object_score``, ordered by
          combined score descending with fresh strictly increasing ranks;
          ``video_id`` and ``frame_index`` are preserved. Ties keep the input
          (original ranking) order.

    Raises:
        - ValueError: if ``predictions`` contains values that are not
          :class:`RankedKisPrediction`.
    """
    _require_ranked_predictions(predictions)
    query_tokens = normalize_tokens(query_text, config.normalization_method)

    combined: list[tuple[float, RankedKisPrediction]] = []
    for prediction in predictions:
        frame = object_source.frame_objects(
            prediction.video_id,
            prediction.frame_index,
            config.confidence_threshold,
        )
        object_score = compute_object_score(query_tokens, frame, config)
        final_score = float(prediction.score) + config.lambda_weight * object_score
        combined.append((final_score, prediction))

    combined.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(
        RankedKisPrediction(
            rank=rank,
            video_id=item.video_id,
            frame_index=item.frame_index,
            score=final_score,
        )
        for rank, (final_score, item) in enumerate(combined, start=1)
    )


def build_reranked_artifact(
    artifact: KisPredictionArtifact,
    object_source: FrameObjectSource,
    config: ObjectRerankingConfig,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Build a new artifact from object-reranked predictions.

    Expected input:
        - artifact: a validated KIS prediction artifact.
        - object_source: resolves predicted frames to object data.
        - config: reranking configuration.
        - run_id: optional explicit run id for the output artifact; defaults
          to a deterministic id derived from the source run id and the
          reranking configuration.

    Returns:
        - A new :class:`KisPredictionArtifact` that preserves the source
          artifact metadata (evaluation set, index reference, model metadata,
          query text field) and carries the re-ranked predictions with their
          combined scores, so it remains loadable by the standard
          prediction-artifact utilities.

    Raises:
        - ValueError: if any query's predictions are not ranked predictions.
    """
    query_results = tuple(
        QueryPredictionResult(
            query_id=query_result.query_id,
            query_text=query_result.query_text,
            predictions=rerank_query_predictions(
                query_result.predictions,
                query_result.query_text,
                object_source,
                config,
            ),
        )
        for query_result in artifact.query_results
    )
    return KisPredictionArtifact(
        schema_version=artifact.schema_version,
        run_id=run_id or _rerank_run_id(artifact.run_id, config),
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


def rerank_prediction_artifact(
    prediction_artifact: KisPredictionArtifact | Path | str,
    object_source: FrameObjectSource | Path | str,
    config: ObjectRerankingConfig,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Load a prediction artifact and return an object-reranked artifact.

    Expected input:
        - prediction_artifact: a validated :class:`KisPredictionArtifact`, or a
          path to a ``predict-kis`` prediction artifact JSON.
        - object_source: a :class:`FrameObjectSource`, or a path to the raw
          ``objects`` collection directory (a source is built with the default
          sibling ``map-keyframes`` directory).
        - config: reranking configuration.
        - run_id: optional explicit run id for the output artifact.

    Returns:
        - A new :class:`KisPredictionArtifact` with object-reranked
          predictions, preserving source metadata; ready for persistence with
          :func:`write_kis_prediction_artifact` and reloading with
          :func:`load_kis_predictions_for_scoring`.

    Raises:
        - FileNotFoundError: if an input path or collection directory is absent.
        - ValueError: if the artifact payload violates the prediction schema,
          the object source is invalid, or reranking cannot be applied.
    """
    artifact = (
        prediction_artifact
        if isinstance(prediction_artifact, KisPredictionArtifact)
        else read_kis_prediction_artifact(prediction_artifact)
    )
    source = (
        object_source
        if isinstance(object_source, FrameObjectSource)
        else FrameObjectSource(object_source)
    )
    return build_reranked_artifact(artifact, source, config, run_id=run_id)


def write_reranked_prediction_artifact(
    output_path: Path | str,
    source_path: Path | str,
    object_source: FrameObjectSource | Path | str,
    config: ObjectRerankingConfig,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Object-rerank a prediction artifact and persist it atomically.

    Expected input:
        - output_path: target path for the re-ranked prediction artifact JSON.
        - source_path: path to the source ``predict-kis`` prediction artifact.
        - object_source: a :class:`FrameObjectSource`, or a path to the raw
          ``objects`` collection directory.
        - config: reranking configuration.
        - run_id: optional explicit run id for the output artifact.

    Returns:
        - The re-ranked :class:`KisPredictionArtifact` that was written to
          ``output_path`` using the standard atomic artifact writer.

    Raises:
        - FileNotFoundError: if a source path or collection directory is absent.
        - ValueError: if the artifact payload or configuration is invalid.
        - OSError: if writing fails.
    """
    artifact = rerank_prediction_artifact(source_path, object_source, config, run_id=run_id)
    write_kis_prediction_artifact(output_path, artifact)
    return artifact


def _require_ranked_predictions(predictions: Sequence[RankedKisPrediction]) -> None:
    """Fail unless every prediction is a typed ranked prediction."""
    for prediction in predictions:
        if not isinstance(prediction, RankedKisPrediction):
            raise ValueError("predictions must contain only RankedKisPrediction values")


def _rerank_run_id(source_run_id: str, config: ObjectRerankingConfig) -> str:
    """Build a deterministic run id from the source run and configuration."""
    return (
        f"object-rerank-{source_run_id}-"
        f"l{config.lambda_weight:g}-c{config.confidence_threshold:g}-"
        f"n{config.normalization_method}-m{config.object_score_method}"
    )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
